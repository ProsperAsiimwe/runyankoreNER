import os
import argparse
import random
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt

# --------------------
# Config
# --------------------

MODEL_NAME_MAP = {
    "xlmr": "xlm-roberta-base",
    "mbert": "bert-base-multilingual-cased",
}
ENTITY_TAGS = {"PER", "LOC", "ORG", "DATE"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_LANG = "run"

# Logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"experiment_{timestamp}.log")
logging.basicConfig(
    filename=log_file,
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# --------------------
# Reproducibility + math helpers
# --------------------

def set_seed(s: int = 42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def _safe_pearson(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    """
    Pearson correlation that tolerates NaNs and near-constant vectors.
    Returns NaN if correlation is undefined.
    """
    if a is None or b is None:
        return np.nan
    if np.isnan(a).any() or np.isnan(b).any():
        return np.nan
    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    a_mean = a.mean()
    b_mean = b.mean()
    a_center = a - a_mean
    b_center = b - b_mean
    num = np.dot(a_center, b_center)
    den = np.linalg.norm(a_center) * np.linalg.norm(b_center)
    if den < eps:
        return np.nan
    return float(num / den)

# --------------------
# Data reading
# --------------------

def _read_conll_sentences(file_path: str):
    """Yield (tokens, tags) per sentence from a CoNLL-like file."""
    with open(file_path, "r", encoding="utf-8") as f:
        tokens, tags = [], []
        for raw in f:
            line = raw.strip()
            if not line:
                if tokens:
                    yield tokens, tags
                tokens, tags = [], []
            else:
                parts = line.split()
                if len(parts) == 2:
                    tok, tag = parts
                    tokens.append(tok)
                    tags.append(tag)
        if tokens:
            yield tokens, tags

# --------------------
# Embeddings (batched, multi-layer)
# --------------------

def extract_entity_embeddings_batched_multilayer(
    file_path: str,
    tokenizer,
    model,
    layer_indices: list[int],
    entity_tags=ENTITY_TAGS,
    max_tokens_per_type: int = 500,
    use_cls: bool = False,
    use_context: bool = False,
    context_window: int = 2,
    batch_size: int = 64,
    save_examples_path: str | None = None,
):
    """
    Build embeddings for entity tokens at multiple layers (given hidden_states indices).
    For non-CLS mode, mean-pools subwords of the entity token. For CLS mode, takes [CLS].
    Returns:
        per_layer_embeddings: dict[layer_idx][tag] -> list[np.ndarray]
        per_tag_counts: dict[tag] -> count (based on last processed layer; used for logging only)
    """
    assert getattr(tokenizer, "is_fast", False), "Fast tokenizer required (word_ids)."

    # Initialize buffers for each layer and tag
    per_layer_embeddings = {li: {tag: [] for tag in entity_tags} for li in layer_indices}
    per_tag_counts = {tag: 0 for tag in entity_tags}
    per_tag_examples = {tag: [] for tag in entity_tags} if save_examples_path else None

    # Batch buffers
    batch_payload = []   # dict: span_tokens, rel_idx, tag
    def _flush_batch():
        nonlocal batch_payload
        if not batch_payload:
            return

        if use_cls:
            texts = [" ".join(item["span_tokens"]) for item in batch_payload]
            tokenized = tokenizer(
                texts,
                is_split_into_words=False,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True,
            ).to(DEVICE)
        else:
            spans = [item["span_tokens"] for item in batch_payload]
            tokenized = tokenizer(
                spans,
                is_split_into_words=True,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True,
            ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**tokenized, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple len = L+1; index 0=embeddings, 1..L=layers

        # For each selected layer, compute pooled embeddings and distribute
        B = tokenized.input_ids.shape[0]
        for layer_idx in layer_indices:
            layer_tensor = hidden_states[layer_idx]  # (B, T, H)

            if use_cls:
                # Take position 0
                cls_embs = layer_tensor[:, 0, :].detach().cpu().numpy()  # (B, H)
                for i, item in enumerate(batch_payload):
                    tag = item["tag"]
                    if per_tag_counts[tag] < max_tokens_per_type:
                        per_layer_embeddings[layer_idx][tag].append(cls_embs[i])
                        per_tag_counts[tag] += 1
                        if per_tag_examples is not None:
                            per_tag_examples[tag].append(" ".join(item["span_tokens"]))
            else:
                # Mean-pool subwords of the entity word
                for i, item in enumerate(batch_payload):
                    tag = item["tag"]
                    if per_tag_counts[tag] >= max_tokens_per_type:
                        continue
                    word_ids = tokenized.word_ids(batch_index=i)
                    rel_idx = item["rel_idx"]
                    sub_idx = [j for j, wid in enumerate(word_ids) if wid == rel_idx]
                    if not sub_idx:
                        # fallback: first non-special
                        sub_idx = [j for j, wid in enumerate(word_ids) if wid is not None][:1]
                    if not sub_idx:
                        continue
                    emb = layer_tensor[i, sub_idx, :].mean(dim=0).detach().cpu().numpy()
                    per_layer_embeddings[layer_idx][tag].append(emb)
                    per_tag_counts[tag] += 1
                    if per_tag_examples is not None:
                        per_tag_examples[tag].append(" ".join(item["span_tokens"]))

        batch_payload = []

    # Iterate sentences and collect spans
    for tokens, tags in tqdm(_read_conll_sentences(file_path), desc=f"Reading {os.path.basename(file_path)}"):
        for idx, tag in enumerate(tags):
            stripped_tag = tag.replace("B-", "").replace("I-", "")
            if stripped_tag not in entity_tags:
                continue
            if per_tag_counts[stripped_tag] >= max_tokens_per_type:
                continue

            start = max(0, idx - context_window) if use_context else idx
            end = min(len(tokens), idx + context_window + 1) if use_context else idx + 1
            span = tokens[start:end]
            if not span:
                continue
            rel_idx = idx - start

            batch_payload.append({"span_tokens": span, "rel_idx": rel_idx, "tag": stripped_tag})

            if len(batch_payload) >= batch_size:
                _flush_batch()
    _flush_batch()

    # Save examples (only one copy per tag, not per layer)
    if save_examples_path and per_tag_examples is not None:
        os.makedirs(save_examples_path, exist_ok=True)
        for tag in per_tag_examples:
            with open(os.path.join(save_examples_path, f"{tag}_examples.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(per_tag_examples[tag]))

    return per_layer_embeddings, per_tag_counts

# --------------------
# Prototypes per layer
# --------------------

def compute_language_prototypes_multilayer(
    language_files: dict,
    model_type: str,
    layer_indices: list[int],
    max_tokens_per_type: int = 500,
    output_dir: str = "outputs_prototypes",
    use_cls: bool = False,
    use_context: bool = False,
    context_window: int = 2,
    batch_size: int = 64,
):
    model_name = MODEL_NAME_MAP[model_type]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE)
    model.eval()
    assert getattr(tokenizer, "is_fast", False), "Fast tokenizer required (word_ids)."

    os.makedirs(output_dir, exist_ok=True)

    # Structure: all_prototypes[layer_idx][lang][tag] = vector
    all_prototypes = {li: {} for li in layer_indices}
    all_counts = {}

    for lang, path in tqdm(language_files.items(), desc="Computing prototypes (multi-layer)"):
        logging.info(f"Processing {lang}")
        example_path = os.path.join(output_dir, f"{lang}_examples")

        per_layer_embs, counts_by_tag = extract_entity_embeddings_batched_multilayer(
            path,
            tokenizer,
            model,
            layer_indices=layer_indices,
            entity_tags=ENTITY_TAGS,
            max_tokens_per_type=max_tokens_per_type,
            use_cls=use_cls,
            use_context=use_context,
            context_window=context_window,
            batch_size=batch_size,
            save_examples_path=example_path,
        )

        all_counts[lang] = counts_by_tag

        # Compute per-layer prototypes
        H = model.config.hidden_size
        for li in layer_indices:
            tag2embs = per_layer_embs[li]
            prototypes = {}
            for tag, embs in tag2embs.items():
                if len(embs) == 0:
                    prototypes[tag] = np.full((H,), np.nan, dtype=np.float32)
                else:
                    arr = np.stack(embs).astype(np.float32)
                    prototypes[tag] = arr.mean(axis=0)
            # Save language prototypes for this layer
            if lang not in all_prototypes[li]:
                all_prototypes[li][lang] = {}
            all_prototypes[li][lang] = prototypes

        # Optionally save per-language per-layer .npz files
        for li in layer_indices:
            np.savez(os.path.join(output_dir, f"{lang}_prototypes_layer{li}_{model_type}.npz"), **all_prototypes[li][lang])

    # Save counts
    pd.DataFrame.from_dict(all_counts, orient="index").to_csv(os.path.join(output_dir, "prototype_counts.csv"))
    return all_prototypes, model

# --------------------
# Correlations (Runyankore vs others) per layer
# --------------------

def compute_correlations_to_target_multilayer(
    all_prototypes_by_layer: dict,
    target_lang: str = TARGET_LANG,
    output_dir: str = "outputs_prototypes",
    use_hybrid: bool = False,  # kept for API symmetry; Pearson is primary metric here
):
    """
    For each layer, compute Pearson r between target_lang and every other language,
    tag-wise then averaged over available tags (skipping NaNs).
    Returns:
        corr_tables: dict[layer_idx] -> DataFrame (index=language, columns=[tags..., 'Mean'])
    """
    os.makedirs(output_dir, exist_ok=True)
    corr_tables = {}

    for li, lang2protos in all_prototypes_by_layer.items():
        tgt = lang2protos[target_lang]
        rows = {}
        for lang, protos in lang2protos.items():
            if lang == target_lang:
                continue
            scores = {}
            vals = []
            for tag in ENTITY_TAGS:
                r = _safe_pearson(np.array(tgt[tag]), np.array(protos[tag]))
                scores[tag] = r
                if not np.isnan(r):
                    vals.append(r)
            scores["Mean"] = float(np.mean(vals)) if vals else np.nan
            rows[lang] = scores

        df = pd.DataFrame.from_dict(rows, orient="index")
        df = df.sort_values("Mean", ascending=False, na_position="last")
        out_csv = os.path.join(output_dir, f"pearson_to_{target_lang}_layer{li}.csv")
        df.to_csv(out_csv)
        corr_tables[li] = df

    return corr_tables

# --------------------
# Visualization
# --------------------

def _save_sorted_bar(df: pd.DataFrame, layer_idx: int, output_dir: str, title_prefix: str):
    """Full bar chart for all languages at one layer (Part 2)."""
    vals = df["Mean"].fillna(0.0)
    order = vals.sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    plt.bar(order.index, order.values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Pearson r (mean over tags)")
    plt.title(f"{title_prefix} — Layer {layer_idx}")
    plt.tight_layout()
    fname = os.path.join(output_dir, f"{title_prefix.lower().replace(' ', '_')}_layer{layer_idx}.png")
    plt.savefig(fname, dpi=200)
    plt.close()

def _save_heatmap(language_order, layer_order, mat, output_path: str, title: str):
    """Heatmap: languages × layers (Part 1)."""
    plt.figure(figsize=(1.2 + 0.5*len(layer_order), 0.6 + 0.4*len(language_order)))
    im = plt.imshow(mat, aspect="auto", interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.yticks(range(len(language_order)), language_order)
    plt.xticks(range(len(layer_order)), layer_order, rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def create_visualizations_for_corr_tables(
    corr_tables: dict[int, pd.DataFrame],
    model,
    output_dir: str,
    model_type: str,
):
    """
    - Heatmap over selected layers (languages × layers) using Mean Pearson r
    - Full bar chart for each layer (all languages)
    - Print + save top 5 and worst 5 per layer
    """
    os.makedirs(output_dir, exist_ok=True)

    # Ensure a consistent language order based on the output layer (last chosen layer)
    num_layers_total = model.config.num_hidden_layers
    output_layer_idx = num_layers_total  # we always include this
    if output_layer_idx not in corr_tables:
        # fall back to the last available layer in dict order
        output_layer_idx = sorted(corr_tables.keys())[-1]

    base_df = corr_tables[output_layer_idx]
    language_order = list(base_df.index)
    layer_order = sorted(corr_tables.keys())

    # Heatmap matrix: rows=languages, cols=layers
    mat = np.zeros((len(language_order), len(layer_order)), dtype=float)
    mat[:] = np.nan
    for j, li in enumerate(layer_order):
        df = corr_tables[li].reindex(language_order)
        mat[:, j] = df["Mean"].values

    # Save heatmap
    heatmap_path = os.path.join(output_dir, f"heatmap_pearson_layers_{model_type}.png")
    _save_heatmap(language_order, layer_order, mat, heatmap_path,
                  f"Runyankore similarity (Pearson r) across layers — {model_type}")

    # Full bar charts + summaries
    summary_rows = []
    for li in layer_order:
        df = corr_tables[li]
        _save_sorted_bar(df, li, output_dir, f"Pearson correlations vs run ({model_type})")

        # Top 5 and worst 5 (drop NaNs first)
        clean = df["Mean"].dropna()
        top5 = clean.sort_values(ascending=False).head(5)
        bot5 = clean.sort_values(ascending=True).head(5)

        # Print summary to console and log
        print(f"\n=== Layer {li} ({model_type}) — Top 5 ===")
        logging.info(f"Layer {li} ({model_type}) — Top 5")
        for rank, (lang, val) in enumerate(top5.items(), 1):
            print(f"{rank}. {lang}: r={val:.4f}")
            logging.info(f"{rank}. {lang}: r={val:.4f}")

        print(f"=== Layer {li} ({model_type}) — Worst 5 ===")
        logging.info(f"Layer {li} ({model_type}) — Worst 5")
        for rank, (lang, val) in enumerate(bot5.items(), 1):
            print(f"{rank}. {lang}: r={val:.4f}")
            logging.info(f"{rank}. {lang}: r={val:.4f}")

        # stash for a CSV summary
        for lang, val in top5.items():
            summary_rows.append({"layer": li, "model": model_type, "rank_type": "top5", "language": lang, "pearson": val})
        for lang, val in bot5.items():
            summary_rows.append({"layer": li, "model": model_type, "rank_type": "worst5", "language": lang, "pearson": val})

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(os.path.join(output_dir, f"summary_top_worst_{model_type}.csv"), index=False)

# --------------------
# Pipeline
# --------------------

def pick_random_middle_layers(model, num_middle_layers: int, seed: int):
    """
    Choose num_middle_layers from 1..L-1 (exclude 0=embeddings and L=final),
    and always include L (final/output layer).
    """
    L = model.config.num_hidden_layers
    candidates = list(range(1, L))  # 1..L-1
    rng = random.Random(seed)
    chosen = rng.sample(candidates, k=min(num_middle_layers, len(candidates)))
    chosen = sorted(set(chosen + [L]))  # always include final layer L
    return chosen  # these are indices into outputs.hidden_states

def run_one_model(
    model_type: str,
    language_files: dict,
    args,
):
    model_name = MODEL_NAME_MAP[model_type]
    # Build a temp model to read num_hidden_layers
    tmp_model = AutoModel.from_pretrained(model_name)
    layer_indices = pick_random_middle_layers(tmp_model, args.num_middle_layers, args.seed)
    del tmp_model

    logging.info(f"{model_type}: selected layer indices (hidden_states) = {layer_indices}")
    print(f"{model_type}: selected layer indices (hidden_states) = {layer_indices}")

    # Compute prototypes at all selected layers
    all_prototypes_by_layer, model = compute_language_prototypes_multilayer(
        language_files=language_files,
        model_type=model_type,
        layer_indices=layer_indices,
        max_tokens_per_type=args.max_tokens_per_type,
        output_dir=args.output_dir,
        use_cls=args.use_cls,
        use_context=args.use_context,
        context_window=args.context_window,
        batch_size=args.batch_size,
    )

    # Correlations vs run at each layer
    corr_tables = compute_correlations_to_target_multilayer(
        all_prototypes_by_layer,
        target_lang=TARGET_LANG,
        output_dir=args.output_dir,
        use_hybrid=False,  # Pearson primary
    )

    # Visualizations + summaries
    create_visualizations_for_corr_tables(
        corr_tables=corr_tables,
        model=model,
        output_dir=args.output_dir,
        model_type=model_type,
    )

# --------------------
# Main
# --------------------

def main():
    parser = argparse.ArgumentParser(description="Layer-wise correlation visualizations for language similarity (Runyankore target)")
    parser.add_argument("--model_type", type=str, default="xlmr", choices=["xlmr", "mbert"], help="Model to run. Use --run_both to run both.")
    parser.add_argument("--run_both", action="store_true", help="Run both models (xlmr and mbert).")
    parser.add_argument("--use_cls", action="store_true", help="Use CLS token instead of subword pooling")
    parser.add_argument("--use_context", action="store_true", help="Use context window around entity")
    parser.add_argument("--use_hybrid", action="store_true", help="(unused for plots) kept for API symmetry")
    parser.add_argument("--context_window", type=int, default=2, help="Size of context window")
    parser.add_argument("--max_tokens_per_type", type=int, default=500, help="Max entity tokens per tag")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (also used for layer sampling)")
    parser.add_argument("--num_middle_layers", type=int, default=3, help="How many random middle layers to sample per model (final layer always included)")
    parser.add_argument("--output_dir", type=str, default="outputs_prototypes")
    args = parser.parse_args()

    set_seed(args.seed)

    logging.info(
        f"Start: model_type={args.model_type}, run_both={args.run_both}, use_cls={args.use_cls}, "
        f"use_context={args.use_context}, context_window={args.context_window}, "
        f"max_tokens_per_type={args.max_tokens_per_type}, batch_size={args.batch_size}, "
        f"seed={args.seed}, num_middle_layers={args.num_middle_layers}, output_dir={args.output_dir}"
    )

    # Data paths
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data"))
    language_files = {
        "run": os.path.join(BASE_DIR, "MPTC/train.txt"),
        "bam": os.path.join(BASE_DIR, "MasakhaNER2.0/bam/train.txt"),
        "bbj": os.path.join(BASE_DIR, "MasakhaNER2.0/bbj/train.txt"),
        "ewe": os.path.join(BASE_DIR, "MasakhaNER2.0/ewe/train.txt"),
        "fon": os.path.join(BASE_DIR, "MasakhaNER2.0/fon/train.txt"),
        "hau": os.path.join(BASE_DIR, "MasakhaNER2.0/hau/train.txt"),
        "ibo": os.path.join(BASE_DIR, "MasakhaNER2.0/ibo/train.txt"),
        "kin": os.path.join(BASE_DIR, "MasakhaNER2.0/kin/train.txt"),
        "lug": os.path.join(BASE_DIR, "MasakhaNER2.0/lug/train.txt"),
        "luo": os.path.join(BASE_DIR, "MasakhaNER2.0/luo/train.txt"),
        "mos": os.path.join(BASE_DIR, "MasakhaNER2.0/mos/train.txt"),
        "nya": os.path.join(BASE_DIR, "MasakhaNER2.0/nya/train.txt"),
        "pcm": os.path.join(BASE_DIR, "MasakhaNER2.0/pcm/train.txt"),
        "sna": os.path.join(BASE_DIR, "MasakhaNER2.0/sna/train.txt"),
        "swa": os.path.join(BASE_DIR, "MasakhaNER2.0/swa/train.txt"),
        "tsn": os.path.join(BASE_DIR, "MasakhaNER2.0/tsn/train.txt"),
        "twi": os.path.join(BASE_DIR, "MasakhaNER2.0/twi/train.txt"),
        "wol": os.path.join(BASE_DIR, "MasakhaNER2.0/wol/train.txt"),
        "xho": os.path.join(BASE_DIR, "MasakhaNER2.0/xho/train.txt"),
        "yor": os.path.join(BASE_DIR, "MasakhaNER2.0/yor/train.txt"),
        "zul": os.path.join(BASE_DIR, "MasakhaNER2.0/zul/train.txt"),
    }

    # Run one or both models
    model_list = ["xlmr", "mbert"] if args.run_both else [args.model_type]
    for mt in model_list:
        run_one_model(mt, language_files, args)

    logging.info("Experiment finished successfully.")

if __name__ == "__main__":
    main()
