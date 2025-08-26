import os
import argparse
import random
import logging
from datetime import datetime
from itertools import product

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

def _safe_cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    """
    Cosine similarity that tolerates NaNs and near-zero norms.
    Returns NaN if similarity is undefined.
    """
    if a is None or b is None:
        return np.nan
    if np.isnan(a).any() or np.isnan(b).any():
        return np.nan
    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den < eps:
        return np.nan
    return float(np.dot(a, b) / den)

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

    for lang, path in tqdm(language_files.items(), desc=f"Computing prototypes (multi-layer) — {model_type}"):
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
# Cosine similarities (Runyankore vs others) per layer
# --------------------

def compute_cosine_to_target_multilayer(
    all_prototypes_by_layer: dict,
    target_lang: str = TARGET_LANG,
    output_dir: str = "outputs_prototypes",
    config_tag: str = "",
):
    """
    For each layer, compute cosine similarity between target_lang and every other language,
    tag-wise then averaged over available tags (skipping NaNs).
    Returns:
        sim_tables: dict[layer_idx] -> DataFrame (index=language, columns=[tags..., 'Mean'])
    """
    os.makedirs(output_dir, exist_ok=True)
    sim_tables = {}

    for li, lang2protos in all_prototypes_by_layer.items():
        tgt = lang2protos[target_lang]
        rows = {}
        for lang, protos in lang2protos.items():
            if lang == target_lang:
                continue
            scores = {}
            vals = []
            for tag in ENTITY_TAGS:
                r = _safe_cosine(np.array(tgt[tag]), np.array(protos[tag]))
                scores[tag] = r
                if not np.isnan(r):
                    vals.append(r)
            scores["Mean"] = float(np.mean(vals)) if vals else np.nan
            rows[lang] = scores

        df = pd.DataFrame.from_dict(rows, orient="index")
        df = df.sort_values("Mean", ascending=False, na_position="last")
        tag_suffix = f"__{config_tag}" if config_tag else ""
        out_csv = os.path.join(output_dir, f"cosine_to_{target_lang}_layer{li}{tag_suffix}.csv")
        df.to_csv(out_csv)
        sim_tables[li] = df

    return sim_tables

# --------------------
# Visualization
# --------------------

def _save_sorted_bar(df: pd.DataFrame, layer_idx: int, output_dir: str, title_prefix: str, config_tag: str = ""):
    """Full bar chart for all languages at one layer (Cosine)."""
    vals = df["Mean"].fillna(0.0)
    order = vals.sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    plt.bar(order.index, order.values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Cosine similarity (mean over tags)")
    title = f"{title_prefix} — Layer {layer_idx}"
    if config_tag:
        title += f" — {config_tag}"
    plt.title(title)
    plt.tight_layout()
    fname_base = f"{title_prefix.lower().replace(' ', '_')}_layer{layer_idx}"
    if config_tag:
        fname_base += f"__{config_tag}"
    fname = os.path.join(output_dir, f"{fname_base}.png")
    plt.savefig(fname, dpi=200)
    plt.close()

def _save_heatmap(language_order, layer_order, mat, output_path: str, title: str):
    """Heatmap: languages × layers (Cosine)."""
    plt.figure(figsize=(1.2 + 0.5*len(layer_order), 0.6 + 0.4*len(language_order)))
    im = plt.imshow(mat, aspect="auto", interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.yticks(range(len(language_order)), language_order)
    plt.xticks(range(len(layer_order)), layer_order, rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def create_visualizations_for_sim_tables(
    sim_tables: dict[int, pd.DataFrame],
    model,
    output_dir: str,
    model_type: str,
    config_tag: str = "",
):
    """
    - Heatmap over selected layers (languages × layers) using Mean cosine similarity
    - Full bar chart for each layer (all languages)
    - Print + save top 5 and worst 5 per layer
    """
    os.makedirs(output_dir, exist_ok=True)

    # Ensure a consistent language order based on the final layer
    num_layers_total = model.config.num_hidden_layers
    output_layer_idx = num_layers_total
    if output_layer_idx not in sim_tables:
        output_layer_idx = sorted(sim_tables.keys())[-1]

    base_df = sim_tables[output_layer_idx]
    language_order = list(base_df.index)
    layer_order = sorted(sim_tables.keys())

    # Heatmap matrix: rows=languages, cols=layers
    mat = np.zeros((len(language_order), len(layer_order)), dtype=float)
    mat[:] = np.nan
    for j, li in enumerate(layer_order):
        df = sim_tables[li].reindex(language_order)
        mat[:, j] = df["Mean"].values

    # Save heatmap
    heatmap_name = f"heatmap_cosine_layers_{model_type}"
    if config_tag:
        heatmap_name += f"__{config_tag}"
    heatmap_path = os.path.join(output_dir, f"{heatmap_name}.png")
    title = f"Runyankore similarity (cosine) across layers — {model_type}"
    if config_tag:
        title += f" — {config_tag}"
    _save_heatmap(language_order, layer_order, mat, heatmap_path, title)

    # Full bar charts + summaries
    summary_rows = []
    for li in layer_order:
        df = sim_tables[li]
        _save_sorted_bar(df, li, output_dir, f"Cosine similarities vs run ({model_type})", config_tag=config_tag)

        # Top 5 and worst 5 (drop NaNs first)
        clean = df["Mean"].dropna()
        top5 = clean.sort_values(ascending=False).head(5)
        bot5 = clean.sort_values(ascending=True).head(5)

        # Print summary to console and log
        header = f"Layer {li} ({model_type})"
        if config_tag:
            header += f" — {config_tag}"
        print(f"\n=== {header} — Top 5 (cosine) ===")
        logging.info(f"{header} — Top 5 (cosine)")
        for rank, (lang, val) in enumerate(top5.items(), 1):
            print(f"{rank}. {lang}: cos={val:.4f}")
            logging.info(f"{rank}. {lang}: cos={val:.4f}")

        print(f"=== {header} — Worst 5 (cosine) ===")
        logging.info(f"{header} — Worst 5 (cosine)")
        for rank, (lang, val) in enumerate(bot5.items(), 1):
            print(f"{rank}. {lang}: cos={val:.4f}")
            logging.info(f"{rank}. {lang}: cos={val:.4f}")

        for lang, val in top5.items():
            summary_rows.append({"layer": li, "model": model_type, "rank_type": "top5", "language": lang, "cosine": val, "config": config_tag})
        for lang, val in bot5.items():
            summary_rows.append({"layer": li, "model": model_type, "rank_type": "worst5", "language": lang, "cosine": val, "config": config_tag})

    if summary_rows:
        base = f"summary_top_worst_cosine_{model_type}"
        if config_tag:
            base += f"__{config_tag}"
        pd.DataFrame(summary_rows).to_csv(os.path.join(output_dir, f"{base}.csv"), index=False)

# --------------------
# Layer selection (ALL layers)
# --------------------

def get_all_hidden_layers(model):
    """
    Return all transformer layers by hidden_states indexing:
    1..L inclusive (exclude 0 which is embeddings).
    """
    L = model.config.num_hidden_layers
    return list(range(1, L + 1))

# --------------------
# Aggregation helpers
# --------------------

def config_to_tag(cfg: dict) -> str:
    """
    Make a short, filesystem-safe tag that encodes the hyperparameters for filenames.
    """
    bits = []
    for k in ["use_cls", "use_context", "context_window", "max_tokens_per_type", "batch_size"]:
        v = cfg[k]
        if isinstance(v, bool):
            v = int(v)
        bits.append(f"{k[:3]}{v}")
    return "_".join(bits)

def compute_layer_and_overall_scores(sim_tables: dict[int, pd.DataFrame]) -> tuple[dict, float]:
    """
    Returns:
      - layer_scores: dict[layer_idx] -> mean over languages of df['Mean'] (skip NaNs)
      - overall_score: mean over layers of layer_scores (skip NaNs)
    """
    layer_scores = {}
    for li, df in sim_tables.items():
        # df['Mean'] is per-language (mean over tags); take mean over languages
        layer_scores[li] = float(pd.to_numeric(df["Mean"], errors="coerce").mean(skipna=True))
    overall = float(pd.Series(layer_scores).mean(skipna=True)) if layer_scores else float("nan")
    return layer_scores, overall

# --------------------
# Per-model pipeline for one config
# --------------------

def run_one_model_one_config(
    model_type: str,
    language_files: dict,
    args,
    cfg: dict,
    results_aggregator: list,  # append dict rows here
):
    model_name = MODEL_NAME_MAP[model_type]
    # Build a temp model to read num_hidden_layers
    tmp_model = AutoModel.from_pretrained(model_name)
    layer_indices = get_all_hidden_layers(tmp_model)  # *** ALL layers ***
    del tmp_model

    cfg_tag = config_to_tag(cfg)

    # Where to save this config’s outputs
    out_dir_model = os.path.join(args.output_dir, model_type, cfg_tag)
    os.makedirs(out_dir_model, exist_ok=True)

    # Log and print
    logging.info(f"{model_type} :: Running config {cfg_tag} with layers={layer_indices}")
    print(f"{model_type}: layers={layer_indices} | config={cfg_tag}")

    # Compute prototypes at all layers
    all_prototypes_by_layer, model = compute_language_prototypes_multilayer(
        language_files=language_files,
        model_type=model_type,
        layer_indices=layer_indices,
        max_tokens_per_type=cfg["max_tokens_per_type"],
        output_dir=out_dir_model,
        use_cls=cfg["use_cls"],
        use_context=cfg["use_context"],
        context_window=cfg["context_window"],
        batch_size=cfg["batch_size"],
    )

    # Cosine similarities vs run at each layer
    sim_tables = compute_cosine_to_target_multilayer(
        all_prototypes_by_layer,
        target_lang=TARGET_LANG,
        output_dir=out_dir_model,
        config_tag=cfg_tag,
    )

    # Visualizations + summaries (scoped to this config)
    create_visualizations_for_sim_tables(
        sim_tables=sim_tables,
        model=model,
        output_dir=out_dir_model,
        model_type=model_type,
        config_tag=cfg_tag,
    )

    # Compute aggregate scores
    layer_scores, overall_score = compute_layer_and_overall_scores(sim_tables)

    # Record rows for aggregation CSV
    for li, ls in layer_scores.items():
        results_aggregator.append({
            "model": model_type,
            "config": cfg_tag,
            "layer": li,
            "layer_mean_cosine_over_languages": ls,
        })
    results_aggregator.append({
        "model": model_type,
        "config": cfg_tag,
        "layer": "ALL",
        "layer_mean_cosine_over_languages": overall_score,
    })

# --------------------
# Main
# --------------------

def parse_bool_list(s: str) -> list[bool]:
    return [c.strip() in ("1", "true", "True", "yes", "y") for c in s.split(",") if c.strip() != ""]

def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]

def main():
    parser = argparse.ArgumentParser(description="All-layer cosine similarity with config sweep + best-config aggregation (Runyankore target)")
    parser.add_argument("--model_type", type=str, default="xlmr", choices=["xlmr", "mbert"], help="Model to run. Use --run_both to run both.")
    parser.add_argument("--run_both", action="store_true", help="Run both models (xlmr and mbert).")

    # Single-run defaults (also used if corresponding grid arg not provided)
    parser.add_argument("--use_cls", action="store_true", help="Use CLS token instead of subword pooling")
    parser.add_argument("--use_context", action="store_true", help="Use context window around entity")
    parser.add_argument("--context_window", type=int, default=2, help="Size of context window")
    parser.add_argument("--max_tokens_per_type", type=int, default=500, help="Max entity tokens per tag")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference")

    # Grid (sweep) — pass comma-separated lists; if omitted, the single-run defaults are used
    parser.add_argument("--grid_use_cls", type=str, default="", help="Comma list of booleans (e.g., '0,1')")
    parser.add_argument("--grid_use_context", type=str, default="", help="Comma list of booleans (e.g., '0,1')")
    parser.add_argument("--grid_context_window", type=str, default="", help="Comma list of ints (e.g., '0,2,4')")
    parser.add_argument("--grid_max_tokens_per_type", type=str, default="", help="Comma list of ints")
    parser.add_argument("--grid_batch_size", type=str, default="", help="Comma list of ints")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="outputs_prototypes")
    args = parser.parse_args()

    set_seed(args.seed)

    logging.info(
        f"Start: model_type={args.model_type}, run_both={args.run_both}, "
        f"use_cls={args.use_cls}, use_context={args.use_context}, context_window={args.context_window}, "
        f"max_tokens_per_type={args.max_tokens_per_type}, batch_size={args.batch_size}, seed={args.seed}, "
        f"output_dir={args.output_dir}"
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

    # Build config grid
    use_cls_grid = parse_bool_list(args.grid_use_cls) if args.grid_use_cls else [args.use_cls]
    use_ctx_grid = parse_bool_list(args.grid_use_context) if args.grid_use_context else [args.use_context]
    ctx_win_grid = parse_int_list(args.grid_context_window) if args.grid_context_window else [args.context_window]
    max_tok_grid = parse_int_list(args.grid_max_tokens_per_type) if args.grid_max_tokens_per_type else [args.max_tokens_per_type]
    batch_grid   = parse_int_list(args.grid_batch_size) if args.grid_batch_size else [args.batch_size]

    config_grid = []
    for use_cls, use_ctx, ctx_win, max_tok, batch in product(use_cls_grid, use_ctx_grid, ctx_win_grid, max_tok_grid, batch_grid):
        config_grid.append({
            "use_cls": bool(use_cls),
            "use_context": bool(use_ctx),
            "context_window": int(ctx_win),
            "max_tokens_per_type": int(max_tok),
            "batch_size": int(batch),
        })

    # Which models?
    model_list = ["xlmr", "mbert"] if args.run_both else [args.model_type]

    # Aggregate across configs to pick best
    for mt in model_list:
        results_rows = []  # will collect per-layer + overall scores for each config
        for cfg in config_grid:
            run_one_model_one_config(
                model_type=mt,
                language_files=language_files,
                args=args,
                cfg=cfg,
                results_aggregator=results_rows,
            )

        # Save aggregation table per model
        agg_df = pd.DataFrame(results_rows)
        model_out_dir = os.path.join(args.output_dir, mt)
        os.makedirs(model_out_dir, exist_ok=True)
        agg_csv_path = os.path.join(model_out_dir, f"aggregate_results_{mt}.csv")
        agg_df.to_csv(agg_csv_path, index=False)

        # Select best config by the "ALL" row (overall mean over layers)
        overall = agg_df[agg_df["layer"] == "ALL"].copy()
        # Higher is better
        best_row = overall.sort_values("layer_mean_cosine_over_languages", ascending=False).head(1)
        if not best_row.empty:
            best_config_tag = best_row.iloc[0]["config"]
            best_score = float(best_row.iloc[0]["layer_mean_cosine_over_languages"])
            print(f"\n>>> Best config for {mt}: {best_config_tag} (overall mean cosine={best_score:.4f})")

            # Save a small CSV with the best config summary
            best_csv_path = os.path.join(model_out_dir, f"best_config_{mt}.csv")
            best_row.to_csv(best_csv_path, index=False)

            # Also dump per-layer scores for that best config
            best_layers = agg_df[(agg_df["config"] == best_config_tag) & (agg_df["layer"] != "ALL")].copy()
            best_layers_path = os.path.join(model_out_dir, f"best_config_layers_{mt}__{best_config_tag}.csv")
            best_layers.to_csv(best_layers_path, index=False)
        else:
            print(f"\n>>> No results found for model {mt} — check inputs/data.")

    logging.info("Experiment finished successfully.")

if __name__ == "__main__":
    main()
