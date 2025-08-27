import os
import argparse
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from typing import Dict, List, Tuple

# =========================
# Config
# =========================

MODEL_NAME_MAP = {
    "xlmr": "xlm-roberta-base",
    "mbert": "bert-base-multilingual-cased"
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
    level=logging.INFO
)

# =========================
# Helpers
# =========================

def _unit_norm(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (n + eps)

def _mean_unit(vecs: np.ndarray) -> np.ndarray:
    """Normalize each vector -> mean -> re-normalize."""
    if vecs.ndim != 2 or vecs.shape[0] == 0:
        raise ValueError("vecs must be (N,H) with N>0")
    v = _unit_norm(vecs)
    m = v.mean(axis=0)
    return _unit_norm(m)

def _safe_cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    if a is None or b is None:
        return np.nan
    if np.isnan(a).any() or np.isnan(b).any():
        return np.nan
    den = (np.linalg.norm(a) * np.linalg.norm(b))
    if den < eps:
        return np.nan
    return float(np.dot(a, b) / den)

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

# =========================
# Embedding extraction (batched, multi-layer)
# =========================

def extract_entity_embeddings_batched_multilayer(
    file_path: str,
    tokenizer,
    model,
    layer_indices: List[int],
    entity_tags=ENTITY_TAGS,
    max_tokens_per_type: int = 500,
    use_cls: bool = False,
    use_context: bool = False,
    context_window: int = 2,
    batch_size: int = 64,
    save_examples_path: str | None = None,
) -> Tuple[Dict[int, Dict[str, List[np.ndarray]]], Dict[str, int]]:
    """
    Returns:
      per_layer_embeddings: dict[layer_idx][tag] -> list[np.ndarray] (unit vectors)
      per_tag_counts: dict[tag] -> count used (cap max_tokens_per_type)
    """
    assert getattr(tokenizer, "is_fast", False), "Fast tokenizer is required (word_ids)."

    per_layer_embeddings = {li: {tag: [] for tag in entity_tags} for li in layer_indices}
    per_tag_counts = {tag: 0 for tag in entity_tags}
    per_tag_examples = {tag: [] for tag in entity_tags} if save_examples_path else None
    fallback_events = 0

    batch_payload = []  # list of dict(span_tokens, rel_idx, tag)

    def _flush_batch():
        nonlocal batch_payload, fallback_events
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
        hidden_states = outputs.hidden_states  # tuple: [embeddings, layer1, ..., layerL]

        for layer_idx in layer_indices:
            layer_tensor = hidden_states[layer_idx]  # (B, T, H)

            if use_cls:
                embs = layer_tensor[:, 0, :].detach().cpu().numpy()  # (B,H)
                embs = _unit_norm(embs)
                for i, item in enumerate(batch_payload):
                    tag = item["tag"]
                    if per_tag_counts[tag] < max_tokens_per_type:
                        per_layer_embeddings[layer_idx][tag].append(embs[i].astype(np.float32))
                        per_tag_counts[tag] += 1
                        if per_tag_examples is not None:
                            per_tag_examples[tag].append(" ".join(item["span_tokens"]))
            else:
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
                        fallback_events += 1
                    if not sub_idx:
                        continue
                    sub_vecs = layer_tensor[i, sub_idx, :].detach().cpu().numpy()  # (S,H)
                    sub_vecs = _unit_norm(sub_vecs)
                    emb = sub_vecs.mean(axis=0)
                    emb = _unit_norm(emb)
                    per_layer_embeddings[layer_idx][tag].append(emb.astype(np.float32))
                    per_tag_counts[tag] += 1
                    if per_tag_examples is not None:
                        per_tag_examples[tag].append(" ".join(item["span_tokens"]))

        batch_payload = []

    # Iterate sentences, collect spans
    for tokens, tags in tqdm(_read_conll_sentences(file_path), desc=f"Reading {os.path.basename(file_path)}"):
        for idx, tag in enumerate(tags):
            base = tag.replace("B-", "").replace("I-", "")
            if base not in entity_tags:
                continue
            if per_tag_counts[base] >= max_tokens_per_type:
                continue

            # Build span + relative entity index within span
            start = max(0, idx - context_window) if use_context else idx
            end = min(len(tokens), idx + context_window + 1) if use_context else idx + 1
            span = tokens[start:end]
            if not span:
                continue
            rel_idx = idx - start

            batch_payload.append({"span_tokens": span, "rel_idx": rel_idx, "tag": base})

            if len(batch_payload) >= batch_size:
                _flush_batch()
    _flush_batch()

    if fallback_events > 0:
        logging.warning(f"Tokenizer fallback events (no subword match): {fallback_events}")

    if save_examples_path and per_tag_examples is not None:
        os.makedirs(save_examples_path, exist_ok=True)
        for tag in per_tag_examples:
            with open(os.path.join(save_examples_path, f"{tag}_examples.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(per_tag_examples[tag]))

    return per_layer_embeddings, per_tag_counts

# =========================
# Prototypes per layer (unit prototypes)
# =========================

def compute_language_prototypes_multilayer(
    language_files: Dict[str, str],
    model_type: str,
    layer_indices: List[int],
    max_tokens_per_type: int = 500,
    output_dir: str = "outputs_prototypes",
    use_cls: bool = False,
    use_context: bool = False,
    context_window: int = 2,
    batch_size: int = 64,
    save_prototypes: bool = False,
):
    model_name = MODEL_NAME_MAP[model_type]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE)
    model.eval()
    assert getattr(tokenizer, "is_fast", False), "Fast tokenizer required."

    os.makedirs(output_dir, exist_ok=True)

    # all_prototypes[layer][lang][tag] = vector (unit) or NaN-vector if missing
    all_prototypes = {li: {} for li in layer_indices}
    all_counts: Dict[str, Dict[str, int]] = {}

    for lang, path in tqdm(language_files.items(), desc=f"Computing prototypes — {model_type}"):
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

        H = model.config.hidden_size
        for li in layer_indices:
            tag2embs = per_layer_embs[li]
            protos = {}
            for tag, embs in tag2embs.items():
                if len(embs) == 0:
                    proto = np.full((H,), np.nan, dtype=np.float32)
                else:
                    arr = np.stack(embs).astype(np.float32)  # already unit vectors
                    proto = _mean_unit(arr).astype(np.float32)
                protos[tag] = proto
            all_prototypes[li][lang] = protos

            # (NEW) Save only if requested
            if save_prototypes:
                np.savez(os.path.join(output_dir, f"{lang}_prototypes_layer{li}_{model_type}.npz"), **protos)

    # Save counts (always useful + tiny)
    pd.DataFrame.from_dict(all_counts, orient="index").to_csv(os.path.join(output_dir, "prototype_counts.csv"))
    return all_prototypes, model, all_counts

# =========================
# Cosine similarities to target (weighted tag mean) per layer
# =========================

def compute_cosine_to_target_multilayer(
    all_prototypes_by_layer: Dict[int, Dict[str, Dict[str, np.ndarray]]],
    counts_by_lang: Dict[str, Dict[str, int]],
    target_lang: str = TARGET_LANG,
    output_dir: str = "outputs_prototypes",
) -> Dict[int, pd.DataFrame]:
    os.makedirs(output_dir, exist_ok=True)
    sim_tables = {}
    tgt_counts = counts_by_lang[target_lang]

    for li, lang2protos in all_prototypes_by_layer.items():
        tgt = lang2protos[target_lang]
        rows = {}
        for lang, protos in lang2protos.items():
            if lang == target_lang:
                continue
            scores = {}
            numer = 0.0
            denom = 0.0
            for tag in ENTITY_TAGS:
                r = _safe_cosine(np.array(tgt[tag]), np.array(protos[tag]))
                scores[tag] = r
                w = float(min(tgt_counts.get(tag, 0), counts_by_lang[lang].get(tag, 0)))
                if not np.isnan(r) and w > 0:
                    numer += w * r
                    denom += w
            scores["Mean"] = (numer / denom) if denom > 0 else np.nan
            rows[lang] = scores

        df = pd.DataFrame.from_dict(rows, orient="index")
        df = df.sort_values("Mean", ascending=False, na_position="last")
        out_csv = os.path.join(output_dir, f"cosine_to_{target_lang}_layer{li}.csv")
        df.to_csv(out_csv)
        sim_tables[li] = df

    return sim_tables

# =========================
# Visualization
# =========================

def _save_sorted_bar(df: pd.DataFrame, layer_idx: int, output_dir: str, title_prefix: str):
    vals = pd.to_numeric(df["Mean"], errors="coerce").fillna(0.0)
    order = vals.sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    plt.bar(order.index, order.values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Cosine similarity (weighted mean over tags)")
    plt.title(f"{title_prefix} — Layer {layer_idx}")
    plt.tight_layout()
    fname = os.path.join(output_dir, f"{title_prefix.lower().replace(' ', '_')}_layer{layer_idx}.png")
    plt.savefig(fname, dpi=200)
    plt.close()

def _save_heatmap(language_order, layer_order, mat, output_path: str, title: str):
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
    sim_tables: Dict[int, pd.DataFrame],
    model,
    output_dir: str,
    model_type: str,
):
    os.makedirs(output_dir, exist_ok=True)

    # Choose a reference layer to define language order (final layer by default)
    num_layers_total = model.config.num_hidden_layers
    ref_layer = num_layers_total if num_layers_total in sim_tables else sorted(sim_tables.keys())[-1]
    base_df = sim_tables[ref_layer]
    language_order = list(base_df.index)
    layer_order = sorted(sim_tables.keys())

    # Heatmap matrix: rows=languages, cols=layers
    mat = np.zeros((len(language_order), len(layer_order)), dtype=float)
    mat[:] = np.nan
    for j, li in enumerate(layer_order):
        df = sim_tables[li].reindex(language_order)
        mat[:, j] = df["Mean"].values

    heatmap_path = os.path.join(output_dir, f"heatmap_cosine_layers_{model_type}.png")
    _save_heatmap(language_order, layer_order, mat, heatmap_path,
                  f"Runyankore similarity (cosine, weighted) across layers — {model_type}")

    # Per-layer bar charts + top/worst summaries
    summary_rows = []
    for li in layer_order:
        df = sim_tables[li]
        _save_sorted_bar(df, li, output_dir, f"Cosine similarities vs run ({model_type})")

        clean = pd.to_numeric(df["Mean"], errors="coerce").dropna()
        top5 = clean.sort_values(ascending=False).head(5)
        bot5 = clean.sort_values(ascending=True).head(5)

        print(f"\n=== Layer {li} ({model_type}) — Top 5 (cosine) ===")
        for rank, (lang, val) in enumerate(top5.items(), 1):
            print(f"{rank}. {lang}: cos={val:.4f}")

        print(f"=== Layer {li} ({model_type}) — Worst 5 (cosine) ===")
        for rank, (lang, val) in enumerate(bot5.items(), 1):
            print(f"{rank}. {lang}: cos={val:.4f}")

        for lang, val in top5.items():
            summary_rows.append({"layer": li, "model": model_type, "rank_type": "top5", "language": lang, "cosine": val})
        for lang, val in bot5.items():
            summary_rows.append({"layer": li, "model": model_type, "rank_type": "worst5", "language": lang, "cosine": val})

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(os.path.join(output_dir, f"summary_top_worst_cosine_{model_type}.csv"), index=False)

# =========================
# Main
# =========================

def get_all_hidden_layers(model) -> List[int]:
    L = model.config.num_hidden_layers
    # hidden_states indexing: 0=embeddings, 1..L = layers
    return list(range(1, L + 1))

def main():
    parser = argparse.ArgumentParser(description="Multi-layer prototype-based language similarity (Runyankore target)")
    parser.add_argument("--model_type", type=str, default="xlmr", choices=["xlmr", "mbert"])
    parser.add_argument("--use_cls", action="store_true", help="Use CLS token instead of subword pooling")
    parser.add_argument("--use_context", action="store_true", help="Use small context window around entity")
    parser.add_argument("--context_window", type=int, default=2, help="Context window size")
    parser.add_argument("--max_tokens_per_type", type=int, default=500, help="Max entity tokens per tag")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for batched inference")
    parser.add_argument("--output_dir", type=str, default="outputs_prototypes")
    parser.add_argument("--save_prototypes", action="store_true", help="If set, save per-language per-layer NPZ prototypes")
    args = parser.parse_args()

    logging.info(
        f"Start: model_type={args.model_type}, use_cls={args.use_cls}, use_context={args.use_context}, "
        f"context_window={args.context_window}, max_tokens_per_type={args.max_tokens_per_type}, "
        f"batch_size={args.batch_size}, output_dir={args.output_dir}, save_prototypes={args.save_prototypes}"
    )

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data"))
    language_files = {
        "run": os.path.join(BASE_DIR, "SALT/train.txt"),
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

    # Build a temporary model to get layer indices
    tmp_model = AutoModel.from_pretrained(MODEL_NAME_MAP[args.model_type])
    layer_indices = get_all_hidden_layers(tmp_model)
    del tmp_model

    out_dir = os.path.join(args.output_dir, args.model_type)
    os.makedirs(out_dir, exist_ok=True)

    # Compute per-language, per-layer prototypes
    all_prototypes_by_layer, model, counts_by_lang = compute_language_prototypes_multilayer(
        language_files=language_files,
        model_type=args.model_type,
        layer_indices=layer_indices,
        max_tokens_per_type=args.max_tokens_per_type,
        output_dir=out_dir,
        use_cls=args.use_cls,
        use_context=args.use_context,
        context_window=args.context_window,
        batch_size=args.batch_size,
        save_prototypes=args.save_prototypes,
    )

    # Compute cosine similarities vs Runyankore for each layer
    sim_tables = compute_cosine_to_target_multilayer(
        all_prototypes_by_layer=all_prototypes_by_layer,
        counts_by_lang=counts_by_lang,
        target_lang=TARGET_LANG,
        output_dir=out_dir,
    )

    # Visualizations + summaries
    create_visualizations_for_sim_tables(
        sim_tables=sim_tables,
        model=model,
        output_dir=out_dir,
        model_type=args.model_type,
    )

    print("\nDone. Outputs in:", out_dir)
    logging.info("Experiment finished successfully.")

if __name__ == "__main__":
    main()
