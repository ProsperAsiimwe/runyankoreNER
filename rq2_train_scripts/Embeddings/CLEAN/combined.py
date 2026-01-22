#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import random
import contextlib
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import matplotlib.pyplot as plt

# =========================
# Config
# =========================

MODEL_NAME_MAP = {
    "afroxlmr": "Davlan/afro-xlmr-base",
    "xlmr": "xlm-roberta-base",
    "mbert": "bert-base-multilingual-cased"
}
DEFAULT_ENTITY_TAGS: Set[str] = {"PER", "LOC", "ORG", "DATE"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_LANG = "run"

# Logging to file + console
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
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
logging.getLogger("").addHandler(console)

# =========================
# Helpers
# =========================

def set_deterministic(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _unit_norm(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (n + eps)

def _mean_unit(vecs: np.ndarray) -> np.ndarray:
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

def _read_conll_sentences_loose(file_path: str):
    """
    Yield (tokens, tags) from a CoNLL-like file.
    Accepts extra columns; uses first column as token, last column as tag.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        tokens, tags = [], []
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                if tokens:
                    yield tokens, tags
                tokens, tags = [], []
                continue
            parts = line.split()
            tok = parts[0]
            tag = parts[-1]
            tokens.append(tok)
            tags.append(tag)
        if tokens:
            yield tokens, tags

def _coerce_tag(tag: str) -> str:
    if tag == "O":
        return "O"
    return tag.split("-", 1)[-1] if "-" in tag else tag

def _iter_bio_spans(tokens: List[str], tags: List[str], entity_tags: Set[str]):
    """
    Yield (start, end, ent_type) for BIO sequences where ent_type ∈ entity_tags.
    end is exclusive.
    """
    n = len(tokens)
    i = 0
    while i < n:
        t = tags[i]
        if t == "O":
            i += 1
            continue
        ent = _coerce_tag(t)
        if ent not in entity_tags:
            i += 1
            continue
        j = i + 1
        while j < n:
            tj = tags[j]
            if tj == "O":
                break
            ej = _coerce_tag(tj)
            pj = tj.split("-", 1)[0] if "-" in tj else "I"
            if ej != ent or pj not in {"I", "B"}:
                break
            j += 1
        yield i, j, ent
        i = j

# =========================
# Embedding extraction (batched, multi-layer)
# =========================

def extract_entity_embeddings_batched_multilayer(
    file_path: str,
    tokenizer,
    model,
    layer_indices: List[int],
    entity_tags: Set[str],
    max_tokens_per_type: int = 500,
    use_cls: bool = False,
    use_context: bool = False,
    context_window: int = 2,
    batch_size: int = 64,
    save_examples_path: Optional[str] = None,
    span_mode: str = "bio",           # "bio" or "token"
    pool: str = "mean",               # "mean" | "max" | "first" | "last"
    use_amp: bool = False,
    max_length: int = 128,
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

    batch_payload = []  # list of dict: {"span_tokens", "rel_idx_list", "tag"}

    def _pool_vectors(sub_vecs: np.ndarray) -> np.ndarray:
        if sub_vecs.size == 0:
            return sub_vecs
        if pool == "mean":
            return _unit_norm(sub_vecs.mean(axis=0))
        if pool == "max":
            return _unit_norm(sub_vecs.max(axis=0))
        if pool == "first":
            return _unit_norm(sub_vecs[0])
        if pool == "last":
            return _unit_norm(sub_vecs[-1])
        return _unit_norm(sub_vecs.mean(axis=0))

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
                max_length=max_length,
                padding=True,
            ).to(DEVICE)
        else:
            spans = [item["span_tokens"] for item in batch_payload]
            tokenized = tokenizer(
                spans,
                is_split_into_words=True,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            ).to(DEVICE)

        with torch.no_grad():
            if use_amp and DEVICE == "cuda":
                amp_dtype = (
                    torch.bfloat16
                    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                    else torch.float16
                )
                cm = torch.autocast(device_type="cuda", dtype=amp_dtype)
            else:
                cm = contextlib.nullcontext()

            with cm:
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
                    rel_idx_list = item["rel_idx_list"]

                    # Collect subword indices for all entity tokens
                    sub_idx = []
                    for rel_idx in rel_idx_list:
                        sub_idx.extend([j for j, wid in enumerate(word_ids) if wid == rel_idx])

                    if not sub_idx:
                        first_non_special = [j for j, wid in enumerate(word_ids) if wid is not None][:1]
                        if first_non_special:
                            sub_idx = first_non_special
                            fallback_events += 1
                        else:
                            continue

                    sub_vecs = layer_tensor[i, sub_idx, :].detach().cpu().numpy()  # (S,H)
                    sub_vecs = _unit_norm(sub_vecs)
                    emb = _pool_vectors(sub_vecs)
                    per_layer_embeddings[layer_idx][tag].append(emb.astype(np.float32))
                    per_tag_counts[tag] += 1
                    if per_tag_examples is not None:
                        per_tag_examples[tag].append(" ".join(item["span_tokens"]))

        batch_payload = []

    # Iterate sentences, collect spans
    for tokens, tags in tqdm(_read_conll_sentences_loose(file_path), desc=f"Reading {os.path.basename(file_path)}"):
        if span_mode == "bio":
            spans = _iter_bio_spans(tokens, tags, entity_tags)
            for start, end, ent in spans:
                if per_tag_counts[ent] >= max_tokens_per_type:
                    continue
                if use_context:
                    ctx_start = max(0, start - context_window)
                    ctx_end = min(len(tokens), end + context_window)
                else:
                    ctx_start, ctx_end = start, end
                span = tokens[ctx_start:ctx_end]
                if not span:
                    continue
                rel_idx_list = list(range(start - ctx_start, end - ctx_start))  # all entity tokens
                batch_payload.append({"span_tokens": span, "rel_idx_list": rel_idx_list, "tag": ent})
                if len(batch_payload) >= batch_size:
                    _flush_batch()
        else:
            # Token-level (legacy)
            for idx, tag in enumerate(tags):
                base = _coerce_tag(tag)
                if base not in entity_tags:
                    continue
                if per_tag_counts[base] >= max_tokens_per_type:
                    continue
                if use_context:
                    start = max(0, idx - context_window)
                    end = min(len(tokens), idx + context_window + 1)
                else:
                    start, end = idx, idx + 1
                span = tokens[start:end]
                if not span:
                    continue
                rel_idx_list = [idx - start]
                batch_payload.append({"span_tokens": span, "rel_idx_list": rel_idx_list, "tag": base})
                if len(batch_payload) >= batch_size:
                    _flush_batch()

    _flush_batch()

    if fallback_events > 0:
        logging.warning(f"Tokenizer fallback events (no subword match): {fallback_events} [{os.path.basename(file_path)}]")

    if save_examples_path and per_tag_examples is not None:
        os.makedirs(save_examples_path, exist_ok=True)
        for tag in per_tag_examples:
            outp = os.path.join(save_examples_path, f"{tag}_examples.txt")
            with open(outp, "w", encoding="utf-8") as f:
                f.write("\n".join(per_tag_examples[tag]))

    return per_layer_embeddings, per_tag_counts

# =========================
# Prototypes per layer (unit prototypes)
# =========================

def compute_language_prototypes_multilayer(
    language_files: Dict[str, str],
    model_type: str,
    layer_indices: List[int],
    entity_tags: Set[str],
    max_tokens_per_type: int = 500,
    output_dir: str = "outputs_prototypes",
    use_cls: bool = False,
    use_context: bool = False,
    context_window: int = 2,
    batch_size: int = 64,
    save_prototypes: bool = False,
    span_mode: str = "bio",
    pool: str = "mean",
    use_amp: bool = False,
    max_length: int = 128,
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
        if not os.path.isfile(path):
            logging.warning(f"[skip] Missing file for {lang}: {path}")
            continue

        logging.info(f"Processing {lang}")
        example_path = os.path.join(output_dir, f"{lang}_examples")

        per_layer_embs, counts_by_tag = extract_entity_embeddings_batched_multilayer(
            path,
            tokenizer,
            model,
            layer_indices=layer_indices,
            entity_tags=entity_tags,
            max_tokens_per_type=max_tokens_per_type,
            use_cls=use_cls,
            use_context=use_context,
            context_window=context_window,
            batch_size=batch_size,
            save_examples_path=example_path,
            span_mode=span_mode,
            pool=pool,
            use_amp=use_amp,
            max_length=max_length,
        )
        all_counts[lang] = counts_by_tag

        if sum(counts_by_tag.values()) == 0:
            logging.warning(f"No entity examples found for {lang}. Check labels & ENTITY_TAGS.")

        H = model.config.hidden_size
        for li in layer_indices:
            tag2embs = per_layer_embs[li]
            protos = {}
            for tag, embs in tag2embs.items():
                if len(embs) == 0:
                    proto = np.full((H,), np.nan, dtype=np.float32)
                else:
                    arr = np.stack(embs).astype(np.float32)
                    proto = _mean_unit(arr).astype(np.float32)
                protos[tag] = proto
            all_prototypes[li][lang] = protos

            if save_prototypes:
                np.savez(os.path.join(output_dir, f"{lang}_prototypes_layer{li}_{model_type}.npz"), **protos)

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
    entity_tags: Set[str] = DEFAULT_ENTITY_TAGS,
) -> Dict[int, pd.DataFrame]:
    os.makedirs(output_dir, exist_ok=True)
    sim_tables = {}

    if target_lang not in counts_by_lang or target_lang not in next(iter(all_prototypes_by_layer.values())):
        raise ValueError(f"Target language '{target_lang}' missing in prototypes/counts.")

    tgt_counts = counts_by_lang.get(target_lang, {})

    for li, lang2protos in all_prototypes_by_layer.items():
        tgt = lang2protos[target_lang]
        rows = {}
        for lang, protos in lang2protos.items():
            if lang == target_lang:
                continue
            numer = 0.0
            denom = 0.0
            scores = {}
            for tag in entity_tags:
                r = _safe_cosine(np.array(tgt.get(tag)), np.array(protos.get(tag)))
                scores[tag] = r
                w = float(min(tgt_counts.get(tag, 0), counts_by_lang.get(lang, {}).get(tag, 0)))
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

def _save_sorted_bar(df: pd.DataFrame, layer_idx: int, output_dir: str, title_prefix: str, top_k_annot: int = 5):
    vals = pd.to_numeric(df["Mean"], errors="coerce").fillna(0.0)
    order = vals.sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    bars = plt.bar(order.index, order.values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Cosine similarity (weighted mean over tags)")
    plt.title(f"{title_prefix} — Layer {layer_idx}")
    for i, (lang, val) in enumerate(order.head(top_k_annot).items()):
        plt.text(i, val, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
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
    title_suffix: str = "",
):
    os.makedirs(output_dir, exist_ok=True)

    num_layers_total = model.config.num_hidden_layers
    ref_layer = num_layers_total if num_layers_total in sim_tables else sorted(sim_tables.keys())[-1]
    base_df = sim_tables[ref_layer]
    language_order = list(base_df.index)
    layer_order = sorted(sim_tables.keys())

    mat = np.zeros((len(language_order), len(layer_order)), dtype=float)
    mat[:] = np.nan
    for j, li in enumerate(layer_order):
        df = sim_tables[li].reindex(language_order)
        mat[:, j] = df["Mean"].values

    heatmap_basename = f"heatmap_cosine_layers_{model_type}{title_suffix}"
    heatmap_path = os.path.join(output_dir, f"{heatmap_basename}.png")
    _save_heatmap(
        language_order,
        layer_order,
        mat,
        heatmap_path,
        f"Runyankore similarity (cosine, weighted) across layers — {model_type}{title_suffix}"
    )

    heat_df = pd.DataFrame(mat, index=language_order, columns=layer_order)
    heat_df.to_csv(os.path.join(output_dir, f"{heatmap_basename}.csv"))

    summary_rows = []
    for li in layer_order:
        df = sim_tables[li]
        _save_sorted_bar(df, li, output_dir, f"Cosine similarities vs run ({model_type}{title_suffix})")

        clean = pd.to_numeric(df["Mean"], errors="coerce").dropna()
        top5 = clean.sort_values(ascending=False).head(5)
        bot5 = clean.sort_values(ascending=True).head(5)

        logging.info(f"=== Layer {li} ({model_type}) — Top 5 (cosine) ===")
        for rank, (lang, val) in enumerate(top5.items(), 1):
            logging.info(f"{rank}. {lang}: cos={val:.4f}")

        logging.info(f"=== Layer {li} ({model_type}) — Worst 5 (cosine) ===")
        for rank, (lang, val) in enumerate(bot5.items(), 1):
            logging.info(f"{rank}. {lang}: cos={val:.4f}")

        for lang, val in top5.items():
            summary_rows.append({"layer": li, "model": model_type, "rank_type": "top5", "language": lang, "cosine": val})
        for lang, val in bot5.items():
            summary_rows.append({"layer": li, "model": model_type, "rank_type": "worst5", "language": lang, "cosine": val})

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(output_dir, f"summary_top_worst_cosine_{model_type}{title_suffix}.csv"),
            index=False
        )

# =========================
# Main
# =========================

def get_all_hidden_layers(model) -> List[int]:
    L = model.config.num_hidden_layers
    return list(range(1, L + 1))

def main():
    parser = argparse.ArgumentParser(
        description="Multi-layer prototype-based language similarity (Runyankore target)"
    )
    parser.add_argument("--model_type", type=str, default="xlmr", choices=list(MODEL_NAME_MAP.keys()))
    parser.add_argument("--use_cls", action="store_true", help="Use CLS token instead of subword pooling")
    parser.add_argument("--use_context", action="store_true", help="Use small context window around entity")
    parser.add_argument("--context_window", type=int, default=2, help="Context window size")
    parser.add_argument("--max_tokens_per_type", type=int, default=500, help="Max entity tokens per tag")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for batched inference")
    parser.add_argument("--output_dir", type=str, default="outputs_prototypes")
    parser.add_argument("--save_prototypes", action="store_true", help="Save per-language per-layer NPZ prototypes")

    # QoL / robustness flags
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", help="Enable AMP (fp16/bf16) for faster inference")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--langs", type=str, default="all",
                        help="Comma-separated language codes or 'all'")
    parser.add_argument("--layers", type=str, default="all",
                        help="Comma-separated layer indices (1..L) or 'all'")
    parser.add_argument("--span_mode", type=str, default="bio", choices=["bio", "token"],
                        help="BIO span pooling or token-level pooling")
    parser.add_argument("--pool", type=str, default="mean", choices=["mean", "max", "first", "last"],
                        help="Pooling over subword vectors inside entity span")
    parser.add_argument("--entity_tags", type=str, default="PER,LOC,ORG,DATE",
                        help="Comma-separated entity tags to include")
    parser.add_argument("--title_suffix", type=str, default="", help="Optional suffix for plot titles/filenames")

    args = parser.parse_args()

    set_deterministic(args.seed)
    use_amp = args.fp16
    max_length = args.max_length
    title_suffix = f" {args.title_suffix.strip()}" if args.title_suffix.strip() else ""

    entity_tags = {t.strip() for t in args.entity_tags.split(",") if t.strip()}
    if not entity_tags:
        entity_tags = DEFAULT_ENTITY_TAGS

    logging.info(
        f"Start: model_type={args.model_type}, use_cls={args.use_cls}, "
        f"use_context={args.use_context}, context_window={args.context_window}, "
        f"max_tokens_per_type={args.max_tokens_per_type}, batch_size={args.batch_size}, "
        f"output_dir={args.output_dir}, save_prototypes={args.save_prototypes}, "
        f"seed={args.seed}, fp16={args.fp16}, max_length={args.max_length}, "
        f"langs={args.langs}, layers={args.layers}, span_mode={args.span_mode}, "
        f"pool={args.pool}, entity_tags={entity_tags}, title_suffix='{title_suffix.strip()}'"
    )

    # ---- Data paths (adjust BASE_DIR to your environment) ----
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data"))
    language_files = {
        "run": os.path.join(BASE_DIR, "COMBINED/train.txt"),
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

    # Filter languages
    if args.langs != "all":
        keep = {x.strip() for x in args.langs.split(",") if x.strip()}
        language_files = {k: v for k, v in language_files.items() if k in keep}

    # Prune missing files (warn)
    pruned = {}
    for lang, p in language_files.items():
        if os.path.isfile(p):
            pruned[lang] = p
        else:
            logging.warning(f"[skip] Missing file for {lang}: {p}")
    language_files = pruned

    if TARGET_LANG not in language_files:
        logging.error(f"Target language '{TARGET_LANG}' not present after filtering; abort.")
        print(f"Target language '{TARGET_LANG}' not present; nothing to do.")
        return

    # Resolve layers
    tmp_model = AutoModel.from_pretrained(MODEL_NAME_MAP[args.model_type])
    all_layers = get_all_hidden_layers(tmp_model)
    if args.layers != "all":
        try:
            layer_indices = [int(x) for x in args.layers.split(",") if x.strip()]
        except ValueError:
            layer_indices = all_layers
            logging.warning("Invalid --layers; defaulting to all layers")
        layer_indices = [li for li in layer_indices if li in all_layers]
        if not layer_indices:
            layer_indices = all_layers
    else:
        layer_indices = all_layers
    del tmp_model

    out_dir = os.path.join(args.output_dir, args.model_type)
    os.makedirs(out_dir, exist_ok=True)

    # Compute per-language, per-layer prototypes
    all_prototypes_by_layer, model, counts_by_lang = compute_language_prototypes_multilayer(
        language_files=language_files,
        model_type=args.model_type,
        layer_indices=layer_indices,
        entity_tags=entity_tags,
        max_tokens_per_type=args.max_tokens_per_type,
        output_dir=out_dir,
        use_cls=args.use_cls,
        use_context=args.use_context,
        context_window=args.context_window,
        batch_size=args.batch_size,
        save_prototypes=args.save_prototypes,
        span_mode=args.span_mode,
        pool=args.pool,
        use_amp=use_amp,
        max_length=max_length,
    )

    # Compute cosine similarities vs Runyankore for each layer
    sim_tables = compute_cosine_to_target_multilayer(
        all_prototypes_by_layer=all_prototypes_by_layer,
        counts_by_lang=counts_by_lang,
        target_lang=TARGET_LANG,
        output_dir=out_dir,
        entity_tags=entity_tags,
    )

    # Visualizations + summaries
    create_visualizations_for_sim_tables(
        sim_tables=sim_tables,
        model=model,
        output_dir=out_dir,
        model_type=args.model_type,
        title_suffix=title_suffix,
    )

    logging.info("Experiment finished successfully.")
    print("\nDone. Outputs in:", out_dir)

if __name__ == "__main__":
    main()
