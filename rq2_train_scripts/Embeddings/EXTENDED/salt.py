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
from sklearn.metrics.pairwise import cosine_similarity  # kept for API parity; we use a safer cosine below
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

# Setup logging
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
# Utils
# --------------------

def set_seed(s: int = 42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def _safe_cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    """Cosine similarity that gracefully handles NaNs and near-zero norms."""
    if a is None or b is None:
        return np.nan
    if np.isnan(a).any() or np.isnan(b).any():
        return np.nan
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return np.nan
    return float(np.dot(a, b) / (na * nb))

# --------------------
# Core: batched span extraction
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
        # last flush
        if tokens:
            yield tokens, tags


def extract_entity_embeddings_batched(
    file_path: str,
    tokenizer,
    model,
    entity_tags=ENTITY_TAGS,
    max_tokens_per_type: int = 500,
    use_cls: bool = False,
    use_context: bool = False,
    context_window: int = 2,
    batch_size: int = 64,
    save_examples_path: str | None = None,
):
    """
    Build embeddings for entity tokens with correct word-subword alignment.
    Batched for speed. For non-CLS, mean-pools all subwords of the entity token.
    """
    assert getattr(tokenizer, "is_fast", False), "This function requires a fast tokenizer (word_ids support)."

    # Buffers
    per_tag_embeddings = {tag: [] for tag in entity_tags}
    per_tag_examples = {tag: [] for tag in entity_tags}
    per_tag_counts = {tag: 0 for tag in entity_tags}

    # Global buffers for a batch
    batch_payload = []  # list of dicts with fields needed for pooling
    batch_targets = []  # which tag each item belongs to
    batch_examples = []  # example strings to save

    def _flush_batch():
        """Run model on accumulated batch_payload and distribute embeddings."""
        nonlocal batch_payload, batch_targets, batch_examples
        if not batch_payload:
            return

        if use_cls:
            # CLS mode: tokenize packed strings
            texts = [" ".join(item["span_tokens"]) for item in batch_payload]
            tokenized = tokenizer(
                texts,
                is_split_into_words=False,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True,
            ).to(DEVICE)
            with torch.no_grad():
                outputs = model(**tokenized)  # last_hidden_state: (B, T, H)
            cls_embs = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()  # (B, H)

            for emb, tag, ex in zip(cls_embs, batch_targets, batch_examples):
                if per_tag_counts[tag] < max_tokens_per_type:
                    per_tag_embeddings[tag].append(emb)
                    per_tag_examples[tag].append(ex)
                    per_tag_counts[tag] += 1
        else:
            # Non-CLS: split into words and mean-pool subwords for the entity word
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
                outputs = model(**tokenized)  # (B, T, H)
            last_hidden = outputs.last_hidden_state  # (B, T, H)

            # For each item in batch, pool over subword indices where word_ids == rel_idx
            for i, (item, tag, ex) in enumerate(zip(batch_payload, batch_targets, batch_examples)):
                if per_tag_counts[tag] >= max_tokens_per_type:
                    continue

                word_ids = tokenized.word_ids(batch_index=i)
                rel_idx = item["rel_idx"]

                sub_idx = [j for j, wid in enumerate(word_ids) if wid == rel_idx]
                if not sub_idx:
                    # fallback: first non-special token
                    sub_idx = [j for j, wid in enumerate(word_ids) if wid is not None][:1]
                if not sub_idx:
                    continue  # skip if truly nothing to pool

                hidden = last_hidden[i, sub_idx, :]  # (k, H)
                emb = hidden.mean(dim=0).detach().cpu().numpy()
                per_tag_embeddings[tag].append(emb)
                per_tag_examples[tag].append(ex)
                per_tag_counts[tag] += 1

        # reset buffers
        batch_payload = []
        batch_targets = []
        batch_examples = []

    # Iterate sentences and collect spans for batching
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
            rel_idx = idx - start  # entity token's index inside span

            batch_payload.append({"span_tokens": span, "rel_idx": rel_idx})
            batch_targets.append(stripped_tag)
            batch_examples.append(" ".join(span))

            if len(batch_payload) >= batch_size:
                _flush_batch()

    # flush remaining
    _flush_batch()

    # Save examples if requested
    if save_examples_path:
        os.makedirs(save_examples_path, exist_ok=True)
        for tag in per_tag_examples:
            with open(os.path.join(save_examples_path, f"{tag}_examples.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(per_tag_examples[tag]))

    return per_tag_embeddings, per_tag_counts


# --------------------
# Prototypes
# --------------------

def compute_language_prototypes(
    language_files: dict,
    model_type: str,
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
    all_prototypes: dict[str, dict[str, np.ndarray]] = {}
    all_counts: dict[str, dict[str, int]] = {}

    for lang, path in tqdm(language_files.items(), desc="Computing prototypes"):
        logging.info(f"Processing {lang}")
        example_path = os.path.join(output_dir, f"{lang}_examples")

        embeddings_by_tag, counts_by_tag = extract_entity_embeddings_batched(
            path,
            tokenizer,
            model,
            entity_tags=ENTITY_TAGS,
            max_tokens_per_type=max_tokens_per_type,
            use_cls=use_cls,
            use_context=use_context,
            context_window=context_window,
            batch_size=batch_size,
            save_examples_path=example_path,
        )

        prototypes = {}
        H = model.config.hidden_size
        for tag, embs in embeddings_by_tag.items():
            if len(embs) == 0:
                prototypes[tag] = np.full((H,), np.nan, dtype=np.float32)  # mark missing cleanly
            else:
                arr = np.stack(embs).astype(np.float32)
                prototypes[tag] = arr.mean(axis=0)

        all_prototypes[lang] = prototypes
        all_counts[lang] = counts_by_tag

        # save per-language prototypes (.npz with NaNs preserved)
        np.savez(os.path.join(output_dir, f"{lang}_prototypes_{model_type}.npz"), **prototypes)

    # persist counts for transparency/debugging
    pd.DataFrame.from_dict(all_counts, orient="index").to_csv(os.path.join(output_dir, "prototype_counts.csv"))
    return all_prototypes


# --------------------
# Similarity + plots
# --------------------

def compute_similarity_to_target(all_prototypes: dict, output_dir: str = "outputs_prototypes", use_hybrid: bool = False):
    similarities = {}
    run_prototypes = all_prototypes[TARGET_LANG]

    for lang, prototypes in all_prototypes.items():
        if lang == TARGET_LANG:
            continue

        sim_scores = {}
        for tag in ENTITY_TAGS:
            a = run_prototypes[tag]
            b = prototypes[tag]
            cos_sim = _safe_cosine(a, b)
            if use_hybrid and not np.isnan(cos_sim):
                if a is None or b is None or np.isnan(a).any() or np.isnan(b).any():
                    sim_scores[tag] = np.nan
                else:
                    euc = np.linalg.norm(a - b)
                    sim_scores[tag] = (cos_sim + (1.0 / (1.0 + euc))) / 2.0
            else:
                sim_scores[tag] = cos_sim

        valid = [v for v in sim_scores.values() if not np.isnan(v)]
        sim_scores["Mean"] = float(np.mean(valid)) if valid else np.nan
        similarities[lang] = sim_scores

    df = pd.DataFrame.from_dict(similarities, orient="index")
    df = df.sort_values("Mean", ascending=False, na_position="last")
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, f"similarity_to_{TARGET_LANG}.csv"))

    plot_similarity_bar(df, output_dir)
    for tag in ENTITY_TAGS:
        plot_similarity_bar(
            df.sort_values(tag, ascending=False, na_position="last"),
            output_dir,
            title_suffix=tag,
            col=tag,
        )

    logging.info("Top 5 languages most similar to Runyankore:")
    print("\nTop 5 languages most similar to Runyankore:")
    shown = 0
    for i, row in enumerate(df.itertuples(), 1):
        if np.isnan(row.Mean):
            continue
        logging.info(f"{i}. {row.Index} - Mean Similarity: {row.Mean:.4f}")
        print(f"{i}. {row.Index} - Mean Similarity: {row.Mean:.4f}")
        shown += 1
        if shown == 5:
            break

    return df


def plot_similarity_bar(df: pd.DataFrame, output_dir: str, title_suffix: str = "Mean", col: str = "Mean"):
    vals = df[col].fillna(0.0)
    plt.figure(figsize=(10, 6))
    plt.bar(df.index, vals)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Similarity")
    plt.title(f"Similarity to {TARGET_LANG.upper()} by {title_suffix}")
    plt.tight_layout()
    fname_png = f"bar_similarity_{col.lower()}.png"
    fname_svg = f"bar_similarity_{col.lower()}.svg"
    plt.savefig(os.path.join(output_dir, fname_png), dpi=200)
    plt.savefig(os.path.join(output_dir, fname_svg))
    plt.close()

# --------------------
# Main
# --------------------

def main():
    parser = argparse.ArgumentParser(description="Prototype-based Language Similarity (batched, robust)")
    parser.add_argument("--model_type", type=str, default="xlmr", choices=["xlmr", "mbert"])
    parser.add_argument("--use_cls", action="store_true", help="Use CLS token instead of subword pooling")
    parser.add_argument("--use_context", action="store_true", help="Use context window around entity")
    parser.add_argument("--use_hybrid", action="store_true", help="Use cosine + Euclidean hybrid distance")
    parser.add_argument("--context_window", type=int, default=2, help="Size of context window")
    parser.add_argument("--max_tokens_per_type", type=int, default=500, help="Max entity tokens per tag")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for model inference")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="outputs_prototypes")
    args = parser.parse_args()

    set_seed(args.seed)

    logging.info(
        f"Experiment started with model_type={args.model_type}, use_cls={args.use_cls}, "
        f"use_context={args.use_context}, context_window={args.context_window}, "
        f"use_hybrid={args.use_hybrid}, max_tokens_per_type={args.max_tokens_per_type}, "
        f"batch_size={args.batch_size}, seed={args.seed}, output_dir={args.output_dir}"
    )

    # Base data dir (unchanged from your original)
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

    all_prototypes = compute_language_prototypes(
        language_files=language_files,
        model_type=args.model_type,
        max_tokens_per_type=args.max_tokens_per_type,
        output_dir=args.output_dir,
        use_cls=args.use_cls,
        use_context=args.use_context,
        context_window=args.context_window,
        batch_size=args.batch_size,
    )

    compute_similarity_to_target(
        all_prototypes,
        output_dir=args.output_dir,
        use_hybrid=args.use_hybrid,
    )

    logging.info("Experiment finished successfully.")


if __name__ == "__main__":
    main()
