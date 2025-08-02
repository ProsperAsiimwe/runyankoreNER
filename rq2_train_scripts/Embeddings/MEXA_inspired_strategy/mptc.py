import os
import argparse
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import logging
from datetime import datetime

MODEL_NAME_MAP = {
    "xlmr": "xlm-roberta-base",
    "mbert": "bert-base-multilingual-cased"
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
    level=logging.INFO
)

def extract_entity_embeddings(file_path, tokenizer, model, entity_tags=ENTITY_TAGS, max_tokens_per_type=500,
                              use_cls=False, use_context=False, context_window=2, save_examples_path=None):
    entity_embeddings = {tag: [] for tag in entity_tags}
    sentence_samples = {tag: [] for tag in entity_tags}

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    tokens, tags = [], []
    for line in tqdm(lines, desc=f"Reading {os.path.basename(file_path)}"):
        line = line.strip()
        if not line:
            for idx, tag in enumerate(tags):
                stripped_tag = tag.replace("B-", "").replace("I-", "")
                if stripped_tag in entity_tags and len(entity_embeddings[stripped_tag]) < max_tokens_per_type:
                    start = max(0, idx - context_window) if use_context else idx
                    end = min(len(tokens), idx + context_window + 1) if use_context else idx + 1
                    span = tokens[start:end]
                    if not span: continue
                    sentence_samples[stripped_tag].append(" ".join(span))
                    tokenized = tokenizer(span if not use_cls else [" ".join(span)],
                                          is_split_into_words=not use_cls,
                                          return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
                    with torch.no_grad():
                        outputs = model(**tokenized)
                    if use_cls:
                        emb = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
                    else:
                        word_ids = tokenized.word_ids()
                        first_subword_idx = next((i for i, wid in enumerate(word_ids) if wid == 0), None)
                        emb = outputs.last_hidden_state[0, first_subword_idx].cpu().numpy() if first_subword_idx else None
                    if emb is not None:
                        entity_embeddings[stripped_tag].append(emb)
            tokens, tags = [], []
        else:
            parts = line.split()
            if len(parts) == 2:
                token, tag = parts
                tokens.append(token)
                tags.append(tag)

    if save_examples_path:
        os.makedirs(save_examples_path, exist_ok=True)
        for tag in sentence_samples:
            with open(os.path.join(save_examples_path, f"{tag}_examples.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(sentence_samples[tag]))

    return entity_embeddings

def compute_language_prototypes(language_files, model_type, max_tokens_per_type=500, output_dir="mptc_outputs_prototypes",
                                use_cls=False, use_context=False, context_window=2):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_MAP[model_type])
    model = AutoModel.from_pretrained(MODEL_NAME_MAP[model_type]).to(DEVICE)
    model.eval()

    os.makedirs(output_dir, exist_ok=True)
    all_prototypes = {}

    for lang, path in tqdm(language_files.items(), desc="Computing prototypes"):
        logging.info(f"Processing {lang}")
        example_path = os.path.join(output_dir, f"{lang}_examples")
        embeddings = extract_entity_embeddings(path, tokenizer, model, ENTITY_TAGS, max_tokens_per_type,
                                               use_cls, use_context, context_window, save_examples_path=example_path)
        prototypes = {
            tag: np.mean(embs, axis=0) if embs else np.zeros(model.config.hidden_size)
            for tag, embs in embeddings.items()
        }
        all_prototypes[lang] = prototypes
        np.savez(os.path.join(output_dir, f"{lang}_prototypes_{model_type}.npz"), **prototypes)

    return all_prototypes

def compute_similarity_to_target(all_prototypes, output_dir="mptc_outputs_prototypes", use_hybrid=False):
    similarities = {}
    run_prototypes = all_prototypes[TARGET_LANG]
    for lang, prototypes in all_prototypes.items():
        if lang == TARGET_LANG:
            continue
        sim_scores = {}
        for tag in ENTITY_TAGS:
            cos_sim = cosine_similarity(run_prototypes[tag].reshape(1, -1), prototypes[tag].reshape(1, -1))[0][0]
            if use_hybrid:
                euc_dist = np.linalg.norm(run_prototypes[tag] - prototypes[tag])
                hybrid = (cos_sim + (1 / (1 + euc_dist))) / 2
                sim_scores[tag] = hybrid
            else:
                sim_scores[tag] = cos_sim
        sim_scores["Mean"] = np.mean(list(sim_scores.values()))
        similarities[lang] = sim_scores

    df = pd.DataFrame.from_dict(similarities, orient="index").sort_values("Mean", ascending=False)
    df.to_csv(os.path.join(output_dir, f"similarity_to_{TARGET_LANG}.csv"))

    plot_similarity_bar(df, output_dir)
    for tag in ENTITY_TAGS:
        plot_similarity_bar(df.sort_values(tag, ascending=False), output_dir, title_suffix=tag, col=tag)

    logging.info("Top 5 languages most similar to Runyankore:")
    print("\nTop 5 languages most similar to Runyankore:")
    for i, row in enumerate(df.head(5).itertuples(), 1):
        logging.info(f"{i}. {row.Index} - Mean Similarity: {row.Mean:.4f}")
        print(f"{i}. {row.Index} - Mean Similarity: {row.Mean:.4f}")

    return df

def plot_similarity_bar(df, output_dir, title_suffix="Mean", col="Mean"):
    plt.figure(figsize=(10, 6))
    plt.bar(df.index, df[col])
    plt.xticks(rotation=45)
    plt.ylabel("Similarity")
    plt.title(f"Similarity to {TARGET_LANG.upper()} by {title_suffix}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"bar_similarity_{col.lower()}.png"))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Prototype-based Language Similarity with Optional Extensions")
    parser.add_argument("--model_type", type=str, default="xlmr", choices=["xlmr", "mbert"])
    parser.add_argument("--use_cls", action="store_true", help="Use CLS token instead of mean pooling")
    parser.add_argument("--use_context", action="store_true", help="Use context window around entity")
    parser.add_argument("--use_hybrid", action="store_true", help="Use cosine + Euclidean hybrid distance")
    parser.add_argument("--context_window", type=int, default=2, help="Size of context window")
    parser.add_argument("--max_tokens_per_type", type=int, default=500, help="Number of entity tokens per type")
    parser.add_argument("--output_dir", type=str, default="outputs_prototypes")
    args = parser.parse_args()

    logging.info(f"Experiment started with model_type={args.model_type}, use_cls={args.use_cls}, "
                 f"use_context={args.use_context}, context_window={args.context_window}, "
                 f"use_hybrid={args.use_hybrid}, max_tokens_per_type={args.max_tokens_per_type}, "
                 f"output_dir={args.output_dir}")

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

    all_prototypes = compute_language_prototypes(
        language_files,
        model_type=args.model_type,
        max_tokens_per_type=args.max_tokens_per_type,
        output_dir=args.output_dir,
        use_cls=args.use_cls,
        use_context=args.use_context,
        context_window=args.context_window
    )
    compute_similarity_to_target(all_prototypes, output_dir=args.output_dir, use_hybrid=args.use_hybrid)
    logging.info("Experiment finished successfully.")

if __name__ == "__main__":
    main()
