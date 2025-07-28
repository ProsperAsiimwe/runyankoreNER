import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

"""
 Runyankore + Pairwise Debugging + Entity-Only Similarity
 
 1. Pairwise Debugging Mode:
    - Function compare_two_languages(lang1, lang2, ...) compares any two languages directly.
    - Prints similarity score in either full-text or entity-only mode.

2. Entity-Only Similarity (BIO-Based)
    - Flag entity_only=True.
    - Extracts only tokens with BIO tags (B-XXX or I-XXX) before computing embeddings.
    - Saves:
        - *_used_sentences_entityonly.txt
        - *_similarities_entityonly.csv
        - *_similarity_bar_entityonly.png
"""

# -----------------------------
# CONFIG
# -----------------------------
MODEL_NAME_MAP = {
    "xlmr": "xlm-roberta-base",
    "mbert": "bert-base-multilingual-cased"
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_LANG = "run"  # Default target (Runyankore)
MAX_SAMPLES = 2000

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------

def load_model_and_tokenizer(model_type: str):
    """Load tokenizer and model for either XLM-R or mBERT."""
    assert model_type in MODEL_NAME_MAP, f"Model type must be one of {list(MODEL_NAME_MAP.keys())}"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_MAP[model_type])
    model = AutoModel.from_pretrained(MODEL_NAME_MAP[model_type]).to(DEVICE)
    model.eval()
    return tokenizer, model

def extract_sentences_from_bio(file_path: str):
    """
    Extract sentences based only on tokens with BIO entity tags.
    Returns a list of sentences (space-joined entity tokens).
    """
    entity_tokens = []
    current_sentence = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                # End of sentence
                if current_sentence:
                    entity_tokens.append(" ".join(current_sentence))
                    current_sentence = []
            else:
                parts = line.split()
                if len(parts) == 2:
                    token, tag = parts
                    if tag != "O":  # Only keep entity tokens
                        current_sentence.append(token)

    if current_sentence:
        entity_tokens.append(" ".join(current_sentence))

    return entity_tokens

def compute_mean_embedding(file_path: str, tokenizer, model, max_samples: int = MAX_SAMPLES, entity_only: bool = False):
    """
    Compute mean sentence embedding (full text or entity-only).
    Returns mean embedding and the sentences used.
    """
    if entity_only:
        sentences = extract_sentences_from_bio(file_path)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            sentences = [line.strip() for line in f if line.strip()]

    if max_samples:
        sentences = sentences[:max_samples]

    embeddings = []
    for sentence in tqdm(sentences, desc=f"Embedding {os.path.basename(file_path)}"):
        tokens = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
        with torch.no_grad():
            outputs = model(**tokens)
            sentence_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        embeddings.append(sentence_embedding)

    return np.mean(embeddings, axis=0), sentences

# -----------------------------
# RUNYANKORE VS ALL LANGUAGES
# -----------------------------

def compute_runyankore_similarity(language_files: dict, model_type: str, output_dir: str = "outputs_advanced_language_similarity", entity_only: bool = False):
    """
    Compute similarity between Runyankore and other languages.
    Supports entity-only mode (BIO-based).
    """
    tokenizer, model = load_model_and_tokenizer(model_type)
    os.makedirs(output_dir, exist_ok=True)

    # Compute Runyankore embedding
    run_embedding, run_sentences = compute_mean_embedding(language_files[TARGET_LANG], tokenizer, model, entity_only=entity_only)
    run_sentences_path = os.path.join(output_dir, f"{model_type}_{TARGET_LANG}_used_sentences{'_entityonly' if entity_only else ''}.txt")
    with open(run_sentences_path, "w", encoding="utf-8") as sf:
        sf.write("\n".join(run_sentences))
    print(f"[INFO] Saved Runyankore sentences at {run_sentences_path}")

    similarities = {}
    for lang, file_path in language_files.items():
        if lang == TARGET_LANG:
            continue
        embedding, sentences = compute_mean_embedding(file_path, tokenizer, model, entity_only=entity_only)

        # Save sentences used
        sent_path = os.path.join(output_dir, f"{model_type}_{lang}_used_sentences{'_entityonly' if entity_only else ''}.txt")
        with open(sent_path, "w", encoding="utf-8") as sf:
            sf.write("\n".join(sentences))

        sim = cosine_similarity(run_embedding.reshape(1, -1), embedding.reshape(1, -1))[0][0]
        similarities[lang] = sim

    # Save similarity CSV
    suffix = "_entityonly" if entity_only else ""
    df_sim = pd.DataFrame(similarities.items(), columns=["Language", "Cosine_Similarity"])
    df_sim = df_sim.sort_values(by="Cosine_Similarity", ascending=False).reset_index(drop=True)
    csv_path = os.path.join(output_dir, f"{model_type}_{TARGET_LANG}_similarities{suffix}.csv")
    df_sim.to_csv(csv_path, index=False)
    print(f"[INFO] Saved similarity results at {csv_path}")

    return df_sim

def plot_runyankore_similarity(df_sim: pd.DataFrame, model_type: str, entity_only: bool = False, output_dir: str = "outputs_advanced_language_similarity"):
    """Generate a bar plot for Runyankore vs other languages similarity."""
    plt.figure(figsize=(10, 6))
    plt.bar(df_sim["Language"], df_sim["Cosine_Similarity"])
    plt.xticks(rotation=45)
    plt.ylabel("Cosine Similarity")
    title_suffix = "Entity-Only" if entity_only else "Full Text"
    plt.title(f"Runyankore vs Other Languages Similarity ({model_type.upper()} - {title_suffix})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_type}_{TARGET_LANG}_similarity_bar{'_entityonly' if entity_only else ''}.png"))
    plt.close()

# -----------------------------
# PAIRWISE DEBUGGING MODE
# -----------------------------

def compare_two_languages(lang1: str, lang2: str, language_files: dict, model_type: str, entity_only: bool = False):
    """Compare similarity between any two selected languages (debug mode)."""
    tokenizer, model = load_model_and_tokenizer(model_type)

    emb1, _ = compute_mean_embedding(language_files[lang1], tokenizer, model, entity_only=entity_only)
    emb2, _ = compute_mean_embedding(language_files[lang2], tokenizer, model, entity_only=entity_only)

    sim = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
    mode = "Entity-Only" if entity_only else "Full Text"
    print(f"\n[DEBUG] Cosine Similarity between {lang1} and {lang2} ({model_type.upper()}, {mode}): {sim:.6f}")
    return sim

# -----------------------------
# MAIN USAGE
# -----------------------------
if __name__ == "__main__":
    # Update with your actual file paths
    language_files = {
        "run": "../../data/SALT/train.txt",
        "bam": "../../data/MasakhaNER2.0/bam/train.txt",
        "bbj": "../../data/MasakhaNER2.0/bbj/train.txt",     
        "ewe": "../../data/MasakhaNER2.0/ewe/train.txt",     
        "fon": "../../data/MasakhaNER2.0/fon/train.txt",     
        "hau": "../../data/MasakhaNER2.0/hau/train.txt",     
        "ibo": "../../data/MasakhaNER2.0/ibo/train.txt",    
        "kin": "../../data/MasakhaNER2.0/kin/train.txt",     
        "lug": "../../data/MasakhaNER2.0/lug/train.txt",     
        "luo": "../../data/MasakhaNER2.0/luo/train.txt",     
        "mos": "../../data/MasakhaNER2.0/mos/train.txt",     
        "nya": "../../data/MasakhaNER2.0/nya/train.txt",     
        "pcm": "../../data/MasakhaNER2.0/pcm/train.txt",     
        "sna": "../../data/MasakhaNER2.0/sna/train.txt",     
        "swa": "../../data/MasakhaNER2.0/swa/train.txt",     
        "tsn": "../../data/MasakhaNER2.0/tsn/train.txt",     
        "twi": "../../data/MasakhaNER2.0/twi/train.txt",     
        "wol": "../../data/MasakhaNER2.0/wol/train.txt",     
        "xho": "../../data/MasakhaNER2.0/xho/train.txt",     
        "yor": "../../data/MasakhaNER2.0/yor/train.txt",     
        "zul": "../../data/MasakhaNER2.0/zul/train.txt"     
    }

    for model_type in ["xlmr", "mbert"]:
        # === 1) RUNYANKORE FULL TEXT SIMILARITY ===
        df_full = compute_runyankore_similarity(language_files, model_type)
        plot_runyankore_similarity(df_full, model_type)

        # === 2) RUNYANKORE ENTITY-ONLY SIMILARITY ===
        df_entity = compute_runyankore_similarity(language_files, model_type, entity_only=True)
        plot_runyankore_similarity(df_entity, model_type, entity_only=True)

        # === 3) DEBUGGING: PAIRWISE COMPARISON ===
        compare_two_languages("lug", "xho", language_files, model_type)
        compare_two_languages("lug", "xho", language_files, model_type, entity_only=True)
