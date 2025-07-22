import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
MODEL_NAME_MAP = {
    "xlmr": "xlm-roberta-base",
    "mbert": "bert-base-multilingual-cased"
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# FUNCTIONS
# -----------------------------

def load_model_and_tokenizer(model_type: str):
    """Load tokenizer and model for either XLM-R or mBERT."""
    assert model_type in MODEL_NAME_MAP, f"Model type must be one of {list(MODEL_NAME_MAP.keys())}"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_MAP[model_type])
    model = AutoModel.from_pretrained(MODEL_NAME_MAP[model_type]).to(DEVICE)
    model.eval()
    return tokenizer, model

def compute_mean_embedding(file_path: str, tokenizer, model, max_samples: int = 2000):
    """
    Compute mean sentence embedding for a language dataset.
    Each line in file_path is assumed to be a sentence.
    """
    embeddings = []
    with open(file_path, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    if max_samples:
        sentences = sentences[:max_samples]

    for sentence in tqdm(sentences, desc=f"Embedding {os.path.basename(file_path)}"):
        tokens = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
        with torch.no_grad():
            outputs = model(**tokens)
            # Use mean of last hidden state as sentence embedding
            sentence_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        embeddings.append(sentence_embedding)

    return np.mean(embeddings, axis=0)

def build_similarity_matrix(language_files: dict, model_type: str, output_dir: str = "outputs"):
    """
    language_files: dict { 'lang_code': 'path/to/file.txt', ... }
    Returns similarity matrix as pandas DataFrame.
    """
    tokenizer, model = load_model_and_tokenizer(model_type)

    # Compute embeddings for all languages
    lang_embeddings = {}
    for lang, file_path in language_files.items():
        lang_embeddings[lang] = compute_mean_embedding(file_path, tokenizer, model)

    # Compute cosine similarity matrix
    lang_codes = list(lang_embeddings.keys())
    embedding_matrix = np.vstack([lang_embeddings[lang] for lang in lang_codes])
    similarity = cosine_similarity(embedding_matrix)

    df_sim = pd.DataFrame(similarity, index=lang_codes, columns=lang_codes)

    # Save CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{model_type}_similarity_matrix.csv")
    df_sim.to_csv(csv_path)
    print(f"[INFO] Similarity matrix saved at {csv_path}")

    return df_sim

def plot_similarity_heatmap(df_sim, model_type: str, output_dir: str = "outputs"):
    """Generate and save similarity heatmap."""
    plt.figure(figsize=(10, 8))
    plt.imshow(df_sim, cmap="viridis", interpolation="nearest")
    plt.xticks(range(len(df_sim)), df_sim.columns, rotation=90)
    plt.yticks(range(len(df_sim)), df_sim.index)
    plt.colorbar(label="Cosine Similarity")
    plt.title(f"Language Similarity Heatmap ({model_type.upper()})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_type}_similarity_heatmap.png"))
    plt.close()

def get_top_k_languages(df_sim, target_lang: str, k: int = 5):
    """Return top-k similar languages for the target language."""
    sorted_langs = df_sim[target_lang].sort_values(ascending=False)
    top_k = sorted_langs.iloc[1:k+1]  # skip the target language itself
    print(f"\n[INFO] Top-{k} similar languages to {target_lang}:\n{top_k}\n")
    return top_k

# -----------------------------
# MAIN USAGE EXAMPLE
# -----------------------------
if __name__ == "__main__":
    # Example language file mapping (update with actual paths)
    language_files = {
        "run": "../data/SALT/train.txt",
        "bam": "../data/MasakhaNER2.0/bam/train.txt",
        "bbj": "../data/MasakhaNER2.0/bbj/train.txt",     
        "ewe": "../data/MasakhaNER2.0/ewe/train.txt",     
        "fon": "../data/MasakhaNER2.0/fon/train.txt",     
        "hau": "../data/MasakhaNER2.0/hau/train.txt",     
        "ibo": "../data/MasakhaNER2.0/ibo/train.txt",    
        "kin": "../data/MasakhaNER2.0/kin/train.txt",     
        "lug": "../data/MasakhaNER2.0/lug/train.txt",     
        "luo": "../data/MasakhaNER2.0/luo/train.txt",     
        "mos": "../data/MasakhaNER2.0/mos/train.txt",     
        "nya": "../data/MasakhaNER2.0/nya/train.txt",     
        "pcm": "../data/MasakhaNER2.0/pcm/train.txt",     
        "sna": "../data/MasakhaNER2.0/sna/train.txt",     
        "swa": "../data/MasakhaNER2.0/swa/train.txt",     
        "tsn": "../data/MasakhaNER2.0/tsn/train.txt",     
        "twi": "../data/MasakhaNER2.0/twi/train.txt",     
        "wol": "../data/MasakhaNER2.0/wol/train.txt",     
        "xho": "../data/MasakhaNER2.0/xho/train.txt",     
        "yor": "../data/MasakhaNER2.0/yor/train.txt",     
        "zul": "../data/MasakhaNER2.0/zul/train.txt"     
    }

    for model_type in ["xlmr", "mbert"]:
        df_sim = build_similarity_matrix(language_files, model_type)
        plot_similarity_heatmap(df_sim, model_type)
        get_top_k_languages(df_sim, target_lang="run", k=5)
