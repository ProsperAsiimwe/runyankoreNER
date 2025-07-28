
import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

MODEL_NAME_MAP = {
    "xlmr": "xlm-roberta-base",
    "mbert": "bert-base-multilingual-cased"
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_LANG = "run"


def extract_bio_sentences(file_path: str):
    sentences = []
    current_tokens = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_tokens:
                    sentences.append(" ".join(current_tokens))
                    current_tokens = []
            else:
                parts = line.split()
                if len(parts) == 2:
                    token, tag = parts
                    current_tokens.append(token)
    if current_tokens:
        sentences.append(" ".join(current_tokens))
    return sentences


def load_model_and_tokenizer(model_type: str):
    assert model_type in MODEL_NAME_MAP, f"Model type must be one of {list(MODEL_NAME_MAP.keys())}"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_MAP[model_type])
    model = AutoModel.from_pretrained(MODEL_NAME_MAP[model_type]).to(DEVICE)
    model.eval()
    return tokenizer, model


def compute_mean_embedding(file_path: str, tokenizer, model, max_samples: int):
    embeddings = []
    sentences = extract_bio_sentences(file_path)
    if max_samples:
        sentences = sentences[:max_samples]

    for sentence in tqdm(sentences, desc=f"Embedding {os.path.basename(file_path)}"):
        tokens = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
        with torch.no_grad():
            outputs = model(**tokens)
            sentence_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        embeddings.append(sentence_embedding)

    return np.mean(embeddings, axis=0), sentences


def compute_runyankore_similarity(language_files: dict, model_type: str, max_samples: int, base_output_dir: str = "outputs_sample_size_tuning"):
    output_dir = os.path.join(base_output_dir, str(max_samples))
    tokenizer, model = load_model_and_tokenizer(model_type)
    assert TARGET_LANG in language_files, f"{TARGET_LANG} must be in language_files!"

    run_embedding, run_sentences = compute_mean_embedding(language_files[TARGET_LANG], tokenizer, model, max_samples)
    os.makedirs(output_dir, exist_ok=True)
    run_sentences_path = os.path.join(output_dir, f"{model_type}_{TARGET_LANG}_used_sentences.txt")
    with open(run_sentences_path, "w", encoding="utf-8") as sf:
        sf.write("\n".join(run_sentences))
    print(f"[INFO] Saved Runyankore sentences at {run_sentences_path}")

    similarities = {}
    for lang, file_path in language_files.items():
        if lang == TARGET_LANG:
            continue
        embedding, sentences = compute_mean_embedding(file_path, tokenizer, model, max_samples)

        sent_path = os.path.join(output_dir, f"{model_type}_{lang}_used_sentences.txt")
        with open(sent_path, "w", encoding="utf-8") as sf:
            sf.write("\n".join(sentences))
        print(f"[INFO] Saved sentences used for {lang} at {sent_path}")

        sim = cosine_similarity(
            run_embedding.reshape(1, -1),
            embedding.reshape(1, -1)
        )[0][0]
        similarities[lang] = sim

    df_sim = pd.DataFrame(similarities.items(), columns=["Language", "Cosine_Similarity"])
    df_sim = df_sim.sort_values(by="Cosine_Similarity", ascending=False).reset_index(drop=True)
    csv_path = os.path.join(output_dir, f"{model_type}_{TARGET_LANG}_similarities.csv")
    df_sim.to_csv(csv_path, index=False)
    print(f"[INFO] Saved similarity results at {csv_path}")

    return df_sim


def plot_runyankore_similarity(df_sim: pd.DataFrame, model_type: str, max_samples: int, base_output_dir: str = "outputs_sample_size_tuning"):
    output_dir = os.path.join(base_output_dir, str(max_samples))
    plt.figure(figsize=(10, 6))
    plt.bar(df_sim["Language"], df_sim["Cosine_Similarity"])
    plt.xticks(rotation=45)
    plt.ylabel("Cosine Similarity")
    plt.title(f"Runyankore vs Other Languages Similarity ({model_type.upper()} - {max_samples} Samples)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_type}_{TARGET_LANG}_similarity_bar.png"))
    plt.close()


if __name__ == "__main__":
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

    for max_samples in [2000, 2500, 3000, 3500, 4000, 4500, 5000]:
        for model_type in ["xlmr", "mbert"]:
            df_result = compute_runyankore_similarity(language_files, model_type, max_samples)
            plot_runyankore_similarity(df_result, model_type, max_samples)
            print(f"\nTop-5 similar languages to Runyankore ({model_type.upper()}, {max_samples} Samples):\n{df_result.head(5)}\n")
