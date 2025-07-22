import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------
TARGET_LANG = "run"  # Runyankore
TOP_K = 5
SIMILARITY_FILES = {
    "xlmr": "./outputs/xlmr_similarity_matrix.csv",
    "mbert": "./outputs/mbert_similarity_matrix.csv"
}
SAVE_TO_FILE = True
OUTPUT_DIR = "outputs"

# -----------------------------
# FUNCTIONS
# -----------------------------

def extract_top_bottom_k(similarity_file, model_type, target_lang=TARGET_LANG, k=TOP_K):
    """Extract Top-k and Bottom-k languages from similarity matrix."""
    df = pd.read_csv(similarity_file, index_col=0)

    # Sort descending for Top-k, ascending for Bottom-k
    top_k = df[target_lang].sort_values(ascending=False).iloc[1:k+1]
    bottom_k = df[target_lang].sort_values(ascending=True).iloc[:k]

    print(f"\n=== {model_type.upper()} ===")
    print(f"Top-{k} similar languages to {target_lang}:\n{top_k}\n")
    print(f"Bottom-{k} least similar languages to {target_lang}:\n{bottom_k}\n")

    return top_k, bottom_k

def save_top_bottom_k(top_k, bottom_k, model_type, output_dir=OUTPUT_DIR):
    """Save extracted Top-k and Bottom-k to CSV."""
    top_k.to_csv(f"{output_dir}/{model_type}_top_{TOP_K}.csv")
    bottom_k.to_csv(f"{output_dir}/{model_type}_bottom_{TOP_K}.csv")
    print(f"[INFO] Saved top & bottom k CSVs for {model_type.upper()} in {output_dir}/")

# -----------------------------
# MAIN USAGE
# -----------------------------
if __name__ == "__main__":
    for model_type, file_path in SIMILARITY_FILES.items():
        top_k, bottom_k = extract_top_bottom_k(file_path, model_type)
        if SAVE_TO_FILE:
            save_top_bottom_k(top_k, bottom_k, model_type)
