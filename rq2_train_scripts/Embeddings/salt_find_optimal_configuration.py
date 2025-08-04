import os
import re
import pandas as pd

# Define language families for readability (optional use)
language_families = {
    "bam": "Mande", "bbj": "Bantu", "ewe": "Kwa", "fon": "Kwa", "hau": "Chadic",
    "ibo": "Volta-Niger", "kin": "Bantu", "lug": "Bantu", "luo": "Nilotic",
    "mos": "Gur", "nya": "Bantu", "pcm": "English Creole", "sna": "Bantu",
    "swa": "Bantu", "tsn": "Bantu", "twi": "Kwa", "wol": "Atlantic-Congo",
    "xho": "Bantu", "yor": "Volta-Niger", "zul": "Bantu"
}

def extract_config_info(path):
    model_match = re.search(r"salt/([^/]+)/", path)
    config_match = re.search(r"config_\d+_cls(.*?)_ctx(.*?)_hyb(.*?)/tokens_(\d+)_ctx(\d+)", path)

    if not model_match:
        print("Could not extract model from:", path)
    if not config_match:
        print("Could not extract config from:", path)

    if model_match and config_match:
        return {
            "Model": model_match.group(1),
            "CLS": config_match.group(1),
            "CTX": config_match.group(2),
            "HYB": config_match.group(3),
            "Tokens": int(config_match.group(4)),
            "Window": int(config_match.group(5))
        }
    return None

def load_all_results(base_dir="MEXA_inspired_strategy/outputs/salt"):
    rows = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.startswith("similarity_to_run") and file.endswith(".csv"):
                path = os.path.join(root, file)
                config = extract_config_info(path)
                if config is None:
                    continue
                try:
                    df = pd.read_csv(path, index_col=0)
                except Exception as e:
                    print(f"Failed to read {path}: {e}")
                    continue

                if "Mean" not in df.columns:
                    print(f"Skipping file without 'Mean' column: {path}")
                    continue

                for lang, row in df.iterrows():
                    rows.append({
                        **config,
                        "Language": lang,
                        "MeanSim": round(row["Mean"], 4)
                    })

    df = pd.DataFrame(rows)
    print("\nModels processed:", df["Model"].unique())
    return df

def find_best_config(df, model_name):
    group_cols = ["CLS", "CTX", "HYB", "Tokens", "Window"]
    model_df = df[df["Model"] == model_name]

    if model_df.empty:
        print(f"No data found for model: {model_name}")
        return None

    config_avgs = model_df.groupby(group_cols)["MeanSim"].mean().reset_index()
    best_row = config_avgs.loc[config_avgs["MeanSim"].idxmax()]
    return best_row

def print_top_bottom(df, model_name, config_row):
    mask = (df["Model"] == model_name) & \
           (df["CLS"] == config_row["CLS"]) & \
           (df["CTX"] == config_row["CTX"]) & \
           (df["HYB"] == config_row["HYB"]) & \
           (df["Tokens"] == config_row["Tokens"]) & \
           (df["Window"] == config_row["Window"])

    config_df = df[mask].sort_values("MeanSim", ascending=False)

    print(f"\n===== Optimal Configuration for {model_name.upper()} =====")
    print(config_row.to_string(index=True))

    print("\nTop-5 Most Similar Languages:")
    for _, row in config_df.head(5).iterrows():
        family = language_families.get(row["Language"], "Unknown")
        print(f"  {row['Language']} ({row['MeanSim']:.4f}) - {family}")

    print("\nBottom-5 Least Similar Languages:")
    for _, row in config_df.tail(5).iterrows():
        family = language_families.get(row["Language"], "Unknown")
        print(f"  {row['Language']} ({row['MeanSim']:.4f}) - {family}")

def main():
    df = load_all_results()
    if df.empty:
        print("No similarity data found.")
        return

    for model in ["xlmr", "mbert"]:
        best_config = find_best_config(df, model)
        if best_config is not None:
            print_top_bottom(df, model, best_config)

if __name__ == "__main__":
    main()
