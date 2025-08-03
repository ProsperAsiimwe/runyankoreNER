import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Language-to-family mapping
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
    if model_match and config_match:
        return {
            "Model": model_match.group(1),
            "CLS": config_match.group(1),
            "CTX": config_match.group(2),
            "HYB": config_match.group(3),
            "Tokens": config_match.group(4),
            "Window": config_match.group(5)
        }
    return None

def compile_ranked_similarity(base_dir="MEXA_inspired_strategy/outputs/salt"):
    all_rows = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.startswith("similarity_to_run") and file.endswith(".csv"):
                path = os.path.join(root, file)
                config = extract_config_info(path)
                if config is None:
                    continue
                df = pd.read_csv(path, index_col=0)
                df_sorted = df.sort_values("Mean", ascending=False)
                for rank, (lang, row) in enumerate(df_sorted.iterrows(), 1):
                    all_rows.append({
                        **config,
                        "Rank": rank,
                        "Language": lang,
                        "MeanSim": round(row["Mean"], 4),
                        "Family": language_families.get(lang, "Unknown")
                    })
    return pd.DataFrame(all_rows)

def generate_configuration_tables(df, model, output_dir):
    model_df = df[df["Model"] == model].copy()
    model_df["ConfigStr"] = model_df["CLS"] + "-" + model_df["CTX"] + "-" + model_df["HYB"]

    for config_str, group in model_df.groupby("ConfigStr"):
        config_safe = config_str.replace("-", "_")
        config_table = group.sort_values("Language")[["Language", "MeanSim", "Tokens", "Window", "Family"]]
        table_tex = config_table.to_latex(index=False,
                                          caption=f"{model.upper()} Results for Config: {config_str}",
                                          label=f"tab:{model}_{config_safe}_config",
                                          column_format="lcrrl")
        with open(os.path.join(output_dir, f"{model}_config_{config_safe}.tex"), "w") as f:
            f.write(table_tex)

def generate_model_tables(df, model, output_dir="salt_latex"):
    os.makedirs(output_dir, exist_ok=True)
    model_df = df[df["Model"] == model].sort_values(by=["CLS", "CTX", "HYB", "Tokens", "Window", "Rank"])
    model_df["ConfigStr"] = model_df["CLS"] + "-" + model_df["CTX"] + "-" + model_df["HYB"]

    # Full Table
    full_columns = ["CLS", "CTX", "HYB", "Tokens", "Window", "Rank", "Language", "MeanSim", "Family"]
    full_table = model_df[full_columns].to_latex(index=False, longtable=True,
        caption=f"Ranked Language Similarity for {model.upper()}",
        label=f"tab:{model}_full",
        column_format="lllllllrll")
    with open(os.path.join(output_dir, f"{model}_ranked_similarity_full.tex"), "w") as f:
        f.write(full_table)

    # Top-5 Summary
    top5 = (model_df.sort_values(["Language", "MeanSim"], ascending=[True, False])
                  .groupby("Language").first()
                  .sort_values("MeanSim", ascending=False)
                  .head(5)
                  .reset_index())
    top5 = top5[["Language", "MeanSim", "ConfigStr", "Tokens", "Window", "Family"]]
    top5_table = top5.to_latex(index=False, caption=f"Top-5 Languages for {model.upper()}",
                               label=f"tab:{model}_top5", column_format="lccccc")
    with open(os.path.join(output_dir, f"{model}_top5_summary.tex"), "w") as f:
        f.write(top5_table)

    # Bottom-5 Summary
    bottom5 = (model_df.sort_values(["Language", "MeanSim"], ascending=[True, False])
                     .groupby("Language").first()
                     .sort_values("MeanSim")
                     .head(5)
                     .reset_index())
    bottom5 = bottom5[["Language", "MeanSim", "ConfigStr", "Tokens", "Window", "Family"]]
    bottom5_table = bottom5.to_latex(index=False, caption=f"Bottom-5 Languages for {model.upper()}",
                                     label=f"tab:{model}_bottom5", column_format="lccccc")
    with open(os.path.join(output_dir, f"{model}_bottom5_summary.tex"), "w") as f:
        f.write(bottom5_table)

    generate_configuration_tables(df, model, output_dir)

def generate_visualizations(df, output_dir="salt_latex"):
    os.makedirs(output_dir, exist_ok=True)

    # Bar Chart: Top-10
    top_df = (
        df.sort_values(["Language", "MeanSim"], ascending=[True, False])
          .groupby(["Model", "Language"]).first()
          .reset_index()
    )
    top10 = top_df.groupby("Language").apply(lambda g: g.nlargest(1, "MeanSim")).reset_index(drop=True)
    top10 = top10.sort_values("MeanSim", ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=top10, x="Language", y="MeanSim", hue="Model")
    plt.title("Top-10 Language Similarities to Runyankore")
    plt.ylabel("Mean Similarity")
    plt.xlabel("Language")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top10_bar_chart.png"))
    plt.close()

    # Heatmap: Similarity vs Configuration
    df["ConfigStr"] = df["CLS"] + "-" + df["CTX"] + "-" + df["HYB"]
    pivot_df = df.pivot_table(index="Language", columns="ConfigStr", values="MeanSim", aggfunc="max")

    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot_df, cmap="viridis", cbar_kws={'label': 'Mean Similarity'})
    plt.title("Heatmap: Similarity vs Configuration")
    plt.xlabel("Configuration (CLS-CTX-HYB)")
    plt.ylabel("Language")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "similarity_heatmap.png"))
    plt.close()

# === RUNNING SCRIPT ===
df = compile_ranked_similarity()
os.makedirs("salt_latex", exist_ok=True)
df.to_csv("salt_latex/compiled_similarity.csv", index=False)

for model in df["Model"].unique():
    generate_model_tables(df, model)

generate_visualizations(df)
