import os
import re
import pandas as pd
from textwrap import dedent

# Language family mapping
language_families = {
    "bam": "Mande", "bbj": "Bantu", "ewe": "Kwa", "fon": "Kwa", "hau": "Chadic",
    "ibo": "Volta-Niger", "kin": "Bantu", "lug": "Bantu", "luo": "Nilotic",
    "mos": "Gur", "nya": "Bantu", "pcm": "English Creole", "sna": "Bantu",
    "swa": "Bantu", "tsn": "Bantu", "twi": "Kwa", "wol": "Atlantic-Congo",
    "xho": "Bantu", "yor": "Volta-Niger", "zul": "Bantu"
}

def extract_config_info(path):
    # Example: mptc/xlmr/config_5_clsfalse_ctxfalse_hybtrue/tokens_300_ctx2/
    model_match = re.search(r"mptc/([^/]+)/", path)
    config_match = re.search(r"config_\d+_cls(.*?)_ctx(.*?)_hyb(.*?)/tokens_(\d+)_ctx(\d+)", path)
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

def compile_ranked_similarity(base_dir="MEXA_inspired_strategy/outputs/mptc"):
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

def format_inline_list(lang_scores):
    return ", ".join([f"{lang} ({score:.4f})" for lang, score in lang_scores])

def generate_latex_landscape_table(df, model):
    model_df = df[df["Model"] == model]
    output_dir = os.path.join("mptc_latex_detailed", model, "landscape_tables")
    os.makedirs(output_dir, exist_ok=True)

    config_groups = model_df.groupby(["CLS", "CTX", "HYB"])
    for (cls, ctx, hyb), group in config_groups:
        rows = []
        caption = f"Similarity results for {model.upper()} with CLS={cls}, CTX={ctx}, HYB={hyb} on the MPTC dataset."
        label = f"tab:mptc_{model}_cls{cls}_ctx{ctx}_hyb{hyb}"

        for (tokens, window), subdf in group.groupby(["Tokens", "Window"]):
            mean_scores = subdf.groupby("Language")["MeanSim"].mean().sort_values(ascending=False)
            top5 = format_inline_list(mean_scores.head(5).items())
            bottom5 = format_inline_list(mean_scores.tail(5).items())

            rows.append(f"{tokens} & {window} & {top5} & {bottom5} \\\\ \\hline")

        table_content = "\n".join(rows)

        tex_content = dedent(f"""
        \\begin{{landscape}}
        \\small
        \\captionsetup{{type=table}}
        \\captionof{{table}}{{{caption}}}
        \\label{{{label}}}

        \\renewcommand{{\\arraystretch}}{{1.1}}
        \\setlength{{\\tabcolsep}}{{4pt}}

        \\begin{{tabular}}{{|c|c|p{{6.5cm}}|p{{6.5cm}}|}}
        \\hline
        \\textbf{{Tokens}} & \\textbf{{Window}} & \\textbf{{top-5}} & \\textbf{{bottom-5}} \\\\
        \\hline
        {table_content}
        \\end{{tabular}}

        \\end{{landscape}}
        """).strip()

        filename = os.path.join(output_dir, f"{model}_cls{cls}_ctx{ctx}_hyb{hyb}_landscape.tex")
        with open(filename, "w") as f:
            f.write(tex_content)

def main():
    df = compile_ranked_similarity()
    if "Model" not in df.columns:
        print("ERROR: The 'Model' column is missing from the DataFrame. Check folder structure or regex pattern.")
        return
    for model in df["Model"].unique():
        generate_latex_landscape_table(df, model)

if __name__ == "__main__":
    main()
