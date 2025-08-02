import pandas as pd
import os
from collections import defaultdict

# === CONFIGURATION ===
sample_sizes = [2000, 2500, 3000, 3500, 4000, 4500, 5000]
datasets = ["SALT", "MPTC", "COMBINED"]
csv_base_dir = "."
embedding_variants = {
    "XLM-R": "xlmr_run_similarities.csv",
    "mBERT": "mbert_run_similarities.csv"
}

# === Helper: LaTeX Table Generation ===
def generate_latex_table(embedding_name, scores, languages, output_filename):
    def format_row(lang):
        row = [f"\\rowcolor{{yellow!30}}\n{lang}" if lang == "run" else lang]
        for sample in sample_sizes:
            for dataset in datasets:
                score = scores.get(lang, {}).get(sample, {}).get(dataset, "---")
                row.append(str(score))
        return " & ".join(row) + " \\\\"

    total_cols = len(sample_sizes) * len(datasets)

    table_lines = []
    table_lines.append("\\begin{landscape}")
    table_lines.append("\\begin{table}[h!]")
    table_lines.append("\\centering")
    table_lines.append("\\footnotesize")
    table_lines.append(f"\\caption{{Mean Cosine Similarity Scores Between Runyankore and Other Languages Across Datasets and Sentence Sample Sizes ({embedding_name}, Entity-Only Embeddings)}}")
    table_lines.append(f"\\label{{tab:cosine_similarity_comparison_{embedding_name.lower().replace('-', '')}}}")
    table_lines.append("\\begin{adjustbox}{max width=\\linewidth}")
    table_lines.append("\\begin{tabular}{" + "l" + "c" * total_cols + "}")
    table_lines.append("\\toprule")

    # First row: Multicolumn for each sample size
    table_lines.append("\\textbf{Lang} & " + " & ".join(
        [f"\\multicolumn{{3}}{{c}}{{\\textbf{{{size} Sent.}}}}" for size in sample_sizes]) + " \\\\")

    # Second row: cmidrules
    cmidrule_parts = [f"\\cmidrule(lr){{{i*3+2}-{i*3+4}}}" for i in range(len(sample_sizes))]
    table_lines.append("".join(cmidrule_parts))

    # Third row: Datasets
    table_lines.append(" & " + " & ".join(datasets * len(sample_sizes)) + " \\\\")
    table_lines.append("\\midrule")

    # Data rows
    for lang in sorted(languages):
        table_lines.append(format_row(lang))

    table_lines.append("\\bottomrule")
    table_lines.append("\\end{tabular}")
    table_lines.append("\\end{adjustbox}")
    table_lines.append("\\end{table}")
    table_lines.append("\\end{landscape}")

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(table_lines))

    print(f"[✓] LaTeX table written to {output_filename}")

# === Main Loop: Load Data for Both Embeddings ===
for emb_label, emb_filename in embedding_variants.items():
    print(f"[•] Processing: {emb_label}")
    scores = defaultdict(lambda: defaultdict(dict))  # scores[lang][sample_size][dataset] = value
    languages = set()

    for sample in sample_sizes:
        for dataset in datasets:
            file_path = os.path.join(csv_base_dir, dataset, "outputs_entity_only", str(sample), emb_filename)
            if not os.path.isfile(file_path):
                print(f"[!] Missing: {file_path}")
                continue

            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                lang = row["Language"]
                sim = round(float(row["Cosine_Similarity"]), 3)
                scores[lang][sample][dataset] = sim
                languages.add(lang)

    output_tex_file = f"entity_only_cosine_similarity_table_{emb_label.lower().replace('-', '')}.tex"
    generate_latex_table(emb_label, scores, languages, output_tex_file)
