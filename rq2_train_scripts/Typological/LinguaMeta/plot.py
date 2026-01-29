import os
import matplotlib.pyplot as plt


lang_colors = {
    "lug": "#1f77b4",  # blue
    "kin": "#aec7e8",  # light blue
    "swa": "#ff7f0e",  # orange
    "luo": "#ffbb78",  # light orange
    "twi": "#2ca02c",  # green
    "nya": "#98df8a",  # light green
    "sna": "#d62728",  # red
    "bbj": "#ff9896",  # light red
    "ibo": "#9467bd",  # purple
    "tsn": "#c5b0d5",  # lavender
    "yor": "#8c564b",  # brown
    "pcm": "#c49c94",  # light brown
    "hau": "#e377c2",  # pink
    "zul": "#f7b6d2",  # light pink
    "fon": "#7f7f7f",  # dark gray
    "ewe": "#c7c7c7",  # light gray
    "xho": "#bcbd22",  # olive
    "mos": "#dbdb8d",  # light olive
    "bam": "#17becf",  # teal
    "wol": "#9edae5",  # light teal
}


languages = [
    "kin","lug","luo","swa","twi","nya","sna","bbj","ibo","tsn",
    "yor","pcm","hau","zul","fon","ewe","xho","mos","bam","wol"
]

scores = [
    0.951,0.935,0.881,0.669,0.500,0.405,0.385,0.366,0.357,0.353,
    0.348,0.348,0.347,0.346,0.345,0.344,0.344,0.341,0.337,0.335
]


data = list(zip(languages, scores))
data.sort(key=lambda x: x[1], reverse=True)

languages_sorted, scores_sorted = zip(*data)
colors_sorted = [lang_colors[l] for l in languages_sorted]


fig, ax = plt.subplots(figsize=(14, 7))

bars = ax.bar(
    languages_sorted,
    scores_sorted,
    color=colors_sorted,
    edgecolor="black",
    linewidth=0.6
)

# Optional: hatch Twi bar for emphasis
for bar, lang in zip(bars, languages_sorted):
    if lang == "twi":
        bar.set_hatch("//")

ax.set_ylim(0, 1)
ax.set_ylabel("S_total (0–1)")
ax.set_xlabel("Auxiliary language")
ax.set_title("Typology similarity (S_total) — equal weights")

ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.xticks(rotation=45, ha="right")

# Value labels
for bar, val in zip(bars, scores_sorted):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.01,
        f"{val:.3f}",
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.tight_layout()


output_dir = "out_typology/figures"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "Linguameta.png")
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Figure saved to {output_path}")
