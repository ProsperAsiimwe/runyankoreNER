#!/usr/bin/env python3
"""
heatmap_from_csv.py

Reads a CSV of layer-wise scores (rows = languages, columns = hidden layers)
and produces a compact heatmap figure using matplotlib.

- No command-line flags needed. Configure everything in the CONFIG section below.
- Works for both "higher is better" and "lower is better" metrics.
- Adds per-layer rank marks: labels "1", "2", "3" on the best K cells in each column.
"""

import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# =======================
# CONFIGURATION SECTION
# =======================

CSV_PATH = "./combined/summary_best/afro-xlmr/swd/global_config/values_by_layer_swd_afro-xlmr.csv"          # Path to your CSV file
OUT_PATH = "afro-xlmr_combined_swd_heatmap.png"   
FIG_TITLE = "Layer-wise performance per language (Top 4 performing languages shown)"

# If True, the colormap is reversed so LOWER values look visually better (brighter).
LOWER_IS_BETTER = True

# Row sorting for readability. Options: "none", "by_mean", "by_min", "by_max"
SORT_MODE = "by_mean"

# Optional numeric annotations inside cells (can get busy with many rows)
ANNOTATE = False
ANNOT_PRECISION = 4

# Per-layer top-K ranking labels (e.g., 1/2/3). Ties receive the same rank label.
SHOW_TOP_K_PER_LAYER = True
TOP_K = 4                     # Mark top K per layer
RANK_FONT_SIZE = 8            # Font size for the "1/2/3" labels
RANK_FONT_WEIGHT = "bold"     # "normal" or "bold"

# Figure sizing and rendering
DPI = 300                 # Resolution for PNG/JPG outputs
FIGWIDTH = None           # Width in inches (None = auto)
FIGHEIGHT = None          # Height in inches (None = auto)

# Color scale: leave None to auto-fit data range
VMIN = None               # Lower bound for color scale
VMAX = None               # Upper bound for color scale

# Colormap name (Matplotlib name). Examples: "viridis", "plasma", "magma", "cividis"
COLORMAP_NAME = "viridis"

# =======================
# END CONFIG
# =======================


def load_csv(csv_path: str):
    """Load CSV into (row_labels, col_labels, matrix)."""
    with open(csv_path, newline='', encoding='utf-8') as f:
        rows = list(csv.reader(f))

    if not rows:
        raise ValueError("CSV appears empty.")

    header = rows[0]
    # First header cell may be blank — treat as label column
    col_labels = [c.strip() for c in header[1:]]
    data_rows = rows[1:]

    row_labels, matrix_vals = [], []
    for r in data_rows:
        if not r:
            continue
        row_labels.append((r[0] or "").strip())
        nums = []
        for cell in r[1:]:
            try:
                v = float(cell)
            except Exception:
                v = np.nan
            nums.append(v)
        matrix_vals.append(nums)

    M = np.array(matrix_vals, dtype=float)
    return row_labels, col_labels, M


def compute_sort_index(M: np.ndarray, how: str):
    if how == "none":
        return np.arange(M.shape[0])
    if how == "by_mean":
        key = np.nanmean(M, axis=1)
        # For lower-is-better, smaller means are better (so ascending puts best at top)
        return np.argsort(key) if LOWER_IS_BETTER else np.argsort(-key)
    if how == "by_min":
        key = np.nanmin(M, axis=1)
        return np.argsort(key)  # lower→top
    if how == "by_max":
        key = np.nanmax(M, axis=1)
        return np.argsort(-key)  # higher→top
    return np.arange(M.shape[0])


def auto_figsize(n_rows: int, n_cols: int, figwidth, figheight):
    # Heuristic sizing
    if figwidth is None:
        figwidth = max(5.5, min(14.0, 0.6 * n_cols + 3.0))
    if figheight is None:
        figheight = max(3.5, min(12.0, 0.35 * n_rows + 1.8))
    return figwidth, figheight


def top_k_indices_per_column(M: np.ndarray, k: int, lower_is_better: bool):
    """
    Return a dict: col_index -> list of (row_index, rank_label) for the best k rows in that column.
    Ties share the same rank label; we proceed to the next rank after ties.
    """
    n_rows, n_cols = M.shape
    result = {j: [] for j in range(n_cols)}

    for j in range(n_cols):
        col = M[:, j]
        # Consider only finite values
        finite_mask = np.isfinite(col)
        idxs = np.where(finite_mask)[0]
        if idxs.size == 0:
            continue

        vals = col[idxs]
        order = np.argsort(vals) if lower_is_better else np.argsort(-vals)
        sorted_idxs = idxs[order]
        sorted_vals = vals[order]

        # Assign ranks with ties
        ranks = np.empty_like(sorted_vals, dtype=int)
        current_rank = 1
        ranks[0] = current_rank
        for t in range(1, len(sorted_vals)):
            if np.isclose(sorted_vals[t], sorted_vals[t - 1], rtol=1e-9, atol=1e-12):
                ranks[t] = current_rank
            else:
                current_rank += 1
                ranks[t] = current_rank

        # Take all with rank <= K (covers ties). If that exceeds K, we still include them.
        take = np.where(ranks <= k)[0]
        for t in take:
            result[j].append((sorted_idxs[t], int(ranks[t])))

    return result


def make_heatmap(row_labels, col_labels, M):
    # Data range
    data_min = np.nanmin(M)
    data_max = np.nanmax(M)
    vmin = data_min if VMIN is None else VMIN
    vmax = data_max if VMAX is None else VMAX

    # Sort rows
    order = compute_sort_index(M, SORT_MODE)
    M = M[order]
    row_labels = [row_labels[i] for i in order]

    # Figure and colormap
    fw, fh = auto_figsize(len(row_labels), len(col_labels), FIGWIDTH, FIGHEIGHT)
    fig, ax = plt.subplots(figsize=(fw, fh), constrained_layout=True)

    cmap = plt.cm.get_cmap(COLORMAP_NAME)
    if LOWER_IS_BETTER:
        cmap = cmap.reversed()

    im = ax.imshow(M, aspect='auto', vmin=vmin, vmax=vmax, cmap=cmap)

    # Ticks/labels
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)

    ax.set_xlabel("Hidden layer")
    ax.set_ylabel("Language")
    if FIG_TITLE:
        ax.set_title(FIG_TITLE)

    # Colorbar (label reflects "lower is better" if applicable)
    cbar = fig.colorbar(im, ax=ax)
    cbar_label = "score (lower is better)" if LOWER_IS_BETTER else "score"
    cbar.ax.set_ylabel(cbar_label, rotation=90, va="center")

    # Optional numeric annotations
    if ANNOTATE:
        r, c = M.shape
        fmt = "{:0." + str(ANNOT_PRECISION) + "f}"
        for i in range(r):
            for j in range(c):
                val = M[i, j]
                if np.isfinite(val):
                    ax.text(j, i, fmt.format(val), ha="center", va="center", fontsize=7)

    # Per-layer top-K rank labels (1/2/3)
    if SHOW_TOP_K_PER_LAYER and TOP_K > 0:
        per_col = top_k_indices_per_column(M, TOP_K, LOWER_IS_BETTER)
        for j, items in per_col.items():
            for (i, rank_label) in items:
                ax.text(
                    j, i, str(rank_label),
                    ha="center", va="center",
                    fontsize=RANK_FONT_SIZE,
                    fontweight=RANK_FONT_WEIGHT,
                )

    return fig


def main():
    row_labels, col_labels, M = load_csv(CSV_PATH)
    fig = make_heatmap(row_labels, col_labels, M)

    out_path = Path(OUT_PATH)
    if out_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        fig.savefig(out_path, bbox_inches="tight", dpi=DPI)
    else:
        fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved heatmap to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
