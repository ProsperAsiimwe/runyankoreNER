#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import glob
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TARGET_LANG_DEFAULT = "run"

# -------------------------
# Logging
# -------------------------

def init_logger(prefix: str = "swd_reports") -> None:
    os.makedirs("logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join("logs", f"{prefix}_{ts}.log")
    logging.basicConfig(
        filename=log_file,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logging.getLogger("").addHandler(console)

# -------------------------
# IO helpers
# -------------------------

_LAYER_RE = re.compile(r"_layer(\d+)\.csv$")

def discover_swd_layer_csvs(out_dir: str, target_lang: str) -> List[Tuple[int, str]]:
    pattern = os.path.join(out_dir, f"swd_to_{target_lang}_layer*.csv")
    paths = glob.glob(pattern)
    items: List[Tuple[int, str]] = []
    for p in paths:
        m = _LAYER_RE.search(p)
        if not m:
            continue
        items.append((int(m.group(1)), p))
    items.sort(key=lambda x: x[0])
    return items

def load_swd_tables(out_dir: str, target_lang: str) -> Dict[int, pd.DataFrame]:
    layer_files = discover_swd_layer_csvs(out_dir, target_lang)
    if not layer_files:
        raise FileNotFoundError(
            f"No SWD CSVs found in '{out_dir}'. Expected files like swd_to_{target_lang}_layer*.csv"
        )
    tables: Dict[int, pd.DataFrame] = {}
    for li, path in layer_files:
        df = pd.read_csv(path, index_col=0)
        # Normalize expected column name
        if "SWD_mean" not in df.columns:
            # allow a few fallbacks if someone renamed columns
            for alt in ["swd_mean", "Mean", "mean", "SWD", "swd"]:
                if alt in df.columns:
                    df = df.rename(columns={alt: "SWD_mean"})
                    break
        tables[li] = df
    return tables

# -------------------------
# Plot helpers
# -------------------------

def _save_sorted_bar_swd(df: pd.DataFrame, layer_idx: int, out_dir: str, model_type: str, target_lang: str, top_k_annot: int = 5) -> None:
    """
    For SWD: lower = closer/better. So we sort ascending for the bars.
    """
    if "SWD_mean" not in df.columns:
        logging.warning(f"Layer {layer_idx}: missing SWD_mean; skipping bar plot.")
        return

    vals = pd.to_numeric(df["SWD_mean"], errors="coerce")
    vals = vals.dropna()
    if vals.empty:
        logging.warning(f"Layer {layer_idx}: no numeric SWD_mean values; skipping bar plot.")
        return

    order = vals.sort_values(ascending=True)

    plt.figure(figsize=(12, 6))
    plt.bar(order.index, order.values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Sliced Wasserstein Distance (mean over tags)")
    plt.title(f"SWD vs {target_lang} — {model_type} — Layer {layer_idx} (lower is closer)")

    # annotate best few (smallest)
    for i, (lang, val) in enumerate(order.head(top_k_annot).items()):
        plt.text(i, val, f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fname = os.path.join(out_dir, f"swd_sorted_layer{layer_idx}.png")
    plt.savefig(fname, dpi=200)
    plt.close()

def _save_heatmap(language_order: List[str], layer_order: List[int], mat: np.ndarray, output_path: str, title: str) -> None:
    plt.figure(figsize=(1.2 + 0.5 * len(layer_order), 0.6 + 0.4 * len(language_order)))
    im = plt.imshow(mat, aspect="auto", interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.yticks(range(len(language_order)), language_order)
    plt.xticks(range(len(layer_order)), layer_order, rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

# -------------------------
# Reporting
# -------------------------

def create_swd_reports(
    swd_tables: Dict[int, pd.DataFrame],
    out_dir: str,
    model_type: str,
    target_lang: str,
    top_k: int = 5,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    layer_order = sorted(swd_tables.keys())
    # pick a reference layer for consistent language ordering (use last/highest layer available)
    ref_layer = layer_order[-1]
    base_df = swd_tables[ref_layer]

    if "SWD_mean" not in base_df.columns:
        raise ValueError(f"Reference layer {ref_layer} does not contain 'SWD_mean' column.")

    # Language order: keep whatever index ordering appears in ref layer file
    language_order = list(base_df.index)

    # Build heatmap matrix [langs x layers]
    mat = np.full((len(language_order), len(layer_order)), np.nan, dtype=float)
    for j, li in enumerate(layer_order):
        df = swd_tables[li].reindex(language_order)
        mat[:, j] = pd.to_numeric(df.get("SWD_mean"), errors="coerce").values

    heatmap_basename = f"heatmap_swd_layers_{model_type}"
    heatmap_path = os.path.join(out_dir, f"{heatmap_basename}.png")
    _save_heatmap(
        language_order=language_order,
        layer_order=layer_order,
        mat=mat,
        output_path=heatmap_path,
        title=f"Runyankore distance (SWD mean) across layers — {model_type} (lower is closer)",
    )

    heat_df = pd.DataFrame(mat, index=language_order, columns=layer_order)
    heat_df.to_csv(os.path.join(out_dir, f"{heatmap_basename}.csv"))

    summary_rows = []

    for li in layer_order:
        df = swd_tables[li]

        # per-layer bar plot
        _save_sorted_bar_swd(df, li, out_dir, model_type=model_type, target_lang=target_lang, top_k_annot=top_k)

        clean = pd.to_numeric(df["SWD_mean"], errors="coerce").dropna()
        if clean.empty:
            logging.warning(f"Layer {li}: no valid SWD_mean values; skipping top/worst.")
            continue

        # For SWD: "top/best" = smallest distances; "worst" = largest
        best = clean.sort_values(ascending=True).head(top_k)
        worst = clean.sort_values(ascending=False).head(top_k)

        logging.info(f"=== Layer {li} ({model_type}) — Best {top_k} (lowest SWD) ===")
        for rank, (lang, val) in enumerate(best.items(), 1):
            logging.info(f"{rank}. {lang}: swd={val:.6f}")

        logging.info(f"=== Layer {li} ({model_type}) — Worst {top_k} (highest SWD) ===")
        for rank, (lang, val) in enumerate(worst.items(), 1):
            logging.info(f"{rank}. {lang}: swd={val:.6f}")

        for lang, val in best.items():
            summary_rows.append(
                {"layer": li, "model": model_type, "rank_type": f"best{top_k}", "language": lang, "swd_mean": float(val)}
            )
        for lang, val in worst.items():
            summary_rows.append(
                {"layer": li, "model": model_type, "rank_type": f"worst{top_k}", "language": lang, "swd_mean": float(val)}
            )

    if summary_rows:
        out_csv = os.path.join(out_dir, f"summary_top_worst_swd_{model_type}.csv")
        pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
        logging.info(f"Wrote summary: {out_csv}")
    else:
        logging.warning("No summary rows produced (all layers empty/NaN?).")

# -------------------------
# Main
# -------------------------

def main():
    init_logger()

    ap = argparse.ArgumentParser("Create SWD visualizations + top/worst summary reports from swd_to_<target>_layer*.csv")
    ap.add_argument("--model_type", default="xlmr", choices=["afroxlmr", "xlmr", "mbert"])
    ap.add_argument("--output_dir", default="outputs_swd", help="Base SWD output directory (same as SWD script)")
    ap.add_argument("--target_lang", default=TARGET_LANG_DEFAULT)
    ap.add_argument("--top_k", type=int, default=5)
    args = ap.parse_args()

    out_dir = os.path.join(args.output_dir, args.model_type)
    logging.info(f"Reading SWD CSVs from: {out_dir}")

    swd_tables = load_swd_tables(out_dir=out_dir, target_lang=args.target_lang)
    create_swd_reports(
        swd_tables=swd_tables,
        out_dir=out_dir,
        model_type=args.model_type,
        target_lang=args.target_lang,
        top_k=args.top_k,
    )

    print("\nDone. SWD report outputs in:", out_dir)

if __name__ == "__main__":
    main()
