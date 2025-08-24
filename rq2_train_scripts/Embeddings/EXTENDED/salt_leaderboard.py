#!/usr/bin/env python3
import os
import re
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, List, Tuple

# Files look like: pearson_to_run_layer12.csv
LAYER_CSV_RE = re.compile(r"pearson_to_run_layer(\d+)\.csv$", re.IGNORECASE)

def find_final_layer_csv(run_dir: str) -> Optional[Tuple[str, int]]:
    best = (-1, None)
    try:
        for fname in os.listdir(run_dir):
            m = LAYER_CSV_RE.match(fname)
            if m:
                layer = int(m.group(1))
                if layer > best[0]:
                    best = (layer, os.path.join(run_dir, fname))
    except FileNotFoundError:
        return None
    if best[1] is None:
        return None
    return best[1], best[0]

def parse_model_from_path(path_parts: List[str]) -> Optional[str]:
    # look for ".../outputs/salt_extended/<model>/..."
    for i in range(len(path_parts) - 1):
        if path_parts[i].lower() == "salt_extended" and i + 1 < len(path_parts):
            return path_parts[i + 1]
    return None

def parse_config_from_component(comp: str) -> Dict[str, Any]:
    # comp: "config_3_clsfalse_ctxtrue_hybfalse"
    info = {"config_id": None, "use_cls": None, "use_context": None, "use_hybrid": None}
    m = re.match(r"config_(\d+)_cls(true|false)_ctx(true|false)_hyb(true|false)$", comp, re.IGNORECASE)
    if m:
        info["config_id"] = int(m.group(1))
        info["use_cls"] = m.group(2).lower() == "true"
        info["use_context"] = m.group(3).lower() == "true"
        info["use_hybrid"] = m.group(4).lower() == "true"
    return info

def parse_tokens_ctx_from_component(comp: str) -> Dict[str, Any]:
    # comp: "tokens_500_ctx3" or "tokens_500_ctx_3"
    info = {"max_tokens_per_type": None, "context_window": None}
    m = re.match(r"tokens_(\d+)_ctx_?(\d+)$", comp, re.IGNORECASE)
    if m:
        info["max_tokens_per_type"] = int(m.group(1))
        info["context_window"] = int(m.group(2))
    return info

def score_csv(csv_path: str, metric_column: str = "Mean") -> float:
    df = pd.read_csv(csv_path)
    if metric_column not in df.columns:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols:
            metric_column = numeric_cols[-1]
        else:
            return float("nan")
    vals = pd.to_numeric(df[metric_column], errors="coerce")
    return float(vals.mean(skipna=True))

def collect_runs(base_dir: str, metric_column: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for root, dirs, files in os.walk(base_dir):
        # Consider a "run_dir" to be a directory that contains at least one pearson_to_run_layer*.csv
        csvs = [f for f in files if LAYER_CSV_RE.match(f)]
        if not csvs:
            continue

        final = find_final_layer_csv(root)
        if not final:
            continue
        final_csv, final_layer = final
        score = score_csv(final_csv, metric_column=metric_column)

        parts = os.path.normpath(root).split(os.sep)
        model = parse_model_from_path(parts)

        config_info = {}
        tokens_info = {}
        for comp in parts:
            if not config_info:
                parsed = parse_config_from_component(comp)
                if parsed.get("config_id") is not None:
                    config_info = parsed
            if not tokens_info:
                parsed2 = parse_tokens_ctx_from_component(comp)
                if parsed2.get("max_tokens_per_type") is not None:
                    tokens_info = parsed2

        row = {
            "model": model,
            "config_id": config_info.get("config_id"),
            "use_cls": config_info.get("use_cls"),
            "use_context": config_info.get("use_context"),
            "use_hybrid": config_info.get("use_hybrid"),
            "max_tokens_per_type": tokens_info.get("max_tokens_per_type"),
            "context_window": tokens_info.get("context_window"),
            "final_layer": final_layer,
            "score_mean_of_Mean": score,
            "run_dir": root,
            "final_layer_csv": final_csv,
        }
        rows.append(row)

    columns = [
        "model","config_id","use_cls","use_context","use_hybrid",
        "max_tokens_per_type","context_window","final_layer",
        "score_mean_of_Mean","run_dir","final_layer_csv"
    ]
    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows)
    df = df.sort_values(["score_mean_of_Mean"], ascending=[False], na_position="last").reset_index(drop=True)
    return df

# ---------- Plotting ----------

def _short_label(row: pd.Series) -> str:
    # Compact but informative label for the y-axis
    return (f"{row['model']} | cfg={row['config_id']} | "
            f"cls={row['use_cls']} ctx={row['use_context']} hyb={row['use_hybrid']} | "
            f"tok={row['max_tokens_per_type']} cw={row['context_window']} | L={row['final_layer']}")

def _barh_leaderboard(df: pd.DataFrame, out_png: str, out_svg: Optional[str] = None,
                      topk: int = 20, title: str = "Leaderboard: Final-layer Pearson (mean of Mean)"):
    if df.empty:
        return
    d = df[["score_mean_of_Mean","model","config_id","use_cls","use_context","use_hybrid",
            "max_tokens_per_type","context_window","final_layer","run_dir"]].copy()
    d = d.dropna(subset=["score_mean_of_Mean"])
    if d.empty:
        return
    d = d.head(min(topk, len(d))).iloc[::-1]  # reverse for barh (top at top after plotting)
    labels = d.apply(_short_label, axis=1)
    scores = d["score_mean_of_Mean"].values

    plt.figure(figsize=(12, max(4, 0.45*len(d))))
    plt.barh(range(len(d)), scores)
    plt.yticks(range(len(d)), labels)
    plt.xlabel("Final-layer Pearson score (mean over languages)")
    plt.title(title)
    # annotate bars
    for i, v in enumerate(scores):
        plt.text(v, i, f" {v:.4f}", va="center")
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    plt.savefig(out_png, dpi=200)
    if out_svg:
        plt.savefig(out_svg)
    plt.close()

def _per_model_charts(df: pd.DataFrame, out_dir: str, topk: int = 10):
    if df.empty or "model" not in df.columns:
        return
    for model, sub in df.groupby("model"):
        sub_sorted = sub.sort_values("score_mean_of_Mean", ascending=False)
        out_png = os.path.join(out_dir, f"leaderboard_{model}.png")
        out_svg = os.path.join(out_dir, f"leaderboard_{model}.svg")
        _barh_leaderboard(sub_sorted, out_png, out_svg, topk,
                          title=f"Leaderboard ({model}): Final-layer Pearson (mean of Mean)")

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Summarize hyper-parameter runs, rank by final-layer Pearson mean, and plot leaderboards.")
    ap.add_argument("--base-dir", type=str, default="rq2_train_scripts/Embeddings/EXTENDED/outputs/salt_extended",
                    help="Root of the outputs tree written by your grid runs.")
    ap.add_argument("--metric-column", type=str, default="Mean",
                    help="Column to average within each per-layer CSV (default: 'Mean').")
    ap.add_argument("--out-csv", type=str, default=None,
                    help="Optional path to save the leaderboard CSV.")
    ap.add_argument("--per-model-topk", type=int, default=5,
                    help="Also print top-k per model for a quick glance.")
    ap.add_argument("--plot-topk", type=int, default=20,
                    help="How many top runs to include in the global leaderboard plot.")
    ap.add_argument("--plot-out", type=str, default=None,
                    help="Path to save the global leaderboard plot PNG (default: <base-dir>/leaderboard.png).")
    ap.add_argument("--plot-per-model", action="store_true",
                    help="Also save per-model leaderboard plots.")
    args = ap.parse_args()

    df = collect_runs(args.base_dir, args.metric_column)
    if df.empty:
        print(f"[WARN] No runs found under: {args.base_dir}")
        return

    # Print a compact leaderboard
    with pd.option_context("display.max_rows", 50, "display.max_columns", 20, "display.width", 160):
        print("\n=== Global Leaderboard (best to worst) ===")
        print(df[[
            "score_mean_of_Mean","model","config_id","use_cls","use_context","use_hybrid",
            "max_tokens_per_type","context_window","final_layer","run_dir"
        ]].head(50))

    # Optional save CSV
    if args.out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
        df.to_csv(args.out_csv, index=False)
        print(f"\n[INFO] Leaderboard saved to: {args.out_csv}")

    # Global leaderboard plot
    plot_out = args.plot_out or os.path.join(args.base_dir, "leaderboard.png")
    plot_svg = os.path.splitext(plot_out)[0] + ".svg"
    _barh_leaderboard(df, plot_out, plot_svg, topk=args.plot_topk)
    print(f"[INFO] Leaderboard plot saved to: {plot_out} and {plot_svg}")

    # Optional per-model plots
    if args.plot_per_model:
        _per_model_charts(df, args.base_dir, topk=min(args.plot_topk, 10))
        print(f"[INFO] Per-model leaderboard plots saved under: {args.base_dir}")

    # Quick per-model Top-K to console
    if args.per_model_topk and "model" in df.columns:
        print("\n=== Per-model Top-K ===")
        for model, sub in df.groupby("model"):
            sub_sorted = sub.sort_values("score_mean_of_Mean", ascending=False)
            print(f"\nModel: {model}")
            print(sub_sorted[[
                "score_mean_of_Mean","config_id","use_cls","use_context","use_hybrid",
                "max_tokens_per_type","context_window","final_layer","run_dir"
            ]].head(args.per_model_topk))

if __name__ == "__main__":
    main()
