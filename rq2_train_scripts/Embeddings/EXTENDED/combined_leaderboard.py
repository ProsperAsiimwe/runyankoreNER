#!/usr/bin/env python3
import os
import re
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, List, Tuple

# -------------------------
# File pattern helpers
# -------------------------
LAYER_CSV_RE = re.compile(r"pearson_to_run_layer(\d+)\.csv$", re.IGNORECASE)

def find_final_layer_csv(run_dir: str) -> Optional[Tuple[str, int]]:
    """Return (path, layer_index) of the highest-layer CSV in run_dir."""
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

# -------------------------
# Metadata parsing from paths
# -------------------------
def parse_model_from_path(path_parts: List[str]) -> Optional[str]:
    # look for ".../outputs/combined_extended/<model>/..."
    for i in range(len(path_parts) - 1):
        if path_parts[i].lower() == "combined_extended" and i + 1 < len(path_parts):
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

# -------------------------
# Scoring
# -------------------------
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

# -------------------------
# Collect runs (final-layer only)
# -------------------------
def collect_runs_final(base_dir: str, metric_column: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for root, dirs, files in os.walk(base_dir):
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

        config_info, tokens_info = {}, {}
        for comp in parts:
            if not config_info:
                parsed = parse_config_from_component(comp)
                if parsed.get("config_id") is not None:
                    config_info = parsed
            if not tokens_info:
                parsed2 = parse_tokens_ctx_from_component(comp)
                if parsed2.get("max_tokens_per_type") is not None:
                    tokens_info = parsed2

        rows.append({
            "model": model,
            "config_id": config_info.get("config_id"),
            "use_cls": config_info.get("use_cls"),
            "use_context": config_info.get("use_context"),
            "use_hybrid": config_info.get("use_hybrid"),
            "max_tokens_per_type": tokens_info.get("max_tokens_per_type"),
            "context_window": tokens_info.get("context_window"),
            "layer": final_layer,
            "score": score,
            "run_dir": root,
            "csv_path": final_csv,
            "scope": "final_layer_only",
        })

    cols = ["model","config_id","use_cls","use_context","use_hybrid",
            "max_tokens_per_type","context_window","layer","score","run_dir","csv_path","scope"]
    df = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
    df = df.sort_values(["score"], ascending=[False], na_position="last").reset_index(drop=True)
    return df

# -------------------------
# Collect runs (ALL layers)
# -------------------------
def collect_runs_all_layers(base_dir: str, metric_column: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for root, dirs, files in os.walk(base_dir):
        layer_files = [f for f in files if LAYER_CSV_RE.match(f)]
        if not layer_files:
            continue

        parts = os.path.normpath(root).split(os.sep)
        model = parse_model_from_path(parts)
        config_info, tokens_info = {}, {}
        for comp in parts:
            if not config_info:
                parsed = parse_config_from_component(comp)
                if parsed.get("config_id") is not None:
                    config_info = parsed
            if not tokens_info:
                parsed2 = parse_tokens_ctx_from_component(comp)
                if parsed2.get("max_tokens_per_type") is not None:
                    tokens_info = parsed2

        for fname in layer_files:
            m = LAYER_CSV_RE.match(fname)
            if not m:
                continue
            layer = int(m.group(1))
            csv_path = os.path.join(root, fname)
            score = score_csv(csv_path, metric_column=metric_column)

            rows.append({
                "model": model,
                "config_id": config_info.get("config_id"),
                "use_cls": config_info.get("use_cls"),
                "use_context": config_info.get("use_context"),
                "use_hybrid": config_info.get("use_hybrid"),
                "max_tokens_per_type": tokens_info.get("max_tokens_per_type"),
                "context_window": tokens_info.get("context_window"),
                "layer": layer,
                "score": score,
                "run_dir": root,
                "csv_path": csv_path,
                "scope": "all_layers",
            })

    cols = ["model","config_id","use_cls","use_context","use_hybrid",
            "max_tokens_per_type","context_window","layer","score","run_dir","csv_path","scope"]
    df = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
    df = df.sort_values(["score"], ascending=[False], na_position="last").reset_index(drop=True)
    return df

# -------------------------
# Plotting
# -------------------------
def _short_label(row: pd.Series) -> str:
    return (f"{row['model']} | cfg={row['config_id']} | "
            f"cls={row['use_cls']} ctx={row['use_context']} hyb={row['use_hybrid']} | "
            f"tok={row['max_tokens_per_type']} cw={row['context_window']} | L={row['layer']}")

def _barh(df: pd.DataFrame, out_png: str, title: str, topk: int = 20, annotate=True, out_svg: bool = True):
    d = df.dropna(subset=["score"]).head(min(topk, len(df))).iloc[::-1]
    if d.empty:
        return
    labels = d.apply(_short_label, axis=1)
    scores = d["score"].values
    plt.figure(figsize=(12, max(4, 0.45*len(d))))
    plt.barh(range(len(d)), scores)
    plt.yticks(range(len(d)), labels)
    plt.xlabel("Pearson score (mean over languages)")
    plt.title(title)
    if annotate:
        for i, v in enumerate(scores):
            plt.text(v, i, f" {v:.4f}", va="center")
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    plt.savefig(out_png, dpi=200)
    if out_svg:
        base, _ = os.path.splitext(out_png)
        plt.savefig(base + ".svg")
    plt.close()

def _per_model_best_per_layer_plots(df_all_layers: pd.DataFrame, out_dir: str):
    """For each model: bar chart of best score per layer, annotated with cfg id."""
    for model, sub in df_all_layers.groupby("model"):
        # Best per layer for this model
        idx = sub.groupby("layer")["score"].idxmax()
        best_per_layer = sub.loc[idx].sort_values("layer")
        out_png = os.path.join(out_dir, f"best_per_layer_{model}.png")
        title = f"Best configuration per layer — {model}"
        _barh(best_per_layer.sort_values("score", ascending=False), out_png, title, topk=len(best_per_layer))

# -------------------------
# Main CLI
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Summarize runs: final-layer leaderboard, best per layer, best overall across layers, with plots.")
    ap.add_argument("--base-dir", type=str, default="rq2_train_scripts/Embeddings/EXTENDED/outputs/combined_extended",
                    help="Root outputs directory.")
    ap.add_argument("--metric-column", type=str, default="Mean",
                    help="Column to average within each per-layer CSV (default: 'Mean').")
    ap.add_argument("--out-csv", type=str, default=None,
                    help="Path to save the final-layer leaderboard CSV.")
    ap.add_argument("--plot-topk", type=int, default=20,
                    help="How many top runs to include in the global leaderboard plot.")
    ap.add_argument("--plot-per-model", action="store_true",
                    help="Also save per-model leaderboard plots (final-layer).")
    ap.add_argument("--analyze-all-layers", action="store_true",
                    help="Also compute best configuration per layer and best overall across layers (per model), with CSVs and plots.")
    args = ap.parse_args()

    # -------- Final-layer leaderboard (existing behavior) --------
    df_final = collect_runs_final(args.base_dir, args.metric_column)
    if df_final.empty:
        print(f"[WARN] No runs found under: {args.base_dir}")
    else:
        with pd.option_context("display.max_rows", 50, "display.max_columns", 20, "display.width", 160):
            print("\n=== Global Leaderboard (FINAL layer; best to worst) ===")
            print(df_final[[
                "score","model","config_id","use_cls","use_context","use_hybrid",
                "max_tokens_per_type","context_window","layer","run_dir"
            ]].head(50))

        if args.out_csv:
            os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
            df_final.to_csv(args.out_csv, index=False)
            print(f"[INFO] Final-layer leaderboard saved to: {args.out_csv}")

        # Global final-layer plot
        plot_out = os.path.join(args.base_dir, "leaderboard_final_layer.png")
        _barh(df_final, plot_out, "Leaderboard (FINAL layer): Pearson mean over languages", topk=args.plot_topk)
        print(f"[INFO] Final-layer leaderboard plot saved to: {plot_out} and {os.path.splitext(plot_out)[0] + '.svg'}")

        # Optional per-model final-layer plots
        if args.plot_per_model and "model" in df_final.columns:
            for model, sub in df_final.groupby("model"):
                out_png = os.path.join(args.base_dir, f"leaderboard_final_{model}.png")
                _barh(sub, out_png, f"Leaderboard (FINAL layer) — {model}", topk=min(args.plot_topk, 10))
            print(f"[INFO] Per-model final-layer leaderboard plots saved under: {args.base_dir}")

    # -------- All-layers analysis (answers your two questions) --------
    if args.analyze_all_layers:
        df_all = collect_runs_all_layers(args.base_dir, args.metric_column)
        if df_all.empty:
            print(f"[WARN] No per-layer CSVs found under: {args.base_dir}")
            return

        # (1) Best configuration per layer, per model
        best_per_layer_rows = []
        for model, sub in df_all.groupby("model"):
            idx = sub.groupby("layer")["score"].idxmax()
            best_per_layer = sub.loc[idx].copy()
            best_per_layer["rank_within_layer"] = 1
            best_per_layer_rows.append(best_per_layer)

        df_best_per_layer = pd.concat(best_per_layer_rows, ignore_index=True) if best_per_layer_rows else pd.DataFrame()
        df_best_per_layer = df_best_per_layer.sort_values(["model","layer"]).reset_index(drop=True)

        out_csv_bpl = os.path.join(args.base_dir, "best_per_layer_per_model.csv")
        df_best_per_layer.to_csv(out_csv_bpl, index=False)
        print(f"[INFO] Best-per-layer (per model) saved to: {out_csv_bpl}")

        # Plot: for each model, show best score per layer
        _per_model_best_per_layer_plots(df_all, args.base_dir)
        print(f"[INFO] Best-per-layer plots saved under: {args.base_dir}")

        # (2) Best overall across all layers, per model
        best_overall_rows = []
        for model, sub in df_all.groupby("model"):
            j = sub["score"].idxmax()
            best_overall_rows.append(sub.loc[j])

        df_best_overall = pd.DataFrame(best_overall_rows).reset_index(drop=True)
        out_csv_bo = os.path.join(args.base_dir, "best_overall_across_layers_per_model.csv")
        df_best_overall.to_csv(out_csv_bo, index=False)
        print(f"[INFO] Best-overall-across-layers (per model) saved to: {out_csv_bo}")

        # Plot: overall top-k across all (model, layer, config) triples
        plot_out_all = os.path.join(args.base_dir, "leaderboard_all_layers.png")
        _barh(df_all, plot_out_all, "Leaderboard (ALL layers): best configs mixed", topk=args.plot_topk)
        print(f"[INFO] Global all-layers leaderboard plot saved to: {plot_out_all} and {os.path.splitext(plot_out_all)[0] + '.svg'}")

if __name__ == "__main__":
    main()
