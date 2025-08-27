#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Select best hyper-parameter configuration per layer for each technique,
then export summary CSVs and multi-colored line plots (one line per language).

Assumes outputs are laid out like:

<root>/combined/<MODEL>/config_<i>_cls<true|false>_ctx<true|false>/tokens_<T>_ctx<C>/
  core/<MODEL>/cosine_to_run_layer{L}.csv
  per_entity/<MODEL>/per_entity_cosine_to_run_layer{L}.csv
  alt_measures/<MODEL>/alt_measures_to_run_layer{L}.csv
  swd/<MODEL>/swd_to_run_layer{L}.csv

Default root: rq2_train_scripts/Embeddings/CLEAN/outputs/combined
"""

import os
import re
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------
# Helpers
# ------------------------------

def list_dirs(p: str) -> List[str]:
    return [os.path.join(p, d) for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))]

def parse_config_tokens_ctx(path: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extract (config_id, tokens, ctx) from path segments, e.g.
    .../config_3_clsfalse_ctxtrue/tokens_300_ctx2/...
    Returns (config_str, tokens, ctx_window) as strings or None.
    """
    cfg = None
    tlim = None
    cwin = None

    parts = path.replace("\\", "/").split("/")
    for p in parts:
        if p.startswith("config_"):
            cfg = p
        elif p.startswith("tokens_"):
            # tokens_300_ctx2
            m = re.match(r"tokens_([0-9]+)_ctx([0-9]+)", p)
            if m:
                tlim, cwin = m.group(1), m.group(2)
    return cfg, tlim, cwin

def find_layer_from_filename(fn: str) -> Optional[int]:
    # …_layer12.csv  -> 12
    m = re.search(r"layer(\d+)\.csv$", fn)
    return int(m.group(1)) if m else None

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

# ------------------------------
# Metric extraction per technique
# ------------------------------

def load_core_metric(csv_path: str, metric_col: str = "Mean") -> pd.Series:
    """
    core: cosine_to_run_layer{L}.csv with a 'Mean' column (higher is better).
    Index = language codes.
    """
    df = pd.read_csv(csv_path, index_col=0)
    if metric_col not in df.columns:
        raise ValueError(f"{csv_path} missing column '{metric_col}'")
    return pd.to_numeric(df[metric_col], errors="coerce")

def load_per_entity_metric(csv_path: str,
                           tags: List[str],
                           agg: str = "mean",
                           topk: int = 3) -> pd.Series:
    """
    per_entity: per_entity_cosine_to_run_layer{L}.csv with one column per tag (higher is better).
    Aggregates across tags -> a single Series per language.
    """
    df = pd.read_csv(csv_path, index_col=0)
    missing = [t for t in tags if t not in df.columns]
    if missing:
        # soft fail: drop missing tags
        tags = [t for t in tags if t in df.columns]
    sub = df[tags].apply(pd.to_numeric, errors="coerce")

    if agg == "mean":
        s = sub.mean(axis=1)
    elif agg == "median":
        s = sub.median(axis=1)
    elif agg == "topk":
        # average of top-k values across tags
        s = sub.apply(lambda row: row.nlargest(min(topk, row.notna().sum())).mean(), axis=1)
    else:
        raise ValueError(f"Unknown per-entity agg: {agg}")
    return s

def load_alt_measures_metric(csv_path: str,
                             metric: str,
                             tags: List[str]) -> Tuple[pd.Series, bool]:
    """
    alt_measures: alt_measures_to_run_layer{L}.csv
    Columns include: {TAG}_cos, {TAG}_euclid, {TAG}_ccos, and CKA_proto_tags.
    Returns (Series, lower_better_flag).
    """
    df = pd.read_csv(csv_path, index_col=0)

    lower_better = False
    if metric == "CKA_proto_tags":
        if metric not in df.columns:
            raise ValueError(f"{csv_path} missing '{metric}'")
        s = pd.to_numeric(df[metric], errors="coerce")
        lower_better = False

    elif metric in ("cos_mean", "ccos_mean", "euclid_mean"):
        suffix = "_cos" if metric == "cos_mean" else ("_ccos" if metric == "ccos_mean" else "_euclid")
        cols = [f"{t}{suffix}" for t in df.columns for t in tags if f"{t}{suffix}" in df.columns]
        cols = [f"{t}{suffix}" for t in tags if f"{t}{suffix}" in df.columns]
        if not cols:
            raise ValueError(f"{csv_path} has no columns for {metric}")
        sub = df[cols].apply(pd.to_numeric, errors="coerce")
        s = sub.mean(axis=1)
        lower_better = (metric == "euclid_mean")

    else:
        # Direct column name (e.g., PER_cos, PER_ccos, PER_euclid)
        if metric not in df.columns:
            raise ValueError(f"{csv_path} missing '{metric}'")
        s = pd.to_numeric(df[metric], errors="coerce")
        lower_better = metric.endswith("_euclid")

    return s, lower_better

def load_swd_metric(csv_path: str,
                    metric: str = "SWD_mean") -> Tuple[pd.Series, bool]:
    """
    swd: swd_to_run_layer{L}.csv with per-tag columns and SWD_mean (lower is better).
    """
    df = pd.read_csv(csv_path, index_col=0)
    if metric not in df.columns:
        raise ValueError(f"{csv_path} missing '{metric}'")
    s = pd.to_numeric(df[metric], errors="coerce")
    return s, True  # lower is better

# ------------------------------
# Core selection logic
# ------------------------------

def scan_best_by_layer_for_tech(
    model_root: str,
    model: str,
    technique: str,
    options: dict
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, str], bool]:
    """
    Explore all configs under <model_root>/config_*/tokens_*_ctx*/
    For each layer: choose config maximizing (or minimizing) average metric over languages.

    Returns:
      values_by_layer: languages x layers (metric values) from best configs
      best_table:      rows=layers with columns describing winning configs & score
      best_config_map: {layer -> config_id}
      lower_better:    bool for plotting guidance
    """

    # model_root already points to the model directory (e.g., .../combined/xlmr)
    model_dir = model_root
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Missing model dir: {model_dir}")

    cfg_dirs = []
    for cfg_dir in sorted(d for d in list_dirs(model_dir) if os.path.basename(d).startswith("config_")):
        for tok_dir in sorted(d for d in list_dirs(cfg_dir) if os.path.basename(d).startswith("tokens_")):
            tech_dir = os.path.join(tok_dir, technique, model)
            if os.path.isdir(tech_dir):
                cfg_dirs.append((cfg_dir, tok_dir, tech_dir))

    if not cfg_dirs:
        raise RuntimeError(f"No {technique} results found under {model_dir}")

    # Collect all layer numbers available across configs
    layer_set = set()
    for _, _, tech_dir in cfg_dirs:
        for fn in os.listdir(tech_dir):
            if not fn.endswith(".csv"):
                continue
            L = find_layer_from_filename(fn)
            if L is not None:
                layer_set.add(L)

    if not layer_set:
        raise RuntimeError(f"No layer CSVs found for technique={technique}, model={model}")

    layers = sorted(layer_set)

    # Prepare aggregations
    lower_better = False   # technique-dependent; may get updated below
    values_by_layer: Dict[int, pd.Series] = {}
    winners: Dict[int, Tuple[str, str, str, float]] = {}  # layer -> (cfg, tokens, ctx, score)

    # For each layer, scan configs
    for L in layers:
        best_score = -np.inf
        if technique in ("swd",):
            best_score = np.inf  # lower is better for SWD

        best_series = None
        best_cfg = None
        best_tok = None
        best_ctx = None

        for cfg_dir, tok_dir, tech_dir in cfg_dirs:
            # pick the correct filename per technique
            if technique == "core":
                fn = os.path.join(tech_dir, f"cosine_to_run_layer{L}.csv")
                if not os.path.isfile(fn):
                    continue
                s = load_core_metric(fn, metric_col=options.get("core_metric", "Mean"))
                lb = False

            elif technique == "per_entity":
                fn = os.path.join(tech_dir, f"per_entity_cosine_to_run_layer{L}.csv")
                if not os.path.isfile(fn):
                    continue
                tags = options.get("per_entity_tags", ["PER","LOC","ORG","DATE"])
                agg = options.get("per_entity_agg", "mean")
                topk = int(options.get("per_entity_topk", 3))
                s = load_per_entity_metric(fn, tags=tags, agg=agg, topk=topk)
                lb = False

            elif technique == "alt_measures":
                fn = os.path.join(tech_dir, f"alt_measures_to_run_layer{L}.csv")
                if not os.path.isfile(fn):
                    continue
                metric = options.get("alt_metric", "CKA_proto_tags")
                tags = options.get("alt_tags", ["PER","LOC","ORG","DATE"])
                s, lb = load_alt_measures_metric(fn, metric=metric, tags=tags)

            elif technique == "swd":
                fn = os.path.join(tech_dir, f"swd_to_run_layer{L}.csv")
                if not os.path.isfile(fn):
                    continue
                metric = options.get("swd_metric", "SWD_mean")
                s, lb = load_swd_metric(fn, metric=metric)

            else:
                raise ValueError(f"Unknown technique: {technique}")

            # keep track globally for plotting legend note
            lower_better = lower_better or lb

            # Score = average across languages (exclude NaNs)
            score = float(s.dropna().mean()) if s.notna().any() else (np.inf if lb else -np.inf)

            # Choose best: maximize if higher-better; minimize if lower-better
            take = (score < best_score) if lb else (score > best_score)
            if take:
                best_score = score
                best_series = s
                cfg_str, tok, ctx = parse_config_tokens_ctx(tok_dir)
                best_cfg, best_tok, best_ctx = cfg_str or os.path.basename(cfg_dir), tok, ctx

        if best_series is None:
            # layer had no valid data across configs
            continue

        values_by_layer[L] = best_series
        winners[L] = (best_cfg or "", best_tok or "", best_ctx or "", float(best_score))

    # unify language index across layers
    # take union of all languages, then reindex layers (fill NaN where missing)
    lang_all = set()
    for s in values_by_layer.values():
        lang_all.update(list(s.index))
    langs = sorted(lang_all)

    mat = []
    for L in layers:
        s = values_by_layer.get(L)
        if s is None:
            mat.append(pd.Series(index=langs, dtype=float))
        else:
            mat.append(s.reindex(langs))

    values_df = pd.concat(mat, axis=1)
    values_df.columns = layers

    # winners table
    best_rows = []
    for L in layers:
        cfg_str, tok, ctx, score = winners.get(L, ("", "", "", np.nan))
        best_rows.append({
            "layer": L,
            "best_config": cfg_str,
            "tokens": tok,
            "ctx_window": ctx,
            "score_mean_over_langs": score
        })
    best_df = pd.DataFrame(best_rows).sort_values("layer")

    return values_df, best_df, {L: winners[L][0] for L in winners}, lower_better


# ------------------------------
# Plotting
# ------------------------------

def plot_lines(values_df: pd.DataFrame,
               title: str,
               ylabel: str,
               out_png: str,
               invert_for_lower_better: bool):
    """
    values_df: languages x layers
    """
    ensure_dir(os.path.dirname(out_png) or ".")
    plt.figure(figsize=(12, 7))

    X = values_df.columns.astype(int).tolist()
    Y = values_df.copy()

    if invert_for_lower_better:
        # Plot "closeness" = negative distance (so higher lines mean closer)
        Y = -Y
        ylabel = f"{ylabel} (higher=closer; plotted as negative)"

    # One colored line per language
    for lang in Y.index:
        plt.plot(X, Y.loc[lang].values, label=lang)

    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    # Put legend outside if many languages
    n_langs = len(Y.index)
    if n_langs <= 12:
        plt.legend(loc="best", fontsize=9)
    else:
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, ncol=1)
        plt.tight_layout(rect=[0, 0, 0.78, 1])
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


# ------------------------------
# Main
# ------------------------------

def main():
    ap = argparse.ArgumentParser("Select best config per layer and plot per-language lines")
    ap.add_argument("--root", default="rq2_train_scripts/Embeddings/CLEAN/outputs/combined",
                    help="Root dir that contains model subdirs (xlmr/mbert) and config folders")
    ap.add_argument("--models", default="xlmr,mbert", help="Comma-separated models to process")
    ap.add_argument("--techniques", default="core,per_entity,alt_measures,swd",
                    help="Comma-separated list of techniques to process")

    # Technique-specific options
    ap.add_argument("--core_metric", default="Mean")

    ap.add_argument("--per_entity_tags", default="PER,LOC,ORG,DATE")
    ap.add_argument("--per_entity_agg", default="mean", choices=["mean","median","topk"])
    ap.add_argument("--per_entity_topk", type=int, default=3)

    ap.add_argument("--alt_metric", default="CKA_proto_tags",
                    help="CKA_proto_tags | cos_mean | ccos_mean | euclid_mean | or a direct column like PER_cos")
    ap.add_argument("--alt_tags", default="PER,LOC,ORG,DATE")

    ap.add_argument("--swd_metric", default="SWD_mean")

    ap.add_argument("--outdir", default="rq2_train_scripts/Embeddings/CLEAN/outputs/combined/summary_best",
                    help="Where to save selected CSVs/plots")
    args = ap.parse_args()

    root = args.root
    outroot = args.outdir
    ensure_dir(outroot)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    techniques = [t.strip() for t in args.techniques.split(",") if t.strip()]

    # Technique options dict
    tech_opts = {
        "core": {"core_metric": args.core_metric},
        "per_entity": {
            "per_entity_tags": [t.strip() for t in args.per_entity_tags.split(",") if t.strip()],
            "per_entity_agg": args.per_entity_agg,
            "per_entity_topk": args.per_entity_topk,
        },
        "alt_measures": {
            "alt_metric": args.alt_metric,
            "alt_tags": [t.strip() for t in args.alt_tags.split(",") if t.strip()],
        },
        "swd": {"swd_metric": args.swd_metric},
    }

    for model in models:
        print(f"\n>>> Model: {model}")
        model_root = os.path.join(root, model)

        for tech in techniques:
            print(f"  - Technique: {tech}")

            # Compute selections
            values_df, best_df, best_map, lower_better = scan_best_by_layer_for_tech(
                model_root=model_root,
                model=model,
                technique=tech,
                options=tech_opts.get(tech, {})
            )

            # Save CSVs
            out_dir = os.path.join(outroot, model, tech)
            ensure_dir(out_dir)

            values_csv = os.path.join(out_dir, f"values_by_layer_{tech}_{model}.csv")
            best_csv = os.path.join(out_dir, f"best_config_by_layer_{tech}_{model}.csv")
            values_df.to_csv(values_csv)
            best_df.to_csv(best_csv, index=False)

            # Plot
            ylabel = {
                "core": "Cosine similarity",
                "per_entity": "Per-entity cosine (aggregated)",
                "alt_measures": args.alt_metric,
                "swd": args.swd_metric,
            }.get(tech, "Metric")

            title = f"{tech} — {model} (best config per layer)"
            out_png = os.path.join(out_dir, f"lines_{tech}_{model}.png")
            plot_lines(values_df, title=title, ylabel=ylabel, out_png=out_png,
                       invert_for_lower_better=lower_better)

            # Console summary
            print(f"    Saved: {best_csv}")
            print(f"    Saved: {values_csv}")
            print(f"    Plot : {out_png}")

    print("\nDone.")


if __name__ == "__main__":
    main()
