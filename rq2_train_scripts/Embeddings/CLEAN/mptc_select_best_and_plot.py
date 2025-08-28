#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Select best hyper-parameter configuration per layer for each technique,
then export summary CSVs and multi-colored line plots (one line per language).

Assumed layout (relative to a ROOT that ends at ".../outputs/mptc"):
<ROOT>/<MODEL>/config_<i>_cls<true|false>_ctx<true|false>/tokens_<T>_ctx<C>/
  core/<MODEL>/cosine_to_run_layer{L}.csv
  per_entity/<MODEL>/per_entity_cosine_to_run_layer{L}.csv
  alt_measures/<MODEL>/alt_measures_to_run_layer{L}.csv
  swd/<MODEL>/swd_to_run_layer{L}.csv
"""

import os
import re
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------
# Helpers
# ------------------------------

def abspath_from_script(*parts) -> str:
    """Build an absolute path from the directory this script lives in."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, *parts))

def list_dirs(p: str) -> List[str]:
    if not os.path.isdir(p):
        return []
    return [os.path.join(p, d) for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))]

def parse_config_tokens_ctx(path: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extract (config_id, tokens, ctx) from segments like:
    .../config_3_clsfalse_ctxtrue/tokens_300_ctx2/...
    """
    cfg = None; tlim = None; cwin = None
    parts = path.replace("\\", "/").split("/")
    for seg in parts:
        if seg.startswith("config_"):
            cfg = seg
        elif seg.startswith("tokens_"):
            m = re.match(r"tokens_([0-9]+)_ctx([0-9]+)", seg)
            if m:
                tlim, cwin = m.group(1), m.group(2)
    return cfg, tlim, cwin

def find_layer_from_filename(fn: str) -> Optional[int]:
    m = re.search(r"layer(\d+)\.csv$", fn)
    return int(m.group(1)) if m else None

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

# ------------------------------
# Metric extraction per technique
# ------------------------------

def load_core_metric(csv_path: str, metric_col: str = "Mean") -> pd.Series:
    df = pd.read_csv(csv_path, index_col=0)
    if metric_col not in df.columns:
        raise ValueError(f"{csv_path} missing column '{metric_col}'")
    return pd.to_numeric(df[metric_col], errors="coerce")

def load_per_entity_metric(csv_path: str, tags: List[str], agg: str = "mean", topk: int = 3) -> pd.Series:
    df = pd.read_csv(csv_path, index_col=0)
    tags = [t for t in tags if t in df.columns]
    if not tags:
        raise ValueError(f"{csv_path} has none of requested tag columns")
    sub = df[tags].apply(pd.to_numeric, errors="coerce")
    if agg == "mean":
        s = sub.mean(axis=1)
    elif agg == "median":
        s = sub.median(axis=1)
    elif agg == "topk":
        s = sub.apply(lambda row: row.nlargest(min(topk, row.notna().sum())).mean(), axis=1)
    else:
        raise ValueError(f"Unknown per-entity agg: {agg}")
    return s

def load_alt_measures_metric(csv_path: str, metric: str, tags: List[str]) -> Tuple[pd.Series, bool]:
    df = pd.read_csv(csv_path, index_col=0)
    lower_better = False
    if metric == "CKA_proto_tags":
        if metric not in df.columns:
            raise ValueError(f"{csv_path} missing '{metric}'")
        s = pd.to_numeric(df[metric], errors="coerce")
        lower_better = False
    elif metric in ("cos_mean", "ccos_mean", "euclid_mean"):
        suffix = "_cos" if metric == "cos_mean" else ("_ccos" if metric == "ccos_mean" else "_euclid")
        cols = [f"{t}{suffix}" for t in tags if f"{t}{suffix}" in df.columns]
        if not cols:
            raise ValueError(f"{csv_path} has no columns for {metric}")
        sub = df[cols].apply(pd.to_numeric, errors="coerce")
        s = sub.mean(axis=1)
        lower_better = (metric == "euclid_mean")
    else:
        if metric not in df.columns:
            raise ValueError(f"{csv_path} missing '{metric}'")
        s = pd.to_numeric(df[metric], errors="coerce")
        lower_better = metric.endswith("_euclid")
    return s, lower_better

def load_swd_metric(csv_path: str, metric: str = "SWD_mean") -> Tuple[pd.Series, bool]:
    df = pd.read_csv(csv_path, index_col=0)
    if metric not in df.columns:
        raise ValueError(f"{csv_path} missing '{metric}'")
    s = pd.to_numeric(df[metric], errors="coerce")
    return s, True  # lower is better

# ------------------------------
# Core selection logic
# ------------------------------

def scan_best_by_layer_for_tech(model_root: str, model: str, technique: str, options: dict):
    """
    Explore all configs under <model_root>/config_*/tokens_*_ctx*/
    For each layer: choose config maximizing (or minimizing) average metric over languages.
    Returns:
      values_df (langs x layers), best_df (rows=layers), best_config_map {layer->config_id}, lower_better (bool)
    """
    model_dir = model_root  # already ends with .../outputs/mptc/<model>
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Missing model dir: {model_dir}")

    cfg_dirs = []
    for cfg_dir in sorted(d for d in list_dirs(model_dir) if os.path.basename(d).startswith("config_")):
        for tok_dir in sorted(d for d in list_dirs(cfg_dir) if os.path.basename(d).startswith("tokens_")):
            tech_dir = os.path.join(tok_dir, technique, model)
            if os.path.isdir(tech_dir):
                cfg_dirs.append((cfg_dir, tok_dir, tech_dir))

    if not cfg_dirs:
        raise RuntimeError(f"No '{technique}' results found under {model_dir}. "
                           f"Expected e.g. {model_dir}/config_*/tokens_*/{technique}/{model}/...csv")

    # Collect all layer numbers
    layer_set = set()
    for _, _, tech_dir in cfg_dirs:
        for fn in os.listdir(tech_dir):
            if fn.endswith(".csv"):
                L = find_layer_from_filename(fn)
                if L is not None:
                    layer_set.add(L)
    if not layer_set:
        raise RuntimeError(f"No layer CSVs found for technique={technique}, model={model}")

    layers = sorted(layer_set)
    print(f"    Found {len(cfg_dirs)} configs, {len(layers)} layers for '{technique}'")

    lower_better_overall = False
    values_by_layer: Dict[int, pd.Series] = {}
    winners: Dict[int, Tuple[str, str, str, float]] = {}  # layer -> (cfg, tokens, ctx, score)

    for L in layers:
        best_score = -np.inf
        if technique in ("swd",):
            best_score = np.inf  # lower is better

        best_series = None; best_cfg = best_tok = best_ctx = None

        for cfg_dir, tok_dir, tech_dir in cfg_dirs:
            if technique == "core":
                fn = os.path.join(tech_dir, f"cosine_to_run_layer{L}.csv")
                if not os.path.isfile(fn): continue
                s = load_core_metric(fn, metric_col=options.get("core_metric", "Mean"))
                lb = False

            elif technique == "per_entity":
                fn = os.path.join(tech_dir, f"per_entity_cosine_to_run_layer{L}.csv")
                if not os.path.isfile(fn): continue
                tags = options.get("per_entity_tags", ["PER","LOC","ORG","DATE"])
                agg = options.get("per_entity_agg", "mean")
                topk = int(options.get("per_entity_topk", 3))
                s = load_per_entity_metric(fn, tags=tags, agg=agg, topk=topk)
                lb = False

            elif technique == "alt_measures":
                fn = os.path.join(tech_dir, f"alt_measures_to_run_layer{L}.csv")
                if not os.path.isfile(fn): continue
                metric = options.get("alt_metric", "CKA_proto_tags")
                tags = options.get("alt_tags", ["PER","LOC","ORG","DATE"])
                s, lb = load_alt_measures_metric(fn, metric=metric, tags=tags)

            elif technique == "swd":
                fn = os.path.join(tech_dir, f"swd_to_run_layer{L}.csv")
                if not os.path.isfile(fn): continue
                metric = options.get("swd_metric", "SWD_mean")
                s, lb = load_swd_metric(fn, metric=metric)

            else:
                raise ValueError(f"Unknown technique: {technique}")

            lower_better_overall = lower_better_overall or lb
            score = float(s.dropna().mean()) if s.notna().any() else (np.inf if lb else -np.inf)
            take = (score < best_score) if lb else (score > best_score)
            if take:
                best_score = score
                best_series = s
                cfg_str, tok, ctx = parse_config_tokens_ctx(tok_dir)
                best_cfg, best_tok, best_ctx = cfg_str or os.path.basename(cfg_dir), tok, ctx

        if best_series is None:
            continue

        values_by_layer[L] = best_series
        winners[L] = (best_cfg or "", best_tok or "", best_ctx or "", float(best_score))

    # union of languages across layers
    lang_all = set()
    for s in values_by_layer.values():
        lang_all.update(list(s.index))
    langs = sorted(lang_all)

    cols = []
    for L in layers:
        s = values_by_layer.get(L)
        cols.append(s.reindex(langs) if s is not None else pd.Series(index=langs, dtype=float))
    values_df = pd.concat(cols, axis=1)
    values_df.columns = layers

    best_rows = []
    for L in layers:
        cfg_str, tok, ctx, score = winners.get(L, ("", "", "", np.nan))
        best_rows.append({"layer": L, "best_config": cfg_str, "tokens": tok, "ctx_window": ctx,
                          "score_mean_over_langs": score})
    best_df = pd.DataFrame(best_rows).sort_values("layer")

    return values_df, best_df, {L: winners[L][0] for L in winners}, lower_better_overall


# ------------------------------
# Plotting
# ------------------------------

def plot_lines(values_df: pd.DataFrame, title: str, ylabel: str, out_png: str, invert_for_lower_better: bool):
    ensure_dir(os.path.dirname(out_png) or ".")
    plt.figure(figsize=(12, 7))
    X = values_df.columns.astype(int).tolist()
    Y = values_df.copy()
    if invert_for_lower_better:
        Y = -Y
        ylabel = f"{ylabel} (higher=closer; plotted as negative)"
    for lang in Y.index:
        plt.plot(X, Y.loc[lang].values, label=lang)
    plt.xlabel("Layer"); plt.ylabel(ylabel); plt.title(title)
    plt.grid(True, alpha=0.3)
    if len(Y.index) <= 12:
        plt.legend(loc="best", fontsize=9)
    else:
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, ncol=1)
        plt.tight_layout(rect=[0, 0, 0.78, 1])
    plt.tight_layout()
    plt.savefig(out_png, dpi=220); plt.close()


# ------------------------------
# Main
# ------------------------------

def main():
    ap = argparse.ArgumentParser("Select best config per layer and plot per-language lines")
    # Keep a short default. We’ll resolve it relative to this script and try fallbacks.
    ap.add_argument("--root", default="outputs/mptc",
                    help="Root dir that contains model subdirs (xlmr/mbert) and config folders")
    ap.add_argument("--models", default="xlmr,mbert", help="Comma-separated models to process")
    ap.add_argument("--techniques", default="core,per_entity,alt_measures,swd",
                    help="Comma-separated techniques to process")

    # Technique-specific
    ap.add_argument("--core_metric", default="Mean")
    ap.add_argument("--per_entity_tags", default="PER,LOC,ORG,DATE")
    ap.add_argument("--per_entity_agg", default="mean", choices=["mean","median","topk"])
    ap.add_argument("--per_entity_topk", type=int, default=3)
    ap.add_argument("--alt_metric", default="CKA_proto_tags",
                    help="CKA_proto_tags | cos_mean | ccos_mean | euclid_mean | or a direct column (e.g., PER_cos)")
    ap.add_argument("--alt_tags", default="PER,LOC,ORG,DATE")
    ap.add_argument("--swd_metric", default="SWD_mean")
    ap.add_argument("--outdir", default="summary_best",
                    help="Where to save selected CSVs/plots (created under the resolved root)")
    args = ap.parse_args()

    # Resolve root robustly:
    candidates = [
        args.root,                                 # as provided
        abspath_from_script(args.root),            # relative to this script
        abspath_from_script("outputs", "mptc"),    # common repo layout
        "rq2_train_scripts/Embeddings/CLEAN/outputs/mptc",  # legacy path
    ]
    root = None
    for cand in candidates:
        if os.path.isdir(os.path.join(cand, "xlmr")) or os.path.isdir(os.path.join(cand, "mbert")):
            root = os.path.normpath(cand); break
    if root is None:
        msg = ("Could not locate outputs root. Tried:\n  - " +
               "\n  - ".join(os.path.normpath(c) for c in candidates) +
               "\nTip: run with e.g.  --root outputs/mptc   from CLEAN/  (or the full absolute path).")
        raise FileNotFoundError(msg)

    print(f"[INFO] Using ROOT: {root}")

    # Outdir under the same parent as root, unless absolute given
    outroot = args.outdir
    if not os.path.isabs(outroot):
        outroot = os.path.join(root, args.outdir)
    ensure_dir(outroot)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    techniques = [t.strip() for t in args.techniques.split(",") if t.strip()]

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
        if not os.path.isdir(model_root):
            print(f"  [WARN] Skipping model '{model}' — not found under {root}")
            continue

        # Quick visibility: list a few configs
        cfgs = [os.path.basename(d) for d in list_dirs(model_root) if os.path.basename(d).startswith("config_")]
        print(f"  [INFO] Found {len(cfgs)} configs: {', '.join(cfgs[:4])}{' ...' if len(cfgs)>4 else ''}")

        for tech in techniques:
            print(f"  - Technique: {tech}")
            values_df, best_df, best_map, lower_better = scan_best_by_layer_for_tech(
                model_root=model_root, model=model, technique=tech, options=tech_opts.get(tech, {})
            )

            out_dir = os.path.join(outroot, model, tech)
            ensure_dir(out_dir)
            values_csv = os.path.join(out_dir, f"values_by_layer_{tech}_{model}.csv")
            best_csv = os.path.join(out_dir, f"best_config_by_layer_{tech}_{model}.csv")
            values_df.to_csv(values_csv)
            best_df.to_csv(best_csv, index=False)

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

            print(f"    Saved: {best_csv}")
            print(f"    Saved: {values_csv}")
            print(f"    Plot : {out_png}")

    print("\nDone.")


if __name__ == "__main__":
    main()
