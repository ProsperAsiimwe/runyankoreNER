#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Per-layer best selection (upper envelope) + automatic single global-config selection.

Outputs per technique/model:
  Appendix-style (per-layer winners):
    - best_config_by_layer_{tech}_{model}.csv
    - values_by_layer_{tech}_{model}.csv
    - values_with_config_by_layer_{tech}_{model}.csv   (merged, publication-friendly)
    - lines_{tech}_{model}.png                          (annotated with per-layer config labels)
    - README_{tech}_{model}.txt

  Main-figure (single fixed global config across all layers), auto-selected:
    - global_config/values_by_layer_{tech}_{model}.csv
    - global_config/lines_{tech}_{model}.png
    - global_config/README_{tech}_{model}.txt
"""

import os
import re
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
import matplotlib.colors as mcolors


# ------------------------------
# Helpers
# ------------------------------

def abspath_from_script(*parts) -> str:
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

def short_config_id(cfg: str, tok: Optional[str], ctx: Optional[str]) -> str:
    m = re.search(r"config_(\d+)", cfg or "")
    cid = f"c{m.group(1)}" if m else "c?"
    t = f"t{tok}" if tok else "t?"
    c = f"c{ctx}" if ctx else "c?"
    return f"{cid}_{t}_{c}"

def find_layer_from_filename(fn: str) -> Optional[int]:
    m = re.search(r"layer(\d+)\.csv$", fn)
    return int(m.group(1)) if m else None

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# ------------------------------
# Color utilities
# ------------------------------

def _hexify(rgb):
    return mcolors.to_hex(rgb, keep_alpha=False)

def _rel_luminance(rgb):
    def f(c):
        return c/12.92 if c <= 0.04045 else ((c+0.055)/1.055)**2.4
    r, g, b = rgb
    R, G, B = f(r), f(g), f(b)
    return 0.2126*R + 0.7152*G + 0.0722*B

def _contrast_ratio(rgb, bg_rgb=(1.0, 1.0, 1.0)):
    L1 = _rel_luminance(rgb); L2 = _rel_luminance(bg_rgb)
    L_light, L_dark = (L1, L2) if L1 > L2 else (L2, L1)
    return (L_light + 0.05) / (L_dark + 0.05)

def pick_colors(n: int, bg_rgb=(1.0, 1.0, 1.0), min_contrast: float = 3.0) -> List[str]:
    curated = []
    for name in ("tab10", "tab20", "Set1", "Dark2"):
        cmap = plt.get_cmap(name)
        if hasattr(cmap, "colors"):
            curated.extend(list(cmap.colors))
        else:
            curated.extend([cmap(i/9.0) for i in range(10)])
    seen = set()
    curated = [tuple(c[:3]) for c in curated]
    unique_curated = []
    for c in curated:
        if c not in seen:
            seen.add(c); unique_curated.append(c)
    good = [c for c in unique_curated if _contrast_ratio(c, bg_rgb) >= min_contrast]
    i = 0
    while len(good) < n and i < 1000:
        k = len(good) + i + 1
        hue = ((k * 0.61803398875) % 1.0); sat = 0.75; val = 0.9
        rgb = mcolors.hsv_to_rgb((hue, sat, val))
        if _contrast_ratio(rgb, bg_rgb) >= min_contrast and tuple(rgb) not in good:
            good.append(tuple(rgb))
        i += 1
    return [_hexify(c) for c in good[:n]]


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
# Per-layer selection
# ------------------------------

def scan_best_by_layer_for_tech(model_root: str, model: str, technique: str, options: dict):
    """
    For each layer: choose config maximizing (or minimizing) average metric over languages.
    Returns:
      values_df (langs x layers), best_df (rows=layers),
      best_config_map {layer->config_id}, lower_better (bool)
    """
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
        raise RuntimeError(f"No '{technique}' results found under {model_dir}.")

    # Collect layers
    layer_set = set()
    for _, _, tech_dir in cfg_dirs:
        for fn in os.listdir(tech_dir):
            if fn.endswith(".csv"):
                L = find_layer_from_filename(fn)
                if L is not None:
                    layer_set.add(L)
    if not layer_set:
        raise RuntimeError(f"No layer CSVs for technique={technique}, model={model}")

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
                s = load_per_entity_metric(fn, tags=tags, agg=agg, topk=topk); lb = False
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

    # union of languages
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
        best_rows.append({
            "layer": L,
            "best_config": cfg_str,
            "tokens": tok,
            "ctx_window": ctx,
            "score_mean_over_langs": score
        })
    best_df = pd.DataFrame(best_rows).sort_values("layer")

    return values_df, best_df, {L: winners[L][0] for L in winners}, lower_better_overall


# ------------------------------
# Plotting
# ------------------------------

def plot_lines(values_df: pd.DataFrame,
               title: str,
               ylabel: str,
               out_png: str,
               invert_for_lower_better: bool,
               best_df: Optional[pd.DataFrame] = None,
               show_config_labels: bool = True):
    ensure_dir(os.path.dirname(out_png) or ".")
    plt.figure(figsize=(12, 7))

    X = values_df.columns.astype(int).tolist()
    Y = values_df.copy()
    if invert_for_lower_better:
        Y = -Y
        ylabel = f"{ylabel} (higher=closer; plotted as negative)"

    langs: List[str] = list(Y.index)

    palette = pick_colors(len(langs), bg_rgb=(1.0, 1.0, 1.0), min_contrast=3.0)
    ax = plt.gca()
    ax.set_prop_cycle(cycler(color=palette))

    for lang in langs:
        plt.plot(X, Y.loc[lang].values, label=lang, linewidth=1.8)

    plt.xlabel("Layer"); plt.ylabel(ylabel); plt.title(title)
    plt.grid(True, alpha=0.3)

    if show_config_labels and best_df is not None and not best_df.empty:
        ymax = np.nanmax(Y.values); ymin = np.nanmin(Y.values)
        ypad = 0.03 * (ymax - ymin + 1e-8)
        for _, row in best_df.iterrows():
            L = int(row["layer"])
            cfg = str(row["best_config"])
            tok = str(row.get("tokens", "")) or None
            ctx = str(row.get("ctx_window", "")) or None
            label = short_config_id(cfg, tok, ctx)
            if L in values_df.columns:
                xi = list(values_df.columns).index(L)
                x = X[xi]
                ax.text(x, ymax + ypad, label, rotation=90, fontsize=8,
                        ha="center", va="bottom", alpha=0.8)

    if len(langs) <= 12:
        plt.legend(loc="best", fontsize=9)
    else:
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, ncol=1)
        plt.tight_layout(rect=[0, 0, 0.78, 1])

    plt.tight_layout()
    plt.savefig(out_png, dpi=220); plt.close()


# ------------------------------
# Merging for clarity
# ------------------------------

def merge_values_with_best(values_df: pd.DataFrame, best_df: pd.DataFrame) -> pd.DataFrame:
    values_T = values_df.transpose()
    merged = best_df.merge(values_T, left_on="layer", right_index=True, how="left")
    meta_cols = ["layer", "best_config", "tokens", "ctx_window", "score_mean_over_langs"]
    lang_cols = [c for c in merged.columns if c not in meta_cols]
    return merged[meta_cols + lang_cols]


# ------------------------------
# GLOBAL CONFIG (auto) selection & loading
# ------------------------------

def choose_global_config(best_df: pd.DataFrame, lower_better: bool) -> Tuple[str, str, str]:
    """
    Choose (config_id, tokens, ctx) that:
      1) appears most frequently in best_df rows;
      2) if tie, pick the one with the best 'score_mean_over_langs'
         among its occurrences (max if higher-better, min if lower-better).
    """
    # Operate on full triple
    triples = best_df[["best_config", "tokens", "ctx_window", "score_mean_over_langs"]].copy()

    # Count frequency
    freq = (triples
            .groupby(["best_config", "tokens", "ctx_window"])
            .size()
            .reset_index(name="count"))

    max_count = freq["count"].max()
    tied = freq[freq["count"] == max_count][["best_config", "tokens", "ctx_window"]]

    if len(tied) == 1:
        cfg, tok, ctx = tied.iloc[0].tolist()
        return str(cfg), str(tok), str(ctx)

    # Tie-break by best score across the layers where the triple occurs
    scores = (triples
              .groupby(["best_config", "tokens", "ctx_window"])["score_mean_over_langs"]
              .agg(lambda s: np.nanmax(s) if not lower_better else np.nanmin(s))
              .reset_index(name="tie_break_score"))

    tied_scores = scores.merge(tied, on=["best_config", "tokens", "ctx_window"])
    if lower_better:
        best_row = tied_scores.sort_values("tie_break_score", ascending=True).iloc[0]
    else:
        best_row = tied_scores.sort_values("tie_break_score", ascending=False).iloc[0]
    return str(best_row["best_config"]), str(best_row["tokens"]), str(best_row["ctx_window"])


def build_values_for_specific_config(model_root: str,
                                     model: str,
                                     technique: str,
                                     options: dict,
                                     cfg_id: str,
                                     tokens: str,
                                     ctx: str) -> Tuple[pd.DataFrame, bool, List[int]]:
    """
    Load per-language series across ALL layers for a fixed (cfg_id, tokens, ctx).
    Returns (values_df [langs x layers], lower_better, layers_list).
    """
    tech_dir = os.path.join(model_root, cfg_id, f"tokens_{tokens}_ctx{ctx}", technique, model)
    if not os.path.isdir(tech_dir):
        raise FileNotFoundError(f"Missing directory for chosen global config: {tech_dir}")

    # Discover layers
    layer_set = set()
    for fn in os.listdir(tech_dir):
        if fn.endswith(".csv"):
            L = find_layer_from_filename(fn)
            if L is not None:
                layer_set.add(L)
    layers = sorted(layer_set)
    if not layers:
        raise RuntimeError(f"No layer CSVs for global config under {tech_dir}")

    # Helper to load a single layer series + lower-better
    def load_series_for_layer(L: int) -> Tuple[pd.Series, bool]:
        if technique == "core":
            fn = os.path.join(tech_dir, f"cosine_to_run_layer{L}.csv")
            s = load_core_metric(fn, metric_col=options.get("core_metric", "Mean")); lb = False
        elif technique == "per_entity":
            fn = os.path.join(tech_dir, f"per_entity_cosine_to_run_layer{L}.csv")
            tags = options.get("per_entity_tags", ["PER","LOC","ORG","DATE"])
            agg  = options.get("per_entity_agg", "mean")
            topk = int(options.get("per_entity_topk", 3))
            s = load_per_entity_metric(fn, tags=tags, agg=agg, topk=topk); lb = False
        elif technique == "alt_measures":
            fn = os.path.join(tech_dir, f"alt_measures_to_run_layer{L}.csv")
            metric = options.get("alt_metric", "CKA_proto_tags")
            tags   = options.get("alt_tags", ["PER","LOC","ORG","DATE"])
            s, lb  = load_alt_measures_metric(fn, metric=metric, tags=tags)
        elif technique == "swd":
            fn = os.path.join(tech_dir, f"swd_to_run_layer{L}.csv")
            metric = options.get("swd_metric", "SWD_mean")
            s, lb  = load_swd_metric(fn, metric=metric)
        else:
            raise ValueError(f"Unknown technique: {technique}")
        return s, lb

    # Load all layers
    values_by_layer = {}
    lower_better_overall = False
    langs_set = set()
    for L in layers:
        s, lb = load_series_for_layer(L)
        values_by_layer[L] = s
        lower_better_overall = lower_better_overall or lb
        langs_set.update(list(s.index))
    langs = sorted(langs_set)

    cols = []
    for L in layers:
        s = values_by_layer.get(L)
        cols.append(s.reindex(langs) if s is not None else pd.Series(index=langs, dtype=float))
    values_df = pd.concat(cols, axis=1)
    values_df.columns = layers
    values_df.index = langs
    return values_df, lower_better_overall, layers


# ------------------------------
# Main
# ------------------------------

def main():
    ap = argparse.ArgumentParser("Per-layer best + auto global-config selection")
    ap.add_argument("--root", default="outputs/salt",
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
                    help="Where to save outputs (created under the resolved root)")
    args = ap.parse_args()

    # Resolve root robustly
    candidates = [
        args.root,
        abspath_from_script(args.root),
        abspath_from_script("outputs", "salt"),
        "rq2_train_scripts/Embeddings/CLEAN/outputs/salt",
    ]
    root = None
    for cand in candidates:
        if os.path.isdir(os.path.join(cand, "xlmr")) or os.path.isdir(os.path.join(cand, "mbert")):
            root = os.path.normpath(cand); break
    if root is None:
        msg = ("Could not locate outputs root. Tried:\n  - " +
               "\n  - ".join(os.path.normpath(c) for c in candidates) +
               "\nTip: run with e.g.  --root outputs/salt   from CLEAN/  (or the full absolute path).")
        raise FileNotFoundError(msg)

    print(f"[INFO] Using ROOT: {root}")

    # Outdir resolution
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

        cfgs = [os.path.basename(d) for d in list_dirs(model_root) if os.path.basename(d).startswith("config_")]
        print(f"  [INFO] Found {len(cfgs)} configs: {', '.join(cfgs[:4])}{' ...' if len(cfgs)>4 else ''}")

        for tech in techniques:
            print(f"  - Technique: {tech}")
            values_df, best_df, best_map, lower_better = scan_best_by_layer_for_tech(
                model_root=model_root, model=model, technique=tech, options=tech_opts.get(tech, {})
            )

            out_dir = os.path.join(outroot, model, tech)
            ensure_dir(out_dir)

            # Per-layer (upper envelope) outputs
            values_csv = os.path.join(out_dir, f"values_by_layer_{tech}_{model}.csv")
            best_csv   = os.path.join(out_dir, f"best_config_by_layer_{tech}_{model}.csv")
            values_df.to_csv(values_csv)
            best_df.to_csv(best_csv, index=False)

            merged_df = merge_values_with_best(values_df, best_df)
            merged_csv = os.path.join(out_dir, f"values_with_config_by_layer_{tech}_{model}.csv")
            merged_df.to_csv(merged_csv, index=False)

            ylabel = {
                "core": "Cosine similarity",
                "per_entity": "Per-entity cosine (aggregated)",
                "alt_measures": args.alt_metric,
                "swd": args.swd_metric,
            }.get(tech, "Metric")

            title = f"{tech} — {model} (per-layer best; labels show winner per layer)"
            perlayer_png = os.path.join(out_dir, f"lines_{tech}_{model}.png")
            plot_lines(values_df, title=title, ylabel=ylabel, out_png=perlayer_png,
                       invert_for_lower_better=lower_better, best_df=best_df, show_config_labels=True)

            readme = os.path.join(out_dir, f"README_{tech}_{model}.txt")
            with open(readme, "w", encoding="utf-8") as f:
                f.write(
                    "Per-layer 'upper envelope' view:\n"
                    "- best_config_by_layer_{tech}_{model}.csv: winner per layer with tokens, ctx_window, mean score.\n"
                    "- values_by_layer_{tech}_{model}.csv: per-language values; each column uses the winner for that layer.\n"
                    "- values_with_config_by_layer_{tech}_{model}.csv: merged table (layer + winner + per-language values).\n"
                    "- lines_{tech}_{model}.png: one line per language; labels above x-axis show the winner for each layer.\n"
                )

            print(f"    Saved: {best_csv}")
            print(f"    Saved: {values_csv}")
            print(f"    Saved: {merged_csv}")
            print(f"    Plot : {perlayer_png}")
            print(f"    Info : {readme}")

            # ------------------------------
            # Global-config (auto) selection & outputs
            # ------------------------------
            g_cfg, g_tok, g_ctx = choose_global_config(best_df, lower_better)
            print(f"    [GLOBAL] Selected config: {short_config_id(g_cfg, g_tok, g_ctx)}")

            g_values_df, g_lower_better, g_layers = build_values_for_specific_config(
                model_root=model_root, model=model, technique=tech,
                options=tech_opts.get(tech, {}), cfg_id=g_cfg, tokens=g_tok, ctx=g_ctx
            )

            g_out_dir = os.path.join(out_dir, "global_config")
            ensure_dir(g_out_dir)

            g_values_csv = os.path.join(g_out_dir, f"values_by_layer_{tech}_{model}.csv")
            g_values_df.to_csv(g_values_csv)

            g_title = (f"{tech} — {model} (single global config: "
                       f"{short_config_id(g_cfg, g_tok, g_ctx)})")
            g_png = os.path.join(g_out_dir, f"lines_{tech}_{model}.png")
            # For the global plot, no per-layer labels (single config), but we can show a header
            plot_lines(g_values_df, title=g_title, ylabel=ylabel, out_png=g_png,
                       invert_for_lower_better=g_lower_better, best_df=None, show_config_labels=False)

            g_readme = os.path.join(g_out_dir, f"README_{tech}_{model}.txt")
            with open(g_readme, "w", encoding="utf-8") as f:
                f.write(
                    "Global-config view (single fixed configuration across all layers):\n"
                    f"- Chosen automatically as the (config,tokens,ctx) triple that appears most often as a per-layer winner;\n"
                    f"  if tied (or all once), pick the tied triple with the best score_mean_over_langs "
                    f"({'lowest' if g_lower_better else 'highest'}).\n"
                    f"- Selected: {short_config_id(g_cfg, g_tok, g_ctx)}\n"
                    "- values_by_layer_{tech}_{model}.csv: per-language values using ONLY the global config for every layer.\n"
                    "- lines_{tech}_{model}.png: one line per language; SAME configuration at every layer.\n"
                )

            print(f"    [GLOBAL] Saved: {g_values_csv}")
            print(f"    [GLOBAL] Plot : {g_png}")
            print(f"    [GLOBAL] Info : {g_readme}")

    print("\nDone.")


if __name__ == "__main__":
    main()
