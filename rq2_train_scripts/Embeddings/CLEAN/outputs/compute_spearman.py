#!/usr/bin/env python3
"""
Compute Spearman correlations between layer-wise similarity ranks and per-language F1 ranks,
and produce per-language×layer heatmaps of rank errors.

Inputs (set paths in DATA below):
- Similarity CSVs: rows=languages, cols=layers (often "1..12"), values = cosine or SWD
- F1 CSVs per model: 'language,f1' for zero-shot and co-training

Outputs:
- spearman_out/correlations.csv: model,metric,setting,layer,rho,n_effective,n_langs,ci_low,ci_high
- spearman_out/{model}_{metric}_spearman.png : ρ vs. layer (zero-shot & co-train)
- spearman_out/heatmap_{rankdiff|abserr}_{model}_{metric}_{zeroshot|cotr}.png (diagnostic)
- spearman_out/{rankdiff|abserr}_{model}_{metric}_{zeroshot|cotr}.csv (diagnostic)
- spearman_out/{model}_{metric}_heatmap.png  (MAIN: zero-shot |rank error| for paper)
"""

from pathlib import Path
import sys
import re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

# =======================
# CONFIG
# =======================
DATA = {
    "mbert": {
        "core_csv": "./combined/summary_best/mbert/core/global_config/values_by_layer_core_mbert.csv",
        "swd_csv":  "./combined/summary_best/mbert/swd/global_config/values_by_layer_swd_mbert.csv",
        "zs_f1":    "./spearman_files/zs_f1_mbert.csv",
        "ct_f1":    "./spearman_files/ct_f1_mbert.csv",
    },
    "xlmr": {
        "core_csv": "./combined/summary_best/xlmr/core/global_config/values_by_layer_core_xlmr.csv",
        "swd_csv":  "./combined/summary_best/xlmr/swd/global_config/values_by_layer_swd_xlmr.csv",
        "zs_f1":    "./spearman_files/zs_f1_xlmr.csv",
        "ct_f1":    "./spearman_files/ct_f1_xlmr.csv",
    },
    "afro-xlmr": {
        "core_csv": "./combined/summary_best/afro-xlmr/core/global_config/values_by_layer_core_xlmr.csv",
        "swd_csv":  "./combined/summary_best/afro-xlmr/swd/global_config/values_by_layer_swd_xlmr.csv",
        "zs_f1":    "./spearman_files/zs_f1_xlmr.csv",
        "ct_f1":    "./spearman_files/ct_f1_xlmr.csv",
    },
}
LANG_LIST = "./spearman_files/languages.txt"   # one code per line (e.g., kin, lug, nya, ...)
OUT_DIR = "./spearman_files/spearman_out"

# Plotting & stats
PLOT_BOOTSTRAP = False
BOOT_N = 2000
RANDOM_SEED = 7
FIGSIZE = (7.5, 4.5)

# Verbose diagnostics
VERBOSE = True
# =======================


def dbg(msg: str):
    if VERBOSE:
        print(f"[spearman] {msg}", file=sys.stderr)


def read_lang_list(path: str):
    langs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                langs.append(s)
    # normalize: lowercase + strip
    langs = [x.strip().lower() for x in langs]
    dbg(f"Loaded {len(langs)} languages from {path}: {langs}")
    return langs


def read_any_csv(path: str) -> pd.DataFrame:
    # Try comma, then semicolon
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, sep=";")
    # Strip spaces on headers
    df.columns = [c.strip() for c in df.columns]
    return df


def read_similarity_csv(path: str) -> pd.DataFrame:
    df = read_any_csv(path)
    if df.shape[1] < 2:
        raise ValueError(f"{path}: expected at least 2 columns (language + layers). Got {df.shape[1]}.")

    # First column is language labels
    df = df.rename(columns={df.columns[0]: "language"})
    df["language"] = df["language"].astype(str).str.strip().str.lower()

    # Candidate layer columns = everything except 'language'
    layer_cols = [c for c in df.columns if c != "language"]

    # Convert to numeric (coerce errors to NaN)
    for c in layer_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Sort layer columns numerically by trailing digits or the whole token if purely digits
    def layer_key(c):
        s = str(c).strip()
        if re.fullmatch(r"\d+", s):     # "1", "2", ...
            return int(s)
        m = re.search(r"(\d+)$", s)     # L1, h12, layer_7...
        return int(m.group(1)) if m else 10**9

    layer_cols_sorted = sorted(layer_cols, key=layer_key)
    df = df[["language"] + layer_cols_sorted]

    dbg(f"[{path}] Detected {len(layer_cols_sorted)} layer columns: {layer_cols_sorted[:6]}{' ...' if len(layer_cols_sorted)>6 else ''}")
    return df


def read_f1_csv(path: str) -> pd.DataFrame:
    df = read_any_csv(path)
    # Normalize header names for detection
    norm = {c.lower().strip(): c for c in df.columns}
    lang_col = norm.get("language") or norm.get("lang") or list(df.columns)[0]
    f1_col = (norm.get("f1") or norm.get("f1_score") or norm.get("score")
              or norm.get("zs_f1") or norm.get("ct_f1"))
    if not f1_col:
        raise ValueError(f"{path}: could not find an F1 column among {list(df.columns)}")

    df = df.rename(columns={lang_col: "language", f1_col: "f1"})
    df["language"] = df["language"].astype(str).str.strip().str.lower()
    df["f1"] = pd.to_numeric(df["f1"], errors="coerce")

    dbg(f"[{path}] F1 rows: {len(df)}, NaN F1s: {int(df['f1'].isna().sum())}")
    return df[["language", "f1"]]


def align_on_languages(df: pd.DataFrame, all_langs, label: str) -> pd.DataFrame:
    """Filter to languages in all_langs, preserve order of all_langs, and report misses."""
    if "language" not in df.columns:
        df = df.reset_index().rename(columns={df.columns[0]: "language"})
    df["language"] = df["language"].astype(str).str.strip().str.lower()
    present = set(df["language"])
    missing = [l for l in all_langs if l not in present]
    if missing:
        dbg(f"[{label}] Missing {len(missing)} languages (dropped): {missing}")
    else:
        dbg(f"[{label}] All languages present.")

    df = df[df["language"].isin(all_langs)].copy()
    df = df.set_index("language").reindex(all_langs)
    return df


def rank_vector(values, higher_is_better=True):
    """
    Convert a numeric vector -> rank vector (float ranks; average ties).
    higher_is_better=True => highest value gets best rank (smallest number).
    """
    arr = np.asarray(values, dtype=float)
    mask = np.isfinite(arr)
    ranks = np.full_like(arr, fill_value=np.nan, dtype=float)
    if mask.sum() == 0:
        return ranks

    # pandas.rank ranks ascending; invert if higher is better
    arr_eff = -arr[mask] if higher_is_better else arr[mask]
    tmp = pd.Series(arr_eff).rank(method="average", ascending=True)  # 1 = best
    ranks[mask] = tmp.values
    return ranks


def spearman_between(sim_values, f1_values, higher_is_better):
    sim_ranks = rank_vector(sim_values, higher_is_better=higher_is_better)
    f1_ranks  = rank_vector(f1_values,  higher_is_better=True)  # higher F1 is better

    mask = np.isfinite(sim_ranks) & np.isfinite(f1_ranks)
    n = int(mask.sum())
    if n < 3:
        return np.nan, n

    rho, _ = spearmanr(sim_ranks[mask], f1_ranks[mask])
    return float(rho), n


def bootstrap_ci(sim_vals, f1_vals, higher_is_better, n=2000, seed=7):
    rng = np.random.default_rng(seed)
    sim = np.asarray(sim_vals, float)
    f1 = np.asarray(f1_vals, float)
    mask = np.isfinite(sim) & np.isfinite(f1)
    sim, f1 = sim[mask], f1[mask]
    if len(sim) < 3:
        return (np.nan, np.nan)
    N = len(sim)
    out = []
    for _ in range(n):
        idx = rng.integers(0, N, size=N)
        r, _ = spearman_between(sim[idx], f1[idx], higher_is_better)
        out.append(r)
    lo, hi = np.nanpercentile(out, [2.5, 97.5])
    return (float(lo), float(hi))


# ======= Additions for per-language heatmaps =======

def rank_series(vals, higher_is_better=True) -> pd.Series:
    arr = pd.to_numeric(pd.Series(vals), errors="coerce")
    arr_eff = -arr if higher_is_better else arr
    return arr_eff.rank(method="average", ascending=True)  # 1 = best


def compute_rank_error_matrix(sim_df: pd.DataFrame, f1_df: pd.DataFrame, higher_is_better: bool):
    """
    sim_df: index=language, columns=layer names (numeric strings like '1','2',...)
    f1_df : index=language, single column 'f1'
    Returns:
      rank_diff (DataFrame): F1_rank - SIM_rank (signed)
      abs_err   (DataFrame): |rank_diff|
    """
    langs = sim_df.index.tolist()
    layers = [c for c in sim_df.columns if c != "language"]

    f1_ranks = rank_series(f1_df['f1'].values, higher_is_better=True).values
    rank_diff = pd.DataFrame(index=langs, columns=layers, dtype=float)
    abs_err   = pd.DataFrame(index=langs, columns=layers, dtype=float)

    for layer in layers:
        sim_vals = sim_df[layer].values
        sim_ranks = rank_series(sim_vals, higher_is_better=higher_is_better).values
        diff = f1_ranks - sim_ranks
        rank_diff[layer] = diff
        abs_err[layer] = np.abs(diff)

    return rank_diff, abs_err


def plot_heatmap(df: pd.DataFrame, title: str, outpath: Path, center=None, cmap="coolwarm"):
    """
    df: index=languages, columns=layers; values numeric
    """
    # Try natural numeric sort of columns
    cols = list(df.columns)
    try:
        cols_sorted = sorted(cols, key=lambda c: int(str(c)))
        df = df[cols_sorted]
    except Exception:
        pass

    mat = df.values.astype(float)
    plt.figure(figsize=(max(6, 0.6*len(df.columns)+3), max(4, 0.35*len(df.index)+1.8)))
    im = plt.imshow(mat, aspect='auto', cmap=cmap)
    if center is not None:
        v = np.nanmax(np.abs(mat))
        im.set_clim(-v, v)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(np.arange(len(df.columns)), df.columns)
    plt.yticks(np.arange(len(df.index)), df.index)
    plt.xlabel("Layer")
    plt.ylabel("Language")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


# =======================
# MAIN
# =======================
def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_langs = read_lang_list(LANG_LIST)

    rows = []
    for model, paths in DATA.items():
        for metric_name, sim_path in [("core", paths["core_csv"]), ("swd", paths["swd_csv"])]:
            dbg(f"Processing model={model}, metric={metric_name}")
            # Load data
            sim_df = read_similarity_csv(sim_path)
            zs_df  = read_f1_csv(paths["zs_f1"])
            ct_df  = read_f1_csv(paths["ct_f1"])

            # Align on language list (order enforced)
            sim_df = align_on_languages(sim_df, all_langs, f"{model}/{metric_name}/sim")
            zs_df  = align_on_languages(zs_df.reset_index(),  all_langs, f"{model}/zs_f1")
            ct_df  = align_on_languages(ct_df.reset_index(),  all_langs, f"{model}/ct_f1")

            # Use detected layer columns
            layer_cols = [c for c in sim_df.columns if c != "language"]
            if not layer_cols:
                dbg(f"[FATAL] No layer columns available after alignment for {model}/{metric_name}. Columns: {list(sim_df.columns)}")
                continue

            higher_is_better = (metric_name == "core")

            # Prepare plotting containers
            plot_layers, plot_zs, plot_ct, zs_ci, ct_ci = [], [], [], [], []

            # Quick peeks
            dbg(f"{model}/{metric_name}: sim_df shape={sim_df.shape}, zs_df={zs_df.shape}, ct_df={ct_df.shape}")
            dbg(f"{model}/{metric_name}: using layers={layer_cols}")

            for layer in layer_cols:
                sim_vals = sim_df[layer].values
                zs_vals  = zs_df["f1"].values
                ct_vals  = ct_df["f1"].values

                rho_zs, n_zs = spearman_between(sim_vals, zs_vals, higher_is_better)
                rho_ct, n_ct = spearman_between(sim_vals, ct_vals, higher_is_better)

                if PLOT_BOOTSTRAP:
                    lo_zs, hi_zs = bootstrap_ci(sim_vals, zs_vals, higher_is_better, n=BOOT_N, seed=RANDOM_SEED)
                    lo_ct, hi_ct = bootstrap_ci(sim_vals, ct_vals, higher_is_better, n=BOOT_N, seed=RANDOM_SEED)
                else:
                    lo_zs = hi_zs = np.nan
                    lo_ct = hi_ct = np.nan

                rows.append({
                    "model": model,
                    "metric": metric_name,
                    "setting": "zero_shot",
                    "layer": str(layer),
                    "rho": rho_zs,
                    "n_effective": n_zs,
                    "n_langs": int(len(sim_vals)),
                    "ci_low": lo_zs,
                    "ci_high": hi_zs,
                })
                rows.append({
                    "model": model,
                    "metric": metric_name,
                    "setting": "co_train",
                    "layer": str(layer),
                    "rho": rho_ct,
                    "n_effective": n_ct,
                    "n_langs": int(len(sim_vals)),
                    "ci_low": lo_ct,
                    "ci_high": hi_ct,
                })

                dbg(f"{model}/{metric_name}/layer {layer}: rho_zs={rho_zs:.3f} (n={n_zs}), rho_ct={rho_ct:.3f} (n={n_ct})")

                plot_layers.append(str(layer))
                plot_zs.append(rho_zs)
                plot_ct.append(rho_ct)
                zs_ci.append((lo_zs, hi_zs))
                ct_ci.append((lo_ct, hi_ct))

            # -------- Rank-error heatmaps & CSVs --------
            rankdiff_zs, abserr_zs = compute_rank_error_matrix(sim_df, zs_df, higher_is_better)
            rankdiff_ct, abserr_ct = compute_rank_error_matrix(sim_df, ct_df, higher_is_better)

            # Save diagnostic CSVs
            rankdiff_zs.to_csv(out_dir / f"rankdiff_{model}_{metric_name}_zeroshot.csv")
            abserr_zs.to_csv(out_dir / f"abserr_{model}_{metric_name}_zeroshot.csv")
            rankdiff_ct.to_csv(out_dir / f"rankdiff_{model}_{metric_name}_cotr.csv")
            abserr_ct.to_csv(out_dir / f"abserr_{model}_{metric_name}_cotr.csv")

            # Diagnostic heatmaps
            title_metric = "Cosine" if higher_is_better else "SWD"
            plot_heatmap(rankdiff_zs,
                         title=f"{model.upper()} · {title_metric} · Signed rank error (Zero-shot)",
                         outpath=out_dir / f"heatmap_rankdiff_{model}_{metric_name}_zeroshot.png",
                         center=0, cmap="coolwarm")
            plot_heatmap(abserr_zs,
                         title=f"{model.upper()} · {title_metric} · |rank error| (Zero-shot)",
                         outpath=out_dir / f"heatmap_abserr_{model}_{metric_name}_zeroshot.png",
                         center=None, cmap="viridis")
            plot_heatmap(rankdiff_ct,
                         title=f"{model.upper()} · {title_metric} · Signed rank error (Co-train)",
                         outpath=out_dir / f"heatmap_rankdiff_{model}_{metric_name}_cotr.png",
                         center=0, cmap="coolwarm")
            plot_heatmap(abserr_ct,
                         title=f"{model.upper()} · {title_metric} · |rank error| (Co-train)",
                         outpath=out_dir / f"heatmap_abserr_{model}_{metric_name}_cotr.png",
                         center=None, cmap="viridis")

            # -------- MAIN heatmap for paper (zero-shot |rank error|) --------
            # These are the four files you requested for the main text:
            # mbert_core_heatmap.png, mbert_swd_heatmap.png, xlmr_core_heatmap.png, xlmr_swd_heatmap.png
            main_heatmap_path = out_dir / f"{model}_{metric_name}_heatmap.png"
            plot_heatmap(
                abserr_zs,
                title=f"{model.upper()} · {title_metric} · |rank error| (Zero-shot)",
                outpath=main_heatmap_path,
                center=None, cmap="viridis"
            )
            
            
            # -------- MAIN heatmap for paper (co-train |rank error|) --------
            main_heatmap_cotr = out_dir / f"{model}_{metric_name}_heatmap_cotr.png"
            plot_heatmap(
                abserr_ct,
                title=f"{model.upper()} · {title_metric} · |rank error| (Co-train)",
                outpath=main_heatmap_cotr,
                center=None, cmap="viridis"
            )

            # -------- Spearman line plot --------
            plt.figure(figsize=FIGSIZE)
            x = np.arange(len(plot_layers))
            plt.plot(x, plot_zs, marker="o", label="Zero-shot ρ")
            plt.plot(x, plot_ct, marker="s", label="Co-training ρ")
            if PLOT_BOOTSTRAP:
                for xi, (lo, hi) in zip(x, zs_ci):
                    if np.isfinite(lo) and np.isfinite(hi):
                        plt.fill_between([xi-0.1, xi+0.1], [lo, lo], [hi, hi], alpha=0.2)
                for xi, (lo, hi) in zip(x, ct_ci):
                    if np.isfinite(lo) and np.isfinite(hi):
                        plt.fill_between([xi-0.1, xi+0.1], [lo, lo], [hi, hi], alpha=0.2)

            plt.xticks(x, plot_layers)
            plt.xlabel("Layer")
            plt.ylabel("Spearman ρ")
            plt.title(f"{model.upper()} · {title_metric}: Similarity vs F1 rank")
            plt.grid(True, linestyle="--", alpha=0.4)
            plt.legend()
            plt.tight_layout()
            Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
            plt.savefig(Path(OUT_DIR) / f"{model}_{metric_name}_spearman.png", dpi=300)
            plt.close()

    # Write correlations
    out_csv = Path(OUT_DIR) / "correlations.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved: {out_csv} and figures in {OUT_DIR}/")


if __name__ == "__main__":
    main()

