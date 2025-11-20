#!/usr/bin/env python3
"""
Typology-based similarity + grouping for Runyankore vs 20 auxiliaries,
with optional weight learning from *multiple* performance CSVs:

  ct_f1_mbert.csv       (Co-training, mBERT)
  ct_f1_xlmr.csv        (Co-training, XLM-R)
  ct_f1_afroxlmr.csv    (Co-training, AFRO-XLMR)
  zs_f1_mbert.csv       (Zero-shot, mBERT)
  zs_f1_xlmr.csv        (Zero-shot, XLM-R)
  zs_f1_afroxlmr.csv    (Zero-shot, AFRO-XLMR)

Each perf file must have:
  language,f1
  bam,0.277
  ...

We infer:
  setup ∈ {co-train, zero-shot}          from filename prefix {ct, zs}
  model ∈ {mbert, xlmr, afroxlmr}        from filename suffix {mbert, xlmr, afroxlmr}

Outputs in --outdir:
  - typology_similarity_scores.csv               (equal-weight baseline)
  - scores_with_groups_equal.csv
  - groups_equal.csv
  - grouping_equal_report.txt

  For each (setup, model):
    - typology_similarity_scores_learned_{variant}.csv
    - scores_with_groups_learned_{variant}.csv
    - groups_{variant}.csv
    - weight_learning_{variant}.txt
    - grouping_learned_report_{variant}.txt

PLUS, integrated visualizations (PNG + SVG) written to --outdir/figures/:
  - fig1_bar_equal.[png|svg]                     (S_total ranking, equal)
  - fig2_heatmap_features_equal.[png|svg]        (three feature scores)
  - fig3_scatter_s_total_vs_f1_{variant}.[png|svg]   (S_total vs F1, per variant if perf files given)
  - fig4_weights_{variant}.[png|svg]             (learned weights bar chart)
  - fig5_pca_groups_equal.[png|svg]              (2D PCA of features, equal groups)
  - fig6_pca_groups_{variant}.[png|svg]          (2D PCA per variant)
  - fig7_group_sizes_{tag}.[png|svg]             (cluster sizes, equal + per variant)
  - figures/tables_summary.csv                   (aux_code, S_total, group, F1s wide)
  - figures/thesis_figure_captions.md            (starter captions)

(C) Optional: quick τ sensitivity in the report (no code change needed)

Run with --tau 500, --tau 1000, --tau 2000 and confirm the top-k auxiliaries and groups are stable; report one sentence in the thesis.

Example:
python3 typology_similarity.py \
  --languages languages_template.csv \
  --features lingua_features.csv \
  --outdir out_typology \
  --k 3 4 5 6 \
  --performance_files ct_f1_mbert.csv ct_f1_xlmr.csv ct_f1_afroxlmr.csv zs_f1_mbert.csv zs_f1_xlmr.csv zs_f1_afroxlmr.csv
"""

import argparse, os, math, re
import pandas as pd
import numpy as np

# -----------------------
# Optional plotting (no seaborn; safe fallbacks)
# -----------------------
HAS_MPL = True
try:
    import matplotlib.pyplot as plt
except Exception:
    HAS_MPL = False

# THREE features only
FEATURE_COLS = ["S_script","S_locale","S_geo"]

# -----------------------
# Utility computations
# -----------------------

def haversine_km(lat1, lon1, lat2, lon2):
    if any(pd.isna([lat1, lon1, lat2, lon2])):
        return np.nan
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2*R*math.asin(min(1, math.sqrt(a)))

def norm_text(x):
    if isinstance(x, str):
        return x.strip().lower()
    return ""

def jaccard(a, b):
    A = set([norm_text(x) for x in a if norm_text(x)])
    B = set([norm_text(x) for x in b if norm_text(x)])
    if not A and not B:
        return np.nan  # policy: no info -> no signal
    return len(A & B) / max(1, len(A | B))

def exp_kernel(dist_km, tau=1000.0):
    if pd.isna(dist_km): return np.nan
    return math.exp(- dist_km / tau)

def clip01(x):
    return max(0.0, min(1.0, x))

def script_score(s_t_primary, s_t_alts, s_l_primary, s_l_alts):
    stp = norm_text(s_t_primary); slp = norm_text(s_l_primary)
    t_alts = set([norm_text(x) for x in s_t_alts if norm_text(x)]) if s_t_alts else set()
    l_alts = set([norm_text(x) for x in s_l_alts if norm_text(x)]) if s_l_alts else set()
    if stp and slp and stp == slp: return 1.0
    if (stp and stp in l_alts) or (slp and slp in t_alts): return 0.5
    return 0.0

# -----------------------
# IO helpers
# -----------------------

def read_languages(path):
    return pd.read_csv(path)

def parse_countries(cell):
    if pd.isna(cell): return []
    return [x.strip() for x in str(cell).split(";") if str(x).strip()]

def read_features(path):
    df = pd.read_csv(path)
    feat = {}
    for _, row in df.iterrows():
        code = str(row["code"]).strip()
        feat[code] = {
            "countries": parse_countries(row["countries"]),
            "lat": pd.to_numeric(row["lat"], errors="coerce"),
            "lon": pd.to_numeric(row["lon"], errors="coerce"),
            "script_primary": row["script_primary"],
            "script_alt": [x.strip() for x in str(row["script_alt"]).split(";") if str(x).strip()] if not pd.isna(row["script_alt"]) else [],
        }
    return feat

# -----------------------
# Core scoring
# -----------------------

def compute_scores(lang_df, feat, tau=1000.0, weights=None):
    if weights is None:
        weights = dict(script=1, locale=1, geo=1)
    s = sum(weights.values())
    weights = {k: v/s for k, v in weights.items()}

    target_row = lang_df[lang_df["target"]==1]
    if target_row.empty:
        raise ValueError("languages CSV must include one row with target=1 (Runyankore).")
    T_code = target_row.iloc[0]["code"]
    if T_code not in feat:
        raise ValueError(f"Target code '{T_code}' missing from features CSV.")
    T = feat[T_code]

    out = []
    for _, r in lang_df[lang_df["target"]==0].iterrows():
        code = r["code"]
        if code not in feat:
            continue
        L = feat[code]
        s_script = script_score(T["script_primary"], T["script_alt"], L["script_primary"], L["script_alt"])
        s_locale = jaccard(T["countries"], L["countries"])
        d_km = haversine_km(T["lat"], T["lon"], L["lat"], L["lon"])
        s_geo = exp_kernel(d_km, tau=tau) if not pd.isna(d_km) else np.nan

        feats = {"S_script": s_script, "S_locale": s_locale, "S_geo": s_geo}

        valid = {k: v for k, v in feats.items() if not pd.isna(v)}
        if not valid:
            s_total = np.nan
        else:
            wsub = {
                "S_script": weights["script"],
                "S_locale": weights["locale"],
                "S_geo": weights["geo"],
            }
            wsub = {k: v for k, v in wsub.items() if k in valid}
            if not wsub:
                s_total = np.nan
            else:
                ws = sum(wsub.values())
                if (ws is None) or (not np.isfinite(ws)) or (ws <= 0):
                    wsub = {k: 1.0 / len(wsub) for k in wsub}  # uniform fallback
                else:
                    wsub = {k: v / ws for k, v in wsub.items()}
                s_total = sum(valid[k] * wsub[k] for k in valid)

        out.append({
            "aux_code": code,
            "aux_name": r.get("name",""),
            **feats,
            "S_total": s_total,
            "distance_km": d_km
        })
    return pd.DataFrame(out).sort_values("S_total", ascending=False, na_position="last")

# -----------------------
# Weight learning (pure NumPy OLS + non-negativity heuristic)
# -----------------------

def read_performance_files(perf_files):
    """
    Expect list like:
      ['ct_f1_mbert.csv','ct_f1_xlmr.csv','ct_f1_afroxlmr.csv',
       'zs_f1_mbert.csv','zs_f1_xlmr.csv','zs_f1_afroxlmr.csv']

    Each file has: language,f1
    Infer (setup, model) from filename.
    """
    rows = []
    # UPDATED: now also supports afroxlmr
    pat = re.compile(r'(?P<setup>ct|zs).*?(?P<model>mbert|xlmr|afroxlmr)', re.IGNORECASE)
    for path in perf_files:
        fn = os.path.basename(path)
        m = pat.search(fn)
        if not m:
            raise ValueError(f"Cannot infer (setup/model) from filename: {fn}. Expected patterns like ct_f1_mbert.csv")
        setup = "co-train" if m.group("setup").lower()=="ct" else "zero-shot"
        model = m.group("model").lower()
        df = pd.read_csv(path)
        if not {"language","f1"}.issubset(df.columns):
            raise ValueError(f"{fn} must have columns: language,f1")
        df = df.rename(columns={"language":"aux_code"})
        df["aux_code"] = df["aux_code"].astype(str).str.strip()
        df["f1"] = pd.to_numeric(df["f1"], errors="coerce")
        df["setup"] = setup
        df["model"] = model
        rows.append(df[["aux_code","f1","setup","model"]])
    if not rows:
        return pd.DataFrame(columns=["aux_code","f1","setup","model"])
    return pd.concat(rows, ignore_index=True)

def learn_weights_for_variant(scores_df, perf_df):
    """Return normalized non-negative weights and diagnostics text for the 3 features."""
    if perf_df.empty:
        return None, "No performance rows for this variant."

    merged = pd.merge(perf_df, scores_df, on="aux_code", how="inner")
    if len(merged) < 5:
        return None, "Not enough overlapping auxiliaries to learn weights (need ≥5)."

    Xdf = merged[["S_script","S_locale","S_geo"]]
    y = merged["f1"].astype(float).to_numpy()

    Xdf = Xdf.copy()
    col_means = Xdf.mean(axis=0, skipna=True)
    Xdf = Xdf.fillna(col_means)

    X = Xdf.to_numpy(dtype=float)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, ddof=0, keepdims=True); sd[sd == 0] = 1.0
    Xz = (X - mu) / sd

    order = ["script","locale","geo"]
    weights = None
    diag = []
    try:
        from scipy.optimize import nnls
        w_raw, _ = nnls(Xz, y)
        w = w_raw / w_raw.sum() if w_raw.sum() != 0 else np.ones_like(w_raw)/len(w_raw)
        weights = dict(zip(order, [float(v) for v in w]))
        diag.append("Fitted via NNLS on z-scored features (non-negative, sum=1).")
    except Exception:
        coefs, *_ = np.linalg.lstsq(Xz, y, rcond=None)
        coefs_raw = coefs.copy()
        coefs = np.clip(coefs, 0, None)
        coefs = coefs / coefs.sum() if coefs.sum() != 0 else np.ones_like(coefs)/len(coefs)
        weights = dict(zip(order, [float(v) for v in coefs]))
        diag.append("Fitted via OLS on z-scored features; clipped to ≥0 and normalized.")
        diag.append("Raw (pre-clip) coefficients: " + ", ".join(f"{v:.4f}" for v in coefs_raw))

    # Diagnostics (optional, no SciPy dependency)
    try:
        from scipy.stats import spearmanr, pearsonr
        y_pred = Xz @ np.array([weights[k] for k in order])
        rho, p_rho = spearmanr(y, y_pred)
        r, p_r = pearsonr(y, y_pred)
        ss_res = float(np.sum((y - y_pred)**2))
        ss_tot = float(np.sum((y - float(np.mean(y)))**2))
        r2 = 0.0 if ss_tot == 0 else 1 - ss_res/ss_tot
        diag.append(f"Fit diagnostics: R^2={r2:.3f}, ρ={rho:.3f} (p={p_rho:.3f}), r={r:.3f} (p={p_r:.3f})")
    except Exception:
        pass

    return weights, "\n".join(diag)

# -----------------------
# Grouping (with robust fallback if sklearn unavailable)
# -----------------------

def _quantile_bucket_groups(scores_df, ks):
    """Fallback: assign groups by S_total quantiles."""
    df = scores_df.copy()
    best = None
    for k in ks:
        try:
            df["group"] = pd.qcut(df["S_total"].rank(method="first"), q=k, labels=False, duplicates="drop")
            score = df.groupby("group")["S_total"].mean().var()
            if (best is None) or (score > best["score"]):
                best = {"k": k, "labels": df["group"].to_numpy(), "score": score}
        except Exception:
            continue
    if best is None:
        best = {"k": 1, "labels": np.zeros(len(df), dtype=int), "score": 0.0}
    return best

def cluster_and_write(scores_df, out_csv, out_groups_csv, ks=(3,4,5)):
    # Try sklearn-extra KMedoids, then sklearn KMeans, else quantile fallback
    used_algo = None
    try:
        from sklearn_extra.cluster import KMedoids
        from sklearn.metrics import silhouette_score
        X = scores_df[FEATURE_COLS].fillna(0.0).values
        best = None
        for k in ks:
            model = KMedoids(n_clusters=k, random_state=0)
            labels = model.fit_predict(X)
            sil = silhouette_score(X, labels)
            if (best is None) or (sil > best["sil"]):
                best = {"k":k,"labels":labels,"sil":sil}
        used_algo = "KMedoids (sklearn-extra)"
    except Exception:
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            X = scores_df[FEATURE_COLS].fillna(0.0).values
            best = None
            for k in ks:
                model = KMeans(n_clusters=k, random_state=0, n_init=10)
                labels = model.fit_predict(X)
                sil = silhouette_score(X, labels)
                if (best is None) or (sil > best["sil"]):
                    best = {"k":k,"labels":labels,"sil":sil}
            used_algo = "KMeans (sklearn)"
        except Exception:
            best = _quantile_bucket_groups(scores_df, ks)
            used_algo = "Quantile buckets over S_total (fallback)"
            best["sil"] = float("nan")

    out = scores_df.copy()
    out["group"] = best["labels"]
    out.to_csv(out_csv, index=False)
    out[["aux_code","aux_name","group","S_total"]].to_csv(out_groups_csv, index=False)
    return best["k"], best["sil"], used_algo, out

# -----------------------
# Plot helpers (matplotlib only; no seaborn)
# -----------------------

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def savefig(basepath):
    for ext in ("png","svg"):
        plt.savefig(f"{basepath}.{ext}", bbox_inches="tight", dpi=300)

def merge_names(df, langs_df):
    names = langs_df.rename(columns={"code":"aux_code","name":"aux_name"})
    if "aux_name" in df.columns:
        df = df.merge(names[["aux_code","aux_name"]], on="aux_code", how="left", suffixes=("","_fromlangs"))
        df["aux_name"] = df["aux_name"].fillna(df["aux_name_fromlangs"])
        df = df.drop(columns=[c for c in df.columns if c.endswith("_fromlangs")])
    else:
        df = df.merge(names[["aux_code","aux_name"]], on="aux_code", how="left")
    return df

def plot_bar_ranking(df, outpath, title="Typology similarity (S_total)  (equal weights)"):
    plt.figure(figsize=(8, 6))
    plotdf = df[["aux_code","S_total"]].sort_values("S_total", ascending=True)
    y = np.arange(len(plotdf))
    plt.barh(y, plotdf["S_total"].values)
    plt.yticks(y, plotdf["aux_code"].tolist())
    plt.xlabel("S_total (0–1)")
    plt.title(title)
    for yi, val in zip(y, plotdf["S_total"].values):
        plt.text(float(val) + 0.005, yi, f"{float(val):.2f}", va="center", fontsize=8)
    plt.tight_layout()
    savefig(outpath)
    plt.close()

def plot_feature_heatmap(df, outpath, title="Feature scores by language (equal weights)"):
    plotdf = df[["aux_code"] + FEATURE_COLS].reset_index(drop=True)
    data = np.asarray(plotdf[FEATURE_COLS].values, dtype=float)
    plt.figure(figsize=(8, max(5, 0.35*len(plotdf))))
    im = plt.imshow(data, aspect="auto", interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.yticks(np.arange(len(plotdf)), plotdf["aux_code"].tolist())
    plt.xticks(np.arange(len(FEATURE_COLS)), [c.replace("S_","") for c in FEATURE_COLS], rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    savefig(outpath)
    plt.close()

def _linreg(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return None, None
    A = np.vstack([x, np.ones_like(x)]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, b

def plot_s_total_vs_f1(df_scores, df_perf, variant, outpath):
    merged = df_scores[["aux_code","S_total"]].merge(
        df_perf[df_perf["variant"]==variant][["aux_code","f1"]], on="aux_code", how="inner"
    )
    if merged.empty:
        return
    plt.figure(figsize=(6.5, 5.5))
    plt.scatter(merged["S_total"].values, merged["f1"].values)
    for _, r in merged.iterrows():
        plt.annotate(str(r["aux_code"]), (float(r["S_total"]), float(r["f1"])), xytext=(3,3), textcoords="offset points", fontsize=8)
    plt.xlabel("S_total")
    plt.ylabel("F1")
    plt.title(f"S_total vs F1 — {variant.replace('_',' / ')}")
    m, b = _linreg(merged["S_total"].values, merged["f1"].values)
    if m is not None:
        xs = np.linspace(float(merged["S_total"].min()), float(merged["S_total"].max()), 100)
        ys = m*xs + b
        plt.plot(xs, ys)
    # Try to annotate correlations
    try:
        from scipy.stats import spearmanr, pearsonr
        r, _ = pearsonr(merged["S_total"].values, merged["f1"].values)
        rho, _ = spearmanr(merged["S_total"].values, merged["f1"].values)
        plt.text(0.02, 0.98, f"r={r:.2f}, ρ={rho:.2f}", transform=plt.gca().transAxes, va="top")
    except Exception:
        pass
    plt.tight_layout()
    savefig(outpath)
    plt.close()

def plot_weights_bar(weights, outpath, title):
    order = ["script","locale","geo"]
    vals = [float(weights.get(k, 0.0)) for k in order]
    plt.figure(figsize=(7,4))
    plt.bar(np.arange(len(order)), vals)
    plt.xticks(np.arange(len(order)), order, rotation=30, ha="right")
    for i, v in enumerate(vals):
        plt.text(i, float(v) + 0.01, f"{float(v):.2f}", ha="center", va="bottom", fontsize=9)
    plt.ylim(0, max(0.5, max(vals)+0.1 if vals else 0.6))
    plt.title(title)
    plt.ylabel("Weight (sum=1)")
    plt.tight_layout()
    savefig(outpath)
    plt.close()

def pca_2d(features):
    X = np.asarray(features, dtype=float)
    X = np.nan_to_num(X, nan=0.0)
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T

def plot_pca_groups(df, outpath, title="Group structure (PCA of features)"):
    X = df[FEATURE_COLS].fillna(0.0).to_numpy(dtype=float)
    Z = pca_2d(X)
    labels = df["group"].to_numpy() if "group" in df.columns else np.zeros(len(df), dtype=int)
    codes = df["aux_code"].tolist()
    plt.figure(figsize=(6.5,5.5))
    for g in sorted(set(labels)):
        idx = np.where(labels==g)[0]
        plt.scatter(Z[idx,0], Z[idx,1], label=f"group {int(g)}")
        for i in idx:
            plt.annotate(str(codes[i]), (float(Z[i,0]), float(Z[i,1])), xytext=(3,3), textcoords="offset points", fontsize=8)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.legend(loc="best", fontsize=9, frameon=False)
    plt.tight_layout()
    savefig(outpath)
    plt.close()

def plot_group_sizes(df, outpath, title):
    if "group" not in df.columns:
        return
    ct = df.groupby("group")["aux_code"].count().reset_index(name="count")
    plt.figure(figsize=(5.5,4))
    plt.bar(ct["group"].astype(str).values, ct["count"].values)
    for i, v in enumerate(ct["count"].values):
        plt.text(i, float(v)+0.1, str(int(v)), ha="center", va="bottom", fontsize=10)
    plt.xlabel("Group")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    savefig(outpath)
    plt.close()

# -----------------------
# CLI
# -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--languages", required=True, help="languages_template.csv")
    ap.add_argument("--features", required=True, help="lingua_features.csv (or template)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tau", type=float, default=1000.0)
    ap.add_argument("--k", nargs="+", type=int, default=[3,4,5], help="candidate cluster sizes")
    ap.add_argument(
        "--performance_files",
        nargs="*",
        default=[],
        help="paths to ct/zs mbert/xlmr/afroxlmr CSVs",
    )
    ap.add_argument("--no_figures", action="store_true", help="skip figure generation")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    figs_dir = os.path.join(args.outdir, "figures")
    langs = read_languages(args.languages)
    feat = read_features(args.features)

    # 1) Equal-weight baseline (3 features)
    base_weights = dict(script=1, locale=1, geo=1)
    scores_equal = compute_scores(langs, feat, tau=args.tau, weights=base_weights)
    scores_equal.to_csv(os.path.join(args.outdir, "typology_similarity_scores.csv"), index=False)

    k_best, sil_best, algo, df_equal_labeled = cluster_and_write(
        scores_equal,
        out_csv=os.path.join(args.outdir, "scores_with_groups_equal.csv"),
        out_groups_csv=os.path.join(args.outdir, "groups_equal.csv"),
        ks=args.k
    )
    with open(os.path.join(args.outdir, "grouping_equal_report.txt"), "w") as f:
        f.write(f"Algorithm: {algo}\n")
        f.write(f"Best k={k_best} with silhouette={sil_best}\n")

    # 2) Optional: learn weights from multiple performance CSVs
    perf_all = read_performance_files(args.performance_files) if args.performance_files else pd.DataFrame(columns=["aux_code","f1","setup","model"])
    variant_results = {}  # tag -> dict(df, weights)

    if not perf_all.empty:
        for (setup, model), df_subset in perf_all.groupby(["setup","model"]):
            weights, diag = learn_weights_for_variant(scores_equal, df_subset)
            tag = f"{setup.replace('-','')}_{model}".lower()  # e.g., cotrain_mbert, zeroshot_afroxlmr
            with open(os.path.join(args.outdir, f"weight_learning_{tag}.txt"), "w") as f:
                f.write(
                    "Learned weights (script, locale, geo): "
                    + ", ".join(f"{k}={v:.4f}" for k, v in weights.items())
                    +"\n"
                )                
                f.write(diag + "\n")
            if weights is None:
                continue

            scores_learned = compute_scores(langs, feat, tau=args.tau, weights=weights)
            scores_learned.to_csv(os.path.join(args.outdir, f"typology_similarity_scores_learned_{tag}.csv"), index=False)

            k_best_l, sil_best_l, algo_l, df_learned_labeled = cluster_and_write(
                scores_learned,
                out_csv=os.path.join(args.outdir, f"scores_with_groups_learned_{tag}.csv"),
                out_groups_csv=os.path.join(args.outdir, f"groups_{tag}.csv"),
                ks=args.k
            )
            with open(os.path.join(args.outdir, f"grouping_learned_report_{tag}.txt"), "w") as f:
                f.write(f"Algorithm: {algo_l}\n")
                f.write(f"Best k={k_best_l} with silhouette={sil_best_l}\n")

            variant_results[tag] = {"df": df_learned_labeled, "weights": weights}

    # 3) Make figures (unless disabled or matplotlib missing)
    if not args.no_figures:
        if not HAS_MPL:
            print("[warn] matplotlib not available; skipping figures")
        else:
            ensure_dir(figs_dir)

            # Merge names for nicer labels
            df_equal_plot = merge_names(df_equal_labeled.sort_values("S_total", ascending=False), langs)

            # Equal-weight plots
            plot_bar_ranking(df_equal_plot, os.path.join(figs_dir, "fig1_bar_equal"))
            plot_feature_heatmap(df_equal_plot, os.path.join(figs_dir, "fig2_heatmap_features_equal"))
            plot_pca_groups(df_equal_plot, os.path.join(figs_dir, "fig5_pca_groups_equal"), "Group structure (equal weights)")
            plot_group_sizes(df_equal_plot, os.path.join(figs_dir, "fig7_group_sizes_equal"), "Cluster sizes (equal weights)")

            # Build a wide summary table with F1s
            summary = df_equal_plot[["aux_code","aux_name","S_total","group"]].copy()
            if not perf_all.empty:
                perf_tag = perf_all.copy()
                perf_tag["variant"] = perf_tag.apply(
                    lambda r: f"{'cotrain' if r['setup']=='co-train' else 'zeroshot'}_{r['model']}",
                    axis=1
                )
                piv = perf_tag.pivot_table(index="aux_code", columns="variant", values="f1", aggfunc="mean").reset_index()
                summary = summary.merge(piv, on="aux_code", how="left")
            summary.to_csv(os.path.join(figs_dir, "tables_summary.csv"), index=False)

            # Variant plots: scatter S_total vs F1, weights bar, PCA, sizes
            if not perf_all.empty:
                perf_all["variant"] = perf_all.apply(
                    lambda r: f"{'cotrain' if r['setup']=='co-train' else 'zeroshot'}_{r['model']}",
                    axis=1
                )

            for tag, pack in variant_results.items():
                dfv = merge_names(pack["df"].sort_values("S_total", ascending=False), langs)
                if not perf_all.empty and (perf_all["variant"]==tag).any():
                    plot_s_total_vs_f1(dfv, perf_all, tag, os.path.join(figs_dir, f"fig3_scatter_s_total_vs_f1_{tag}"))
                plot_weights_bar(pack["weights"], os.path.join(figs_dir, f"fig4_weights_{tag}"), f"Learned weights — {tag.replace('_',' / ')}")
                plot_pca_groups(dfv, os.path.join(figs_dir, f"fig6_pca_groups_{tag}"), f"Group structure — {tag.replace('_',' / ')}")
                plot_group_sizes(dfv, os.path.join(figs_dir, f"fig7_group_sizes_{tag}"), f"Cluster sizes — {tag.replace('_',' / ')}")

            # Captions stub (only written if missing)
            cap_path = os.path.join(figs_dir, "thesis_figure_captions.md")
            if not os.path.exists(cap_path):
                with open(cap_path, "w", encoding="utf-8") as f:
                    f.write(
"""# Thesis Figure Captions (starter)
- Fig. 1 — Typology similarity ranking (equal weights). Bars show S_total for each auxiliary language relative to Runyankore.
- Fig. 2 — Feature heatmap (equal weights). Rows are auxiliaries; columns are three typology features used to compute S_total.
- Fig. 3 — S_total vs F1 (per variant). Correlation between typology similarity and downstream NER F1 (co-train / zero-shot × model).
- Fig. 4 — Learned feature weights (per variant). Data-driven weights that best align similarity with observed F1s.
- Fig. 5/6 — PCA view of groups. 2D projection of the three-dimensional feature space with cluster labels.
- Fig. 7 — Cluster sizes. Number of auxiliaries assigned to each typology cluster (equal or learned-weights).
"""
                    )
            print(f"[viz] wrote figures & tables to: {figs_dir}")

if __name__ == "__main__":
    main()
