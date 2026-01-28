#!/usr/bin/env python3
"""
Typology-based similarity + grouping for Runyankore vs auxiliaries,
with optional weight learning from multiple performance CSVs.

Key fixes (for thesis/supervisor):
  1) Full silhouette sweep is reported (per k) in grouping reports.
  2) Canonical S_total is equal-weight and is used consistently across plots,
     especially all scatter plots (S_total vs F1) across variants.
  3) Reduce "S_total looks like 0" issues by:
       - normalizing script labels (Latin/Latn/etc)
       - adding diagnostics for missingness/zero-mass features
       - mean-imputing for clustering/PCA/heatmap instead of fillna(0)
  4) Locale similarity uses macro-region overlap (region column) rather than country overlap.

Outputs in --outdir:
  - typology_similarity_scores.csv
  - scores_with_groups_equal.csv
  - groups_equal.csv
  - grouping_equal_report.txt
  - diagnostics_equal.txt

If performance files given, per (setup, model) variant tag:
  - weight_learning_{tag}.txt
  - typology_similarity_scores_learned_{tag}.csv
  - scores_with_groups_learned_{tag}.csv
  - groups_{tag}.csv
  - grouping_learned_report_{tag}.txt

Figures in --outdir/figures/:
  - fig1_bar_equal.(png|svg)
  - fig2_heatmap_features_equal.(png|svg)
  - fig3_scatter_s_total_vs_f1_{tag}.(png|svg)     [S_total always from equal-weight]
  - fig4_weights_{tag}.(png|svg)
  - fig5_pca_groups_equal.(png|svg)
  - fig6_pca_groups_{tag}.(png|svg)
  - fig7_group_sizes_equal.(png|svg)
  - fig7_group_sizes_{tag}.(png|svg)
  - tables_summary.csv
  - thesis_figure_captions.md 
  
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
# Optional plotting (no seaborn)
# -----------------------
HAS_MPL = True
try:
    import matplotlib.pyplot as plt
except Exception:
    HAS_MPL = False

FEATURE_COLS = ["S_script", "S_locale", "S_geo"]

# -----------------------
# Script normalization
# -----------------------

SCRIPT_ALIASES = {
    "latin": "latn",
    "latn": "latn",
    "latin script": "latn",
    "roman": "latn",
    "roman script": "latn",

    "arabic": "arab",
    "arab": "arab",
    "arabic script": "arab",

    "cyrillic": "cyrl",
    "cyrl": "cyrl",
    "cyrillic script": "cyrl",

    "ethiopic": "ethi",
    "ethi": "ethi",
    "geez": "ethi",
}

def norm_text(x):
    if isinstance(x, str):
        return x.strip().lower()
    return ""

def norm_script(x):
    t = norm_text(x)
    if not t:
        return ""
    return SCRIPT_ALIASES.get(t, t)

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
    return 2 * R * math.asin(min(1, math.sqrt(a)))

def jaccard(a, b):
    A = set([norm_text(x) for x in a if norm_text(x)])
    B = set([norm_text(x) for x in b if norm_text(x)])
    if not A and not B:
        return np.nan
    return len(A & B) / max(1, len(A | B))

def exp_kernel(dist_km, tau=1000.0):
    if pd.isna(dist_km):
        return np.nan
    return math.exp(-dist_km / tau)

def script_score(s_t_primary, s_t_alts, s_l_primary, s_l_alts):
    stp = norm_script(s_t_primary)
    slp = norm_script(s_l_primary)

    t_alts = set([norm_script(x) for x in (s_t_alts or []) if norm_script(x)])
    l_alts = set([norm_script(x) for x in (s_l_alts or []) if norm_script(x)])

    if stp and slp and stp == slp:
        return 1.0
    if (stp and stp in l_alts) or (slp and slp in t_alts):
        return 0.5
    return 0.0

# -----------------------
# IO helpers
# -----------------------

def read_languages(path):
    return pd.read_csv(path)

def parse_semicolon_list(cell):
    if pd.isna(cell):
        return []
    return [x.strip() for x in str(cell).split(";") if str(x).strip()]

def read_features(path):
    """
    Expects lingua_features.csv with at least:
      code, countries, region, lat, lon, script_primary, script_alt, ...

    region can be a single value (e.g., "East Africa") or multiple separated by ';'
    (e.g., "East Africa; Central Africa").
    """
    df = pd.read_csv(path)
    feat = {}
    for _, row in df.iterrows():
        code = str(row["code"]).strip()
        feat[code] = {
            "countries": parse_semicolon_list(row.get("countries", np.nan)),
            "regions": parse_semicolon_list(row.get("region", np.nan)),   # <-- NEW
            "lat": pd.to_numeric(row.get("lat", np.nan), errors="coerce"),
            "lon": pd.to_numeric(row.get("lon", np.nan), errors="coerce"),
            "script_primary": row.get("script_primary", np.nan),
            "script_alt": parse_semicolon_list(row.get("script_alt", np.nan)),
        }
    return feat

# -----------------------
# Core scoring
# -----------------------

def compute_scores(lang_df, feat, tau=1000.0, weights=None):
    if weights is None:
        weights = dict(script=1, locale=1, geo=1)
    s = sum(weights.values())
    weights = {k: v / s for k, v in weights.items()}

    target_row = lang_df[lang_df["target"] == 1]
    if target_row.empty:
        raise ValueError("languages CSV must include one row with target=1 (Runyankore).")
    T_code = target_row.iloc[0]["code"]
    if T_code not in feat:
        raise ValueError(f"Target code '{T_code}' missing from features CSV.")
    T = feat[T_code]

    out = []
    for _, r in lang_df[lang_df["target"] == 0].iterrows():
        code = r["code"]
        if code not in feat:
            continue
        L = feat[code]

        s_script = script_score(T["script_primary"], T["script_alt"], L["script_primary"], L["script_alt"])

        # UPDATED: macro-region overlap instead of country overlap
        s_locale = jaccard(T.get("regions", []), L.get("regions", []))

        d_km = haversine_km(T["lat"], T["lon"], L["lat"], L["lon"])
        s_geo = exp_kernel(d_km, tau=tau) if not pd.isna(d_km) else np.nan

        feats = {"S_script": s_script, "S_locale": s_locale, "S_geo": s_geo}

        valid = {k: v for k, v in feats.items() if not pd.isna(v)}
        if not valid:
            s_total = np.nan
        else:
            wsub = {"S_script": weights["script"], "S_locale": weights["locale"], "S_geo": weights["geo"]}
            wsub = {k: v for k, v in wsub.items() if k in valid}
            ws = sum(wsub.values())
            if (not np.isfinite(ws)) or (ws <= 0):
                wsub = {k: 1.0 / len(wsub) for k in wsub}
            else:
                wsub = {k: v / ws for k, v in wsub.items()}
            s_total = sum(valid[k] * wsub[k] for k in valid)

        out.append({
            "aux_code": code,
            "aux_name": r.get("name", ""),
            **feats,
            "S_total": s_total,
            "distance_km": d_km
        })

    return pd.DataFrame(out).sort_values("S_total", ascending=False, na_position="last")

# -----------------------
# Diagnostics
# -----------------------

def write_diagnostics(scores_df, outpath, title="Diagnostics"):
    df = scores_df.copy()
    lines = []
    lines.append(title)
    lines.append("=" * len(title))
    lines.append("")
    lines.append("Missingness per feature:")
    for c in FEATURE_COLS + ["S_total", "distance_km"]:
        if c in df.columns:
            lines.append(f"- {c}: missing={int(df[c].isna().sum())} / {len(df)}")
    lines.append("")

    lines.append("Zero fractions (exact zeros) per feature (ignoring NaN):")
    for c in FEATURE_COLS:
        if c in df.columns:
            s = df[c].dropna()
            frac0 = float((s == 0).mean()) if len(s) else float("nan")
            lines.append(f"- {c}: frac_zero={frac0:.3f}")
    lines.append("")

    lines.append("S_total summary:")
    if "S_total" in df.columns:
        lines.append(str(df["S_total"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])))
    lines.append("")

    only_script = df["S_locale"].isna() & df["S_geo"].isna() & df["S_script"].notna()
    none_avail = df["S_script"].isna() & df["S_locale"].isna() & df["S_geo"].isna()
    lines.append(f"Rows where only script is available (locale+geo missing): {int(only_script.sum())} / {len(df)}")
    lines.append(f"Rows where ALL features missing: {int(none_avail.sum())} / {len(df)}")
    lines.append("")

    with open(outpath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

# -----------------------
# Weight learning
# -----------------------

def read_performance_files(perf_files):
    rows = []
    pat = re.compile(r"(?P<setup>ct|zs).*?(?P<model>mbert|xlmr|afroxlmr)", re.IGNORECASE)
    for path in perf_files:
        fn = os.path.basename(path)
        m = pat.search(fn)
        if not m:
            raise ValueError(f"Cannot infer (setup/model) from filename: {fn}. Expected patterns like ct_f1_mbert.csv")
        setup = "co-train" if m.group("setup").lower() == "ct" else "zero-shot"
        model = m.group("model").lower()

        df = pd.read_csv(path)
        if not {"language", "f1"}.issubset(df.columns):
            raise ValueError(f"{fn} must have columns: language,f1")
        df = df.rename(columns={"language": "aux_code"})
        df["aux_code"] = df["aux_code"].astype(str).str.strip()
        df["f1"] = pd.to_numeric(df["f1"], errors="coerce")
        df["setup"] = setup
        df["model"] = model
        rows.append(df[["aux_code", "f1", "setup", "model"]])

    if not rows:
        return pd.DataFrame(columns=["aux_code", "f1", "setup", "model"])
    return pd.concat(rows, ignore_index=True)

def learn_weights_for_variant(scores_df, perf_df):
    if perf_df.empty:
        return None, "No performance rows for this variant."

    merged = pd.merge(perf_df, scores_df, on="aux_code", how="inner")
    if len(merged) < 5:
        return None, "Not enough overlapping auxiliaries to learn weights (need ≥5)."

    Xdf = merged[["S_script", "S_locale", "S_geo"]].copy()
    y = merged["f1"].astype(float).to_numpy()

    col_means = Xdf.mean(axis=0, skipna=True)
    Xdf = Xdf.fillna(col_means).fillna(0.0)

    X = Xdf.to_numpy(dtype=float)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, ddof=0, keepdims=True)
    sd[sd == 0] = 1.0
    Xz = (X - mu) / sd

    order = ["script", "locale", "geo"]
    diag = []
    weights = None

    try:
        from scipy.optimize import nnls
        w_raw, _ = nnls(Xz, y)
        w = w_raw / w_raw.sum() if w_raw.sum() != 0 else np.ones_like(w_raw) / len(w_raw)
        weights = dict(zip(order, [float(v) for v in w]))
        diag.append("Fitted via NNLS on z-scored features (non-negative, sum=1).")
    except Exception:
        coefs, *_ = np.linalg.lstsq(Xz, y, rcond=None)
        raw = coefs.copy()
        coefs = np.clip(coefs, 0, None)
        coefs = coefs / coefs.sum() if coefs.sum() != 0 else np.ones_like(coefs) / len(coefs)
        weights = dict(zip(order, [float(v) for v in coefs]))
        diag.append("Fitted via OLS on z-scored features; clipped to ≥0 and normalized.")
        diag.append("Raw (pre-clip) coefficients: " + ", ".join(f"{v:.4f}" for v in raw))

    try:
        from scipy.stats import spearmanr, pearsonr
        y_pred = Xz @ np.array([weights[k] for k in order])
        rho, p_rho = spearmanr(y, y_pred)
        r, p_r = pearsonr(y, y_pred)
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
        r2 = 0.0 if ss_tot == 0 else 1 - ss_res / ss_tot
        diag.append(f"Fit diagnostics: R^2={r2:.3f}, ρ={rho:.3f} (p={p_rho:.3f}), r={r:.3f} (p={p_r:.3f})")
    except Exception:
        pass

    return weights, "\n".join(diag)

# -----------------------
# Grouping (silhouette sweep)
# -----------------------

def _impute_feature_matrix(scores_df):
    Xdf = scores_df[FEATURE_COLS].copy()
    Xdf = Xdf.fillna(Xdf.mean(axis=0, skipna=True)).fillna(0.0)
    return Xdf.to_numpy(dtype=float)

def _quantile_bucket_groups(scores_df, ks):
    df = scores_df.copy()
    best = None
    sil_table = []
    for k in ks:
        try:
            df["group"] = pd.qcut(df["S_total"].rank(method="first"), q=k, labels=False, duplicates="drop")
            score = df.groupby("group")["S_total"].mean().var()
            sil_table.append((k, float("nan")))
            if (best is None) or (score > best["score"]):
                best = {"k": k, "labels": df["group"].to_numpy(), "score": score, "sil": float("nan")}
        except Exception:
            continue
    if best is None:
        best = {"k": 1, "labels": np.zeros(len(df), dtype=int), "score": 0.0, "sil": float("nan")}
        sil_table = [(1, float("nan"))]
    return best, sil_table

def cluster_and_write(scores_df, out_csv, out_groups_csv, ks=(3, 4, 5)):
    used_algo = None
    sil_table = []

    try:
        from sklearn_extra.cluster import KMedoids
        from sklearn.metrics import silhouette_score
        X = _impute_feature_matrix(scores_df)
        best = None
        for k in ks:
            model = KMedoids(n_clusters=k, random_state=0)
            labels = model.fit_predict(X)
            sil = float(silhouette_score(X, labels)) if len(set(labels)) > 1 else float("nan")
            sil_table.append((k, sil))
            if (best is None) or (sil > best["sil"]):
                best = {"k": k, "labels": labels, "sil": sil}
        used_algo = "KMedoids (sklearn-extra)"
    except Exception:
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            X = _impute_feature_matrix(scores_df)
            best = None
            for k in ks:
                model = KMeans(n_clusters=k, random_state=0, n_init=10)
                labels = model.fit_predict(X)
                sil = float(silhouette_score(X, labels)) if len(set(labels)) > 1 else float("nan")
                sil_table.append((k, sil))
                if (best is None) or (sil > best["sil"]):
                    best = {"k": k, "labels": labels, "sil": sil}
            used_algo = "KMeans (sklearn)"
        except Exception:
            best, sil_table = _quantile_bucket_groups(scores_df, ks)
            used_algo = "Quantile buckets over S_total (fallback)"

    out = scores_df.copy()
    out["group"] = best["labels"]
    out.to_csv(out_csv, index=False)
    out[["aux_code", "aux_name", "group", "S_total"]].to_csv(out_groups_csv, index=False)

    return best["k"], best.get("sil", float("nan")), used_algo, out, sil_table

# -----------------------
# Plot helpers
# -----------------------

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def savefig(basepath):
    for ext in ("png", "svg"):
        plt.savefig(f"{basepath}.{ext}", bbox_inches="tight", dpi=300)

def merge_names(df, langs_df):
    names = langs_df.rename(columns={"code": "aux_code", "name": "aux_name"})
    if "aux_name" in df.columns:
        df = df.merge(names[["aux_code", "aux_name"]], on="aux_code", how="left", suffixes=("", "_fromlangs"))
        df["aux_name"] = df["aux_name"].fillna(df["aux_name_fromlangs"])
        df = df.drop(columns=[c for c in df.columns if c.endswith("_fromlangs")])
    else:
        df = df.merge(names[["aux_code", "aux_name"]], on="aux_code", how="left")
    return df

def compute_and_write_correlations_metadata_vs_f1(df_scores_equal, perf_all, out_csv):
    """
    Writes a structured correlation table:
      variant, setup, model, n, pearson_r, pearson_p, spearman_rho, spearman_p

    - Uses canonical equal-weight S_total from df_scores_equal (as per supervisor expectation).
    - Uses perf_all which must contain columns: aux_code, f1, setup, model, variant
    """
    if perf_all is None or perf_all.empty:
        return

    required = {"aux_code", "f1", "variant", "setup", "model"}
    if not required.issubset(set(perf_all.columns)):
        return

    # Merge once to ensure consistent S_total per aux_code
    base = df_scores_equal[["aux_code", "S_total"]].copy()

    rows = []
    for variant, dfv in perf_all.groupby("variant"):
        merged = base.merge(dfv[["aux_code", "f1", "setup", "model"]], on="aux_code", how="inner")
        merged = merged.dropna(subset=["S_total", "f1"])

        n = int(len(merged))
        if n < 3:
            # not enough points for correlation
            setup = str(dfv["setup"].iloc[0]) if "setup" in dfv.columns and len(dfv) else ""
            model = str(dfv["model"].iloc[0]) if "model" in dfv.columns and len(dfv) else ""
            rows.append({
                "variant": variant,
                "setup": setup,
                "model": model,
                "n": n,
                "pearson_r": np.nan,
                "pearson_p": np.nan,
                "spearman_rho": np.nan,
                "spearman_p": np.nan,
            })
            continue

        setup = str(merged["setup"].iloc[0]) if "setup" in merged.columns else ""
        model = str(merged["model"].iloc[0]) if "model" in merged.columns else ""

        # Prefer SciPy if available (gives p-values). Fallback computes correlations without p-values.
        try:
            from scipy.stats import pearsonr, spearmanr
            pr, pp = pearsonr(merged["S_total"].to_numpy(dtype=float), merged["f1"].to_numpy(dtype=float))
            sr, sp = spearmanr(merged["S_total"].to_numpy(dtype=float), merged["f1"].to_numpy(dtype=float))
        except Exception:
            x = merged["S_total"].to_numpy(dtype=float)
            y = merged["f1"].to_numpy(dtype=float)
            # Pearson via numpy
            pr = float(np.corrcoef(x, y)[0, 1])
            # Spearman via ranking then Pearson
            xr = pd.Series(x).rank(method="average").to_numpy(dtype=float)
            yr = pd.Series(y).rank(method="average").to_numpy(dtype=float)
            sr = float(np.corrcoef(xr, yr)[0, 1])
            pp = np.nan
            sp = np.nan

        rows.append({
            "variant": variant,
            "setup": setup,
            "model": model,
            "n": n,
            "pearson_r": float(pr),
            "pearson_p": float(pp) if np.isfinite(pp) else np.nan,
            "spearman_rho": float(sr),
            "spearman_p": float(sp) if np.isfinite(sp) else np.nan,
        })

    out_df = pd.DataFrame(rows).sort_values(["setup", "model", "variant"])
    out_df.to_csv(out_csv, index=False)

def plot_bar_ranking(df, outpath, title="Typology similarity (S_total) — equal weights"):
    plt.figure(figsize=(8, 6))
    plotdf = df[["aux_code", "S_total"]].sort_values("S_total", ascending=True)
    y = np.arange(len(plotdf))
    plt.barh(y, plotdf["S_total"].values)
    plt.yticks(y, plotdf["aux_code"].tolist())
    plt.xlabel("S_total (0–1)")
    plt.title(title)
    for yi, val in zip(y, plotdf["S_total"].values):
        if pd.isna(val):
            continue
        plt.text(float(val) + 0.005, yi, f"{float(val):.3f}", va="center", fontsize=8)
    plt.tight_layout()
    savefig(outpath)
    plt.close()

def plot_feature_heatmap(df, outpath, title="Feature scores by language — equal weights"):
    plotdf = df[["aux_code"] + FEATURE_COLS].reset_index(drop=True)
    data = plotdf[FEATURE_COLS].copy()
    data = data.fillna(data.mean(axis=0, skipna=True)).fillna(0.0)
    mat = np.asarray(data.values, dtype=float)

    plt.figure(figsize=(8, max(5, 0.35 * len(plotdf))))
    im = plt.imshow(mat, aspect="auto", interpolation="nearest", vmin=0.0, vmax=1.0)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.yticks(np.arange(len(plotdf)), plotdf["aux_code"].tolist())
    plt.xticks(np.arange(len(FEATURE_COLS)), [c.replace("S_", "") for c in FEATURE_COLS], rotation=45, ha="right")
    plt.title(title + " (macro-region for locale)")
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

def plot_s_total_vs_f1(df_scores_equal, df_perf, variant, outpath):
    if "variant" not in df_perf.columns:
        return

    merged = df_scores_equal[["aux_code", "S_total"]].merge(
        df_perf[df_perf["variant"] == variant][["aux_code", "f1"]],
        on="aux_code",
        how="inner"
    )
    if merged.empty:
        return

    plt.figure(figsize=(6.5, 5.5))
    plt.scatter(merged["S_total"].values, merged["f1"].values)

    for _, r in merged.iterrows():
        plt.annotate(
            str(r["aux_code"]),
            (float(r["S_total"]), float(r["f1"])),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=8
        )

    # ---- Axis readability fix (Option 1) ----
    ax = plt.gca()
    ax.set_xlim(0.0, 1.0)
    ticks = np.linspace(0.0, 1.0, 11)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.1f}" for t in ticks])
    ax.grid(True, axis="x", linestyle="--", linewidth=0.7, alpha=0.5)

    plt.xlabel("S_total (equal-weight metadata similarity)")
    plt.ylabel("F1")
    plt.title(f"S_total (equal) vs F1 — {variant.replace('_', ' / ')}")

    m, b = _linreg(merged["S_total"].values, merged["f1"].values)
    if m is not None:
        xs = np.linspace(0.0, 1.0, 200)
        ys = m * xs + b
        plt.plot(xs, ys)

    try:
        from scipy.stats import spearmanr, pearsonr
        r, _ = pearsonr(merged["S_total"].values, merged["f1"].values)
        rho, _ = spearmanr(merged["S_total"].values, merged["f1"].values)
        plt.text(0.02, 0.98, f"r={r:.2f}, ρ={rho:.2f}",
                 transform=plt.gca().transAxes, va="top")
    except Exception:
        pass

    plt.tight_layout()
    savefig(outpath)
    plt.close()


def plot_weights_bar(weights, outpath, title):
    order = ["script", "locale", "geo"]
    vals = [float(weights.get(k, 0.0)) for k in order]
    plt.figure(figsize=(7, 4))
    plt.bar(np.arange(len(order)), vals)
    plt.xticks(np.arange(len(order)), order, rotation=30, ha="right")
    for i, v in enumerate(vals):
        plt.text(i, float(v) + 0.01, f"{float(v):.2f}", ha="center", va="bottom", fontsize=9)
    plt.ylim(0, max(0.5, max(vals) + 0.1 if vals else 0.6))
    plt.title(title)
    plt.ylabel("Weight (sum=1)")
    plt.tight_layout()
    savefig(outpath)
    plt.close()

def pca_2d(features):
    X = np.asarray(features, dtype=float)
    X = np.nan_to_num(X, nan=0.0)
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T

def plot_pca_groups(df, outpath, title="Group structure (PCA of features)"):
    Xdf = df[FEATURE_COLS].copy()
    Xdf = Xdf.fillna(Xdf.mean(axis=0, skipna=True)).fillna(0.0)
    X = Xdf.to_numpy(dtype=float)

    Z = pca_2d(X)
    labels = df["group"].to_numpy() if "group" in df.columns else np.zeros(len(df), dtype=int)
    codes = df["aux_code"].tolist()

    plt.figure(figsize=(6.5, 5.5))
    for g in sorted(set(labels)):
        idx = np.where(labels == g)[0]
        plt.scatter(Z[idx, 0], Z[idx, 1], label=f"group {int(g)}")
        for i in idx:
            plt.annotate(str(codes[i]), (float(Z[i, 0]), float(Z[i, 1])),
                         xytext=(3, 3), textcoords="offset points", fontsize=8)
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
    plt.figure(figsize=(5.5, 4))
    plt.bar(ct["group"].astype(str).values, ct["count"].values)
    for i, v in enumerate(ct["count"].values):
        plt.text(i, float(v) + 0.1, str(int(v)), ha="center", va="bottom", fontsize=10)
    plt.xlabel("Group")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    savefig(outpath)
    plt.close()

def write_captions_stub(path):
    if os.path.exists(path):
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write(
"""# Thesis Figure Captions (starter)
- Fig. 1 — Typology similarity ranking (equal weights). Bars show S_total for each auxiliary language relative to Runyankore.
- Fig. 2 — Feature heatmap (equal weights). Rows are auxiliaries; columns are three typology features used to compute S_total. Locale is macro-region overlap.
- Fig. 3 — S_total (equal weights) vs F1 (per variant). Correlation between metadata similarity and downstream NER F1 (co-train / zero-shot × model).
- Fig. 4 — Learned feature weights (per variant). Data-driven weights that best align typology features with observed F1s (feature-importance analysis).
- Fig. 5/6 — PCA view of groups. 2D projection of the three-dimensional feature space with cluster labels.
- Fig. 7 — Cluster sizes. Number of auxiliaries assigned to each typology cluster (equal or learned-weights).
"""
        )

# -----------------------
# CLI
# -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--languages", required=True, help="languages_template.csv")
    ap.add_argument("--features", required=True, help="lingua_features.csv (with region column)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tau", type=float, default=1000.0)
    ap.add_argument("--k", nargs="+", type=int, default=[3, 4, 5], help="candidate cluster sizes")
    ap.add_argument("--performance_files", nargs="*", default=[], help="paths to ct/zs mbert/xlmr/afroxlmr CSVs")
    ap.add_argument("--no_figures", action="store_true", help="skip figure generation")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    figs_dir = os.path.join(args.outdir, "figures")

    langs = read_languages(args.languages)
    feat = read_features(args.features)

    # 1) Canonical equal-weight baseline
    base_weights = dict(script=1, locale=1, geo=1)
    scores_equal = compute_scores(langs, feat, tau=args.tau, weights=base_weights)
    scores_equal.to_csv(os.path.join(args.outdir, "typology_similarity_scores.csv"), index=False)

    write_diagnostics(scores_equal, os.path.join(args.outdir, "diagnostics_equal.txt"),
                      title="Diagnostics — equal-weight typology similarity (macro-region locale)")

    k_best, sil_best, algo, df_equal_labeled, sil_table = cluster_and_write(
        scores_equal,
        out_csv=os.path.join(args.outdir, "scores_with_groups_equal.csv"),
        out_groups_csv=os.path.join(args.outdir, "groups_equal.csv"),
        ks=args.k
    )

    with open(os.path.join(args.outdir, "grouping_equal_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Algorithm: {algo}\n")
        f.write("Silhouette scores by k:\n")
        for k, sil in sil_table:
            f.write(f"  k={k}: silhouette={sil}\n")
        f.write(f"\nSelected k={k_best} with silhouette={sil_best}\n")

    # 2) Optional: learn weights from performance CSVs (feature-importance analysis)
    perf_all = read_performance_files(args.performance_files) if args.performance_files else pd.DataFrame(
        columns=["aux_code", "f1", "setup", "model"]
    )
    variant_results = {}

    if not perf_all.empty:
        perf_all = perf_all.copy()
        perf_all["variant"] = perf_all.apply(
            lambda r: f"{'cotrain' if r['setup'] == 'co-train' else 'zeroshot'}_{r['model']}",
            axis=1
        )
        
        compute_and_write_correlations_metadata_vs_f1(
            df_scores_equal=scores_equal,
            perf_all=perf_all,
            out_csv=os.path.join(args.outdir, "correlations_metadata_vs_f1.csv")
        )

        for (setup, model), df_subset in perf_all.groupby(["setup", "model"]):
            weights, diag = learn_weights_for_variant(scores_equal, df_subset)
            tag = f"{setup.replace('-', '')}_{model}".lower()

            with open(os.path.join(args.outdir, f"weight_learning_{tag}.txt"), "w", encoding="utf-8") as f:
                if weights is None:
                    f.write("Learned weights: <none>\n")
                    f.write(diag + "\n")
                    continue
                order = ["script", "locale", "geo"]
                f.write("Learned weights (script, locale, geo): " +
                        ", ".join(f"{k}={weights[k]:.4f}" for k in order) + "\n")
                f.write(diag + "\n")

            scores_learned = compute_scores(langs, feat, tau=args.tau, weights=weights)
            scores_learned.to_csv(os.path.join(args.outdir, f"typology_similarity_scores_learned_{tag}.csv"), index=False)

            k_best_l, sil_best_l, algo_l, df_learned_labeled, sil_table_l = cluster_and_write(
                scores_learned,
                out_csv=os.path.join(args.outdir, f"scores_with_groups_learned_{tag}.csv"),
                out_groups_csv=os.path.join(args.outdir, f"groups_{tag}.csv"),
                ks=args.k
            )

            with open(os.path.join(args.outdir, f"grouping_learned_report_{tag}.txt"), "w", encoding="utf-8") as f:
                f.write(f"Algorithm: {algo_l}\n")
                f.write("Silhouette scores by k:\n")
                for k, sil in sil_table_l:
                    f.write(f"  k={k}: silhouette={sil}\n")
                f.write(f"\nSelected k={k_best_l} with silhouette={sil_best_l}\n")

            variant_results[tag] = {"df": df_learned_labeled, "weights": weights}

    # 3) Figures
    if not args.no_figures:
        if not HAS_MPL:
            print("[warn] matplotlib not available; skipping figures")
        else:
            ensure_dir(figs_dir)

            df_equal_plot = merge_names(df_equal_labeled.sort_values("S_total", ascending=False), langs)

            plot_bar_ranking(df_equal_plot, os.path.join(figs_dir, "fig1_bar_equal"))
            plot_feature_heatmap(df_equal_plot, os.path.join(figs_dir, "fig2_heatmap_features_equal"))
            plot_pca_groups(df_equal_plot, os.path.join(figs_dir, "fig5_pca_groups_equal"),
                            "Group structure (equal weights)")
            plot_group_sizes(df_equal_plot, os.path.join(figs_dir, "fig7_group_sizes_equal"),
                             "Cluster sizes (equal weights)")

            summary = df_equal_plot[["aux_code", "aux_name", "S_total", "group"]].copy()
            if not perf_all.empty:
                piv = perf_all.pivot_table(index="aux_code", columns="variant", values="f1", aggfunc="mean").reset_index()
                summary = summary.merge(piv, on="aux_code", how="left")
            summary.to_csv(os.path.join(figs_dir, "tables_summary.csv"), index=False)

            if not perf_all.empty:
                for tag in sorted(perf_all["variant"].unique().tolist()):
                    plot_s_total_vs_f1(df_equal_plot, perf_all, tag,
                                       os.path.join(figs_dir, f"fig3_scatter_s_total_vs_f1_{tag}"))

            for tag, pack in variant_results.items():
                w = pack.get("weights")
                dfv = merge_names(pack["df"].sort_values("S_total", ascending=False), langs)

                if w is not None:
                    plot_weights_bar(w, os.path.join(figs_dir, f"fig4_weights_{tag}"),
                                     f"Learned weights — {tag.replace('_', ' / ')}")

                plot_pca_groups(dfv, os.path.join(figs_dir, f"fig6_pca_groups_{tag}"),
                                f"Group structure — {tag.replace('_', ' / ')}")
                plot_group_sizes(dfv, os.path.join(figs_dir, f"fig7_group_sizes_{tag}"),
                                 f"Cluster sizes — {tag.replace('_', ' / ')}")

            write_captions_stub(os.path.join(figs_dir, "thesis_figure_captions.md"))
            print(f"[viz] wrote figures & tables to: {figs_dir}")

if __name__ == "__main__":
    main()
