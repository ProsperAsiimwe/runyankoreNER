#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Typology pipeline (URIEL-only, cosine similarity)

What’s included:
- URIEL/lang2vec cosine similarities to target (block-wise + concatenated ALL).
- Optional light priors: +genetic (family/genus), +geo (lat/lon RFF or country/region one-hot).
- Ablation correlations (per block + ALL) vs. four F1 CSVs.
- Simple learned block weights (per F1 file) -> weighted similarity and ranks.
- Divisibility-aware strata discovery (equal-size groups).

Example
=======
python typology_pipeline.py \
  --features syntax_knn+phonology_knn \
  --langs_csv languages_masakha_run.csv \
  --ct_f1_mbert ct_f1_mbert.csv \
  --ct_f1_xlmr ct_f1_xlmr.csv \
  --ct_f1_afroxlmr ct_f1_afroxlmr.csv \
  --zs_f1_mbert zs_f1_mbert.csv \
  --zs_f1_xlmr zs_f1_xlmr.csv \
  --zs_f1_afroxlmr zs_f1_afroxlmr.csv \
  --out_dir results_uriel --target nyn --desired_group_size 4 \
  --learn_block_weights
"""
import argparse
import json
import os
import sys
import inspect
from importlib import import_module
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from scipy.stats import spearmanr, zscore, pearsonr
import matplotlib.pyplot as plt


# ---------- Resolve a working "get_features" callable (URIEL/lang2vec) ----------
def resolve_lang2vec_get_features() -> Callable[[List[str], str], Dict[str, List[float]]]:
    try:
        mod = import_module("lang2vec")
        print("Imported module 'lang2vec' from:", inspect.getfile(mod))
    except Exception as e:
        raise SystemExit(f"Failed to import 'lang2vec': {e}")

    if hasattr(mod, "Lang2Vec"):
        try:
            l2v = getattr(mod, "Lang2Vec")()
            print("Using class-based API: lang2vec.Lang2Vec().get_features")
            return l2v.get_features
        except Exception as e:
            print("Found top-level Lang2Vec but failed to instantiate/use it:", repr(e))

    if hasattr(mod, "get_features") and callable(getattr(mod, "get_features")):
        print("Using functional API: lang2vec.get_features")
        return getattr(mod, "get_features")

    try:
        sub = import_module("lang2vec.lang2vec")
        print("Imported submodule 'lang2vec.lang2vec' from:", inspect.getfile(sub))
        if hasattr(sub, "Lang2Vec"):
            try:
                l2v = getattr(sub, "Lang2Vec")()
                print("Using class-based API: lang2vec.lang2vec.Lang2Vec().get_features")
                return l2v.get_features
            except Exception as e:
                print("Found submodule Lang2Vec but failed to instantiate/use it:", repr(e))
        if hasattr(sub, "get_features") and callable(getattr(sub, "get_features")):
            print("Using functional API: lang2vec.lang2vec.get_features")
            return getattr(sub, "get_features")
    except Exception as e:
        print("Import 'lang2vec.lang2vec' failed:", repr(e))

    msg = (
        "Could not obtain a working 'get_features' from lang2vec.\n"
        "Try a known-good build:\n"
        "  pip uninstall -y lang2vec\n"
        "  pip install --no-cache-dir git+https://github.com/antonisa/lang2vec.git@master\n"
        "Or set a local URIEL bundle via LANG2VEC_DIR.\n"
        f"sys.path[0]: {sys.path[0]}\n"
    )
    raise SystemExit(msg)


GET_FEATURES = resolve_lang2vec_get_features()


# ---------- Utilities ----------
def load_langs(csv_path: str) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(csv_path)
    if "code" not in df.columns:
        raise ValueError("languages CSV must contain a 'code' column.")
    codes = df["code"].astype(str).tolist()
    return df, codes


def cosine_sim_matrix(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    D = cosine_distances(X, X)
    S = 1.0 - D
    np.fill_diagonal(S, 1.0)
    return S, D


def nearest_neighbors_to_target(
    codes: List[str], sims: np.ndarray, target_code: str, k: int = 3
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    idx = {c: i for i, c in enumerate(codes)}
    if target_code not in idx:
        raise ValueError(f"Target code '{target_code}' not in language list.")
    t = idx[target_code]
    scores = [(codes[i], sims[i, t]) for i in range(len(codes)) if i != t]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k], scores  # (top-k, full)


def spearman_corr(x: pd.Series, y: pd.Series) -> Tuple[float, float, int]:
    df = pd.concat([x, y], axis=1).dropna()
    if df.shape[0] < 3:
        return float("nan"), float("nan"), df.shape[0]
    rho, p = spearmanr(df.iloc[:, 0].values, df.iloc[:, 1].values)
    return float(rho), float(p), int(df.shape[0])


def plot_scatter(
    sim_ser: pd.Series,
    f1_ser: pd.Series,
    title: str,
    out_path: Path,
    xlabel: str = "Similarity to target (higher = closer)",
    ylabel: str = "F1",
):
    merged = pd.concat([sim_ser.rename("sim"), f1_ser.rename("f1")], axis=1).dropna()

    x = merged["sim"].values
    y = merged["f1"].values

    plt.figure()
    plt.scatter(x, y)

    # Label each point with language code
    for lang, row in merged.iterrows():
        plt.annotate(
            lang,
            (row["sim"], row["f1"]),
            fontsize=8,
            xytext=(3, 3),
            textcoords="offset points",
        )

    # --- Fit and plot linear regression line (OLS) ---
    if len(x) >= 2:
        a, b = np.polyfit(x, y, deg=1)  # slope, intercept
        xs = np.linspace(x.min(), x.max(), 200)
        ys = a * xs + b
        plt.plot(xs, ys, linestyle="--", color="red", label=f"OLS fit (slope={a:.3f})")

    # --- Spearman correlation for title ---
    if len(x) >= 2:
        from scipy.stats import spearmanr
        rho, p = spearmanr(x, y)
        n = len(x)
        plt.title(f"{title}\nSpearman $\\rho={rho:.3f}$, $p={p:.4f}$, $n={n}$")
        plt.legend()
    else:
        plt.title(title)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()



def load_f1_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"language", "f1"}.issubset(df.columns):
        raise ValueError(f"{path} must have columns: language,f1")
    df["language"] = df["language"].astype(str)
    return df


def correlate_with_similarity(
    f1_df: pd.DataFrame,
    sim_to_target: pd.Series,
    label: str,
    out_dir: Path,
) -> Dict[str, float]:
    f1_ser = f1_df.set_index("language")["f1"]
    merged = pd.concat([sim_to_target.rename("sim"), f1_ser.rename("f1")], axis=1)
    (out_dir / f"merged_{label}.csv").write_text(merged.to_csv())
    merged = merged.dropna()
    n = merged.shape[0]

    if n < 3:
        rho = p_rho = r = p_r = float("nan")
    else:
        x = merged["sim"].values
        y = merged["f1"].values
        rho, p_rho = spearmanr(x, y)
        r, p_r = pearsonr(x, y)

    # scatter plot (can still mention Spearman in the title)
    plot_scatter(
        sim_ser=merged["sim"],
        f1_ser=merged["f1"],
        title=f"{label}: F1 vs. similarity",
        out_path=out_dir / f"scatter_{label}.png",
    )

    # log both correlations
    (out_dir / f"correlations_{label}.txt").write_text(
        f"Spearman rho={rho:.4f}, p={p_rho:.4g}, n={n}\n"
        f"Pearson r={r:.4f}, p={p_r:.4g}, n={n}\n"
    )

    return {
        "label": label,
        "rho": float(rho),
        "p_rho": float(p_rho),
        "r": float(r),
        "p_r": float(p_r),
        "n": int(n),
    }



# ---------- Build URIEL block vectors ----------
def get_uriel_block_vectors(codes: List[str], features_str: str) -> Dict[str, np.ndarray]:
    """
    Returns a dict: block_name -> X (len(codes) x d_block)
    Supports pseudo-blocks 'genetic' and 'geo' (from langs_csv; injected later).
    """
    blocks = [b for b in features_str.split("+") if b.strip()]
    out: Dict[str, np.ndarray] = {}
    for b in blocks:
        if b in {"genetic", "geo"}:
            # handled later via local priors
            continue
        vecs = GET_FEATURES(codes, b)  # dict[code]->list[float]
        Xb = np.array([vecs[c] for c in codes], dtype=float)
        out[b] = Xb
    return out


# ---------- Optional priors from langs_csv ----------
def build_genetic_block(langs_df: pd.DataFrame, codes: List[str]) -> Optional[np.ndarray]:
    # Uses one-hot for family + genus if columns present
    cols = [c for c in ["family", "genus"] if c in langs_df.columns]
    if not cols:
        return None
    sub = langs_df.set_index("code").loc[codes, cols].fillna("UNK").astype(str)
    oh = pd.get_dummies(sub, prefix=cols)
    return oh.values.astype(float)


def build_geo_block(langs_df: pd.DataFrame, codes: List[str]) -> Optional[np.ndarray]:
    # Tries lat/lon RBF features if available; else one-hot for country/region.
    if {"lat", "lon"}.issubset(langs_df.columns):
        sub = langs_df.set_index("code").loc[codes, ["lat", "lon"]].astype(float)
        X = sub.values
        # Random Fourier Features for an RBF kernel (small D to avoid overfit)
        rng = np.random.default_rng(0)
        D = 32
        gamma = 1.0 / (np.median(np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)) + 1e-6)
        W = rng.normal(0.0, np.sqrt(2 * gamma), size=(X.shape[1], D))
        b = rng.uniform(0, 2 * np.pi, size=(D,))
        Z = np.sqrt(2.0 / D) * np.cos(X @ W + b)
        return Z.astype(float)
    for col in ["country", "region"]:
        if col in langs_df.columns:
            sub = langs_df.set_index("code").loc[codes, [col]].fillna("UNK").astype(str)
            oh = pd.get_dummies(sub, prefix=[col])
            return oh.values.astype(float)
    return None


# ---------- Helpers for similarities ----------
def sim_series_from_matrix(codes: List[str], S: np.ndarray, target: str) -> pd.Series:
    idx = {c: i for i, c in enumerate(codes)}
    t = idx[target]
    sim = pd.Series({c: float(S[idx[c], t]) for c in codes if c != target})
    sim.index.name = "language"
    sim.name = "sim"
    return sim


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    # I/O + task
    ap.add_argument("--langs_csv", type=str, required=True, help="CSV with at least a 'code' column.")
    ap.add_argument("--ct_f1_mbert", type=str, required=True)
    ap.add_argument("--ct_f1_xlmr", type=str, required=True)
    ap.add_argument("--ct_f1_afroxlmr", type=str, required=True)
    ap.add_argument("--zs_f1_mbert", type=str, required=True)
    ap.add_argument("--zs_f1_xlmr", type=str, required=True)
    ap.add_argument("--zs_f1_afroxlmr", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="results")
    ap.add_argument("--target", type=str, default="nyn", help="Target code (e.g., 'nyn' or 'run').")

    # URIEL features (+ optional priors)
    ap.add_argument("--features", type=str,
                    default="syntax_knn+phonology_knn",
                    help="URIEL blocks joined by '+'. Supports 'genetic' and 'geo' as priors.")

    # Exploratory clusters (optional; not used in final experiments)
    ap.add_argument("--export_triplets", action="store_true",
                    help="Also export cluster_triplets.json (exploratory).")
    ap.add_argument("--k_clusters", type=int, default=6)

    # Dynamic grouping
    ap.add_argument("--desired_group_size", type=int, default=4,
                    help="Equal group size per stratum (e.g., 4 for quartets).")

    # Ablations / learned weights
    ap.add_argument("--learn_block_weights", action="store_true",
                    help="Fit simple weights per F1 file to combine blocks.")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("LANG2VEC_DIR =", os.environ.get("LANG2VEC_DIR", "(not set)"))
    print("Python exe   =", sys.executable)

    # Load language list and keep target at index 0
    langs_df, codes = load_langs(args.langs_csv)
    if args.target not in codes:
        codes = [args.target] + [c for c in codes if c != args.target]

    # ---------------- Build URIEL/prior blocks ----------------
    uriel_blocks = get_uriel_block_vectors(codes, args.features)

    feat_names = [b for b in args.features.split("+") if b.strip()]
    if "genetic" in feat_names:
        G = build_genetic_block(langs_df, codes)
        if G is not None:
            uriel_blocks["genetic"] = G
            print("Added 'genetic' prior block from langs_csv.")
        else:
            print("[warn] 'genetic' block requested but no family/genus columns found; skipping.")
    if "geo" in feat_names:
        Geo = build_geo_block(langs_df, codes)
        if Geo is not None:
            uriel_blocks["geo"] = Geo
            print("Added 'geo' prior block from langs_csv.")
        else:
            print("[warn] 'geo' block requested but no suitable geo columns found; skipping.")

    if not uriel_blocks:
        raise SystemExit("No usable URIEL/prior blocks were constructed. Check --features and data.")

    # ---------------- Per-block similarities (cosine) ----------------
    block_sims: Dict[str, pd.Series] = {}
    for b, Xb in uriel_blocks.items():
        # Safety checks
        if not np.isfinite(Xb).all():
            raise SystemExit(f"Non-finite values found in block '{b}'")
        if np.linalg.norm(Xb, axis=1).min() == 0.0:
            raise SystemExit(f"Zero-norm vectors found in block '{b}'")
        Sb = cosine_similarity(Xb, Xb)
        block_sims[b] = sim_series_from_matrix(codes, Sb, args.target)

    # Save blockwise similarities and rank lists
    pd.DataFrame(block_sims).to_csv(out / "blockwise_sim_to_target.csv")
    rank_lists = {b: block_sims[b].sort_values(ascending=False).index.tolist() for b in block_sims}
    pd.DataFrame(rank_lists).to_csv(out / "blockwise_rank_lists.csv", index=False)

    # ---------------- Concatenated ALL (original behavior) ----------------
    X_all = np.concatenate([uriel_blocks[b] for b in uriel_blocks], axis=1)
    S_all, D_all = cosine_sim_matrix(X_all)
    pd.DataFrame(S_all, index=codes, columns=codes).to_csv(out / "cosine_similarity_typology.csv")
    pd.DataFrame(D_all, index=codes, columns=codes).to_csv(out / "cosine_distance_typology.csv")
    sim_to_target = sim_series_from_matrix(codes, S_all, args.target)

    # Quick bar plot
    sim_sorted = sim_to_target.sort_values(ascending=True)
    plt.figure()
    plt.barh(sim_sorted.index.values, sim_sorted.values)
    plt.xlabel(f"Cosine similarity to {args.target} (ALL blocks)")
    plt.ylabel("Language")
    plt.tight_layout()
    plt.savefig(out / "similarity_to_target_barh.png", dpi=180)
    plt.close()

    # ---------------- (Optional) Cluster-based triplets ----------------
    if args.export_triplets:
        km = KMeans(n_clusters=args.k_clusters, n_init=20, random_state=0)
        labels = km.fit_predict(X_all[[i for i in range(len(codes)) if codes[i] != args.target]])
        codes_sub = [c for c in codes if c != args.target]
        idx = {c: i for i, c in enumerate(codes)}
        t = idx[args.target]
        triplets = []
        for cl in range(args.k_clusters):
            members = [codes_sub[i] for i in range(len(codes_sub)) if labels[i] == cl]
            ranked = sorted(members, key=lambda c: S_all[idx[c], t], reverse=True)
            triplets.append(ranked[:3])
        (out / "cluster_triplets.json").write_text(json.dumps(triplets, indent=2))
        print("Wrote exploratory cluster_triplets.json")

    # ---------------- Dynamic strata with divisibility-aware cuts ----------------
    ranked = sim_to_target.sort_values(ascending=False)
    ranked_codes = list(ranked.index)

    def discover_divisible_strata(
        ranked_codes: List[str],
        ranked_sims: pd.Series,
        desired_group_size: int = 4,
    ) -> Tuple[List[List[str]], Dict]:
        n = len(ranked_codes)
        if n < 3 * desired_group_size:
            raise SystemExit(
                f"Not enough languages ({n}) to form 3 strata with group size {desired_group_size}."
            )
        sims = [float(ranked_sims[c]) for c in ranked_codes]
        deltas = [sims[i] - sims[i + 1] for i in range(n - 1)]
        order = sorted(range(len(deltas)), key=lambda i: (-deltas[i], i))
        rank_of = {i: r for r, i in enumerate(order)}

        best = None
        best_strata = None
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                s1, s2, s3 = i + 1, j - i, n - (j + 1)
                if s1 <= 0 or s2 <= 0 or s3 <= 0:
                    continue
                if (s1 % desired_group_size) or (s2 % desired_group_size) or (s3 % desired_group_size):
                    continue
                d1, d2 = deltas[i], deltas[j]
                score = (d1 + d2, min(d1, d2), -(i + j), -i)
                if (best is None) or (score > best[0]):
                    best = (score, (i, j))
                    strata = [
                        ranked_codes[: i + 1],
                        ranked_codes[i + 1 : j + 1],
                        ranked_codes[j + 1 :],
                    ]
                    best_strata = strata
        if best_strata is None:
            i0, i1 = sorted(order[:2])
            strata = [
                ranked_codes[: i0 + 1],
                ranked_codes[i0 + 1 : i1 + 1],
                ranked_codes[i1 + 1 :],
            ]
            meta = {
                "mode": "fallback_no_divisible_pair",
                "cuts": [i0, i1],
                "stratum_sizes": [len(s) for s in strata],
                "desired_group_size": args.desired_group_size,
                "note": "Strata sizes not divisible; groups will produce reserves.",
            }
            return strata, meta
        (i, j) = best[1]
        meta = {
            "mode": "divisible_pair",
            "cuts": [i, j],
            "stratum_sizes": [len(s) for s in best_strata],
            "desired_group_size": args.desired_group_size,
            "gap_values": [deltas[i], deltas[j]],
            "gap_ranks": [rank_of[i], rank_of[j]],
        }
        return best_strata, meta

    def chunk_equal_groups(stratum: List[str], g: int) -> Tuple[List[List[str]], List[str]]:
        k = len(stratum) // g
        groups = [stratum[i * g : (i + 1) * g] for i in range(k)]
        reserve = stratum[k * g :]
        return groups, reserve

    def label_groups(strata_groups: List[List[List[str]]]) -> Dict[str, List[str]]:
        names = ["High", "Intermediate", "Low"]
        out: Dict[str, List[str]] = {}
        for base, groups in zip(names, strata_groups):
            for j, g in enumerate(groups):
                suffix = chr(ord("A") + j)
                key = f"{base}-{suffix}" if len(groups) > 1 else base
                out[key] = g
        return out

    strata, meta = discover_divisible_strata(
        ranked_codes=ranked_codes,
        ranked_sims=ranked,
        desired_group_size=args.desired_group_size,
    )
    stratum_groups: List[List[List[str]]] = []
    reserves: List[List[str]] = []
    for s in strata:
        groups, reserve = chunk_equal_groups(s, args.desired_group_size)
        stratum_groups.append(groups)
        reserves.append(reserve)
    named_groups = label_groups(stratum_groups)

    # Compute stats (min/max/mean similarity per group)
    def gstats(members: List[str]) -> Dict[str, float]:
        vals = [float(ranked[c]) for c in members]
        return {
            "min": float(np.min(vals)) if vals else float("nan"),
            "max": float(np.max(vals)) if vals else float("nan"),
            "mean": float(np.mean(vals)) if vals else float("nan"),
        }

    group_rows = []
    for name, members in named_groups.items():
        st = gstats(members)
        group_rows.append({
            "group": name,
            "members": ",".join(members),
            "min_sim": st["min"],
            "max_sim": st["max"],
            "mean_sim": st["mean"],
        })
    (out / "strata_groups_dynamic.json").write_text(json.dumps(named_groups, indent=2))
    pd.DataFrame(group_rows).to_csv(out / "strata_groups_dynamic.csv", index=False)

    meta_out = {
        "target": args.target,
        "desired_group_size": args.desired_group_size,
        **meta,
    }
    (out / "strata_groups_meta.json").write_text(json.dumps(meta_out, indent=2))
    reserves_dict = {"High_reserve": reserves[0], "Intermediate_reserve": reserves[1], "Low_reserve": reserves[2]}
    (out / "strata_reserve.json").write_text(json.dumps(reserves_dict, indent=2))

    print(f"Wrote strata_groups_dynamic.json/csv with desired_group_size={args.desired_group_size}")
    print(f"Stratum sizes: {meta['stratum_sizes']} (mode={meta['mode']})")
    if any(len(r) > 0 for r in reserves):
        print(f"[Note] Some strata produced reserves: see {out/'strata_reserve.json'}")

    # ---------------- Correlate with my 4 F1 files ----------------
    summaries = []
    pairs = [
        ("ct_f1_mbert", args.ct_f1_mbert, "Co-train mBERT"),
        ("ct_f1_xlmr", args.ct_f1_xlmr, "Co-train XLM-R"),
        ("ct_f1_afroxlmr", args.ct_f1_afroxlmr, "Co-train AFRO-XLMR"),
        ("zs_f1_mbert", args.zs_f1_mbert, "Zero-shot mBERT"),
        ("zs_f1_xlmr", args.zs_f1_xlmr, "Zero-shot XLM-R"),
        ("zs_f1_afroxlmr", args.zs_f1_afroxlmr, "Zero-shot AFRO-XLMR"),
    ]
    for _, path, label in pairs:
        f1_df = load_f1_csv(path)
        summ = correlate_with_similarity(
            f1_df=f1_df,
            sim_to_target=sim_to_target,
            label=label.replace(" ", "_").lower(),
            out_dir=out,
        )
        summaries.append(summ)
    pd.DataFrame(summaries).to_csv(out / "correlation_summary.csv", index=False)

    # ---------------- Ablations + learned weights (URIEL blocks) ----------------
    ab_summ_rows = []
    for b, sim_b in block_sims.items():
        for _, path, label in pairs:
            f1_df = load_f1_csv(path)
            r = correlate_with_similarity(
                f1_df=f1_df,
                sim_to_target=sim_b,
                label=f"{label.replace(' ','_').lower()}_block_{b}",
                out_dir=out,
            )
            ab_summ_rows.append({"block": b, **r})
    # ALL already computed above as sim_to_target
    for _, path, label in pairs:
        f1_df = load_f1_csv(path)
        r = correlate_with_similarity(
            f1_df=f1_df,
            sim_to_target=sim_to_target,
            label=f"{label.replace(' ','_').lower()}_block_ALL",
            out_dir=out,
        )
        ab_summ_rows.append({"block": "ALL", **r})
    pd.DataFrame(ab_summ_rows).to_csv(out / "spearman_summary_per_block.csv", index=False)

    if args.learn_block_weights and len(block_sims) >= 2:
        weights_report = {}
        Zblocks = pd.DataFrame(block_sims)  # langs x blocks
        Zblocks = Zblocks.apply(lambda s: pd.Series(zscore(s.dropna()), index=s.dropna().index)) \
                         .reindex(Zblocks.index)  # keep alignment

        for _, path, label in pairs:
            f1_ser = load_f1_csv(path).set_index("language")["f1"]
            A = pd.concat([Zblocks, f1_ser.rename("f1")], axis=1).dropna()
            if A.shape[0] < 5:
                continue
            X = A[Zblocks.columns].values
            y = A["f1"].values
            w, *_ = np.linalg.lstsq(X, y, rcond=None)  # unconstrained
            w = np.clip(w, 0, None)                   # non-negative clamp
            if w.sum() == 0:
                w = np.ones_like(w) / len(w)
            else:
                w = w / w.sum()
            weights = dict(zip(Zblocks.columns.tolist(), [float(x) for x in w]))
            weights_report[label] = weights

            # Weighted z-combo across blocks → final similarity
            Z_for_all = pd.DataFrame(block_sims).apply(
                lambda s: pd.Series(zscore(s.dropna()), index=s.dropna().index)
            )
            # Align indices and combine
            common_idx = Z_for_all.dropna().index
            sim_weighted = pd.Series(0.0, index=common_idx)
            for b, wb in weights.items():
                if b in Z_for_all.columns:
                    sim_weighted = sim_weighted.add(wb * Z_for_all.loc[common_idx, b], fill_value=0.0)
            sim_weighted.name = "sim"

            # Save ranking and correlation
            (out / f"weighted_rank_{label.replace(' ','_').lower()}.csv").write_text(
                sim_weighted.sort_values(ascending=False).to_csv(header=False)
            )
            corr = correlate_with_similarity(
                f1_df=load_f1_csv(path),
                sim_to_target=sim_weighted,
                label=f"{label.replace(' ','_').lower()}_weighted",
                out_dir=out,
            )
            corr["block"] = "LEARNED"
            ab_summ_rows.append(corr)

        (out / "learned_block_weights.json").write_text(json.dumps(weights_report, indent=2))
        pd.DataFrame(ab_summ_rows).to_csv(out / "spearman_summary_per_block.csv", index=False)

    print("Done. Results saved in:", str(out))


if __name__ == "__main__":
    main()
