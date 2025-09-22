# typology_pipeline.py
# Typology similarities (URIEL/lang2vec) + correlations with 4 F1 CSVs
#
# Example:
#   python typology_pipeline.py \
#     --langs_csv languages_masakha_run.csv \
#     --ct_f1_mbert ct_f1_mbert.csv \
#     --ct_f1_xlmr ct_f1_xlmr.csv \
#     --zs_f1_mbert zs_f1_mbert.csv \
#     --zs_f1_xlmr zs_f1_xlmr.csv \
#     --out_dir results --target nyn
#
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
from sklearn.metrics.pairwise import cosine_distances
from scipy.stats import spearmanr
import matplotlib.pyplot as plt


# ---------- Resolve a working "get_features" callable ----------
def resolve_lang2vec_get_features() -> Callable[[List[str], str], Dict[str, List[float]]]:
    """
    Return a callable get_features(langs, features) no matter how lang2vec is packaged.
    Tries:
      1) class-based API: Lang2Vec().get_features
      2) functional API: lang2vec.get_features
      3) functional API: lang2vec.lang2vec.get_features
    Prints diagnostics to help debugging.
    """
    # Try top-level module
    try:
        mod = import_module("lang2vec")
        print("Imported module 'lang2vec' from:", inspect.getfile(mod))
    except Exception as e:
        raise SystemExit(f"Failed to import 'lang2vec': {e}")

    # 1) Class-based API at top-level
    if hasattr(mod, "Lang2Vec"):
        try:
            l2v = getattr(mod, "Lang2Vec")()
            print("Using class-based API: lang2vec.Lang2Vec().get_features")
            return l2v.get_features
        except Exception as e:
            print("Found top-level Lang2Vec but failed to instantiate/use it:", repr(e))

    # 2) Functional API at top-level
    if hasattr(mod, "get_features"):
        gf = getattr(mod, "get_features")
        if callable(gf):
            print("Using functional API: lang2vec.get_features")
            return gf

    # 3) Submodule variants
    try:
        sub = import_module("lang2vec.lang2vec")
        print("Imported submodule 'lang2vec.lang2vec' from:", inspect.getfile(sub))
        # Class-based in submodule?
        if hasattr(sub, "Lang2Vec"):
            try:
                l2v = getattr(sub, "Lang2Vec")()
                print("Using class-based API: lang2vec.lang2vec.Lang2Vec().get_features")
                return l2v.get_features
            except Exception as e:
                print("Found submodule Lang2Vec but failed to instantiate/use it:", repr(e))
        # Functional in submodule?
        if hasattr(sub, "get_features") and callable(getattr(sub, "get_features")):
            print("Using functional API: lang2vec.lang2vec.get_features")
            return getattr(sub, "get_features")
    except Exception as e:
        print("Import 'lang2vec.lang2vec' failed:", repr(e))

    # Nothing worked
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


def get_uriel_vectors(
    codes: List[str],
    features: str = "syntax_knn+phonology_knn+inventory_knn",
    extra: Optional[List[str]] = None,
):
    vecs = GET_FEATURES(codes, features)  # dict[code] -> list[float]
    X = np.array([vecs[c] for c in codes], dtype=float)
    extra_cache = {}
    if extra:
        for name in extra:
            v = GET_FEATURES(codes, name)
            extra_cache[name] = np.array([v[c] for c in codes], dtype=float)
    return X, extra_cache


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


def kmeans_triplets_excl_target(
    codes: List[str],
    X: np.ndarray,
    sims: np.ndarray,
    target_code: str,
    k: int = 6,
    random_state: int = 0,
):
    idx = {c: i for i, c in enumerate(codes)}
    t = idx[target_code]
    mask = [i for i in range(len(codes)) if i != t]
    X_sub = X[mask]
    codes_sub = [codes[i] for i in mask]
    km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
    labels = km.fit_predict(X_sub)
    triplets = []
    for cl in range(k):
        members = [codes_sub[i] for i in range(len(codes_sub)) if labels[i] == cl]
        ranked = sorted(members, key=lambda c: sims[idx[c], t], reverse=True)
        triplets.append(ranked[:3])
    return triplets, labels, codes_sub


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
    xlabel: str = "Typology cosine similarity to target",
    ylabel: str = "F1",
):
    merged = pd.concat([sim_ser.rename("sim"), f1_ser.rename("f1")], axis=1).dropna()
    plt.figure()
    plt.scatter(merged["sim"].values, merged["f1"].values)
    for lang, row in merged.iterrows():
        plt.annotate(lang, (row["sim"], row["f1"]), fontsize=8, xytext=(3, 3), textcoords="offset points")
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
    merged_path = out_dir / f"merged_{label}.csv"
    merged.to_csv(merged_path)

    rho, p, n = spearman_corr(merged["sim"], merged["f1"])

    plot_scatter(
        sim_ser=merged["sim"],
        f1_ser=merged["f1"],
        title=f"{label}: F1 vs. typology similarity",
        out_path=out_dir / f"scatter_{label}.png",
    )

    with open(out_dir / f"spearman_{label}.txt", "w") as f:
        f.write(f"Spearman rho={rho:.4f}, p={p:.4g}, n={n}\n")

    return {"label": label, "rho": rho, "p": p, "n": n}


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs_csv", type=str, required=True, help="CSV with at least a 'code' column.")
    ap.add_argument("--ct_f1_mbert", type=str, required=True)
    ap.add_argument("--ct_f1_xlmr", type=str, required=True)
    ap.add_argument("--zs_f1_mbert", type=str, required=True)
    ap.add_argument("--zs_f1_xlmr", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="results")
    ap.add_argument("--target", type=str, default="nyn", help="Target ISO-639-3 (Runyankore = 'nyn').")
    ap.add_argument("--features", type=str, default="syntax_knn+phonology_knn+inventory_knn")
    ap.add_argument("--k_triplet", type=int, default=3)
    ap.add_argument("--k_clusters", type=int, default=6)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("LANG2VEC_DIR =", os.environ.get("LANG2VEC_DIR", "(not set)"))
    print("Python exe   =", sys.executable)

    # 1) Languages + URIEL vectors
    langs_df, codes = load_langs(args.langs_csv)
    if args.target not in codes:
        codes = [args.target] + [c for c in codes if c != args.target]

    X, _ = get_uriel_vectors(codes, features=args.features, extra=None)
    S, D = cosine_sim_matrix(X)

    # Save matrices
    pd.DataFrame(S, index=codes, columns=codes).to_csv(out / "cosine_similarity_typology.csv")
    pd.DataFrame(D, index=codes, columns=codes).to_csv(out / "cosine_distance_typology.csv")

    # 2) Ranking to target
    _, fullrank = nearest_neighbors_to_target(codes, S, args.target, k=args.k_triplet)
    rank_df = pd.DataFrame(fullrank, columns=["code", "sim_to_target"])
    rank_df.to_csv(out / "ranking_to_target.csv", index=False)

    # Helpful series of similarity to target (exclude target itself)
    idx = {c: i for i, c in enumerate(codes)}
    t = idx[args.target]
    sim_to_target = pd.Series({c: S[idx[c], t] for c in codes if c != args.target})
    sim_to_target.index.name = "language"

    # Quick bar plot of similarities
    sim_sorted = sim_to_target.sort_values(ascending=True)
    plt.figure()
    plt.barh(sim_sorted.index.values, sim_sorted.values)
    plt.xlabel(f"Cosine similarity to {args.target}")
    plt.ylabel("Language")
    plt.tight_layout()
    plt.savefig(out / "similarity_to_target_barh.png", dpi=180)
    plt.close()

    # 3) Cluster-based triplets (optional)
    triplets, labels, codes_sub = kmeans_triplets_excl_target(
        codes, X, S, args.target, k=args.k_clusters, random_state=0
    )
    with open(out / "cluster_triplets.json", "w") as f:
        json.dump(triplets, f, indent=2)

    # 4) Correlate with your 4 F1 files
    summaries = []
    pairs = [
        ("ct_f1_mbert", args.ct_f1_mbert, "Co-train mBERT"),
        ("ct_f1_xlmr", args.ct_f1_xlmr, "Co-train XLM-R"),
        ("zs_f1_mbert", args.zs_f1_mbert, "Zero-shot mBERT"),
        ("zs_f1_xlmr", args.zs_f1_xlmr, "Zero-shot XLM-R"),
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

    pd.DataFrame(summaries).to_csv(out / "spearman_summary.csv", index=False)
    print("Done. Results saved in:", str(out))


if __name__ == "__main__":
    main()
