"""
Configuration clustering from *set overlap* (Jaccard per dimension), not counts.

Dimensions (set-overlap / Jaccard; Data Your Way and Unmapped 3PA are excluded):
  - oem_codes: BOB manufacturer_stmt tokens (union across cnubers)
  - vendors_3pa_internal | external: distinct VENDOR from 3PA usage + key TYPE; internal
    also unions Titan DataLake COMPONENT integrations (suffix after pip./dsapi., prefixed
    Titan Data Lake - ; external excludes 3PA_PRODUCTS)
  - enterprise_setup: Enterprise_Setup_Map ``Bucket`` (typically one org-setup label per enterprise)
  - spooler_types: MICR | ADP_CDK | OTHER buckets (UNSET excluded)
  - formnames: canonical form keys (trailing numeric suffix stripped)
  - profile_tokens: canonical user profile keys (comma-split, trailing numeric suffix stripped like forms)

The dataframe also keeps ``Bucket`` as plain text for exports (same values as enterprise_setup).

Run on project data (repo root — either form works):
  python -m segmentation.jaccard_similarity_clustering --algorithm hierarchical --n-clusters 6
  python segmentation/jaccard_similarity_clustering.py --algorithm hierarchical --n-clusters 6

Weighted Jaccard (equal default — one weight per dimension, 7 values):
  python -m segmentation.jaccard_similarity_clustering --weights 2,1,1,1,1,1,1

Cosine on concatenated multi-hot (bonus):
  python -m segmentation.jaccard_similarity_clustering --similarity cosine

Synthetic demo only:
  python -m segmentation.jaccard_similarity_clustering --demo
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Support running this file directly (`python segmentation/jaccard_...py`): put repo root on path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

# Reuse encoding + parsing helpers from the main pipeline
from segmentation.pipeline import (
    ROOT,
    csv_text_encoding,
    enterprise_setup_frozenset,
    load_edges,
    load_spine,
    oem_tokens,
    rollup_form_canonical_sets,
    rollup_spooler_bucket_sets,
    split_profile_tokens,
    three_pa_vendor_frozensets_by_type,
)

# Column order for weights / similarity aggregation (must stay consistent)
DIMENSION_KEYS: list[str] = [
    "oem_codes",
    "vendors_3pa_internal",
    "vendors_3pa_external",
    "enterprise_setup",
    "spooler_types",
    "formnames",
    "profile_tokens",
]


# ---------------------------------------------------------------------------
# Jaccard helpers
# ---------------------------------------------------------------------------


def jaccard_similarity(a: frozenset[str], b: frozenset[str]) -> float:
    """
    Jaccard = |A ∩ B| / |A ∪ B|.
    - Both empty: 1.0 (identical absence of codes).
    - Exactly one empty: 0.0 (no overlap possible).
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    uni = len(a | b)
    return float(inter) / float(uni) if uni else 1.0


def pairwise_jaccard_matrix(sets: list[frozenset[str]]) -> np.ndarray:
    """Full n×n symmetric matrix of Jaccard similarities (explicit; fine for small n)."""
    n = len(sets)
    m = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            s = jaccard_similarity(sets[i], sets[j])
            m[i, j] = m[j, i] = s
    return m


def _binary_indicator_matrix(df_sets: pd.DataFrame, col: str) -> sparse.csr_matrix:
    """Sparse rows = enterprises, cols = distinct codes (memory-safe for large vocabularies)."""
    vocab = sorted({x for s in df_sets[col] for x in s})
    n = len(df_sets)
    if not vocab:
        return sparse.csr_matrix((n, 0), dtype=np.float64)
    idx = {v: i for i, v in enumerate(vocab)}
    rows: list[int] = []
    cols: list[int] = []
    for r, (_, row) in enumerate(df_sets.iterrows()):
        for x in row[col]:
            if x in idx:
                rows.append(r)
                cols.append(idx[x])
    data = np.ones(len(rows), dtype=np.float64)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, len(vocab)), dtype=np.float64)


def jaccard_similarity_matrix_from_binary(X: sparse.csr_matrix | np.ndarray) -> np.ndarray:
    """
    Jaccard for rows interpreted as sets: J_ij = |row_i ∧ row_j| / |row_i ∨ row_j|.
    Vectorized via S = X X', union = r_i + r_j - S_ij.
    Both-empty rows: similarity 1; one empty: 0.
    """
    n = X.shape[0]
    if X.shape[1] == 0:
        return np.ones((n, n), dtype=np.float64)
    if sparse.issparse(X):
        S = (X @ X.T).toarray()
        r = np.asarray(X.sum(axis=1)).ravel()
    else:
        S = (X @ X.T).astype(np.float64)
        r = X.sum(axis=1)
    U = r[:, None] + r[None, :] - S
    both_empty = (r[:, None] == 0) & (r[None, :] == 0)
    J = np.zeros((n, n), dtype=np.float64)
    pos = U > 0
    J[pos] = S[pos] / U[pos]
    J[both_empty] = 1.0
    return J


def pairwise_jaccard_matrix_for_column(df_sets: pd.DataFrame, col: str) -> np.ndarray:
    """Preferred for large n: builds binary matrix then vectorized Jaccard."""
    X = _binary_indicator_matrix(df_sets, col)
    return jaccard_similarity_matrix_from_binary(X)


def weighted_average_similarity(
    per_dim_matrices: list[np.ndarray],
    weights: np.ndarray,
) -> np.ndarray:
    """
    Combined similarity = sum_d w_d * sim_d / sum(w_d).
    `weights` length must match len(per_dim_matrices).
    """
    w = np.asarray(weights, dtype=np.float64)
    if w.shape[0] != len(per_dim_matrices):
        raise ValueError("weights length must match number of dimensions")
    w_sum = w.sum()
    if w_sum <= 0:
        raise ValueError("sum of weights must be positive")
    acc = np.zeros_like(per_dim_matrices[0], dtype=np.float64)
    for mat, wd in zip(per_dim_matrices, w, strict=True):
        acc += wd * mat
    return acc / w_sum


def similarity_to_distance(sim: np.ndarray, clip: bool = True) -> np.ndarray:
    """Distance = 1 - similarity (optionally clip to [0, 1])."""
    d = 1.0 - sim
    if clip:
        np.clip(d, 0.0, 1.0, out=d)
    return d


# ---------------------------------------------------------------------------
# Load enterprise × dimension as frozensets (same roll-up rules as pipeline)
# ---------------------------------------------------------------------------


def _bobs_oem_sets(edges: pd.DataFrame, encoding: str) -> pd.Series:
    bobs = pd.read_csv(
        ROOT / "bobs_account_stuff" / "bobs_account_stuff.csv",
        encoding=encoding,
        usecols=["cnumber", "manufacturer_stmt"],
        low_memory=False,
    )
    bobs["cnumber"] = bobs["cnumber"].astype(str).str.strip().str.upper()
    ex = bobs.assign(_oem=bobs["manufacturer_stmt"].map(oem_tokens)).explode("_oem")
    ex = ex.dropna(subset=["_oem"])
    ex = ex[ex["_oem"].astype(str).str.len() > 0]
    d = ex.merge(edges, on="cnumber", how="inner")
    if len(d) == 0:
        return pd.Series(dtype=object)
    return d.groupby("Enterprise ID")["_oem"].agg(frozenset)


def _profile_sets(edges: pd.DataFrame, encoding: str) -> pd.Series:
    path = ROOT / "users_plus_profiles" / "users_plus_profiles.csv"
    kw = dict(chunksize=200_000, low_memory=False, encoding=encoding)
    chunks = []
    for ch in pd.read_csv(path, usecols=["cnumber", "profiles"], **kw):
        ch["cnumber"] = ch["cnumber"].astype(str).str.strip().str.upper()
        pe = ch.assign(_p=ch["profiles"].map(split_profile_tokens)).explode("_p")
        pe = pe.dropna(subset=["_p"])
        chunks.append(pe.merge(edges, on="cnumber", how="inner"))
    if not chunks:
        return pd.Series(dtype=object)
    long_p = pd.concat(chunks, ignore_index=True)
    return long_p.groupby("Enterprise ID")["_p"].agg(frozenset)


def build_enterprise_dimension_sets() -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Rows = Enterprise ID (with ≥1 mapped cnumber).
    Columns: Enterprise Name, Bucket (plain text), plus DIMENSION_KEYS (frozenset each).
    ``enterprise_setup`` is the frozenset form of ``Bucket`` (included in Jaccard).
    Returns (df, encodings_used).
    """
    edges = load_edges()
    ent_ok = set(edges["Enterprise ID"].unique())
    spine = load_spine(ent_ok).set_index("Enterprise ID")

    enc = {
        "bobs": csv_text_encoding(ROOT / "bobs_account_stuff" / "bobs_account_stuff.csv"),
        "forms": csv_text_encoding(ROOT / "form_data" / "form_data.csv"),
        "spooler": csv_text_encoding(ROOT / "spooler_data" / "spooler_data.csv"),
        "users": csv_text_encoding(ROOT / "users_plus_profiles" / "users_plus_profiles.csv"),
    }

    oem_s = _bobs_oem_sets(edges, enc["bobs"])
    v3 = three_pa_vendor_frozensets_by_type(edges)
    sp_s = rollup_spooler_bucket_sets(edges, enc["spooler"])
    fn_s = rollup_form_canonical_sets(edges, enc["forms"])
    prof_s = _profile_sets(edges, enc["users"])

    def fz_for(series: pd.Series, eid: str) -> frozenset[str]:
        if eid in series.index:
            return frozenset(series.loc[eid])
        return frozenset()

    rows = []
    for eid in spine.index:
        b_raw = spine.loc[eid, "Bucket"]
        bucket_str = "" if pd.isna(b_raw) else str(b_raw).strip()
        rows.append(
            {
                "Enterprise ID": eid,
                "Enterprise Name": spine.loc[eid, "Enterprise Name"],
                "Bucket": bucket_str,
                "oem_codes": fz_for(oem_s, eid),
                "vendors_3pa_internal": fz_for(v3["internal"], eid),
                "vendors_3pa_external": fz_for(v3["external"], eid),
                "enterprise_setup": enterprise_setup_frozenset(b_raw),
                "spooler_types": fz_for(sp_s, eid),
                "formnames": fz_for(fn_s, eid),
                "profile_tokens": fz_for(prof_s, eid),
            }
        )
    df = pd.DataFrame(rows).set_index("Enterprise ID")
    return df, enc


# ---------------------------------------------------------------------------
# Cosine similarity on block-wise multi-hot (bonus)
# ---------------------------------------------------------------------------


def multi_hot_cosine_similarity(df_sets: pd.DataFrame) -> np.ndarray:
    """
    For each dimension, build binary columns for each distinct code, concatenate
    blocks, then cosine similarity between rows (L2-normalized dot product).
    """
    blocks: list[sparse.csr_matrix] = []
    for col in DIMENSION_KEYS:
        m = _binary_indicator_matrix(df_sets, col)
        if m.shape[1] > 0:
            blocks.append(m)
    if not blocks:
        return np.eye(len(df_sets))
    X = sparse.hstack(blocks, format="csr")
    return cosine_similarity(X)


def multi_hot_cosine_similarity_weighted(
    df_sets: pd.DataFrame, weights: np.ndarray
) -> np.ndarray:
    """Same as multi_hot_cosine_similarity but scales each dimension block by sqrt(w_d / sum_w)."""
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    blocks: list[sparse.csr_matrix] = []
    for col, wd in zip(DIMENSION_KEYS, w, strict=True):
        m = _binary_indicator_matrix(df_sets, col)
        if m.shape[1] > 0:
            m = m * np.sqrt(wd)
            blocks.append(m)
    if not blocks:
        return np.eye(len(df_sets))
    X = sparse.hstack(blocks, format="csr")
    return cosine_similarity(X)


# ---------------------------------------------------------------------------
# Clustering + plots
# ---------------------------------------------------------------------------


def cluster_hierarchical_precomputed(
    distance: np.ndarray,
    n_clusters: int,
    linkage_method: str = "average",
) -> np.ndarray:
    """Agglomerative clustering with precomputed distances."""
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage=linkage_method,
    )
    return model.fit_predict(distance)


def cluster_dbscan_precomputed(
    distance: np.ndarray,
    eps: float,
    min_samples: int,
) -> np.ndarray:
    """DBSCAN with precomputed metric (distance matrix)."""
    model = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    return model.fit_predict(distance)


def plot_dendrogram(
    distance: np.ndarray,
    labels: list[str],
    linkage_method: str,
    out_path: Path,
    truncate_mode: str | None = "lastp",
    p: int = 50,
) -> None:
    """Dendrogram from condensed pairwise distances (scipy)."""
    # Ensure zero diagonal, symmetric
    d = np.asarray(distance, dtype=np.float64)
    condensed = squareform(d, checks=False)
    Z = linkage(condensed, method=linkage_method)
    plt.figure(figsize=(14, 8))
    dendrogram(
        Z,
        labels=labels,
        leaf_rotation=90.0,
        leaf_font_size=4,
        truncate_mode=truncate_mode,
        p=p,
    )
    plt.title(f"Hierarchical clustering ({linkage_method} linkage, distance = 1 - similarity)")
    plt.xlabel("Enterprise ID")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_2d_mds(distance: np.ndarray, labels: np.ndarray, out_path: Path, title: str) -> None:
    """Classical MDS (metric) stress on distance matrix — no extra deps."""
    from sklearn.manifold import MDS

    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=42,
        normalized_stress="auto",
        n_init=4,
        max_iter=300,
    )
    xy = mds.fit_transform(distance)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(xy[:, 0], xy[:, 1], c=labels, cmap="tab10", s=12, alpha=0.7)
    plt.colorbar(scatter, label="cluster")
    plt.title(title)
    plt.xlabel("MDS-1")
    plt.ylabel("MDS-2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_2d_umap_optional(
    distance: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    title: str,
) -> bool:
    try:
        import umap
    except ImportError:
        return False
    reducer = umap.UMAP(metric="precomputed", random_state=42, n_neighbors=15, min_dist=0.1)
    xy = reducer.fit_transform(distance)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(xy[:, 0], xy[:, 1], c=labels, cmap="tab10", s=12, alpha=0.7)
    plt.colorbar(scatter, label="cluster")
    plt.title(title + " (UMAP)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True


# ---------------------------------------------------------------------------
# Synthetic example (user table)
# ---------------------------------------------------------------------------


def make_example_dataframe() -> pd.DataFrame:
    """
    Small toy dataset matching the user's A/B example style.

      ID | D1        | D2       | D3
      A  | [a,b]     | [x]      | [k,m]
      B  | [a,c]     | [x,y]    | [k]
    """
    return pd.DataFrame(
        {
            "Enterprise ID": ["A", "B", "C"],
            "Enterprise Name": ["Alpha", "Beta", "Gamma"],
            "Bucket": ["t1", "t1", "t2"],
            "oem_codes": [
                frozenset({"a", "b"}),
                frozenset({"a", "c"}),
                frozenset({"a", "b"}),
            ],
            "vendors_3pa_internal": [frozenset({"x"}), frozenset(), frozenset({"x"})],
            "vendors_3pa_external": [frozenset(), frozenset({"y"}), frozenset()],
            "enterprise_setup": [
                frozenset({"T1"}),
                frozenset({"T1"}),
                frozenset({"T2"}),
            ],
            "spooler_types": [
                frozenset({"k", "m"}),
                frozenset({"k"}),
                frozenset({"k", "m"}),
            ],
            "formnames": [frozenset(), frozenset(), frozenset()],
            "profile_tokens": [frozenset(), frozenset(), frozenset()],
        }
    ).set_index("Enterprise ID")


def run_demo() -> None:
    print("=== Synthetic 3-enterprise example ===")
    df = make_example_dataframe()
    mats = []
    for col in DIMENSION_KEYS:
        mats.append(pairwise_jaccard_matrix_for_column(df, col))
    w = np.ones(len(DIMENSION_KEYS)) / len(DIMENSION_KEYS)
    sim = weighted_average_similarity(mats, w)
    print("Combined similarity matrix:\n", np.round(sim, 3))
    dist = similarity_to_distance(sim)
    print("Distance matrix:\n", np.round(dist, 3))
    lab = cluster_hierarchical_precomputed(dist, n_clusters=2, linkage_method="average")
    print("Hierarchical labels (k=2):", lab)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_weights(s: str, n: int) -> np.ndarray:
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) != n:
        raise argparse.ArgumentTypeError(f"expected {n} comma-separated weights, got {len(parts)}")
    return np.array(parts, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Jaccard / cosine configuration similarity clustering")
    parser.add_argument("--demo", action="store_true", help="Run toy example only")
    parser.add_argument(
        "--algorithm",
        choices=("hierarchical", "dbscan"),
        default="hierarchical",
    )
    parser.add_argument("--n-clusters", type=int, default=6, help="For hierarchical")
    parser.add_argument(
        "--linkage",
        default="average",
        help="For hierarchical + dendrogram (single|complete|average|ward not for precomputed)",
    )
    parser.add_argument("--eps", type=float, default=0.35, help="DBSCAN eps on distance")
    parser.add_argument("--min-samples", type=int, default=5, help="DBSCAN min_samples")
    parser.add_argument(
        "--weights",
        type=str,
        default=",".join(["1"] * len(DIMENSION_KEYS)),
        help=f"{len(DIMENSION_KEYS)} comma-separated weights, order: {DIMENSION_KEYS}",
    )
    parser.add_argument(
        "--similarity",
        choices=("jaccard", "cosine"),
        default="jaccard",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "charts" / "jaccard_clusters")
    parser.add_argument("--save-similarity", type=Path, default=None, help="Optional .npz path")
    parser.add_argument("--mds-2d", action="store_true", help="Save MDS scatter colored by cluster")
    parser.add_argument("--umap-2d", action="store_true", help="UMAP if umap-learn installed")
    parser.add_argument(
        "--no-dendrogram",
        action="store_true",
        help="Skip dendrogram (faster for very large n)",
    )
    args = parser.parse_args()

    if args.demo:
        run_demo()
        return

    weights = parse_weights(args.weights, len(DIMENSION_KEYS))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading enterprise × dimension sets…")
    df, enc = build_enterprise_dimension_sets()
    print("Encodings:", enc)
    n = len(df)
    ids = list(df.index)

    if args.similarity == "jaccard":
        per_dim = [
            pairwise_jaccard_matrix_for_column(df, col) for col in DIMENSION_KEYS
        ]
        sim = weighted_average_similarity(per_dim, weights)
    else:
        if np.allclose(weights, weights[0]):
            sim = multi_hot_cosine_similarity(df[DIMENSION_KEYS])
        else:
            sim = multi_hot_cosine_similarity_weighted(df[DIMENSION_KEYS], weights)

    dist = similarity_to_distance(sim)

    if args.save_similarity:
        np.savez_compressed(
            args.save_similarity,
            similarity=sim,
            distance=dist,
            enterprise_ids=np.array(ids, dtype=object),
        )
        print("Saved:", args.save_similarity)

    if args.algorithm == "hierarchical":
        labels = cluster_hierarchical_precomputed(dist, args.n_clusters, args.linkage)
        method_name = f"agglomerative_{args.linkage}_k{args.n_clusters}"
    else:
        labels = cluster_dbscan_precomputed(dist, eps=args.eps, min_samples=args.min_samples)
        method_name = f"dbscan_eps{args.eps}_min{args.min_samples}"

    out_csv = ROOT / "jaccard_cluster_assignments.csv"
    pd.DataFrame(
        {
            "Enterprise ID": ids,
            "Enterprise Name": [df.loc[eid, "Enterprise Name"] for eid in ids],
            "cluster": labels,
            "similarity_metric": args.similarity,
            "clustering_method": method_name,
        }
    ).to_csv(out_csv, index=False)
    print("Wrote:", out_csv)

    if not args.no_dendrogram:
        print("Plotting dendrogram (may truncate leaves for readability)…")
        plot_dendrogram(
            dist,
            ids,
            linkage_method=args.linkage,
            out_path=out_dir / "dendrogram.png",
            truncate_mode="lastp",
            p=min(50, max(30, n // 40)),
        )
        print("Wrote:", out_dir / "dendrogram.png")

    if args.mds_2d:
        plot_2d_mds(
            dist,
            labels,
            out_dir / "mds2d_clusters.png",
            title=f"MDS-2D ({method_name})",
        )
        print("Wrote:", out_dir / "mds2d_clusters.png")

    if args.umap_2d:
        if plot_2d_umap_optional(dist, labels, out_dir / "umap2d_clusters.png", method_name):
            print("Wrote:", out_dir / "umap2d_clusters.png")
        else:
            print("UMAP skipped (install umap-learn).")

    # Silhouette on distance-derived space is unconventional; skip or use precomputed kernel — omit.


if __name__ == "__main__":
    main()
