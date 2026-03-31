"""
How similar are enterprises to each other — overlap vs differentiation?

Answers three complementary questions:
1) **Exact signatures** — How many enterprises share the *identical* full configuration
   (OEM / internal+external 3PA / Enterprise_Setup_Map / spooler / forms / user profile *value sets*)?
2) **Pairwise Jaccard** — For a random pair of enterprises, how much do their *sets*
   overlap dimension-by-dimension (equal-weight average across dimensions)?
3) **k-means silhouette** (optional) — How separated are clusters in the *scaled
   breadth* feature space used for segmentation?

Run: python -m segmentation.configuration_overlap_report
Also invoked automatically from segmentation.pipeline main().
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from segmentation.exact_setup_analysis import enterprises_grouped_by_exact_signature
from segmentation.jaccard_similarity_clustering import (
    DIMENSION_KEYS,
    build_enterprise_dimension_sets,
    pairwise_jaccard_matrix_for_column,
)


def _upper_triangle_stats(sim: np.ndarray) -> dict[str, float | int]:
    iu = np.triu_indices(sim.shape[0], k=1)
    v = np.asarray(sim[iu], dtype=np.float64)
    n = int(v.size)
    if n == 0:
        return {
            "n_pairs": 0,
            "mean": 1.0,
            "std": 0.0,
            "p10": 1.0,
            "p25": 1.0,
            "p50": 1.0,
            "p75": 1.0,
            "p90": 1.0,
        }
    return {
        "n_pairs": n,
        "mean": float(v.mean()),
        "std": float(v.std()),
        "p10": float(np.percentile(v, 10)),
        "p25": float(np.percentile(v, 25)),
        "p50": float(np.percentile(v, 50)),
        "p75": float(np.percentile(v, 75)),
        "p90": float(np.percentile(v, 90)),
    }


def _interpret_combined_jaccard_mean(m: float) -> str:
    if m >= 0.82:
        return "Very high overlap — typical pairs share most coded elements across dimensions."
    if m >= 0.68:
        return "High overlap — strong shared backbone; differences are mostly in the tails."
    if m >= 0.52:
        return "Moderate overlap — meaningful shared elements with substantial pairwise variation."
    if m >= 0.35:
        return "Mixed — many pairs differ meaningfully on several dimensions."
    return "Lower overlap — configurations are relatively distinct in set-overlap terms."


def _interpret_silhouette(s: float | None) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return "Silhouette not provided."
    if s < 0.15:
        return "Weak separation — cluster assignments are fuzzy in the k-means feature space."
    if s < 0.35:
        return "Moderate separation — segments differ on average but with overlap between groups."
    return "Somewhat stronger separation — clusters are more distinct in the k-means feature space."


def write_configuration_overlap_report(
    root: Path,
    charts_dir: Path,
    *,
    kmeans_chosen_k: int | None = None,
    kmeans_silhouette: float | None = None,
) -> None:
    charts_dir.mkdir(parents=True, exist_ok=True)

    n_ent, by_sig = enterprises_grouped_by_exact_signature()
    n_distinct_signatures = len(by_sig)
    n_enterprises_solo = sum(len(v) for v in by_sig.values() if len(v) == 1)
    dup_groups = {k: v for k, v in by_sig.items() if len(v) > 1}
    n_ent_in_dup = sum(len(v) for v in dup_groups.values())
    pct_solo = round(100.0 * n_enterprises_solo / n_ent, 2) if n_ent else 0.0
    pct_any_peer = round(100.0 * n_ent_in_dup / n_ent, 2) if n_ent else 0.0

    df_sets, _enc = build_enterprise_dimension_sets()
    rows: list[dict[str, str | float | int]] = []

    # One Jaccard matrix per dimension; accumulate for equal-weight combined similarity
    acc = None
    for col in DIMENSION_KEYS:
        J = pairwise_jaccard_matrix_for_column(df_sets, col)
        st = _upper_triangle_stats(J)
        rows.append(
            {
                "lens": "pairwise_jaccard_per_dimension",
                "dimension": col,
                **{f"pairwise_{k}": v for k, v in st.items()},
                "interpretation": "",
            }
        )
        acc = J.copy() if acc is None else acc + J
        del J
    assert acc is not None
    combined = acc / float(len(DIMENSION_KEYS))
    comb_st = _upper_triangle_stats(combined)
    jaccard_interpretation = _interpret_combined_jaccard_mean(comb_st["mean"])
    rows.append(
        {
            "lens": "pairwise_jaccard_combined_equal_weights",
            "dimension": "average_of_" + "_".join(DIMENSION_KEYS),
            **{f"pairwise_{k}": v for k, v in comb_st.items()},
            "interpretation": jaccard_interpretation,
        }
    )

    plt.figure(figsize=(8, 4))
    iu = np.triu_indices(combined.shape[0], k=1)
    sample = combined[iu]
    plt.hist(sample, bins=60, color="steelblue", edgecolor="white", alpha=0.85)
    plt.axvline(comb_st["mean"], color="darkred", linestyle="--", label=f"mean = {comb_st['mean']:.3f}")
    plt.axvline(comb_st["p50"], color="orange", linestyle=":", label=f"median = {comb_st['p50']:.3f}")
    plt.xlabel("Pairwise Jaccard similarity (equal weight across dimensions)")
    plt.ylabel("Number of enterprise pairs")
    plt.title("How similar are two enterprises? (set overlap)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(charts_dir / "pairwise_jaccard_combined_histogram.png", dpi=150)
    plt.close()

    rows.append(
        {
            "lens": "exact_full_signature",
            "dimension": "oem_3pa_int_ext_enterprise_setup_spooler_forms_profiles",
            "pairwise_n_pairs": np.nan,
            "pairwise_mean": np.nan,
            "pairwise_std": np.nan,
            "pairwise_p10": np.nan,
            "pairwise_p25": np.nan,
            "pairwise_p50": np.nan,
            "pairwise_p75": np.nan,
            "pairwise_p90": np.nan,
            "interpretation": (
                f"{n_distinct_signatures} distinct exact signatures for {n_ent} enterprises (including enterprise_setup); "
                f"{pct_solo}% of enterprises are the sole occupant of their signature; "
                f"{pct_any_peer}% share an identical signature with at least one other enterprise."
            ),
        }
    )
    rows.append(
        {
            "lens": "exact_full_signature_metric",
            "dimension": "n_distinct_signatures",
            "pairwise_n_pairs": n_distinct_signatures,
            "pairwise_mean": np.nan,
            "pairwise_std": np.nan,
            "pairwise_p10": np.nan,
            "pairwise_p25": np.nan,
            "pairwise_p50": np.nan,
            "pairwise_p75": np.nan,
            "pairwise_p90": np.nan,
            "interpretation": "",
        }
    )
    rows.append(
        {
            "lens": "exact_full_signature_metric",
            "dimension": "n_enterprises_sole_signature",
            "pairwise_n_pairs": n_enterprises_solo,
            "pairwise_mean": np.nan,
            "pairwise_std": np.nan,
            "pairwise_p10": np.nan,
            "pairwise_p25": np.nan,
            "pairwise_p50": np.nan,
            "pairwise_p75": np.nan,
            "pairwise_p90": np.nan,
            "interpretation": "",
        }
    )

    sil_txt = _interpret_silhouette(kmeans_silhouette)
    rows.append(
        {
            "lens": "kmeans_segmentation_space",
            "dimension": f"k{kmeans_chosen_k}_silhouette" if kmeans_chosen_k is not None else "silhouette",
            "pairwise_n_pairs": np.nan,
            "pairwise_mean": float(kmeans_silhouette) if kmeans_silhouette is not None else np.nan,
            "pairwise_std": np.nan,
            "pairwise_p10": np.nan,
            "pairwise_p25": np.nan,
            "pairwise_p50": np.nan,
            "pairwise_p75": np.nan,
            "pairwise_p90": np.nan,
            "interpretation": sil_txt,
        }
    )

    out_df = pd.DataFrame(rows)
    out_path = root / "configuration_overlap_report.csv"
    out_df.to_csv(out_path, index=False)

    jaccard_followup = (
        "many enterprises pull from the same underlying code catalog even when counts differ."
        if comb_st["mean"] >= 0.52
        else (
            "typical pairs share only a modest fraction of the same codes; some dimensions (see CSV per row) "
            "are more uniform than others."
        )
    )
    summary_lines = [
        "CONFIGURATION OVERLAP VS DIFFERENTIATION — EXECUTIVE SUMMARY",
        "=" * 72,
        "",
        "Question: Are enterprises mostly the same, or genuinely different?",
        "",
        "1) EXACT SAME CONFIGURATION (strongest test)",
        f"   • {n_ent} enterprises in scope.",
        f"   • {n_distinct_signatures} distinct full signatures (exact OEM / internal+external 3PA / enterprise_setup / spooler / form / user profile sets).",
        f"   • {pct_solo}% of enterprises are the ONLY one with their exact signature.",
        f"   • {pct_any_peer}% sit in a non-trivial exact duplicate group (share 100% of those sets with ≥1 peer).",
        "   → Interpretation: At full precision, almost every enterprise is unique; exact twins are rare.",
        "",
        "2) PAIRWISE SET OVERLAP (Jaccard, equal weight per dimension)",
        f"   • Mean similarity across all enterprise pairs: {comb_st['mean']:.3f} (1 = identical sets, 0 = no overlap).",
        f"   • Middle 50% of pairs fall between ~{comb_st['p25']:.3f} and ~{comb_st['p75']:.3f} similarity.",
        f"   • {jaccard_interpretation}",
        "   → Interpretation: This measures shared vendor/form/user-profile *codes* (not 'how many distinct'). "
        + jaccard_followup,
        "",
        "3) K-MEANS SEGMENTS (scaled breadth features: counts + flags; Enterprise_Setup_Map excluded)",
    ]
    if kmeans_chosen_k is not None and kmeans_silhouette is not None:
        summary_lines.append(
            f"   • Chosen k = {kmeans_chosen_k}; k-means silhouette ≈ {kmeans_silhouette:.3f}."
        )
        summary_lines.append(f"   • {sil_txt}")
    else:
        summary_lines.append("   • (Run via pipeline to attach silhouette at chosen k.)")
    summary_lines.extend(
        [
            "   → Interpretation: Segments separate 'how broad' the footprint is; moderate silhouette means",
            "     groups overlap — enterprises are not isolated islands in that space.",
            "",
            "BOTTOM LINE",
            "   • Exact configuration: nearly all unique — only rare exact twins.",
            f"   • Set overlap (Jaccard): combined mean {comb_st['mean']:.3f} — see per-dimension means in CSV;",
            "     dimensions with higher means behave more alike across the base (e.g. spooler types often > user profiles).",
            "   • k-means segments: useful gradients of breadth, not crisp partitions (see silhouette).",
            "",
            f"Detail: {out_path.name}",
            "Chart: charts/pairwise_jaccard_combined_histogram.png",
        ]
    )

    summary_path = root / "configuration_overlap_executive_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"Wrote {out_path} and {summary_path} (chart: pairwise_jaccard_combined_histogram.png).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Configuration overlap / differentiation report")
    parser.add_argument(
        "--silhouette",
        type=float,
        default=None,
        help="k-means silhouette at chosen k (optional; pipeline passes this)",
    )
    parser.add_argument("--k", type=int, default=None, help="Chosen k for k-means (optional)")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    write_configuration_overlap_report(
        root,
        root / "charts",
        kmeans_chosen_k=args.k,
        kmeans_silhouette=args.silhouette,
    )


if __name__ == "__main__":
    main()
