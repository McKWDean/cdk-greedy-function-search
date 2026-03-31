"""
Accounting Segment 1 — cohort incremental coverage vs segment and population.

Reads ``20260326 Accounting Segment 1 Enterprise IDs.xlsx`` (sheet ``Accounting Segment 1``):
  - ``Enterprise ID``
  - ``Cohort`` (numeric cohort id; processed in sorted order 1, 2, 3, …)

Outputs (repo root):

1) ``accounting_segment1_cohort_incremental_by_dimension.csv``  
   Per Jaccard dimension:
   - Segment union vs **population** (% of all distinct values in the model that Segment 1 touches)
   - Cohort 1 union vs **segment union** (% of segment’s distinct values that Cohort 1 alone has)
   - For each later cohort: **incremental** new distinct values vs cumulative through prior cohorts,
     as **% of segment union** and counts; cumulative % of segment union after each cohort

2) ``accounting_segment1_cohort_incremental_tagged_union.csv``  
   Same logic on a **tagged** atom space ``dimension|value`` (no collisions across dimensions).

3) ``accounting_segment1_cohort_greedy_80pct_summary.csv``  
   Per cohort: greedy count to reach threshold vs **that cohort’s union** (all 9 dimensions).

4) ``accounting_segment1_cohort_greedy_80pct_picks.csv``  
   Pick order and per-step coverage for each cohort.

Also runs a **greedy cover within each cohort** (same 9 Jaccard dimensions, 80% default):
reference universe on dimension *d* = that cohort’s own union on *d*; pool = cohort enterprises
in the model. Reports how many customers (minimum greedy count) are needed vs cohort size.

Run: ``python -m segmentation.accounting_segment1_cohort_analysis``
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

from segmentation.enterprise_dimension_export import (
    _distinct_values_in_column,
    _sanitize_cell_text,
    _superset_for_enterprises,
)
from segmentation.accounting_segment1_analysis import (
    COVER_PCT_THRESHOLD,
    greedy_cover_to_threshold,
)
from segmentation.jaccard_similarity_clustering import (
    DIMENSION_KEYS,
    build_enterprise_dimension_sets,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_XLSX = ROOT / "20260326 Accounting Segment 1 Enterprise IDs.xlsx"
DEFAULT_SHEET = "Accounting Segment 1"

DIMENSION_LABELS: dict[str, str] = {
    "oem_codes": "OEMs",
    "vendors_3pa_internal": "Internal integrations",
    "vendors_3pa_external": "External integrations",
    "enterprise_setup": "Org setups (Enterprise_Setup_Map)",
    "spooler_types": "Peripherals / printers",
    "formnames": "Forms",
    "profile_tokens": "User profiles",
}


def _find_cohort_column(columns: list) -> str:
    for c in columns:
        if str(c).strip().lower() == "cohort":
            return str(c)
    raise ValueError(f"No Cohort column found; columns are {columns!r}")


def load_cohort_groups(
    path: Path = DEFAULT_XLSX,
    sheet: str = DEFAULT_SHEET,
) -> tuple[dict[int, list[str]], tuple[str, ...]]:
    """
    Returns (cohort_id -> list of Enterprise IDs in stable file order within cohort,
    all segment IDs in file order for union).
    """
    df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    if "Enterprise ID" not in df.columns:
        raise ValueError(f"Expected 'Enterprise ID'; got {list(df.columns)}")
    cohort_col = _find_cohort_column(list(df.columns))
    df = df.dropna(subset=["Enterprise ID", cohort_col])
    df["Enterprise ID"] = (
        df["Enterprise ID"].astype(str).map(lambda x: _sanitize_cell_text(x.strip()))
    )
    df = df[df["Enterprise ID"] != ""]
    df["_cohort"] = pd.to_numeric(df[cohort_col], errors="coerce")
    bad = df["_cohort"].isna()
    if bad.any():
        raise ValueError(f"Non-numeric Cohort rows: {df.loc[bad, cohort_col].tolist()[:10]!r} …")
    df["_cohort"] = df["_cohort"].astype(int)

    groups: dict[int, list[str]] = defaultdict(list)
    all_ids: list[str] = []
    for _, row in df.iterrows():
        eid = row["Enterprise ID"]
        cid = int(row["_cohort"])
        groups[cid].append(eid)
        all_ids.append(eid)

    ordered_cohorts = dict(sorted(groups.items(), key=lambda x: x[0]))
    return ordered_cohorts, tuple(all_ids)


def _tagged_set_for_enterprises(
    df_sets: pd.DataFrame,
    enterprise_ids: tuple[str, ...],
) -> set[str]:
    out: set[str] = set()
    for col in DIMENSION_KEYS:
        for t in _superset_for_enterprises(df_sets, col, enterprise_ids):
            out.add(f"{col}|{t}")
    return out


def tagged_population_universe(df_sets: pd.DataFrame) -> set[str]:
    out: set[str] = set()
    for eid in df_sets.index:
        eid = str(eid).strip()
        for col in DIMENSION_KEYS:
            for t in df_sets.loc[eid, col]:
                tok = _sanitize_cell_text(str(t).strip())
                if tok:
                    out.add(f"{col}|{tok}")
    return out


def build_per_dimension_rows(
    df_sets: pd.DataFrame,
    cohort_groups: dict[int, list[str]],
    segment_ids: tuple[str, ...],
) -> pd.DataFrame:
    cohort_ids = sorted(cohort_groups.keys())
    rows: list[dict] = []

    for col in DIMENSION_KEYS:
        pop = _distinct_values_in_column(df_sets, col)
        seg = _superset_for_enterprises(df_sets, col, segment_ids)
        n_pop, n_seg = len(pop), len(seg)
        pct_seg_of_pop = round(100.0 * n_seg / n_pop, 4) if n_pop else float("nan")

        row: dict = {
            "dimension": col,
            "dimension_label": DIMENSION_LABELS.get(col, col),
            "n_population_universe": n_pop,
            "n_segment_union": n_seg,
            "pct_segment_union_of_population": pct_seg_of_pop,
        }

        cumulative: set[str] = set()
        for i, cid in enumerate(cohort_ids):
            c_eids = tuple(cohort_groups[cid])
            c_union = _superset_for_enterprises(df_sets, col, c_eids)
            incremental = c_union - cumulative
            cumulative |= c_union

            row[f"n_cohort_{cid}_union"] = len(c_union)
            row[f"n_cohort_{cid}_incremental"] = len(incremental)
            row[f"pct_cohort_{cid}_incremental_of_segment_union"] = (
                round(100.0 * len(incremental) / n_seg, 4) if n_seg else float("nan")
            )
            if i == 0:
                row[f"pct_cohort_{cid}_union_of_segment_union"] = (
                    round(100.0 * len(c_union) / n_seg, 4) if n_seg else float("nan")
                )
            else:
                row[f"pct_cohort_{cid}_union_of_segment_union"] = float("nan")

            row[f"n_cumulative_after_cohort_{cid}"] = len(cumulative)
            row[f"pct_cumulative_after_cohort_{cid}_of_segment_union"] = (
                round(100.0 * len(cumulative) / n_seg, 4) if n_seg else float("nan")
            )

        rows.append(row)

    return pd.DataFrame(rows)


def build_tagged_row_table(
    df_sets: pd.DataFrame,
    cohort_groups: dict[int, list[str]],
    segment_ids: tuple[str, ...],
) -> pd.DataFrame:
    cohort_ids = sorted(cohort_groups.keys())
    pop = tagged_population_universe(df_sets)
    seg = _tagged_set_for_enterprises(df_sets, segment_ids)
    n_pop, n_seg = len(pop), len(seg)
    pct_seg_of_pop = round(100.0 * n_seg / n_pop, 4) if n_pop else float("nan")

    row: dict = {
        "dimension": "all_dimensions_tagged_union",
        "dimension_label": "All dimensions (tagged dimension|value)",
        "n_population_universe": n_pop,
        "n_segment_union": n_seg,
        "pct_segment_union_of_population": pct_seg_of_pop,
    }

    cumulative: set[str] = set()
    for i, cid in enumerate(cohort_ids):
        c_eids = tuple(cohort_groups[cid])
        c_union = _tagged_set_for_enterprises(df_sets, c_eids)
        incremental = c_union - cumulative
        cumulative |= c_union

        row[f"n_cohort_{cid}_union"] = len(c_union)
        row[f"n_cohort_{cid}_incremental"] = len(incremental)
        row[f"pct_cohort_{cid}_incremental_of_segment_union"] = (
            round(100.0 * len(incremental) / n_seg, 4) if n_seg else float("nan")
        )
        if i == 0:
            row[f"pct_cohort_{cid}_union_of_segment_union"] = (
                round(100.0 * len(c_union) / n_seg, 4) if n_seg else float("nan")
            )
        else:
            row[f"pct_cohort_{cid}_union_of_segment_union"] = float("nan")

        row[f"n_cumulative_after_cohort_{cid}"] = len(cumulative)
        row[f"pct_cumulative_after_cohort_{cid}_of_segment_union"] = (
            round(100.0 * len(cumulative) / n_seg, 4) if n_seg else float("nan")
        )

    return pd.DataFrame([row])


def _all_target_dims_ge_threshold(
    covered: dict[str, set[str]],
    universe: dict[str, set[str]],
    target_dims: list[str],
    threshold: float,
) -> bool:
    for d in target_dims:
        uu = universe[d]
        if not uu:
            continue
        if len(covered[d] & uu) / len(uu) < threshold - 1e-12:
            return False
    return True


def run_greedy_within_each_cohort(
    df_sets: pd.DataFrame,
    cohort_groups: dict[int, list[str]],
    *,
    target_dims: list[str] | None = None,
    threshold: float = COVER_PCT_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each cohort, greedy-select enterprises from that cohort until every ``target_dim``
    reaches ``threshold`` coverage of **that cohort’s union** on that dimension (or no
    marginal gain remains).
    """
    dims = target_dims if target_dims is not None else list(DIMENSION_KEYS)
    summary_rows: list[dict] = []
    pick_rows: list[dict] = []

    for cid in sorted(cohort_groups.keys()):
        raw_list = cohort_groups[cid]
        pool_unique = list(dict.fromkeys(e for e in raw_list if e in df_sets.index))
        n_file = len(raw_list)
        n_model = len(pool_unique)

        if not pool_unique:
            summary_rows.append(
                {
                    "cohort": cid,
                    "n_enterprises_in_cohort_file": n_file,
                    "n_enterprises_in_model": 0,
                    "n_greedy_selected": 0,
                    "pct_of_in_model_enterprises_needed": float("nan"),
                    "all_dimensions_ge_threshold_vs_cohort_union": True,
                }
            )
            continue

        u_t = tuple(pool_unique)
        chosen, trace, covered_final = greedy_cover_to_threshold(
            df_sets,
            pool_unique,
            dims,
            threshold=threshold,
            universe_source="segment",
            segment_ids_for_universe=u_t,
        )
        U = {d: _superset_for_enterprises(df_sets, d, u_t) for d in dims}
        all_ok = _all_target_dims_ge_threshold(covered_final, U, dims, threshold)

        row: dict = {
            "cohort": cid,
            "n_enterprises_in_cohort_file": n_file,
            "n_enterprises_in_model": n_model,
            "n_greedy_selected": len(chosen),
            "pct_of_in_model_enterprises_needed": (
                round(100.0 * len(chosen) / n_model, 4) if n_model else float("nan")
            ),
            "all_dimensions_ge_threshold_vs_cohort_union": all_ok,
        }
        for d in dims:
            uu = U[d]
            if not uu:
                row[f"pct_final_{d}_vs_cohort_union"] = 100.0
            else:
                row[f"pct_final_{d}_vs_cohort_union"] = round(
                    100.0 * len(covered_final[d] & uu) / len(uu), 4
                )
        summary_rows.append(row)

        for tr in trace:
            pick_rows.append({"cohort": cid, **tr})

    return pd.DataFrame(summary_rows), pd.DataFrame(pick_rows)


def run(
    xlsx: Path | None = None,
    out_dir: Path | None = None,
    *,
    greedy_threshold: float = COVER_PCT_THRESHOLD,
) -> None:
    path = xlsx or DEFAULT_XLSX
    out = out_dir or ROOT
    cohort_groups, segment_order = load_cohort_groups(path)
    segment_ids = tuple(dict.fromkeys(segment_order))

    df_sets, _enc = build_enterprise_dimension_sets()

    by_dim = build_per_dimension_rows(df_sets, cohort_groups, segment_ids)
    tagged = build_tagged_row_table(df_sets, cohort_groups, segment_ids)

    p1 = out / "accounting_segment1_cohort_incremental_by_dimension.csv"
    p2 = out / "accounting_segment1_cohort_incremental_tagged_union.csv"
    by_dim.to_csv(p1, index=False)
    tagged.to_csv(p2, index=False)

    g_sum, g_picks = run_greedy_within_each_cohort(
        df_sets,
        cohort_groups,
        target_dims=list(DIMENSION_KEYS),
        threshold=greedy_threshold,
    )
    p3 = out / "accounting_segment1_cohort_greedy_80pct_summary.csv"
    p4 = out / "accounting_segment1_cohort_greedy_80pct_picks.csv"
    g_sum.to_csv(p3, index=False)
    g_picks.to_csv(p4, index=False)

    summary = out / "accounting_segment1_cohort_summary.txt"
    lines = [
        "Accounting Segment 1 - cohort incremental analysis",
        "=" * 72,
        f"Source: {path.name}",
        f"Cohort ids (in order): {sorted(cohort_groups.keys())}",
        f"Enterprises per cohort: " + ", ".join(f"{k}={len(v)}" for k, v in sorted(cohort_groups.items())),
        f"Unique segment enterprises: {len(segment_ids)}",
        "",
        "pct_segment_union_of_population: share of full-model distinct values Segment 1 touches.",
        "pct_cohort_1_union_of_segment_union: Cohort 1 alone vs full Segment 1 union (same dimension).",
        "pct_cohort_k_incremental_of_segment_union: values NEW when adding cohort k (not in cohorts 1..k-1), as % of segment union.",
        "pct_cumulative_after_cohort_k_of_segment_union: running union through cohort k vs segment union.",
        "",
        f"Wrote {p1.name}",
        f"Wrote {p2.name}",
        "",
        f"Greedy within cohort (threshold={greedy_threshold:.0%} of each cohort's own union, all 9 dimensions):",
        f"Wrote {p3.name}",
        f"Wrote {p4.name}",
    ]
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    ap.add_argument("--out-dir", type=Path, default=ROOT)
    ap.add_argument(
        "--greedy-threshold",
        type=float,
        default=COVER_PCT_THRESHOLD,
        help="Coverage threshold per dimension vs cohort union (default 0.8)",
    )
    args = ap.parse_args()
    run(xlsx=args.xlsx, out_dir=args.out_dir, greedy_threshold=args.greedy_threshold)
