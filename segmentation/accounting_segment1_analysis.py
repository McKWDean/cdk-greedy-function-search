"""
Accounting Segment 1 — population coverage + greedy test set for 80% on key dimensions.

Reads: ``20260326 Accounting Segment 1 Enterprise IDs.xlsx`` (sheet ``Accounting Segment 1``,
column ``Enterprise ID``).

1) **Segment superset vs full population** — For each Jaccard dimension, the % of *all distinct
   values in the model population* that appear in the union of Segment 1 enterprises (in scope).

2) **Greedy minimum test set (>=80%)** — From Segment 1 enterprises only, greedily add enterprises
   until the union covers **at least 80%** of the population universe for each of:
   OEMs, internal 3PA, external 3PA, Enterprise_Setup_Map (org setups), printers, forms, user profiles.

Run from repo root:
  python -m segmentation.accounting_segment1_analysis

Outputs (repo root):
  - accounting_segment1_dimension_coverage.csv
  - accounting_segment1_greedy_80pct_population_universe.csv
  - accounting_segment1_greedy_80pct_segment_universe.csv
  - accounting_segment1_summary.txt
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from segmentation.enterprise_dimension_export import (
    _distinct_values_in_column,
    _sanitize_cell_text,
    _superset_for_enterprises,
)
from segmentation.jaccard_similarity_clustering import (
    DIMENSION_KEYS,
    build_enterprise_dimension_sets,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SEGMENT_XLSX = ROOT / "20260326 Accounting Segment 1 Enterprise IDs.xlsx"
DEFAULT_SHEET = "Accounting Segment 1"
DEFAULT_COL = "Enterprise ID"

# Greedy / 80% target applies to these dimensions only (labels for reporting).
TARGET_DIMENSIONS: list[str] = [
    "oem_codes",
    "vendors_3pa_internal",
    "vendors_3pa_external",
    "enterprise_setup",
    "spooler_types",
    "formnames",
    "profile_tokens",
]

DIMENSION_DISPLAY: dict[str, str] = {
    "oem_codes": "OEMs",
    "vendors_3pa_internal": "Internal integrations",
    "vendors_3pa_external": "External integrations",
    "enterprise_setup": "Org setups (Enterprise_Setup_Map)",
    "spooler_types": "Peripherals / printers",
    "formnames": "Forms",
    "profile_tokens": "User profiles",
}

COVER_PCT_THRESHOLD = 0.80


def load_segment_enterprise_ids(
    path: Path = DEFAULT_SEGMENT_XLSX,
    sheet: str = DEFAULT_SHEET,
    col: str = DEFAULT_COL,
) -> tuple[str, ...]:
    df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    if col not in df.columns:
        raise ValueError(f"Column {col!r} not found; have {list(df.columns)}")
    ids = (
        df[col]
        .dropna()
        .astype(str)
        .map(lambda x: _sanitize_cell_text(x.strip()))
    )
    ids = ids[ids != ""].unique()
    return tuple(ids)


def enterprise_values_for_dim(df_sets: pd.DataFrame, eid: str, col: str) -> set[str]:
    eid = str(eid).strip()
    if eid not in df_sets.index:
        return set()
    out: set[str] = set()
    for t in df_sets.loc[eid, col]:
        tok = _sanitize_cell_text(str(t).strip())
        if tok:
            out.add(tok)
    return out


def segment_coverage_vs_population(
    df_sets: pd.DataFrame,
    segment_ids: tuple[str, ...],
) -> pd.DataFrame:
    """Per dimension: population |U|, segment union |S|, pct = |S|/|U|."""
    rows: list[dict] = []
    for col in DIMENSION_KEYS:
        uni = _distinct_values_in_column(df_sets, col)
        sup = _superset_for_enterprises(df_sets, col, segment_ids)
        nu, ns = len(uni), len(sup)
        pct = round(100.0 * ns / nu, 4) if nu else (0.0 if ns == 0 else float("nan"))
        rows.append(
            {
                "dimension": col,
                "dimension_label": DIMENSION_DISPLAY.get(col, col),
                "n_distinct_values_population": nu,
                "n_distinct_in_segment_superset": ns,
                "pct_of_population_universe_covered": pct,
            }
        )
    # Tagged union row (all dimensions)
    u_all = 0
    s_all = 0
    for col in DIMENSION_KEYS:
        uni = _distinct_values_in_column(df_sets, col)
        sup = _superset_for_enterprises(df_sets, col, segment_ids)
        u_all += len(uni)
        s_all += len(sup & uni)
    pct_all = round(100.0 * s_all / u_all, 4) if u_all else float("nan")
    rows.append(
        {
            "dimension": "all_dimensions_tagged_sum",
            "dimension_label": "Sum over dims (not deduped across dims)",
            "n_distinct_values_population": u_all,
            "n_distinct_in_segment_superset": s_all,
            "pct_of_population_universe_covered": pct_all,
        }
    )
    return pd.DataFrame(rows)


def _dim_satisfied(covered: set[str], universe: set[str], threshold: float) -> bool:
    if not universe:
        return True
    return len(covered & universe) / len(universe) >= threshold - 1e-12


def _atoms_needed(universe: set[str], covered: set[str], threshold: float) -> int:
    """Count of population atoms in this dimension still missing to reach threshold coverage."""
    if not universe:
        return 0
    u = len(universe)
    have = len(covered & universe)
    need_count = int(math.ceil(threshold * u))
    return max(0, need_count - have)


def greedy_cover_to_threshold(
    df_sets: pd.DataFrame,
    pool: list[str],
    target_dims: list[str],
    threshold: float = COVER_PCT_THRESHOLD,
    *,
    universe_source: str = "population",
    segment_ids_for_universe: tuple[str, ...] | None = None,
) -> tuple[list[str], list[dict], dict[str, set[str]]]:
    """
    Greedily add enterprises from ``pool`` until each target dimension has >= threshold
    coverage of its reference universe.

    - ``universe_source='population'``: U_d = all distinct values in the model (all enterprises).
    - ``universe_source='segment'``: U_d = union of values over ``segment_ids_for_universe``
      (typically full Segment 1). Then 80% means 80% of *Segment 1's own* diversity on d.
    """
    if universe_source == "population":
        U = {d: _distinct_values_in_column(df_sets, d) for d in target_dims}
    elif universe_source == "segment":
        if not segment_ids_for_universe:
            raise ValueError("segment_ids_for_universe required for universe_source='segment'")
        U = {
            d: _superset_for_enterprises(df_sets, d, segment_ids_for_universe)
            for d in target_dims
        }
    else:
        raise ValueError(f"unknown universe_source: {universe_source!r}")
    covered = {d: set() for d in target_dims}
    chosen: list[str] = []
    trace: list[dict] = []

    def all_satisfied() -> bool:
        return all(_dim_satisfied(covered[d], U[d], threshold) for d in target_dims)

    pool_set = list(dict.fromkeys(pool))  # stable unique

    while not all_satisfied():
        best_e: str | None = None
        best_score = -1.0
        for e in pool_set:
            if e in chosen:
                continue
            score = 0.0
            for d in target_dims:
                if not U[d]:
                    continue
                new_atoms = enterprise_values_for_dim(df_sets, e, d) & U[d] - covered[d]
                w = float(_atoms_needed(U[d], covered[d], threshold))
                if w <= 0:
                    continue
                score += len(new_atoms) * max(1.0, w)
            if score > best_score or (score == best_score and best_e is not None and e < best_e):
                best_score = score
                best_e = e
        if best_e is None or best_score <= 0:
            break
        chosen.append(best_e)
        row: dict = {
            "pick_order": len(chosen),
            "Enterprise ID": best_e,
            "Enterprise Name": _sanitize_cell_text(str(df_sets.loc[best_e, "Enterprise Name"])),
        }
        for d in target_dims:
            new_atoms = enterprise_values_for_dim(df_sets, best_e, d) & U[d] - covered[d]
            covered[d] |= enterprise_values_for_dim(df_sets, best_e, d) & U[d]
            row[f"n_new_atoms_{d}"] = len(new_atoms)
        for d in target_dims:
            uu = U[d]
            if not uu:
                row[f"pct_{d}"] = 100.0
            else:
                row[f"pct_{d}"] = round(100.0 * len(covered[d] & uu) / len(uu), 4)
        trace.append(row)

    return chosen, trace, covered


def _final_lines_for_greedy(
    covered_final: dict[str, set[str]],
    U_eval: dict[str, set[str]],
    threshold: float,
) -> list[str]:
    lines = []
    for d in TARGET_DIMENSIONS:
        uu = U_eval[d]
        if not uu:
            lines.append(f"  {DIMENSION_DISPLAY.get(d, d)}: N/A (empty universe)")
            continue
        pct = 100.0 * len(covered_final[d] & uu) / len(uu)
        ok = pct >= 100 * threshold - 1e-6
        lines.append(
            f"  {DIMENSION_DISPLAY.get(d, d)}: {pct:.2f}% of reference universe "
            f"({len(covered_final[d] & uu)}/{len(uu)}) {'OK' if ok else 'BELOW TARGET'}"
        )
    return lines


def run_analysis(
    xlsx_path: Path | None = None,
    threshold: float = COVER_PCT_THRESHOLD,
) -> None:
    root = ROOT
    path = xlsx_path or DEFAULT_SEGMENT_XLSX
    segment_ids = load_segment_enterprise_ids(path)

    df_sets, _enc = build_enterprise_dimension_sets()
    in_scope = [e for e in segment_ids if e in df_sets.index]
    missing = [e for e in segment_ids if e not in df_sets.index]

    cov_df = segment_coverage_vs_population(df_sets, segment_ids)
    cov_path = root / "accounting_segment1_dimension_coverage.csv"
    cov_df.to_csv(cov_path, index=False)

    # Greedy A: 80% vs *full population* per dimension (may be infeasible for narrow segments).
    chosen_pop, trace_pop, covered_pop = greedy_cover_to_threshold(
        df_sets,
        in_scope,
        TARGET_DIMENSIONS,
        threshold=threshold,
        universe_source="population",
    )
    trace_pop_df = pd.DataFrame(trace_pop)
    trace_pop_path = root / "accounting_segment1_greedy_80pct_population_universe.csv"
    trace_pop_df.to_csv(trace_pop_path, index=False)

    U_pop = {d: _distinct_values_in_column(df_sets, d) for d in TARGET_DIMENSIONS}

    # Greedy B: 80% vs *Segment 1 union* per dimension (always feasible when segment has data).
    chosen_seg, trace_seg, covered_seg = greedy_cover_to_threshold(
        df_sets,
        in_scope,
        TARGET_DIMENSIONS,
        threshold=threshold,
        universe_source="segment",
        segment_ids_for_universe=tuple(in_scope),
    )
    trace_seg_df = pd.DataFrame(trace_seg)
    trace_seg_path = root / "accounting_segment1_greedy_80pct_segment_universe.csv"
    trace_seg_df.to_csv(trace_seg_path, index=False)

    U_seg = {
        d: _superset_for_enterprises(df_sets, d, tuple(in_scope)) for d in TARGET_DIMENSIONS
    }

    lines = [
        "Accounting Segment 1 - analysis summary",
        "=" * 72,
        f"Source file: {path.name}",
        f"Segment IDs in file: {len(segment_ids)}",
        f"In model spine (mapped cnumber): {len(in_scope)}",
        f"Not in spine: {len(missing)}",
        "",
        "PART 1 - Segment superset vs FULL POPULATION (see accounting_segment1_dimension_coverage.csv):",
        "  For each dimension: pct = (distinct values appearing anywhere in Segment 1) / (distinct in all enterprises).",
        "",
        "PART 2a - Greedy test set: >=80% of POPULATION universe on each target dimension",
        f"  Target dimensions: {', '.join(DIMENSION_DISPLAY.get(d, d) for d in TARGET_DIMENSIONS)}",
        f"  Enterprises selected: {len(chosen_pop)} (stops when no candidate adds new needed atoms)",
        "",
        "  Final vs POPULATION reference:",
    ]
    lines.extend(_final_lines_for_greedy(covered_pop, U_pop, threshold))
    lines.extend(
        [
            "",
            "  If BELOW TARGET: Segment 1 never exhibits enough distinct values vs the full base",
            "  (e.g. rare OEMs/forms/user profiles). No subset of Segment 1 can reach 80% of population.",
            "",
            "PART 2b - Greedy test set: >=80% of SEGMENT-LOCAL universe (diversity inside Segment 1)",
            f"  Enterprises selected: {len(chosen_seg)}",
            "",
            "  Final vs SEGMENT union reference (max diversity Segment 1 can show on each dimension):",
        ]
    )
    lines.extend(_final_lines_for_greedy(covered_seg, U_seg, threshold))

    lines.extend(
        [
            "",
            f"Outputs: {cov_path.name}, {trace_pop_path.name}, {trace_seg_path.name}",
        ]
    )
    summary_path = root / "accounting_segment1_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Accounting Segment 1 coverage + greedy 80% test set")
    p.add_argument(
        "--xlsx",
        type=Path,
        default=DEFAULT_SEGMENT_XLSX,
        help="Path to Segment 1 Enterprise IDs workbook",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=COVER_PCT_THRESHOLD,
        help="Coverage threshold per target dimension (default 0.8)",
    )
    args = p.parse_args()
    run_analysis(xlsx_path=args.xlsx, threshold=args.threshold)
