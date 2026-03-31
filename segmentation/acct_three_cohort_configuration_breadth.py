"""
Configuration breadth for three cohorts (from ``Segment 1 IDs.xlsx``): Accounting or Service.

For each cohort, emits the same views as the manual Segment 1 analysis:

1) **Collective breadth vs full population** — Per Jaccard dimension: distinct values in the
   model universe vs distinct values in the union of cohort enterprises; % covered.

2) **Per-enterprise breadth vs population** — Mean/median distinct values per enterprise for the
   cohort vs all enterprises on the model spine.

**Accounting** (``--vertical acct``, default) — sheets:
  - T&L (Micro-segment 0) → ``ACCT T&L Customers``
  - Micro-segment 1 → ``ACCT M-Segment 1 Customer``
  - Segment 1 → ``ACCT Segment 1 Customers``

**Service** (``--vertical service``) — sheets:
  - T&L (Micro-segment 0) → ``Service T&L Customers``
  - Micro-segment 1 → ``Service M-Segment 1 Customers``
  - Segment 1 → ``Service Segment 1 Customers``

Run from repo root:
  python -m segmentation.acct_three_cohort_configuration_breadth
  python -m segmentation.acct_three_cohort_configuration_breadth --vertical service

Outputs (repo root; prefix depends on vertical):
  - {acct|service}_three_cohort_dimension_coverage.csv
  - {acct|service}_three_cohort_per_enterprise_breadth.csv
  - {acct|service}_three_cohort_configuration_breadth_summary.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from segmentation.accounting_segment1_analysis import (
    DIMENSION_DISPLAY,
    load_segment_enterprise_ids,
    segment_coverage_vs_population,
)
from segmentation.jaccard_similarity_clustering import DIMENSION_KEYS, build_enterprise_dimension_sets

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_IDS_XLSX = ROOT / "Segment 1 IDs.xlsx"

# (cohort_key, display name, Excel sheet name)
ACCT_COHORTS: tuple[tuple[str, str, str], ...] = (
    ("tl_microsegment_0", "T&L customers (Micro-segment 0)", "ACCT T&L Customers"),
    ("microsegment_1", "Micro-segment 1 customers", "ACCT M-Segment 1 Customer"),
    ("segment_1", "Segment 1 customers", "ACCT Segment 1 Customers"),
)

SERVICE_COHORTS: tuple[tuple[str, str, str], ...] = (
    ("tl_microsegment_0", "T&L customers (Micro-segment 0)", "Service T&L Customers"),
    ("microsegment_1", "Micro-segment 1 customers", "Service M-Segment 1 Customers"),
    ("segment_1", "Segment 1 customers", "Service Segment 1 Customers"),
)

DEFAULT_COHORTS = ACCT_COHORTS


def _breadth_counts_for_enterprise(row: pd.Series) -> dict[str, int]:
    return {d: len(row[d]) if hasattr(row[d], "__len__") else 0 for d in DIMENSION_KEYS}


def per_enterprise_breadth_comparison(
    df_sets: pd.DataFrame,
    cohort_ids: tuple[str, ...],
) -> pd.DataFrame:
    """Mean/median |set| per dimension: full population vs cohort (in-scope only)."""
    pop_idx = list(df_sets.index)
    in_scope = [e for e in cohort_ids if e in df_sets.index]

    rows_pop = [_breadth_counts_for_enterprise(df_sets.loc[e]) for e in pop_idx]
    rows_coh = [_breadth_counts_for_enterprise(df_sets.loc[e]) for e in in_scope]
    df_pop = pd.DataFrame(rows_pop)
    df_coh = pd.DataFrame(rows_coh)

    summ: list[dict] = []
    for d in DIMENSION_KEYS:
        pm, pmed = float(df_pop[d].mean()), float(df_pop[d].median())
        cm = float(df_coh[d].mean()) if len(df_coh) else float("nan")
        cmed = float(df_coh[d].median()) if len(df_coh) else float("nan")
        ratio = (cm / pm) if pm and len(df_coh) else float("nan")
        summ.append(
            {
                "dimension": d,
                "dimension_label": DIMENSION_DISPLAY.get(d, d),
                "population_n_enterprises": len(df_pop),
                "cohort_n_enterprises_in_scope": len(df_coh),
                "population_mean_distinct": round(pm, 4),
                "population_median_distinct": round(pmed, 4),
                "cohort_mean_distinct": round(cm, 4) if cm == cm else np.nan,
                "cohort_median_distinct": round(cmed, 4) if cmed == cmed else np.nan,
                "ratio_mean_cohort_over_population": round(ratio, 4) if ratio == ratio else np.nan,
            }
        )

    sum_pop = df_pop.sum(axis=1)
    sum_coh = df_coh.sum(axis=1)
    pm, pmed = float(sum_pop.mean()), float(sum_pop.median())
    cm = float(sum_coh.mean()) if len(sum_coh) else float("nan")
    cmed = float(sum_coh.median()) if len(sum_coh) else float("nan")
    ratio = (cm / pm) if pm and len(sum_coh) else float("nan")
    summ.append(
        {
            "dimension": "sum_dim_counts_per_enterprise",
            "dimension_label": "Sum of per-dimension counts (same enterprise)",
            "population_n_enterprises": len(df_pop),
            "cohort_n_enterprises_in_scope": len(df_coh),
            "population_mean_distinct": round(pm, 4),
            "population_median_distinct": round(pmed, 4),
            "cohort_mean_distinct": round(cm, 4) if cm == cm else np.nan,
            "cohort_median_distinct": round(cmed, 4) if cmed == cmed else np.nan,
            "ratio_mean_cohort_over_population": round(ratio, 4) if ratio == ratio else np.nan,
        }
    )
    return pd.DataFrame(summ)


def run(
    xlsx_path: Path = DEFAULT_IDS_XLSX,
    cohorts: tuple[tuple[str, str, str], ...] = DEFAULT_COHORTS,
    out_dir: Path | None = None,
    *,
    vertical: str = "acct",
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    out_dir = out_dir or ROOT
    vertical = vertical.strip().lower()
    if vertical not in ("acct", "service"):
        raise ValueError("vertical must be 'acct' or 'service'")
    out_prefix = "acct" if vertical == "acct" else "service"
    title = "Accounting" if vertical == "acct" else "Service"

    df_sets, _enc = build_enterprise_dimension_sets()

    cov_parts: list[pd.DataFrame] = []
    breadth_parts: list[pd.DataFrame] = []
    lines: list[str] = [
        f"{title} - three-cohort configuration breadth",
        "=" * 72,
        f"Source: {xlsx_path.name}",
        f"Vertical: {vertical}",
        f"Model enterprises (spine): {len(df_sets)}",
        "",
    ]

    for key, display, sheet in cohorts:
        ids = load_segment_enterprise_ids(xlsx_path, sheet=sheet)
        in_scope = [e for e in ids if e in df_sets.index]
        missing = [e for e in ids if e not in df_sets.index]

        cov = segment_coverage_vs_population(df_sets, ids)
        cov = cov.assign(cohort_key=key, cohort_label=display, excel_sheet=sheet)
        cov_parts.append(cov)

        br = per_enterprise_breadth_comparison(df_sets, ids)
        br = br.assign(cohort_key=key, cohort_label=display, excel_sheet=sheet)
        breadth_parts.append(br)

        lines.extend(
            [
                f"## {display}",
                f"    Sheet: {sheet!r}  |  cohort_key: {key}",
                f"    IDs in file: {len(ids)}  |  In model spine: {len(in_scope)}  |  Not in spine: {len(missing)}",
                "",
                "    Collective breadth vs full population (% of population distinct values touched):",
            ]
        )
        for _, r in cov.iterrows():
            if r["dimension"] == "all_dimensions_tagged_sum":
                lines.append(
                    f"      [tagged sum across dims] {r['pct_of_population_universe_covered']:.2f}% "
                    f"({int(r['n_distinct_in_segment_superset'])}/{int(r['n_distinct_values_population'])})"
                )
            else:
                lines.append(
                    f"      {r['dimension_label']}: {r['pct_of_population_universe_covered']:.2f}% "
                    f"({int(r['n_distinct_in_segment_superset'])}/{int(r['n_distinct_values_population'])})"
                )
        lines.extend(["", "    Per-enterprise breadth (cohort mean vs population mean, ratio):"])
        for _, r in br.iterrows():
            lines.append(
                f"      {r['dimension_label']}: pop_mean={r['population_mean_distinct']:.3f}, "
                f"cohort_mean={r['cohort_mean_distinct']:.3f}, ratio={r['ratio_mean_cohort_over_population']}"
            )
        lines.append("")

    cov_all = pd.concat(cov_parts, ignore_index=True)
    br_all = pd.concat(breadth_parts, ignore_index=True)

    # Stable column order for coverage
    front = ["cohort_key", "cohort_label", "excel_sheet"]
    cov_all = cov_all[front + [c for c in cov_all.columns if c not in front]]
    br_all = br_all[front + [c for c in br_all.columns if c not in front]]

    cov_path = out_dir / f"{out_prefix}_three_cohort_dimension_coverage.csv"
    br_path = out_dir / f"{out_prefix}_three_cohort_per_enterprise_breadth.csv"
    summ_path = out_dir / f"{out_prefix}_three_cohort_configuration_breadth_summary.txt"

    cov_all.to_csv(cov_path, index=False)
    br_all.to_csv(br_path, index=False)
    summary = "\n".join(lines)
    summ_path.write_text(summary, encoding="utf-8")

    return cov_all, br_all, summary


def main() -> None:
    p = argparse.ArgumentParser(description="Configuration breadth for T&L / M-Seg1 / Segment 1 cohorts")
    p.add_argument(
        "--vertical",
        choices=("acct", "service"),
        default="acct",
        help="Workbook tabs: acct=ACCT T&L / M-Seg1 / Seg1; service=Service T&L / M-Seg1 / Seg1",
    )
    p.add_argument(
        "--xlsx",
        type=Path,
        default=DEFAULT_IDS_XLSX,
        help="Workbook with cohort sheets (default: Segment 1 IDs.xlsx in repo root)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT,
        help="Directory for CSV + summary (default: repo root)",
    )
    args = p.parse_args()
    cohorts = ACCT_COHORTS if args.vertical == "acct" else SERVICE_COHORTS
    _, _, summary = run(
        xlsx_path=args.xlsx,
        cohorts=cohorts,
        out_dir=args.out_dir,
        vertical=args.vertical,
    )
    print(summary)


if __name__ == "__main__":
    main()
