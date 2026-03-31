"""
Coverage of five Enterprise IDs vs **Cohort 1’s** configuration union (per dimension).

For each Jaccard dimension *d*:
  - ``U_c1`` = distinct values appearing on **any** Cohort 1 enterprise (from the Segment 1 workbook).
  - ``U_5`` = distinct values appearing on **any** of the five listed enterprises (in model spine).
  - Report ``pct_of_cohort1_union_covered`` = |U_5 ∩ U_c1| / |U_c1| (100% if the five jointly
    exhibit every value Cohort 1 ever shows on *d*).

Reads the same workbook as other Segment 1 tools:
  ``20260326 Accounting Segment 1 Enterprise IDs.xlsx`` → sheet ``Accounting Segment 1``.

Run: ``python -m segmentation.accounting_segment1_first_five_cohort1_coverage``

Output (repo root): ``accounting_segment1_first_five_vs_cohort1_by_dimension.csv``
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from segmentation.accounting_segment1_cohort_analysis import load_cohort_groups
from segmentation.enterprise_dimension_export import _superset_for_enterprises
from segmentation.jaccard_similarity_clustering import DIMENSION_KEYS, build_enterprise_dimension_sets

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_XLSX = ROOT / "20260326 Accounting Segment 1 Enterprise IDs.xlsx"

FIRST_FIVE_ENTERPRISE_IDS: tuple[str, ...] = (
    "E213761",
    "E211404",
    "E101858",
    "E220385",
    "E216435",
)

DIMENSION_LABELS: dict[str, str] = {
    "oem_codes": "OEMs",
    "vendors_3pa_internal": "Internal integrations",
    "vendors_3pa_external": "External integrations",
    "enterprise_setup": "Org setups (Enterprise_Setup_Map)",
    "spooler_types": "Peripherals / printers",
    "formnames": "Forms",
    "profile_tokens": "User profiles",
}


def run(
    xlsx: Path | None = None,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    path = xlsx or DEFAULT_XLSX
    out = out_dir or ROOT
    cohort_groups, _ = load_cohort_groups(path)
    if 1 not in cohort_groups:
        raise ValueError("Cohort 1 not found in workbook")
    cohort1_ids = tuple(dict.fromkeys(cohort_groups[1]))
    five = FIRST_FIVE_ENTERPRISE_IDS

    in_cohort1_file = {e: e in cohort1_ids for e in five}
    df_sets, _ = build_enterprise_dimension_sets()
    in_model = {e: e in df_sets.index for e in five}

    rows: list[dict] = []
    for col in DIMENSION_KEYS:
        u_c1 = _superset_for_enterprises(df_sets, col, cohort1_ids)
        u_5 = _superset_for_enterprises(df_sets, col, five)
        inter = u_5 & u_c1
        n_c1, n_5, n_both = len(u_c1), len(u_5), len(inter)
        pct = round(100.0 * n_both / n_c1, 4) if n_c1 else float("nan")
        rows.append(
            {
                "dimension": col,
                "dimension_label": DIMENSION_LABELS.get(col, col),
                "n_distinct_cohort1_union": n_c1,
                "n_distinct_five_union": n_5,
                "n_distinct_intersection_five_and_cohort1": n_both,
                "pct_of_cohort1_union_covered_by_five": pct,
            }
        )

    # Tagged union summary (same atom style as other reports)
    c1_atoms: set[str] = set()
    five_atoms: set[str] = set()
    for col in DIMENSION_KEYS:
        for t in _superset_for_enterprises(df_sets, col, cohort1_ids):
            c1_atoms.add(f"{col}|{t}")
        for t in _superset_for_enterprises(df_sets, col, five):
            five_atoms.add(f"{col}|{t}")
    inter_atoms = five_atoms & c1_atoms
    n_c1a, n_5a, n_ia = len(c1_atoms), len(five_atoms), len(inter_atoms)
    pct_a = round(100.0 * n_ia / n_c1a, 4) if n_c1a else float("nan")
    rows.append(
        {
            "dimension": "all_dimensions_tagged_union",
            "dimension_label": "All dimensions (tagged)",
            "n_distinct_cohort1_union": n_c1a,
            "n_distinct_five_union": n_5a,
            "n_distinct_intersection_five_and_cohort1": n_ia,
            "pct_of_cohort1_union_covered_by_five": pct_a,
        }
    )

    df = pd.DataFrame(rows)
    meta_path = out / "accounting_segment1_first_five_vs_cohort1_meta.txt"
    meta_lines = [
        "Five Enterprise IDs vs Cohort 1 union (per dimension)",
        "=" * 60,
        f"Workbook: {path.name}",
        f"Cohort 1 enterprises in file: {len(cohort1_ids)}",
        "",
        "Membership of the five IDs:",
    ]
    for e in five:
        meta_lines.append(
            f"  {e}: in Cohort 1 file={in_cohort1_file[e]}, in model spine={in_model[e]}"
        )
    meta_lines.extend(["", f"Wrote: accounting_segment1_first_five_vs_cohort1_by_dimension.csv"])
    meta_path.write_text("\n".join(meta_lines) + "\n", encoding="utf-8")

    csv_path = out / "accounting_segment1_first_five_vs_cohort1_by_dimension.csv"
    df.to_csv(csv_path, index=False)
    print("\n".join(meta_lines))
    return df


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    ap.add_argument("--out-dir", type=Path, default=ROOT)
    args = ap.parse_args()
    run(xlsx=args.xlsx, out_dir=args.out_dir)
