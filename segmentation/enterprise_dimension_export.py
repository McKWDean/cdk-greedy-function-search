"""
Export enterprise × configuration dimensions to Excel.

- **dimension_coverage** (first tab): per dimension, how many enterprise IDs have at least
  one mapped value (non-empty set).
- **overview**: pairwise Jaccard summary **restricted** to enterprises with data on that
  dimension (only those rows enter the matrix). **combined_equal_weight** uses enterprises
  that have ≥1 value in **every** dimension (intersection), then averages Jaccard matrices.
- **bucket_reference**: Enterprise ID, Name, Bucket from Enterprise_Setup_Map.csv (reference only).
- One sheet per clustering dimension: Enterprise ID, Name, Bucket, values, n_distinct.
- **frequency_Bucket** + **frequency_<dimension>**: value × n_enterprises.
- **First 5**: spotlight on five Enterprise IDs — each row shows dimension values/counts;
  a coverage table gives, per dimension, what share of *all distinct values in the population*
  is covered by the union (superset) of those five enterprises, plus one row for the union
  across dimensions (values tagged as ``dimension|value`` so overlap across dims does not collide).
- **Cohort 78**: same layout twice on one sheet — (1) first five IDs from the cohort list only:
  superset coverage table + enterprise detail rows; (2) full ~78-ID list: coverage + all detail rows.
- **Cohort 101**: same four-block layout for the next list (numeric suffixes stored as ``E{suffix}``).

Run: python -m segmentation.enterprise_dimension_export
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from segmentation.jaccard_similarity_clustering import (
    DIMENSION_KEYS,
    build_enterprise_dimension_sets,
    pairwise_jaccard_matrix_for_column,
)

# openpyxl rejects ASCII control characters in cell text
_ILLEGAL_XLSX_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

COVERAGE_SHEET = "dimension_coverage"
OVERVIEW_SHEET = "overview"

# Spotlight cohort for the "First 5" sheet (order preserved in the workbook).
FIRST_FIVE_ENTERPRISE_IDS: tuple[str, ...] = (
    "E213761",
    "E211404",
    "E101858",
    "E220385",
    "E216435",
)
FIRST_FIVE_SHEET = "First 5"

# Second spotlight list (order preserved). First five = first five rows here.
COHORT_78_ENTERPRISE_IDS: tuple[str, ...] = (
    "E311824",
    "E213857",
    "E204657",
    "E200370",
    "E210302",
    "E215615",
    "E220130",
    "E100786",
    "E107948",
    "E201402",
    "E202869",
    "E203234",
    "E207610",
    "E218778",
    "E226187",
    "E228440",
    "E232779",
    "E310872",
    "E201995",
    "E205838",
    "E205961",
    "E210719",
    "E213761",
    "E214714",
    "E214985",
    "E215627",
    "E217403",
    "E218723",
    "E219808",
    "E219992",
    "E220593",
    "E220899",
    "E304603",
    "E308497",
    "E201298",
    "E202245",
    "E202750",
    "E203381",
    "E203868",
    "E205108",
    "E205170",
    "E207700",
    "E208464",
    "E210230",
    "E210373",
    "E211404",
    "E211553",
    "E212243",
    "E212314",
    "E212738",
    "E212813",
    "E212826",
    "E214121",
    "E214942",
    "E216455",
    "E216934",
    "E218842",
    "E218975",
    "E221959",
    "E226998",
    "E227879",
    "E228081",
    "E307022",
    "E307703",
    "E500191",
    "E200158",
    "E202654",
    "E203470",
    "E203652",
    "E204094",
    "E206662",
    "E207594",
    "E213605",
    "E213940",
    "E219110",
    "E233895",
    "E500194",
    "E205297",
)
COHORT_78_FIRST_FIVE: tuple[str, ...] = COHORT_78_ENTERPRISE_IDS[:5]
COHORT_78_SHEET = "Cohort 78"

# Next cohort: image supplied numeric suffixes only; Enterprise ID = "E" + suffix (same as other tabs).
_COHORT_101_SUFFIXES: tuple[int, ...] = (
    683,
    749,
    858,
    906,
    932,
    937,
    944,
    1162,
    1198,
    1220,
    1284,
    1317,
    1344,
    1350,
    1464,
    1535,
    1564,
    1662,
    1673,
    1718,
    1719,
    1725,
    1736,
    1865,
    1869,
    1893,
    2032,
    2036,
    2069,
    2112,
    2162,
    2208,
    2328,
    2343,
    2365,
    2377,
    2474,
    2532,
    2541,
    2555,
    2591,
    2620,
    2659,
    2688,
    2695,
    2734,
    2752,
    2774,
    2778,
    2789,
    2814,
    2830,
    2832,
    2839,
    2845,
    2847,
    2850,
    2864,
    2871,
    2872,
    2913,
    2914,
    2916,
    2923,
    2924,
    2935,
    2948,
    2956,
    2958,
    2961,
    2963,
    2964,
    2965,
    2968,
    2971,
    2972,
    2973,
    2983,
    2984,
    2991,
    2994,
    2995,
    2996,
    2997,
    3001,
    3010,
    3011,
    3015,
    3016,
    3017,
    3019,
    3020,
    3021,
    3022,
    3023,
    3024,
    3025,
    3026,
    3027,
    3028,
    3030,
)
COHORT_101_ENTERPRISE_IDS: tuple[str, ...] = tuple(f"E{n}" for n in _COHORT_101_SUFFIXES)
COHORT_101_FIRST_FIVE: tuple[str, ...] = COHORT_101_ENTERPRISE_IDS[:5]
COHORT_101_SHEET = "Cohort 101"


def _sanitize_cell_text(s: str) -> str:
    if not s:
        return s
    return _ILLEGAL_XLSX_CHARS.sub("", s)


def _upper_triangle_stats(sim: np.ndarray) -> dict[str, float | int]:
    iu = np.triu_indices(sim.shape[0], k=1)
    v = np.asarray(sim[iu], dtype=np.float64)
    n = int(v.size)
    if n == 0:
        return {
            "n_pairs": 0,
            "mean": np.nan,
            "std": np.nan,
            "p10": np.nan,
            "p25": np.nan,
            "p50": np.nan,
            "p75": np.nan,
            "p90": np.nan,
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


def _dimension_coverage_df(df_sets: pd.DataFrame) -> pd.DataFrame:
    """Counts enterprises with ≥1 value per dimension (DIMENSION_KEYS only)."""
    n_total = len(df_sets)
    rows = []
    for col in DIMENSION_KEYS:
        n_with = int((df_sets[col].map(len) > 0).sum())
        pct = round(100.0 * n_with / n_total, 2) if n_total else 0.0
        rows.append(
            {
                "dimension": col,
                "n_enterprise_ids_with_data": n_with,
                "n_enterprise_ids_total": n_total,
                "pct_of_enterprises_with_data": pct,
            }
        )
    return pd.DataFrame(rows)


def _pairwise_jaccard_summary_restricted(df_sets: pd.DataFrame) -> pd.DataFrame:
    """
    Per dimension: Jaccard only among enterprises with a non-empty set on that dimension.
    Combined row: enterprises with non-empty sets on *all* dimensions; mean Jaccard matrix.
    """
    n_total = len(df_sets)
    rows: list[dict[str, str | float | int]] = []

    for col in DIMENSION_KEYS:
        df_sub = df_sets.loc[df_sets[col].map(len) > 0]
        n_sub = len(df_sub)
        base = {
            "dimension": col,
            "n_enterprises_used_in_jaccard": n_sub,
            "n_enterprises_total_in_scope": n_total,
            "note": "Jaccard computed only on rows with ≥1 value in this dimension.",
        }
        if n_sub < 2:
            rows.append(
                {
                    **base,
                    "n_pairs": 0,
                    "mean": np.nan,
                    "std": np.nan,
                    "p10": np.nan,
                    "p25": np.nan,
                    "p50": np.nan,
                    "p75": np.nan,
                    "p90": np.nan,
                }
            )
            continue
        J = pairwise_jaccard_matrix_for_column(df_sub, col)
        st = _upper_triangle_stats(J)
        rows.append({**base, **st})
        del J

    # Combined: intersection of enterprises non-empty on every dimension
    mask = pd.Series(True, index=df_sets.index)
    for col in DIMENSION_KEYS:
        mask &= df_sets[col].map(len) > 0
    df_all = df_sets.loc[mask]
    n_int = len(df_all)
    base_c = {
        "dimension": "combined_equal_weight",
        "n_enterprises_used_in_jaccard": n_int,
        "n_enterprises_total_in_scope": n_total,
        "note": "Mean of per-dimension Jaccard matrices; only enterprises with ≥1 value in every dimension.",
    }
    if n_int < 2:
        rows.append(
            {
                **base_c,
                "n_pairs": 0,
                "mean": np.nan,
                "std": np.nan,
                "p10": np.nan,
                "p25": np.nan,
                "p50": np.nan,
                "p75": np.nan,
                "p90": np.nan,
            }
        )
    else:
        acc = None
        for col in DIMENSION_KEYS:
            J = pairwise_jaccard_matrix_for_column(df_all, col)
            acc = J.copy() if acc is None else acc + J
            del J
        assert acc is not None
        combined = acc / float(len(DIMENSION_KEYS))
        st = _upper_triangle_stats(combined)
        rows.append({**base_c, **st})

    out = pd.DataFrame(rows)
    for c in ("mean", "std", "p10", "p25", "p50", "p75", "p90"):
        if c in out.columns:
            out[c] = out[c].round(6)
    return out


def _join_sorted(s: frozenset, sep: str = "; ") -> str:
    if not s:
        return ""
    parts = [_sanitize_cell_text(str(x).strip()) for x in sorted(s, key=str)]
    parts = [p for p in parts if p]
    return sep.join(parts)


def _dataframe_for_export(df_sets: pd.DataFrame) -> pd.DataFrame:
    """Enterprise ID, Enterprise Name, Bucket, string columns for each dimension + counts."""
    out = df_sets.reset_index()
    if "Bucket" not in out.columns:
        out["Bucket"] = ""
    out["Bucket"] = out["Bucket"].fillna("").astype(str).map(_sanitize_cell_text)
    for col in DIMENSION_KEYS:
        out[f"{col}_str"] = out[col].map(_join_sorted)
        out[f"{col}_n"] = out[col].map(len)
    return out


def _bucket_label_frequency(df_sets: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct Bucket label; n_enterprises with that reference bucket."""
    s = df_sets["Bucket"].fillna("").astype(str).map(lambda x: _sanitize_cell_text(x.strip()) or "(blank)")
    vc = s.value_counts()
    return pd.DataFrame({"value": vc.index.astype(str), "n_enterprises": vc.values})


def _dimension_value_frequency(df_sets: pd.DataFrame, col: str) -> pd.DataFrame:
    """Distinct values in that dimension's sets × count of enterprises whose set contains the value."""
    ctr: Counter[str] = Counter()
    for s in df_sets[col]:
        for t in s:
            tok = _sanitize_cell_text(str(t).strip())
            if tok:
                ctr[tok] += 1
    rows = [{"value": k, "n_enterprises": v} for k, v in ctr.most_common()]
    return pd.DataFrame(rows)


def _frequency_sheet_name(dim: str) -> str:
    base = f"frequency_{dim}"
    if len(base) > 31:
        return base[:31]
    return base


def _distinct_values_in_column(df_sets: pd.DataFrame, col: str) -> set[str]:
    out: set[str] = set()
    for s in df_sets[col]:
        for t in s:
            tok = _sanitize_cell_text(str(t).strip())
            if tok:
                out.add(tok)
    return out


def _superset_for_enterprises(df_sets: pd.DataFrame, col: str, enterprise_ids: tuple[str, ...]) -> set[str]:
    out: set[str] = set()
    for eid in enterprise_ids:
        eid = str(eid).strip()
        if eid not in df_sets.index:
            continue
        for t in df_sets.loc[eid, col]:
            tok = _sanitize_cell_text(str(t).strip())
            if tok:
                out.add(tok)
    return out


def _tagged_distinct_universe_and_superset(
    df_sets: pd.DataFrame,
    enterprise_ids: tuple[str, ...],
    dimension_keys: list[str],
) -> tuple[set[str], set[str]]:
    """Atoms are 'dimension|value' so the same token in two dims counts twice."""
    universe: set[str] = set()
    for col in dimension_keys:
        for t in _distinct_values_in_column(df_sets, col):
            universe.add(f"{col}|{t}")
    superset: set[str] = set()
    for col in dimension_keys:
        for t in _superset_for_enterprises(df_sets, col, enterprise_ids):
            superset.add(f"{col}|{t}")
    return universe, superset


def first_five_superset_coverage_df(
    df_sets: pd.DataFrame,
    enterprise_ids: tuple[str, ...] = FIRST_FIVE_ENTERPRISE_IDS,
) -> pd.DataFrame:
    """
    Per dimension: population distinct count vs union of the cohort; % = superset / universe.
    Final row: tagged union across all dimensions (see module docstring).
    """
    rows: list[dict[str, str | float | int]] = []
    for col in DIMENSION_KEYS:
        uni = _distinct_values_in_column(df_sets, col)
        sup = _superset_for_enterprises(df_sets, col, enterprise_ids)
        nu, ns = len(uni), len(sup)
        pct = round(100.0 * ns / nu, 2) if nu else (0.0 if ns == 0 else float("nan"))
        rows.append(
            {
                "dimension": col,
                "n_distinct_values_all_enterprises": nu,
                "n_distinct_in_cohort_superset": ns,
                "pct_of_universe_covered": pct,
            }
        )
    u_tag, s_tag = _tagged_distinct_universe_and_superset(df_sets, enterprise_ids, DIMENSION_KEYS)
    nut, nst = len(u_tag), len(s_tag)
    pct_t = round(100.0 * nst / nut, 2) if nut else float("nan")
    rows.append(
        {
            "dimension": "all_dimensions_tagged_union",
            "n_distinct_values_all_enterprises": nut,
            "n_distinct_in_cohort_superset": nst,
            "pct_of_universe_covered": pct_t,
        }
    )
    return pd.DataFrame(rows)


def first_five_enterprises_detail_df(
    base: pd.DataFrame,
    enterprise_ids: tuple[str, ...] = FIRST_FIVE_ENTERPRISE_IDS,
) -> pd.DataFrame:
    """One row per requested Enterprise ID (order preserved); missing IDs get a placeholder row."""
    rows: list[dict[str, str | int | float]] = []
    base = base.copy()
    base["_eid"] = base["Enterprise ID"].astype(str).str.strip()
    for eid in enterprise_ids:
        eid = str(eid).strip()
        sub = base.loc[base["_eid"] == eid]
        if len(sub) == 0:
            row: dict[str, str | int | float] = {
                "Enterprise ID": _sanitize_cell_text(eid),
                "Enterprise Name": "(not in scope — no mapped cnumber in spine)",
                "Bucket": "",
            }
            for k in DIMENSION_KEYS:
                row[f"{k}_values"] = ""
                row[f"{k}_n_distinct"] = 0
            rows.append(row)
            continue
        r = sub.iloc[0]
        row = {
            "Enterprise ID": _sanitize_cell_text(str(r["Enterprise ID"])),
            "Enterprise Name": _sanitize_cell_text(str(r.get("Enterprise Name", ""))),
            "Bucket": _sanitize_cell_text(str(r.get("Bucket", ""))),
        }
        for k in DIMENSION_KEYS:
            row[f"{k}_values"] = r[f"{k}_str"]
            row[f"{k}_n_distinct"] = int(r[f"{k}_n"])
        rows.append(row)
    return pd.DataFrame(rows)


def write_enterprise_dimension_workbook(
    root: Path,
    filename: str = "enterprise_configuration_by_dimension.xlsx",
) -> Path:
    df_sets, _enc = build_enterprise_dimension_sets()
    base = _dataframe_for_export(df_sets)
    base = base.sort_values("Enterprise ID", kind="stable")

    out_path = root / filename

    id_s = base["Enterprise ID"].fillna("").astype(str).map(_sanitize_cell_text)
    name_s = base["Enterprise Name"].fillna("").astype(str).map(_sanitize_cell_text)
    bucket_s = base["Bucket"]

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        _dimension_coverage_df(df_sets).to_excel(writer, sheet_name=COVERAGE_SHEET, index=False)
        _pairwise_jaccard_summary_restricted(df_sets).to_excel(
            writer, sheet_name=OVERVIEW_SHEET, index=False
        )

        cov5 = first_five_superset_coverage_df(df_sets, FIRST_FIVE_ENTERPRISE_IDS)
        det5 = first_five_enterprises_detail_df(base, FIRST_FIVE_ENTERPRISE_IDS)
        cov5.to_excel(writer, sheet_name=FIRST_FIVE_SHEET, index=False, startrow=0)
        gap = 2
        det5.to_excel(
            writer,
            sheet_name=FIRST_FIVE_SHEET,
            index=False,
            startrow=len(cov5) + gap + 1,
        )

        cov_c5 = first_five_superset_coverage_df(df_sets, COHORT_78_FIRST_FIVE)
        det_c5 = first_five_enterprises_detail_df(base, COHORT_78_FIRST_FIVE)
        cov_c_all = first_five_superset_coverage_df(df_sets, COHORT_78_ENTERPRISE_IDS)
        det_c_all = first_five_enterprises_detail_df(base, COHORT_78_ENTERPRISE_IDS)
        row_c = 0
        cov_c5.to_excel(writer, sheet_name=COHORT_78_SHEET, index=False, startrow=row_c)
        row_c += len(cov_c5) + 1 + gap
        det_c5.to_excel(writer, sheet_name=COHORT_78_SHEET, index=False, startrow=row_c)
        row_c += len(det_c5) + 1 + gap
        cov_c_all.to_excel(writer, sheet_name=COHORT_78_SHEET, index=False, startrow=row_c)
        row_c += len(cov_c_all) + 1 + gap
        det_c_all.to_excel(writer, sheet_name=COHORT_78_SHEET, index=False, startrow=row_c)

        cov_101_5 = first_five_superset_coverage_df(df_sets, COHORT_101_FIRST_FIVE)
        det_101_5 = first_five_enterprises_detail_df(base, COHORT_101_FIRST_FIVE)
        cov_101_all = first_five_superset_coverage_df(df_sets, COHORT_101_ENTERPRISE_IDS)
        det_101_all = first_five_enterprises_detail_df(base, COHORT_101_ENTERPRISE_IDS)
        row_101 = 0
        cov_101_5.to_excel(writer, sheet_name=COHORT_101_SHEET, index=False, startrow=row_101)
        row_101 += len(cov_101_5) + 1 + gap
        det_101_5.to_excel(writer, sheet_name=COHORT_101_SHEET, index=False, startrow=row_101)
        row_101 += len(det_101_5) + 1 + gap
        cov_101_all.to_excel(writer, sheet_name=COHORT_101_SHEET, index=False, startrow=row_101)
        row_101 += len(cov_101_all) + 1 + gap
        det_101_all.to_excel(writer, sheet_name=COHORT_101_SHEET, index=False, startrow=row_101)

        pd.DataFrame(
            {"Enterprise ID": id_s, "Enterprise Name": name_s, "Bucket": bucket_s}
        ).to_excel(writer, sheet_name="bucket_reference", index=False)

        for key in DIMENSION_KEYS:
            per = pd.DataFrame(
                {
                    "Enterprise ID": id_s,
                    "Enterprise Name": name_s,
                    "Bucket": bucket_s,
                    "values_concatenated": base[f"{key}_str"],
                    "n_distinct": base[f"{key}_n"],
                }
            )
            per.to_excel(writer, sheet_name=key, index=False)

        _bucket_label_frequency(df_sets).to_excel(writer, sheet_name="frequency_Bucket", index=False)

        for key in DIMENSION_KEYS:
            freq = _dimension_value_frequency(df_sets, key)
            freq.to_excel(writer, sheet_name=_frequency_sheet_name(key), index=False)

    return out_path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    path = write_enterprise_dimension_workbook(root)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
