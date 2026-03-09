"""
Greedy function sequencing: choose which functions to modernize first to unlock
customers (and ARR) as quickly as possible.

INPUT DATA
----------
You can provide data in either of two ways:

Option A — Two Excel files
  • Function mapping: one sheet with columns (names must match exactly):
      ENTERPRISE ID   — unique ID per customer
      NAME            — display name for the customer
      FUNCTION        — function name (e.g. "Payroll", "Inventory")
      USAGE_PATTERN   — "1 - daily" | "2 - weekly" | "3 - monthly" (others ignored)
    COMPLETED is taken from WorkflowMapping (see below). Functions not listed in WorkflowMapping are out of scope.
    Multiple rows per (ENTERPRISE ID, FUNCTION) are allowed; all must be "Yes"
    (or "Partial" if treat_partial_as_done) for that pair to count as done.

  • Customer ARR: one sheet with columns:
      ENTERPRISE ID   — same IDs as in function mapping
      ARR             — numeric annual recurring revenue (non-numeric → 0)

Option B — One Excel file with multiple sheets
  • Default sheet names:
      "FunctionMapping"  — function usage/completion data
      "EnterpriseARR"    — ENTERPRISE ID and ARR (one row per customer)
      "Rooftops"         — (optional) ENTERPRISE ID and ROOFTOPS (locations per customer)
      "WorkflowMapping"  — WORKFLOW, FUNCTION, and COMPLETED; defines in-scope functions (only these are considered)
                           and completion status per function; functions not listed here are out of scope
  • Use --input-excel path.xlsx and optionally --sheet-function-mapping,
    --sheet-enterprise-arr, --sheet-rooftops, --sheet-workflow-mapping to override sheet names.
  • If a Rooftops sheet exists, it is merged into the ARR data by ENTERPRISE ID.

Naming: blend weights are always "Xcust_Yarr" = X% weight on customers, Y% on ARR.

For .xlsx files, openpyxl is required: pip install pandas openpyxl
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np

from typing import Tuple, List, Set, Dict, Optional


def _read_excel(path: Path) -> pd.DataFrame:
    """Read first sheet of an Excel file; use openpyxl for .xlsx."""
    path = Path(path)
    kwargs = {}
    if path.suffix.lower() in (".xlsx", ".xlsm"):
        kwargs["engine"] = "openpyxl"
    return pd.read_excel(path, **kwargs)


def read_excel_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    """Read a specific sheet by name; use openpyxl for .xlsx."""
    path = Path(path)
    kwargs = {"sheet_name": sheet_name}
    if path.suffix.lower() in (".xlsx", ".xlsm"):
        kwargs["engine"] = "openpyxl"
    return pd.read_excel(path, **kwargs)


def get_excel_sheet_names(path: Path) -> List[str]:
    """Return list of sheet names in an Excel file."""
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xlsm"):
        return pd.ExcelFile(path, engine="openpyxl").sheet_names
    return pd.ExcelFile(path).sheet_names


REQUIRED_FUNCTION_COLS = ("ENTERPRISE ID", "NAME", "FUNCTION", "USAGE_PATTERN")
REQUIRED_ARR_COLS = ("ENTERPRISE ID", "ARR")
OPTIONAL_ARR_COLS = ("ROOFTOPS",)  # if present, used in customer-facing outputs


# Optional column name aliases (input header -> canonical name)
COLUMN_ALIASES = {
    "COMPLETE": "COMPLETED",
    "Complete": "COMPLETED",
    "EnterpriseID": "ENTERPRISE ID",
    "Enterprise Id": "ENTERPRISE ID",
}


def _normalize_columns(df: pd.DataFrame, canonical: tuple) -> pd.DataFrame:
    """Rename columns to canonical names using case-insensitive strip matching (and collapse inner spaces)."""
    df = df.copy()
    # Apply explicit aliases first (e.g. COMPLETE -> COMPLETED, EnterpriseID -> ENTERPRISE ID)
    for alias, canonical_name in COLUMN_ALIASES.items():
        if alias in df.columns and canonical_name not in df.columns:
            df = df.rename(columns={alias: canonical_name})
    # Build lookup: normalized key (lower, collapse spaces) -> original column name
    def norm_key(s: str) -> str:
        return " ".join(str(s).strip().lower().split())
    col_by_norm = {norm_key(c): c for c in df.columns}
    rename = {}
    for can in canonical:
        key = norm_key(can)
        if can in df.columns:
            continue
        if key in col_by_norm:
            rename[col_by_norm[key]] = can
    if rename:
        df = df.rename(columns=rename)
    return df


def _validate_input_dfs(df_mapping: pd.DataFrame, df_arr: pd.DataFrame, require_completed: bool = True) -> None:
    """Raise ValueError with clear message if required columns are missing."""
    missing_mapping = [c for c in REQUIRED_FUNCTION_COLS if c not in df_mapping.columns]
    if require_completed and "COMPLETED" not in df_mapping.columns:
        missing_mapping.append("COMPLETED (must be in FunctionMapping or supplied via WorkflowMapping)")
    missing_arr = [c for c in REQUIRED_ARR_COLS if c not in df_arr.columns]
    if missing_mapping or missing_arr:
        parts = []
        if missing_mapping:
            parts.append(f"Function mapping sheet needs columns: {list(REQUIRED_FUNCTION_COLS)}; missing: {missing_mapping}")
        if missing_arr:
            parts.append(f"Customer ARR sheet needs columns: {list(REQUIRED_ARR_COLS)}; missing: {missing_arr}")
        raise ValueError("\n".join(parts))


def _reduce_to_most_frequent_usage(
    d: pd.DataFrame,
    enterprise_col: str,
    function_col: str,
    usage_col: str,
) -> pd.DataFrame:
    """Keep one row per (enterprise, function): the row with the most frequent USAGE_PATTERN (mode)."""
    mode_ser = (
        d.groupby([enterprise_col, function_col])[usage_col]
        .apply(lambda s: s.value_counts().index[0])
        .reset_index()
    )
    mode_ser = mode_ser.rename(columns={usage_col: "_usage_mode"})
    d = d.merge(mode_ser, on=[enterprise_col, function_col], how="left")
    d = d[d[usage_col] == d["_usage_mode"]].drop_duplicates(
        subset=[enterprise_col, function_col], keep="first"
    ).drop(columns=["_usage_mode"])
    return d


def _prepare_enterprise_arr(
    enterprise_arr: pd.DataFrame,
    arr_threshold: float = 0.0,
    enterprise_col: str = "ENTERPRISE ID",
    arr_col: str = "ARR",
) -> Tuple[pd.DataFrame, Set, Dict, Optional[Dict]]:
    """Build ea (subset of cols), eligible set, arr_map, rooftops_map (or None)."""
    arr_cols = [enterprise_col, arr_col]
    if "ROOFTOPS" in enterprise_arr.columns:
        arr_cols = [enterprise_col, arr_col, "ROOFTOPS"]
    ea = enterprise_arr[arr_cols].copy()
    ea[arr_col] = pd.to_numeric(ea[arr_col], errors="coerce").fillna(0.0)
    eligible = set(ea.loc[ea[arr_col] >= arr_threshold, enterprise_col].unique())
    arr_map = ea.set_index(enterprise_col)[arr_col].to_dict()
    rooftops_map = None
    if "ROOFTOPS" in ea.columns:
        ea["ROOFTOPS"] = pd.to_numeric(ea["ROOFTOPS"], errors="coerce").fillna(0).astype(int)
        rooftops_map = ea.set_index(enterprise_col)["ROOFTOPS"].to_dict()
    return ea, eligible, arr_map, rooftops_map


def _compute_name_map(d: pd.DataFrame, enterprise_col: str, name_col: str) -> Dict:
    """Enterprise -> most frequent name (mode)."""
    return (
        d.loc[d[name_col].ne(""), [enterprise_col, name_col]]
        .groupby(enterprise_col)[name_col]
        .agg(lambda s: s.value_counts().index[0] if len(s) else "")
        .to_dict()
    )


def _prepare_mapping_ef(
    df_mapping: pd.DataFrame,
    eligible_enterprises: Set,
    valid_patterns: Optional[tuple],
    treat_partial_as_done: bool,
    enterprise_col: str = "ENTERPRISE ID",
    name_col: str = "NAME",
    function_col: str = "FUNCTION",
    completed_col: str = "COMPLETED",
    usage_col: str = "USAGE_PATTERN",
) -> Tuple[pd.DataFrame, Dict]:
    """Filter mapping to eligible + valid_patterns (if not None), collapse to one row per (ent, func); return (ef with is_done, name_map)."""
    d = df_mapping[[enterprise_col, name_col, function_col, completed_col, usage_col]].copy()
    d = d[d[enterprise_col].isin(eligible_enterprises)]
    for col in (enterprise_col, function_col, completed_col, usage_col, name_col):
        d[col] = d[col].astype(str).str.strip()
    d = d[d[completed_col].ne("Outside baseline")]
    if valid_patterns is not None:
        d = d[d[usage_col].isin(valid_patterns)]
    d = d.dropna(subset=[enterprise_col, function_col])
    name_map = _compute_name_map(d, enterprise_col, name_col)
    d["_done_flag"] = d[completed_col].isin(["Yes", "Partial"]) if treat_partial_as_done else d[completed_col].eq("Yes")
    d = _reduce_to_most_frequent_usage(d, enterprise_col, function_col, usage_col)
    ef = d[[enterprise_col, function_col, "_done_flag"]].rename(columns={"_done_flag": "is_done"})
    return ef, name_map


def greedy_unlock_functions_with_arr_outputs(
    df: pd.DataFrame,
    enterprise_arr: pd.DataFrame,
    k: int = 10,
    arr_threshold: float = 0,
    enterprise_col: str = "ENTERPRISE ID",
    name_col: str = "NAME",
    function_col: str = "FUNCTION",
    completed_col: str = "COMPLETED",
    usage_col: str = "USAGE_PATTERN",
    arr_col: str = "ARR",
    valid_patterns=("1 - daily", "2 - weekly", "3 - monthly"),
    treat_partial_as_done: bool = False,
    objective: str = "customers",  # "customers" | "arr" | "blend" | "rooftops"
    blend_weight_customers: float = 0.5,  # for "blend": weight on customers (1 - this = weight on ARR)
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], Set[str]]:
    """
    Same greedy algorithm, but returns TWO outputs shaped exactly as requested:

    Output 1 (steps_df):
      - step
      - function_to_modernize
      - customers_unlocked
      - cumulative_customers_unlocked

    Output 2 (customer_unlock_df):
      - step
      - function_to_modernize
      - customer_name
      - total_arr_of_customer
      - cumulative_arr_unlocked

    Plus: selected_functions (ordered) and unlocked (final set).

    Objective options:
      - "customers": maximize number of customers completed.
      - "arr": maximize ARR unlocked.
      - "rooftops": maximize number of rooftops (locations) unlocked; uses customer count if no rooftops data.
      - "blend": weighted combination; blend_weight_customers in [0,1] = % weight on customers (rest = ARR).
    """

    # Normalize legacy objective names
    if objective == "logo_progress":
        objective = "customers"
    elif objective == "arr_progress":
        objective = "arr"

    # ---- 0) ARR eligibility + mapping collapsed to (ent, func) with is_done ----
    _, eligible_enterprises, arr_map, rooftops_map = _prepare_enterprise_arr(
        enterprise_arr, arr_threshold, enterprise_col, arr_col
    )
    ef, name_map = _prepare_mapping_ef(
        df, eligible_enterprises, valid_patterns, treat_partial_as_done,
        enterprise_col, name_col, function_col, completed_col, usage_col,
    )
    blockers = ef.loc[~ef["is_done"], [enterprise_col, function_col]]

    all_enterprises = set(ef[enterprise_col].unique())
    enterprises_with_blockers = set(blockers[enterprise_col].unique())
    already_unlocked = all_enterprises - enterprises_with_blockers

    blockers_by_ent = (
        blockers.groupby(enterprise_col)[function_col]
        .apply(lambda s: set(s.values))
        .to_dict()
    )
    remaining: Dict = {e: len(fs) for e, fs in blockers_by_ent.items()}

    ents_by_func = (
        blockers.groupby(function_col)[enterprise_col]
        .apply(lambda s: s.unique().tolist())
        .to_dict()
    )
    candidate_functions = set(ents_by_func.keys())

    # ---- 3) Greedy selection with progress scoring ----
    selected_functions: List[str] = []
    unlocked: Set[str] = set(already_unlocked)

    # Step 0: already-unlocked customers (no functions remaining) and their ARR/rooftops
    initial_cumulative_arr = float(sum(arr_map.get(e, 0.0) for e in already_unlocked))
    initial_cumulative_rooftops = int(sum(rooftops_map.get(e, 0) for e in already_unlocked)) if rooftops_map else 0

    steps_rows = [
        {
            "step": 0,
            "function_to_modernize": "",
            "customers_unlocked": len(already_unlocked),
            "cumulative_customers_unlocked": len(already_unlocked),
        }
    ]
    enterprise_rows = []
    for e in sorted(already_unlocked):
        row = {
            "step": 0,
            "ENTERPRISE ID": e,
            "function_to_modernize": "",
            "customer_name": name_map.get(e, ""),
            "total_arr_of_customer": float(arr_map.get(e, 0.0)),
            "cumulative_arr_unlocked": initial_cumulative_arr,
        }
        if rooftops_map is not None:
            row["rooftops"] = int(rooftops_map.get(e, 0))
        enterprise_rows.append(row)

    cumulative_customers_unlocked = len(unlocked)
    cumulative_arr_unlocked = initial_cumulative_arr

    # Greedy steps: always output a row for each step so there are no gaps (step 1, 2, 3, ...)
    for step in range(1, k + 1):
        if not candidate_functions:
            steps_rows.append({
                "step": step,
                "function_to_modernize": "",
                "customers_unlocked": 0,
                "cumulative_customers_unlocked": len(unlocked),
            })
            break

        best_f: Optional[str] = None
        best_score: float = -1.0

        # For "blend" we need per-candidate customer_score and arr_score, then normalize and combine
        use_blend = objective == "blend"
        candidate_scores_customers: Dict[str, float] = {}
        candidate_scores_arr: Dict[str, float] = {}

        for f in candidate_functions:
            customer_score = 0.0
            arr_score = 0.0
            rooftops_score = 0.0
            for e in ents_by_func.get(f, []):
                if e in unlocked:
                    continue
                r = remaining.get(e, 0)
                if r <= 0:
                    continue
                customer_score += 1.0 / r
                arr_score += float(arr_map.get(e, 0.0)) / r
                if rooftops_map is not None:
                    rooftops_score += float(rooftops_map.get(e, 0)) / r

            if use_blend:
                candidate_scores_customers[f] = customer_score
                candidate_scores_arr[f] = arr_score
            else:
                if objective == "arr":
                    score = arr_score
                elif objective == "rooftops":
                    score = rooftops_score if rooftops_map is not None else customer_score
                else:
                    score = customer_score  # "customers" or legacy "logo_progress"
                if score > best_score:
                    best_score = score
                    best_f = f

        if use_blend and candidate_scores_customers:
            max_c = max(candidate_scores_customers.values()) or 1.0
            max_a = max(candidate_scores_arr.values()) or 1.0
            for f in candidate_functions:
                nc = candidate_scores_customers.get(f, 0.0) / max_c
                na = candidate_scores_arr.get(f, 0.0) / max_a
                score = blend_weight_customers * nc + (1.0 - blend_weight_customers) * na
                if score > best_score:
                    best_score = score
                    best_f = f

        if best_f is None or best_score <= 0:
            steps_rows.append({
                "step": step,
                "function_to_modernize": "",
                "customers_unlocked": 0,
                "cumulative_customers_unlocked": len(unlocked),
            })
            break

        selected_functions.append(best_f)
        candidate_functions.discard(best_f)

        newly_unlocked = []
        step_arr_unlocked = 0.0

        for e in ents_by_func.get(best_f, []):
            if e in unlocked:
                continue
            if e not in remaining:
                continue

            remaining[e] -= 1
            if remaining[e] == 0:
                unlocked.add(e)
                newly_unlocked.append(e)
                step_arr_unlocked += float(arr_map.get(e, 0.0))

        # Update cumulative totals
        cumulative_customers_unlocked = len(unlocked)
        cumulative_arr_unlocked += float(step_arr_unlocked)

        # Output 1: step summary
        steps_rows.append({
            "step": step,
            "function_to_modernize": best_f,
            "customers_unlocked": len(newly_unlocked),
            "cumulative_customers_unlocked": cumulative_customers_unlocked,
        })

        # Output 2: per-customer unlock rows with cumulative ARR (and rooftops if present)
        for e in sorted(newly_unlocked):
            row = {
                "step": step,
                "ENTERPRISE ID": e,
                "function_to_modernize": best_f,
                "customer_name": name_map.get(e, ""),
                "total_arr_of_customer": float(arr_map.get(e, 0.0)),
                "cumulative_arr_unlocked": float(cumulative_arr_unlocked),
            }
            if rooftops_map is not None:
                row["rooftops"] = int(rooftops_map.get(e, 0))
            enterprise_rows.append(row)

        # Stop if all customers unlocked (we already appended this step)
        if len(unlocked) == len(all_enterprises):
            break

    steps_df = pd.DataFrame(steps_rows)
    enterprise_unlock_df = pd.DataFrame(enterprise_rows)

    return steps_df, enterprise_unlock_df, selected_functions, unlocked


def _dataframe_to_json_records(df: pd.DataFrame) -> list:
    """Convert DataFrame to JSON-serializable list of dicts (handles NaN, numpy types)."""
    if df is None or df.empty:
        return []
    return json.loads(df.to_json(orient="records", date_format="iso"))


# Dollar columns to round to nearest dollar in outputs
DOLLAR_COLUMNS = ("total_arr_of_customer", "cumulative_arr_unlocked", "arr_unlocked", "ARR")

# Display column names for Excel: capitalized and intuitive (internal name -> display name)
DISPLAY_COLUMNS = {
    "step": "Step",
    "function_to_modernize": "Function To Modernize",
    "customers_unlocked": "Customers Unlocked (This Step)",
    "cumulative_customers_unlocked": "Cumulative Customers Unlocked",
    "rooftops_unlocked": "Rooftops Unlocked (This Step)",
    "cumulative_rooftops_unlocked": "Cumulative Rooftops Unlocked",
    "customer_name": "Customer Name",
    "total_arr_of_customer": "Total ARR Of Customer ($)",
    "cumulative_arr_unlocked": "Cumulative ARR Unlocked ($)",
    "arr_unlocked": "ARR Unlocked ($)",
    "rooftops": "Rooftops (Locations)",
    "scope": "Scope",
    "ENTERPRISE ID": "Enterprise ID",
    "total_functions": "Total Functions (In Scope)",
    "total_functions_in_scope": "Total Functions In Scope",
    "functions_remaining": "Functions Remaining",
    "pct_remaining": "% Remaining",
    "functions_not_modernized": "Functions Not Yet Modernized (Daily/Weekly/Monthly)",
    "function_completed_in_step": "Function Completed (Step)",
    "workflow": "Workflow",
}


def _format_sheet_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to display names for Excel (capitalized, intuitive)."""
    rename = {c: DISPLAY_COLUMNS[c] for c in df.columns if c in DISPLAY_COLUMNS}
    if not rename:
        return df
    return df.rename(columns=rename)


def _round_dollar_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Round dollar columns to nearest integer (whole dollars)."""
    out = df.copy()
    for col in DOLLAR_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).round(0).astype(int)
    return out


def build_tranche_summary(tranche_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a step-level rollup with columns (in order):
    step, function_to_modernize, customers_unlocked, cumulative_customers_unlocked,
    rooftops_unlocked, cumulative_rooftops_unlocked, arr_unlocked, cumulative_arr_unlocked.
    """
    if tranche_df.empty:
        return tranche_df.copy()
    # Aggregate by existing column names (works across pandas versions), then rename
    agg_dict = {
        "customer_name": "nunique",
        "total_arr_of_customer": "sum",
        "cumulative_arr_unlocked": "max",
    }
    if "rooftops" in tranche_df.columns:
        agg_dict["rooftops"] = "sum"
    out = (
        tranche_df
        .groupby(["step", "function_to_modernize"], as_index=False)
        .agg(agg_dict)
        .sort_values(["step"])
    )
    out = out.rename(columns={
        "customer_name": "customers_unlocked",
        "total_arr_of_customer": "arr_unlocked",
    })
    if "rooftops" in out.columns:
        out = out.rename(columns={"rooftops": "rooftops_unlocked"})
    else:
        out["rooftops_unlocked"] = 0
    out["cumulative_customers_unlocked"] = out["customers_unlocked"].cumsum()
    out["cumulative_rooftops_unlocked"] = out["rooftops_unlocked"].cumsum()
    out = out[[
        "step",
        "function_to_modernize",
        "customers_unlocked",
        "cumulative_customers_unlocked",
        "rooftops_unlocked",
        "cumulative_rooftops_unlocked",
        "arr_unlocked",
        "cumulative_arr_unlocked",
    ]]
    return out


def build_enterprise_unlock_view(enterprise_unlock_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per enterprise: ENTERPRISE ID, Name, Rooftops, ARR, Step Unlocked, Function Completed (Step).
    Requires enterprise_unlock_df to have ENTERPRISE ID (and optionally rooftops).
    """
    if enterprise_unlock_df.empty:
        return pd.DataFrame(columns=["ENTERPRISE ID", "Name", "Rooftops", "ARR", "Step Unlocked", "function_completed_in_step"])
    if "ENTERPRISE ID" not in enterprise_unlock_df.columns:
        return pd.DataFrame()
    # One row per enterprise (they appear once when unlocked)
    out = enterprise_unlock_df.drop_duplicates(subset=["ENTERPRISE ID"], keep="first").copy()
    out = out.rename(columns={
        "customer_name": "Name",
        "total_arr_of_customer": "ARR",
        "step": "Step Unlocked",
        "function_to_modernize": "function_completed_in_step",
    })
    if "rooftops" in out.columns:
        out = out.rename(columns={"rooftops": "Rooftops"})
    cols = ["ENTERPRISE ID", "Name", "Rooftops", "ARR", "Step Unlocked", "function_completed_in_step"]
    out = out[[c for c in cols if c in out.columns]]
    return out


def build_enterprise_function_usage_matrix(
    df_mapping: pd.DataFrame,
    enterprise_arr: pd.DataFrame,
    workflow_map: Dict[str, str],
    completed_map: Optional[Dict[str, str]] = None,
    enterprise_col: str = "ENTERPRISE ID",
    name_col: str = "NAME",
    function_col: str = "FUNCTION",
    usage_col: str = "USAGE_PATTERN",
    arr_col: str = "ARR",
) -> pd.DataFrame:
    """
    Matrix: one row per enterprise; columns = Enterprise ID, Name, Rooftops, ARR, then one column
    per in-scope function. Cell value = usage pattern (e.g. "1 - daily", "2 - weekly") or empty.
    Row 1 = column names, row 2 = workflow, row 3 = completion status (Yes/No/Partial), then data.
    Enterprises with ID "0" are excluded. Write with header=False.
    """
    if df_mapping.empty or enterprise_col not in df_mapping.columns or function_col not in df_mapping.columns:
        return pd.DataFrame()

    d = df_mapping[[enterprise_col, name_col, function_col, usage_col]].copy()
    for col in (enterprise_col, function_col, usage_col, name_col):
        d[col] = d[col].astype(str).str.strip()
    d = d.dropna(subset=[enterprise_col, function_col])

    in_scope_functions = sorted(d[function_col].unique().tolist())
    enterprises = sorted(e for e in d[enterprise_col].unique().tolist() if str(e).strip() != "0")
    usage_agg = (
        d.groupby([enterprise_col, function_col])[usage_col]
        .apply(lambda s: s.value_counts().index[0] if len(s) else "")
        .to_dict()
    )
    name_map = _compute_name_map(d, enterprise_col, name_col)
    ea, _, _, _ = _prepare_enterprise_arr(enterprise_arr, -float("inf"), enterprise_col, arr_col)
    ea[enterprise_col] = ea[enterprise_col].astype(str).str.strip()
    ea = ea.set_index(enterprise_col)
    arr_by_ent = ea[arr_col].to_dict()
    rooftops_by_ent = ea["ROOFTOPS"].to_dict() if "ROOFTOPS" in ea.columns else {}

    data_rows = []
    for ent in enterprises:
        row = [
            ent,
            name_map.get(ent, ""),
            int(rooftops_by_ent.get(ent, 0)),
            int(round(float(arr_by_ent.get(ent, 0)), 0)),
        ]
        for func in in_scope_functions:
            row.append(usage_agg.get((ent, func), ""))
        data_rows.append(row)

    # Row 1 = column names, Row 2 = workflow, Row 3 = completion status (Yes/No/Partial)
    header_row_1 = ["Enterprise ID", "Name", "Rooftops", "ARR"] + in_scope_functions
    header_row_2 = ["", "", "", ""] + [workflow_map.get(f, "") for f in in_scope_functions]
    header_row_3 = ["", "", "", ""] + [completed_map.get(f, "") for f in in_scope_functions] if completed_map else ["", "", "", ""] + [""] * len(in_scope_functions)
    all_rows = [header_row_1, header_row_2, header_row_3] + data_rows
    return pd.DataFrame(all_rows)


# Scope definitions for "remaining to modernize" views (by frequency; match input format)
SCOPE_DAILY_WEEKLY = ("1 - daily", "2 - weekly")
SCOPE_DAILY_WEEKLY_MONTHLY = ("1 - daily", "2 - weekly", "3 - monthly")


def build_enterprise_functions_summary(
    df_mapping: pd.DataFrame,
    enterprise_arr: pd.DataFrame,
    arr_threshold: float = 0.0,
    enterprise_col: str = "ENTERPRISE ID",
    name_col: str = "NAME",
    function_col: str = "FUNCTION",
    completed_col: str = "COMPLETED",
    usage_col: str = "USAGE_PATTERN",
    arr_col: str = "ARR",
    treat_partial_as_done: bool = False,
    valid_patterns: tuple = SCOPE_DAILY_WEEKLY_MONTHLY,
) -> pd.DataFrame:
    """
    One row per enterprise: ENTERPRISE ID, Name, Rooftops, ARR, functions_remaining,
    total_functions_in_scope, pct_remaining (functions remaining as % of total in scope).
    valid_patterns: which usage patterns count (e.g. SCOPE_DAILY_WEEKLY or SCOPE_DAILY_WEEKLY_MONTHLY).
    """
    ea, eligible, _, _ = _prepare_enterprise_arr(enterprise_arr, arr_threshold, enterprise_col, arr_col)
    ef, name_map = _prepare_mapping_ef(
        df_mapping, eligible, valid_patterns, treat_partial_as_done,
        enterprise_col, name_col, function_col, completed_col, usage_col,
    )
    total_per_ent = ef.groupby(enterprise_col).size()
    remaining_per_ent = ef.loc[~ef["is_done"]].groupby(enterprise_col).size()

    all_ents = sorted(set(ef[enterprise_col].unique()))
    rows = []
    for ent in all_ents:
        total = int(total_per_ent.get(ent, 0))
        remaining = int(remaining_per_ent.get(ent, 0))
        pct = (100.0 * remaining / total) if total else 0.0
        rows.append({
            "ENTERPRISE ID": ent,
            "Name": name_map.get(ent, ""),
            "functions_remaining": remaining,
            "total_functions_in_scope": total,
            "pct_remaining": round(pct, 2),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Add Rooftops and ARR from enterprise_arr
    out["ARR"] = out["ENTERPRISE ID"].map(ea.set_index(enterprise_col)[arr_col])
    out["ARR"] = pd.to_numeric(out["ARR"], errors="coerce").fillna(0)
    if "ROOFTOPS" in ea.columns:
        out["Rooftops"] = out["ENTERPRISE ID"].map(ea.set_index(enterprise_col)["ROOFTOPS"])
        out["Rooftops"] = pd.to_numeric(out["Rooftops"], errors="coerce").fillna(0).astype(int)
    else:
        out["Rooftops"] = 0

    out = out[["ENTERPRISE ID", "Name", "Rooftops", "ARR", "functions_remaining", "total_functions_in_scope", "pct_remaining"]]
    return out.reset_index(drop=True)


def build_customer_remaining_by_scope(
    df_mapping: pd.DataFrame,
    enterprise_arr: pd.DataFrame,
    arr_threshold: float = 0.0,
    enterprise_col: str = "ENTERPRISE ID",
    name_col: str = "NAME",
    function_col: str = "FUNCTION",
    completed_col: str = "COMPLETED",
    usage_col: str = "USAGE_PATTERN",
    arr_col: str = "ARR",
    treat_partial_as_done: bool = False,
) -> pd.DataFrame:
    """
    Build a DataFrame with three views of how much each customer has left to fully
    modernize, by frequency scope:

      - daily_weekly: only daily + weekly function usage
      - daily_weekly_monthly: daily + weekly + monthly
      - all: all function frequencies

    Each view has: ENTERPRISE ID, customer_name, total_functions (in scope),
    functions_remaining, and pct_remaining.
    Sorted by scope, then by functions_remaining ascending (least left first).
    """
    _, eligible, _, _ = _prepare_enterprise_arr(enterprise_arr, arr_threshold, enterprise_col, arr_col)
    scopes = [
        ("daily_weekly", SCOPE_DAILY_WEEKLY),
        ("daily_weekly_monthly", SCOPE_DAILY_WEEKLY_MONTHLY),
        ("all", None),  # None = all usage patterns
    ]
    rows = []
    for scope_name, scope_patterns in scopes:
        ef, name_map = _prepare_mapping_ef(
            df_mapping, eligible, scope_patterns, treat_partial_as_done,
            enterprise_col, name_col, function_col, completed_col, usage_col,
        )
        if ef.empty:
            continue
        total_per_ent = ef.groupby(enterprise_col).size()
        remaining_per_ent = ef.loc[~ef["is_done"]].groupby(enterprise_col).size()
        for ent in sorted(ef[enterprise_col].unique()):
            total = int(total_per_ent[ent])
            remaining = int(remaining_per_ent.get(ent, 0))
            pct = (100.0 * remaining / total) if total else 0.0
            rows.append({
                "scope": scope_name,
                "ENTERPRISE ID": ent,
                "customer_name": name_map.get(ent, ""),
                "total_functions": total,
                "functions_remaining": remaining,
                "pct_remaining": round(pct, 2),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Sort: by scope (fixed order), then by least remaining first, then by pct
    scope_order = {"daily_weekly": 0, "daily_weekly_monthly": 1, "all": 2}
    out["_scope_ord"] = out["scope"].map(scope_order)
    out = out.sort_values(["_scope_ord", "functions_remaining", "pct_remaining"]).drop(columns=["_scope_ord"])

    # Add rooftops if present in enterprise ARR
    if "ROOFTOPS" in enterprise_arr.columns:
        rooftops_ser = enterprise_arr.set_index(enterprise_col)["ROOFTOPS"]
        rooftops_ser = pd.to_numeric(rooftops_ser, errors="coerce").fillna(0).astype(int)
        out["rooftops"] = out["ENTERPRISE ID"].map(rooftops_ser).fillna(0).astype(int)

    return out.reset_index(drop=True)


def build_customer_functions_not_modernized(
    df_mapping: pd.DataFrame,
    enterprise_arr: pd.DataFrame,
    arr_threshold: float = 0.0,
    enterprise_col: str = "ENTERPRISE ID",
    name_col: str = "NAME",
    function_col: str = "FUNCTION",
    completed_col: str = "COMPLETED",
    usage_col: str = "USAGE_PATTERN",
    arr_col: str = "ARR",
    treat_partial_as_done: bool = False,
) -> pd.DataFrame:
    """
    For Daily/Weekly/Monthly only: one row per customer with a list of functions
    not yet modernized (comma-separated). Same eligibility and completion rules
    as the main unlock logic.
    """
    _, eligible, _, _ = _prepare_enterprise_arr(enterprise_arr, arr_threshold, enterprise_col, arr_col)
    ef, name_map = _prepare_mapping_ef(
        df_mapping, eligible, SCOPE_DAILY_WEEKLY_MONTHLY, treat_partial_as_done,
        enterprise_col, name_col, function_col, completed_col, usage_col,
    )
    not_done = ef.loc[~ef["is_done"], [enterprise_col, function_col]]
    functions_list = (
        not_done.groupby(enterprise_col)[function_col]
        .apply(lambda s: ", ".join(sorted(s.unique())))
        .to_dict()
    )
    all_ents = set(ef[enterprise_col].unique())
    rooftops_map = None
    if "ROOFTOPS" in enterprise_arr.columns:
        rooftops_ser = pd.to_numeric(enterprise_arr.set_index(enterprise_col)["ROOFTOPS"], errors="coerce").fillna(0).astype(int)
        rooftops_map = rooftops_ser.to_dict()

    rows = []
    for ent in sorted(all_ents):
        row = {
            "ENTERPRISE ID": ent,
            "customer_name": name_map.get(ent, ""),
            "functions_not_modernized": funcs_str if (funcs_str := functions_list.get(ent, "")) else "(none — all complete for Daily/Weekly/Monthly)",
        }
        if rooftops_map is not None:
            row["rooftops"] = int(rooftops_map.get(ent, 0))
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Greedy function sequencing with ARR unlock outputs.",
        epilog="Provide an input path (positional or --input-excel) or both --function-mapping and --enterprise-arr.",
    )
    parser.add_argument(
        "input_excel_path",
        nargs="?",
        default=None,
        metavar="INPUT_EXCEL",
        help="Path to single Excel file with two sheets (e.g. CDK_Input.xlsx). Can also use --input-excel.",
    )
    parser.add_argument(
        "--input-excel",
        type=Path,
        metavar="PATH",
        help="Single Excel file with two sheets (FunctionMapping + EnterpriseARR). Same as positional INPUT_EXCEL.",
    )
    parser.add_argument(
        "--function-mapping",
        help="Path to function mapping Excel (use with --enterprise-arr instead of --input-excel).",
    )
    parser.add_argument(
        "--enterprise-arr",
        help="Path to customer ARR Excel (use with --function-mapping instead of --input-excel).",
    )
    parser.add_argument(
        "--sheet-function-mapping",
        default="FunctionMapping",
        help="Sheet name for function data when using --input-excel (default: FunctionMapping).",
    )
    parser.add_argument(
        "--sheet-enterprise-arr",
        default="EnterpriseARR",
        help="Sheet name for customer ARR data when using --input-excel (default: EnterpriseARR).",
    )
    parser.add_argument(
        "--sheet-rooftops",
        default="Rooftops",
        help="Sheet name for Rooftops (ENTERPRISE ID, ROOFTOPS) when using --input-excel (default: Rooftops).",
    )
    parser.add_argument(
        "--sheet-workflow-mapping",
        default="WorkflowMapping",
        help="Sheet name for WorkflowMapping (WORKFLOW, FUNCTION) when using --input-excel (default: WorkflowMapping).",
    )
    parser.add_argument("--k", type=int, default=150, help="Max number of functions to select.")
    parser.add_argument("--arr-threshold", type=float, default=0.0, help="Minimum ARR for customer eligibility.")
    parser.add_argument(
        "--objective",
        choices=["customers", "arr", "blend", "rooftops", "logo_progress", "arr_progress"],
        default="rooftops",
        help="Optimization objective. Default 'rooftops' maximizes rooftops unlocked; also: customers, arr, blend.",
    )
    parser.add_argument(
        "--blend-weight",
        type=float,
        default=None,
        metavar="W",
        help="Single blend weight in [0,1] = %% weight on customers (rest = ARR). If set with objective=blend, only this run.",
    )
    parser.add_argument(
        "--out",
        default="CDK_FunctionSequencingOutput.xlsx",
        help="Output Excel file (one file with multiple sheets). Sheet names use 'Xcust_Yarr' = X%% customers, Y%% ARR.",
    )
    parser.add_argument(
        "--export-json",
        type=Path,
        metavar="PATH",
        help="Also write dashboard data to a JSON file (for the Vercel dashboard).",
    )
    args = parser.parse_args()

    # Accept input Excel from positional arg, --input-excel, or auto-detect in project folder
    input_excel = args.input_excel
    if input_excel is None and args.input_excel_path is not None:
        input_excel = Path(args.input_excel_path)
    if input_excel is None and not (args.function_mapping and args.enterprise_arr):
        # No input given: look for CDK_Input.xlsx next to script or in current directory
        script_dir = Path(__file__).resolve().parent
        for candidate in [script_dir / "CDK_Input.xlsx", Path.cwd() / "CDK_Input.xlsx"]:
            if candidate.exists():
                input_excel = candidate
                print(f"Using input file: {input_excel}")
                break

    if input_excel is not None:
        path = Path(input_excel)
        if not path.exists():
            parser.error(f"Input Excel file not found: {path}")
        df_function_mapping = read_excel_sheet(path, args.sheet_function_mapping)
        df_arr = read_excel_sheet(path, args.sheet_enterprise_arr)
    elif args.function_mapping and args.enterprise_arr:
        df_function_mapping = _read_excel(Path(args.function_mapping))
        df_arr = _read_excel(Path(args.enterprise_arr))
    else:
        parser.error(
            "No input Excel found. Put CDK_Input.xlsx in this script's folder, or run: "
            "python FunctionSequencing.py CDK_Input.xlsx (or use --input-excel / --function-mapping + --enterprise-arr)."
        )

    df_function_mapping = _normalize_columns(df_function_mapping, REQUIRED_FUNCTION_COLS)
    df_arr = _normalize_columns(df_arr, REQUIRED_ARR_COLS + OPTIONAL_ARR_COLS)

    # Load WorkflowMapping early: defines in-scope functions and supplies COMPLETED (and workflow for outputs)
    workflow_map: Dict[str, str] = {}
    completed_map: Dict[str, str] = {}
    if input_excel is not None:
        path = Path(input_excel)
        sheet_names = get_excel_sheet_names(path)
        if args.sheet_workflow_mapping in sheet_names:
            df_workflow = read_excel_sheet(path, args.sheet_workflow_mapping)
            df_workflow = _normalize_columns(df_workflow, ("WORKFLOW", "FUNCTION", "COMPLETED"))
            if "FUNCTION" in df_workflow.columns:
                # In-scope: only functions listed in WorkflowMapping are considered
                in_scope_functions = set(df_workflow["FUNCTION"].dropna().astype(str).str.strip().unique())
                df_function_mapping = df_function_mapping[
                    df_function_mapping["FUNCTION"].astype(str).str.strip().isin(in_scope_functions)
                ].copy()
                if "WORKFLOW" in df_workflow.columns:
                    df_wf_unique = df_workflow.drop_duplicates(subset=["FUNCTION"], keep="first")
                    workflow_map = df_wf_unique.set_index("FUNCTION")["WORKFLOW"].astype(str).str.strip().to_dict()
                if "COMPLETED" in df_workflow.columns:
                    df_wf_c = df_workflow[["FUNCTION", "COMPLETED"]].drop_duplicates(subset=["FUNCTION"], keep="first").copy()
                    df_wf_c["FUNCTION"] = df_wf_c["FUNCTION"].astype(str).str.strip()
                    completed_map = df_wf_c.set_index("FUNCTION")["COMPLETED"].astype(str).str.strip().to_dict()
                    df_function_mapping = df_function_mapping.drop(columns=["COMPLETED"], errors="ignore")
                    df_function_mapping = df_function_mapping.merge(
                        df_workflow[["FUNCTION", "COMPLETED"]].drop_duplicates(subset=["FUNCTION"], keep="first"),
                        on="FUNCTION",
                        how="left",
                    )

    _validate_input_dfs(df_function_mapping, df_arr)

    # If a Rooftops sheet exists (single Excel only), merge it into ARR data by ENTERPRISE ID
    if input_excel is not None:
        path = Path(input_excel)
        sheet_names = get_excel_sheet_names(path)
        if args.sheet_rooftops in sheet_names:
            df_rooftops = read_excel_sheet(path, args.sheet_rooftops)
            df_rooftops = _normalize_columns(df_rooftops, ("ENTERPRISE ID", "ROOFTOPS"))
            if "ENTERPRISE ID" in df_rooftops.columns and "ROOFTOPS" in df_rooftops.columns:
                df_rooftops = df_rooftops[["ENTERPRISE ID", "ROOFTOPS"]].drop_duplicates(subset=["ENTERPRISE ID"], keep="first")
                # Ensure df_arr has "ENTERPRISE ID" for merge (in case Excel header didn't normalize)
                if "ENTERPRISE ID" not in df_arr.columns:
                    _norm = lambda s: " ".join(str(s).strip().lower().split())
                    for c in df_arr.columns:
                        if _norm(c) == "enterprise id":
                            df_arr = df_arr.rename(columns={c: "ENTERPRISE ID"})
                            break
                if "ENTERPRISE ID" in df_arr.columns:
                    df_arr = df_arr.merge(df_rooftops, on="ENTERPRISE ID", how="left")

    # (workflow_map already built from WorkflowMapping above when loading in-scope functions)

    # Output two Excel files: one for Daily+Weekly scope, one for Daily+Weekly+Monthly scope
    blend_runs = [("", 0.0)]
    out_base = Path(args.out).stem
    if out_base.endswith("_DailyWeeklyMonthly"):
        out_base = out_base.replace("_DailyWeeklyMonthly", "")
    elif out_base.endswith("_DailyWeekly"):
        out_base = out_base.replace("_DailyWeekly", "")
    out_dir = Path(args.out).parent
    scope_configs = [
        ("DailyWeekly", SCOPE_DAILY_WEEKLY),
        ("DailyWeeklyMonthly", SCOPE_DAILY_WEEKLY_MONTHLY),
    ]
    saved_paths: List[Path] = []
    export_for_dashboard: Dict[str, dict] = {}

    for scope_label, valid_patterns in scope_configs:
        df_scope = df_function_mapping[
            df_function_mapping["USAGE_PATTERN"].astype(str).str.strip().isin(valid_patterns)
        ].copy()

        sheets: List[Tuple[str, pd.DataFrame]] = []

        # First tab: Enterprise functions summary (for this scope)
        enterprise_summary_df = build_enterprise_functions_summary(
            df_scope,
            df_arr,
            arr_threshold=args.arr_threshold,
            treat_partial_as_done=getattr(args, "treat_partial_as_done", False),
            valid_patterns=valid_patterns,
        )
        enterprise_summary_df = _round_dollar_columns(enterprise_summary_df)
        sheets.append(("EnterpriseFunctionsSummary", _format_sheet_columns(enterprise_summary_df)))

        for suffix, blend_weight in blend_runs:
            steps_df, enterprise_unlock_df, selected_functions, unlocked = greedy_unlock_functions_with_arr_outputs(
                df_scope,
                df_arr,
                k=args.k,
                arr_threshold=args.arr_threshold,
                objective=args.objective,
                blend_weight_customers=blend_weight,
                valid_patterns=valid_patterns,
            )
            tranche_summary_df = build_tranche_summary(enterprise_unlock_df)
            if not steps_df.empty:
                full_steps = steps_df[["step", "function_to_modernize"]].drop_duplicates(subset=["step"], keep="first")
                tranche_summary_df = full_steps.merge(
                    tranche_summary_df.drop(columns=["function_to_modernize"], errors="ignore"),
                    on="step", how="left"
                )
                for col in ("customers_unlocked", "rooftops_unlocked", "arr_unlocked"):
                    if col in tranche_summary_df.columns:
                        tranche_summary_df[col] = tranche_summary_df[col].fillna(0)
                tranche_summary_df["cumulative_customers_unlocked"] = tranche_summary_df["customers_unlocked"].cumsum()
                tranche_summary_df["cumulative_rooftops_unlocked"] = tranche_summary_df["rooftops_unlocked"].cumsum()
                tranche_summary_df["cumulative_arr_unlocked"] = tranche_summary_df["arr_unlocked"].cumsum()
                tranche_summary_df = tranche_summary_df[[
                    "step", "function_to_modernize", "customers_unlocked", "cumulative_customers_unlocked",
                    "rooftops_unlocked", "cumulative_rooftops_unlocked", "arr_unlocked", "cumulative_arr_unlocked",
                ]]
            tranche_summary_df["workflow"] = tranche_summary_df["function_to_modernize"].map(workflow_map).fillna("")
            tranche_summary_df = tranche_summary_df[[
                "step", "function_to_modernize", "workflow", "customers_unlocked", "cumulative_customers_unlocked",
                "rooftops_unlocked", "cumulative_rooftops_unlocked", "arr_unlocked", "cumulative_arr_unlocked",
            ]]
            tranche_summary_df = _round_dollar_columns(tranche_summary_df)

            enterprise_unlock_view_df = build_enterprise_unlock_view(enterprise_unlock_df)
            enterprise_unlock_view_df["workflow"] = enterprise_unlock_view_df["function_completed_in_step"].map(workflow_map).fillna("")
            cols_ent = [c for c in enterprise_unlock_view_df.columns if c != "workflow"]
            idx = cols_ent.index("function_completed_in_step") + 1 if "function_completed_in_step" in cols_ent else len(cols_ent)
            cols_ent.insert(idx, "workflow")
            enterprise_unlock_view_df = enterprise_unlock_view_df[cols_ent]
            enterprise_unlock_view_df = _round_dollar_columns(enterprise_unlock_view_df)

            sheets.append(("TrancheSummary", _format_sheet_columns(tranche_summary_df)))
            sheets.append(("EnterpriseUnlockByStep", _format_sheet_columns(enterprise_unlock_view_df)))

        matrix_df = build_enterprise_function_usage_matrix(
            df_scope,
            df_arr,
            workflow_map,
            completed_map=completed_map,
        )
        if not matrix_df.empty:
            sheets.append(("EnterpriseFunctionUsageMatrix", matrix_df))

        out_path = out_dir / f"{out_base}_{scope_label}.xlsx"
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            for name, frame in sheets:
                sheet_name = name[:31]
                if name == "EnterpriseFunctionUsageMatrix":
                    frame.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
                else:
                    frame.to_excel(writer, sheet_name=sheet_name, index=False)
        saved_paths.append(out_path)
        print(f"\n--- {scope_label} ---")
        print(f"Saved: {out_path} (sheets: {', '.join(s[0][:31] for s in sheets)})")

        if args.export_json is not None:
            export_for_dashboard[scope_label] = {
                "enterpriseSummary": _dataframe_to_json_records(_format_sheet_columns(enterprise_summary_df)),
                "trancheSummary": _dataframe_to_json_records(_format_sheet_columns(tranche_summary_df)),
                "enterpriseUnlockByStep": _dataframe_to_json_records(_format_sheet_columns(enterprise_unlock_view_df)),
                "workflowMap": workflow_map,
            }

    if args.export_json is not None and export_for_dashboard:
        out_json = Path(args.export_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {"scopes": export_for_dashboard, "workflowMap": workflow_map}
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"\nExported dashboard JSON: {out_json}")

    print(f"\nOutputs: {saved_paths[0].name} (Daily+Weekly), {saved_paths[1].name} (Daily+Weekly+Monthly)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
