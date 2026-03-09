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
      USAGE_PATTERN   — "1 - daily" | "2 - weekly" | "3 - monthly" (or "daily", "weekly", "monthly"; others normalized or ignored)
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
from collections import deque
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


def read_all_excel_sheets(path: Path) -> Dict[str, pd.DataFrame]:
    """Read all sheets from an Excel file in one pass. Returns dict sheet_name -> DataFrame."""
    path = Path(path)
    kwargs = {"sheet_name": None}
    if path.suffix.lower() in (".xlsx", ".xlsm"):
        kwargs["engine"] = "openpyxl"
    return pd.read_excel(path, **kwargs)


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
    _prepared_arr: Optional[Tuple[Set, Dict, Optional[Dict]]] = None,  # (eligible, arr_map, rooftops_map) to skip recomputation
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], Set[str]]:
    """
    In-scope = functions not yet modernized (Complete is No or Partial) and used by ≥1 customer
    at this scope's frequency (daily/weekly or daily/weekly/monthly). Greedy picks optimal order
    to maximize rooftops (or chosen objective) unlocked.

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
    if _prepared_arr is not None:
        eligible_enterprises, arr_map, rooftops_map = _prepared_arr
    else:
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
    # In-scope = not yet modernized (Complete is No or Partial) and used by ≥1 customer at this scope's frequency
    candidate_functions = set(ents_by_func.keys())
    k = max(k, len(candidate_functions))

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
        max_c, max_a = 0.0, 0.0

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
                if customer_score > max_c:
                    max_c = customer_score
                if arr_score > max_a:
                    max_a = arr_score
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
            max_c = max_c or 1.0
            max_a = max_a or 1.0
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


def _build_workflow_map_least_functions(df_workflow: pd.DataFrame) -> Dict[str, str]:
    """
    When a function appears in multiple workflows, pick the workflow that has the least
    total functions mapped to it (to spread functions across workflows).
    Returns dict: function_upper -> workflow name.
    """
    if df_workflow is None or df_workflow.empty or "WORKFLOW" not in df_workflow.columns or "FUNCTION" not in df_workflow.columns:
        return {}
    df = df_workflow[["WORKFLOW", "FUNCTION"]].dropna().copy()
    df["FUNCTION"] = df["FUNCTION"].astype(str).str.strip()
    df["WORKFLOW"] = df["WORKFLOW"].astype(str).str.strip()
    # Count distinct functions per workflow (each workflow may have many rows for same function)
    workflow_function_count = df.drop_duplicates(subset=["WORKFLOW", "FUNCTION"]).groupby("WORKFLOW").size()
    workflow_count_map = workflow_function_count.to_dict()
    out: Dict[str, str] = {}
    for func, grp in df.groupby("FUNCTION"):
        workflows = grp["WORKFLOW"].unique().tolist()
        if not workflows:
            continue
        best_wf = min(workflows, key=lambda w: workflow_count_map.get(w, 0))
        out[func.upper()] = best_wf
    return out


def _reorder_sequence_with_workflow_pullup(
    ordered_functions: List[str],
    workflow_map: Dict[str, str],
    threshold: int = 3,
) -> List[str]:
    """
    If a function has fewer than `threshold` other same-workflow functions later in the
    sequence, pull those up to be immediately after it. Modifies only the order.
    """
    if not ordered_functions or threshold <= 0:
        return list(ordered_functions)
    remaining = deque(ordered_functions)
    new_order: List[str] = []
    while remaining:
        f = remaining.popleft()
        f_upper = str(f).strip().upper() if f else ""
        w = workflow_map.get(f_upper, "")
        trailing = [x for x in remaining if workflow_map.get(str(x).strip().upper() if x else "", "") == w]
        if 1 <= len(trailing) < threshold:
            new_order.append(f)
            new_order.extend(trailing)
            trailing_set = set(trailing)
            remaining = deque(x for x in remaining if x not in trailing_set)
        else:
            new_order.append(f)
    return new_order


def _rebuild_steps_and_unlock_from_order(
    new_order: List[str],
    steps_df: pd.DataFrame,
    enterprise_unlock_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild steps_df and enterprise_unlock_df with new step numbers from reordered function list."""
    if not new_order or steps_df.empty or enterprise_unlock_df.empty:
        return steps_df, enterprise_unlock_df
    new_step_for_function = {str(f).strip(): i + 1 for i, f in enumerate(new_order) if f and str(f).strip()}
    # Map old function_to_modernize to new step for each enterprise row (preserve original step if not in new order)
    eu = enterprise_unlock_df.copy()
    eu["_func_key"] = eu["function_to_modernize"].astype(str).str.strip()
    eu["new_step"] = eu["_func_key"].map(new_step_for_function)
    eu.loc[eu["new_step"].isna(), "new_step"] = eu.loc[eu["new_step"].isna(), "step"]
    eu["step"] = eu["new_step"].astype(int)
    eu = eu.drop(columns=["new_step", "_func_key"], errors="ignore")
    enterprise_unlock_df_new = eu

    step_0 = steps_df[steps_df["step"] == 0]
    rows = []
    for i, f in enumerate(new_order):
        s = i + 1
        count = int((eu["step"] == s).sum())
        rows.append({
            "step": s,
            "function_to_modernize": f,
            "customers_unlocked": count,
            "cumulative_customers_unlocked": 0,  # fill below
        })
    if rows:
        steps_new = pd.DataFrame(rows)
        base_cum = int(step_0["cumulative_customers_unlocked"].iloc[0]) if not step_0.empty else 0
        steps_new["cumulative_customers_unlocked"] = base_cum + steps_new["customers_unlocked"].cumsum()
        if not step_0.empty:
            steps_new = pd.concat([step_0, steps_new], ignore_index=True)
        steps_df_new = steps_new.sort_values("step").reset_index(drop=True)
    else:
        steps_df_new = steps_df
    return steps_df_new, enterprise_unlock_df_new


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
    "segment": "Segment",
    "Configuration": "Configuration",
    "Churn Propensity": "Churn Propensity",
    "Risk Category": "Risk Category",
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


def build_segment_summary(
    tranche_summary_df: pd.DataFrame,
    rooftop_segment_size: int = 1000,
) -> pd.DataFrame:
    """
    Segment = group of functions (steps) to achieve 1000 rooftops unlocked.
    Segment 0 = entirely modern customers (step 0). Segment k (k>=1) = floor(cumulative_rooftops/1000) + 1.
    Output: Segment, concatenated functions, count of functions, customers unlocked, rooftops unlocked,
    total Current DMS ARR, average rooftops per customer, average Current DMS ARR per customer.
    """
    if tranche_summary_df is None or tranche_summary_df.empty:
        return pd.DataFrame()
    need = {"step", "function_to_modernize", "customers_unlocked", "rooftops_unlocked", "cumulative_rooftops_unlocked", "arr_unlocked"}
    if not need.issubset(tranche_summary_df.columns):
        return pd.DataFrame()
    df = tranche_summary_df[list(need)].copy()
    # One row per step (take first if duplicate step from any merge)
    df = df.drop_duplicates(subset=["step"], keep="first")
    # Segment 0 = step 0 (entirely modern); segment = floor(cumulative_rooftops / 1000) + 1 for step >= 1
    df["segment"] = np.where(
        df["step"] == 0,
        0,
        (pd.to_numeric(df["cumulative_rooftops_unlocked"], errors="coerce").fillna(0) // rooftop_segment_size).astype(int) + 1,
    )
    agg = df.groupby("segment").agg(
        functions_list=("function_to_modernize", lambda s: " | ".join(str(x).strip() for x in s if str(x).strip())),
        count_functions=("step", "count"),
        customers_unlocked=("customers_unlocked", "sum"),
        rooftops_unlocked=("rooftops_unlocked", "sum"),
        total_current_dms_arr=("arr_unlocked", "sum"),
    ).reset_index()
    agg["average_rooftops_per_customer"] = np.where(
        agg["customers_unlocked"] > 0,
        agg["rooftops_unlocked"] / agg["customers_unlocked"],
        0,
    )
    agg["average_current_dms_arr_per_customer"] = np.where(
        agg["customers_unlocked"] > 0,
        agg["total_current_dms_arr"] / agg["customers_unlocked"],
        0,
    )
    agg = agg.rename(columns={
        "segment": "Segment",
        "functions_list": "Functions (steps in segment)",
        "count_functions": "Count of functions",
        "customers_unlocked": "Count of customers unlocked",
        "rooftops_unlocked": "Count of rooftops unlocked",
        "total_current_dms_arr": "Total Current DMS ARR",
        "average_rooftops_per_customer": "Average rooftops per customer",
        "average_current_dms_arr_per_customer": "Average Current DMS ARR per customer",
    })
    # Round dollars and decimals
    agg["Total Current DMS ARR"] = pd.to_numeric(agg["Total Current DMS ARR"], errors="coerce").fillna(0).round(0).astype(int)
    agg["Average rooftops per customer"] = agg["Average rooftops per customer"].round(2)
    agg["Average Current DMS ARR per customer"] = pd.to_numeric(agg["Average Current DMS ARR per customer"], errors="coerce").fillna(0).round(0).astype(int)
    return agg[
        ["Segment", "Functions (steps in segment)", "Count of functions", "Count of customers unlocked",
         "Count of rooftops unlocked", "Total Current DMS ARR", "Average rooftops per customer",
         "Average Current DMS ARR per customer"]
    ]


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
    header_row_2 = ["", "", "", ""] + [workflow_map.get(f.upper(), "") for f in in_scope_functions]
    header_row_3 = ["", "", "", ""] + [completed_map.get(f.upper(), "") for f in in_scope_functions] if completed_map else ["", "", "", ""] + [""] * len(in_scope_functions)
    all_rows = [header_row_1, header_row_2, header_row_3] + data_rows
    return pd.DataFrame(all_rows)


def _normalize_usage_pattern(usage: str) -> str:
    """Map USAGE_PATTERN to canonical form: 'daily' or '1 - daily' -> '1 - daily', etc."""
    u = str(usage).strip().lower()
    if not u:
        return str(usage).strip()
    if "daily" in u:
        return "1 - daily"
    if "weekly" in u:
        return "2 - weekly"
    if "monthly" in u:
        return "3 - monthly"
    if "quarterly" in u:
        return "4 - quarterly"
    return str(usage).strip()


def _normalize_usage_pattern_series(ser: pd.Series) -> pd.Series:
    """Vectorized: map USAGE_PATTERN column to canonical form."""
    s = ser.astype(str).str.strip()
    u = s.str.lower()
    return pd.Series(
        np.select(
            [
                u.str.contains("daily", na=False),
                u.str.contains("weekly", na=False),
                u.str.contains("monthly", na=False),
                u.str.contains("quarterly", na=False),
            ],
            ["1 - daily", "2 - weekly", "3 - monthly", "4 - quarterly"],
            default=s.values,
        ),
        index=ser.index,
    )


def _usage_bucket(usage: str) -> str:
    """Map USAGE_PATTERN to bucket: daily, weekly, monthly, quarterly, less_than_quarterly."""
    u = str(usage).strip().lower()
    if "daily" in u or u == "1 - daily":
        return "daily"
    if "weekly" in u or u == "2 - weekly":
        return "weekly"
    if "monthly" in u or u == "3 - monthly":
        return "monthly"
    if "quarterly" in u or u == "4 - quarterly":
        return "quarterly"
    return "less_than_quarterly"


def build_function_frequency_summary(
    df_mapping: pd.DataFrame,
    workflow_map: Dict[str, str],
    completed_map: Optional[Dict[str, str]] = None,
    enterprise_col: str = "ENTERPRISE ID",
    function_col: str = "FUNCTION",
    usage_col: str = "USAGE_PATTERN",
    in_scope_functions: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """
    One row per function: Function, Workflow, Complete, usage % by frequency, Total Customers.
    Uses raw mapping data (all usage patterns) so you see how often functions are used.
    Total Customers = unique ENTERPRISE IDs (excl. "0"). Percentages sum to 100% per row.
    """
    if df_mapping.empty or function_col not in df_mapping.columns or usage_col not in df_mapping.columns:
        return pd.DataFrame()

    d = df_mapping[[enterprise_col, function_col, usage_col]].copy()
    d[enterprise_col] = d[enterprise_col].astype(str).str.strip()
    d[function_col] = d[function_col].astype(str).str.strip()
    d[usage_col] = d[usage_col].astype(str).str.strip()
    d = d.dropna(subset=[enterprise_col, function_col])
    d = d[d[enterprise_col] != "0"].copy()
    d = _reduce_to_most_frequent_usage(d, enterprise_col, function_col, usage_col)
    d["_bucket"] = d[usage_col].map(_usage_bucket)
    # Normalize to upper for grouping so casing in FunctionMapping matches in_scope_functions from WorkflowMapping
    d["_func_key"] = d[function_col].str.upper()

    # All functions to report: in-scope set if provided, else unique from data
    if in_scope_functions is not None:
        functions = sorted(in_scope_functions)
    else:
        functions = sorted(d[function_col].unique().tolist())

    total_per_func = d.groupby("_func_key")[enterprise_col].nunique()
    bucket_per_func = d.groupby(["_func_key", "_bucket"])[enterprise_col].nunique().unstack(fill_value=0)

    buckets = ["daily", "weekly", "monthly", "quarterly", "less_than_quarterly"]
    pct_cols = [
        "% Customers (daily)", "% Customers (weekly)", "% Customers (monthly)",
        "% Customers (quarterly)", "% Customers (quarterly plus / no pattern)",
    ]
    for b in buckets:
        if b not in bucket_per_func.columns:
            bucket_per_func[b] = 0

    rows = []
    for func in functions:
        key = func.upper()
        total = int(total_per_func.get(key, 0))
        row = {
            "Function": func,
            "Workflow": workflow_map.get(key, ""),
            "Complete": completed_map.get(key, "") if completed_map else "",
            "Total Customers": total,
        }
        if total > 0 and key in bucket_per_func.index:
            for b, col in zip(buckets, pct_cols):
                row[col] = round(100.0 * bucket_per_func.loc[key, b] / total, 1)
        else:
            for col in pct_cols:
                row[col] = 0
        rows.append(row)

    out = pd.DataFrame(rows)
    out = out[
        ["Function", "Workflow", "Complete",
         "% Customers (daily)", "% Customers (weekly)", "% Customers (monthly)",
         "% Customers (quarterly)", "% Customers (quarterly plus / no pattern)",
         "Total Customers"]
    ]
    return out


# Scope definitions for "remaining to modernize" views (by frequency; match input format)
SCOPE_DAILY = ("1 - daily",)
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
    parser.add_argument(
        "--sheet-enterprise-configs",
        default="EnterpriseConfigs",
        help="Sheet name for EnterpriseConfigs (ENTERPRISE ID, Configuration) when using --input-excel (default: EnterpriseConfigs).",
    )
    parser.add_argument(
        "--sheet-enterprise-churn",
        default="EnterpriseChurn",
        help="Sheet name for EnterpriseChurn (ENTERPRISE ID, Churn Propensity, Risk Category) when using --input-excel (default: EnterpriseChurn).",
    )
    parser.add_argument("--k", type=int, default=300, help="Min number of steps to run. Automatically increased to cover all functions that still block at least one customer (so full modernization has one step per such function).")
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
    parser.add_argument(
        "--dashboard-pretty",
        action="store_true",
        help="Write dashboard JSON with indentation (larger file, human-readable). Default is compact.",
    )
    parser.add_argument(
        "--dashboard-max-unlock-rows",
        type=int,
        metavar="N",
        default=None,
        help="Cap 'Enterprise unlock by step' rows per scope in dashboard JSON (reduces file size). Omit for no cap.",
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

    all_sheets: Optional[Dict[str, pd.DataFrame]] = None
    if input_excel is not None:
        path = Path(input_excel)
        if not path.exists():
            parser.error(f"Input Excel file not found: {path}")
        all_sheets = read_all_excel_sheets(path)
        df_function_mapping = all_sheets[args.sheet_function_mapping].copy()
        df_arr = all_sheets[args.sheet_enterprise_arr].copy()
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
    in_scope_functions: Set[str] = set()
    if all_sheets is not None:
        if args.sheet_workflow_mapping in all_sheets:
            df_workflow = all_sheets[args.sheet_workflow_mapping].copy()
            df_workflow = _normalize_columns(df_workflow, ("WORKFLOW", "FUNCTION", "COMPLETED"))
            if "FUNCTION" in df_workflow.columns:
                # In-scope: only functions listed in WorkflowMapping are considered (case-insensitive match)
                in_scope_functions = set(df_workflow["FUNCTION"].dropna().astype(str).str.strip().unique())
                in_scope_upper = {f.upper() for f in in_scope_functions}
                df_function_mapping = df_function_mapping[
                    df_function_mapping["FUNCTION"].astype(str).str.strip().str.upper().isin(in_scope_upper)
                ].copy()
                if "WORKFLOW" in df_workflow.columns:
                    # Multiple workflows per function: pick the workflow with the least total functions mapped to it
                    workflow_map = _build_workflow_map_least_functions(df_workflow)
                if "COMPLETED" in df_workflow.columns:
                    df_wf_c = df_workflow[["FUNCTION", "COMPLETED"]].drop_duplicates(subset=["FUNCTION"], keep="first").copy()
                    df_wf_c["FUNCTION"] = df_wf_c["FUNCTION"].astype(str).str.strip()
                    df_wf_c["_fk"] = df_wf_c["FUNCTION"].str.upper()
                    completed_map = df_wf_c.set_index("_fk")["COMPLETED"].astype(str).str.strip().to_dict()
                    df_function_mapping = df_function_mapping.drop(columns=["COMPLETED"], errors="ignore")
                    # Merge COMPLETED on upper-case key so different casing in FunctionMapping still matches
                    df_wf_merge = df_workflow[["FUNCTION", "COMPLETED"]].drop_duplicates(subset=["FUNCTION"], keep="first").copy()
                    df_wf_merge["_fk"] = df_wf_merge["FUNCTION"].astype(str).str.strip().str.upper()
                    df_function_mapping["_fk"] = df_function_mapping["FUNCTION"].astype(str).str.strip().str.upper()
                    df_function_mapping = df_function_mapping.merge(
                        df_wf_merge[["_fk", "COMPLETED"]], on="_fk", how="left"
                    ).drop(columns=["_fk"])
    if not in_scope_functions and "FUNCTION" in df_function_mapping.columns:
        in_scope_functions = set(df_function_mapping["FUNCTION"].dropna().astype(str).str.strip().unique())

    # Normalize USAGE_PATTERN so "daily", "weekly", etc. are treated as "1 - daily", "2 - weekly", etc.
    if "USAGE_PATTERN" in df_function_mapping.columns:
        df_function_mapping["USAGE_PATTERN"] = _normalize_usage_pattern_series(df_function_mapping["USAGE_PATTERN"])

    _validate_input_dfs(df_function_mapping, df_arr)

    # If a Rooftops sheet exists (single Excel only), merge it into ARR data by ENTERPRISE ID
    if all_sheets is not None and args.sheet_rooftops in all_sheets:
        df_rooftops = _normalize_columns(all_sheets[args.sheet_rooftops].copy(), ("ENTERPRISE ID", "ROOFTOPS"))
        if "ENTERPRISE ID" in df_rooftops.columns and "ROOFTOPS" in df_rooftops.columns:
            df_rooftops = df_rooftops[["ENTERPRISE ID", "ROOFTOPS"]].drop_duplicates(subset=["ENTERPRISE ID"], keep="first")
            if "ENTERPRISE ID" not in df_arr.columns:
                _norm = lambda s: " ".join(str(s).strip().lower().split())
                for c in df_arr.columns:
                    if _norm(c) == "enterprise id":
                        df_arr = df_arr.rename(columns={c: "ENTERPRISE ID"})
                        break
            if "ENTERPRISE ID" in df_arr.columns:
                df_arr = df_arr.merge(df_rooftops, on="ENTERPRISE ID", how="left")

    # Optional: EnterpriseConfigs (ENTERPRISE ID, Configuration) and EnterpriseChurn (ENTERPRISE ID, Churn Propensity, Risk Category)
    df_enterprise_configs: Optional[pd.DataFrame] = None
    df_enterprise_churn: Optional[pd.DataFrame] = None
    if all_sheets is not None:
        if args.sheet_enterprise_configs in all_sheets:
            _cfg = _normalize_columns(all_sheets[args.sheet_enterprise_configs].copy(), ("ENTERPRISE ID", "Configuration"))
            if "ENTERPRISE ID" in _cfg.columns and "Configuration" in _cfg.columns:
                df_enterprise_configs = _cfg[["ENTERPRISE ID", "Configuration"]].drop_duplicates(subset=["ENTERPRISE ID"], keep="first")
        if args.sheet_enterprise_churn in all_sheets:
            _churn = _normalize_columns(all_sheets[args.sheet_enterprise_churn].copy(), ("ENTERPRISE ID", "Churn Propensity", "Risk Category"))
            if "ENTERPRISE ID" in _churn.columns and "Churn Propensity" in _churn.columns and "Risk Category" in _churn.columns:
                df_enterprise_churn = _churn[["ENTERPRISE ID", "Churn Propensity", "Risk Category"]].drop_duplicates(subset=["ENTERPRISE ID"], keep="first")

    # (workflow_map already built from WorkflowMapping above when loading in-scope functions)

    # Output one Excel file per scope: Daily, Daily+Weekly, Daily+Weekly+Monthly
    blend_runs = [("", 0.0)]
    out_base = Path(args.out).stem
    for suffix in ("_DailyWeeklyMonthly", "_DailyWeekly", "_Daily"):
        if out_base.endswith(suffix):
            out_base = out_base.replace(suffix, "")
            break
    out_dir = Path(args.out).parent
    scope_configs = [
        ("Daily", SCOPE_DAILY),
        ("DailyWeekly", SCOPE_DAILY_WEEKLY),
        ("DailyWeeklyMonthly", SCOPE_DAILY_WEEKLY_MONTHLY),
    ]
    saved_paths: List[Path] = []
    export_for_dashboard: Dict[str, dict] = {}

    print(f"\n--- In-scope and greedy ---")
    print(f"WorkflowMapping: {len(in_scope_functions)} functions. In-scope per scope = not yet modernized (No/Partial) and used by ≥1 customer at that scope.")

    # Precompute ARR/eligible once for all scope runs (avoids 3x _prepare_enterprise_arr)
    _, prepared_eligible, prepared_arr_map, prepared_rooftops_map = _prepare_enterprise_arr(
        df_arr, args.arr_threshold, "ENTERPRISE ID", "ARR"
    )
    prepared_arr = (prepared_eligible, prepared_arr_map, prepared_rooftops_map)

    # 1) In-scope = not yet modernized (Complete No/Partial) and used by ≥1 customer at scope (DW or DWM).
    # 2) Greedy = optimal order of those functions to maximize rooftops unlocked.
    # 3) Function Frequency Summary = raw data (all usage patterns) so you see how often functions are used.
    function_frequency_df = build_function_frequency_summary(
        df_function_mapping,
        workflow_map,
        completed_map=completed_map,
        in_scope_functions=in_scope_functions,
    )

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
                _prepared_arr=prepared_arr,
            )
            n_in_scope = len(selected_functions)
            print(f"  [Scope {scope_label}] In-scope: {n_in_scope} functions (not yet modernized and used by ≥1 customer at this scope)")
            # Pull up same-workflow functions: if <3 other functions on same workflow remain later, move them right after this function
            if not steps_df.empty and workflow_map:
                ordered_functions = [
                    f for f in
                    steps_df[steps_df["step"] >= 1].sort_values("step")["function_to_modernize"].tolist()
                    if f and str(f).strip()
                ]
                new_order = _reorder_sequence_with_workflow_pullup(ordered_functions, workflow_map, threshold=3)
                if new_order != ordered_functions:
                    steps_df, enterprise_unlock_df = _rebuild_steps_and_unlock_from_order(
                        new_order, steps_df, enterprise_unlock_df
                    )
                    selected_functions = new_order
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
            tranche_summary_df["workflow"] = tranche_summary_df["function_to_modernize"].astype(str).str.upper().map(workflow_map).fillna("")
            # Add segment: 0 for step 0, else floor(cumulative_rooftops_unlocked / 1000) + 1
            tranche_summary_df["segment"] = np.where(
                tranche_summary_df["step"] == 0,
                0,
                (pd.to_numeric(tranche_summary_df["cumulative_rooftops_unlocked"], errors="coerce").fillna(0) // 1000).astype(int) + 1,
            )
            tranche_summary_df = tranche_summary_df[[
                "step", "segment", "function_to_modernize", "workflow", "customers_unlocked", "cumulative_customers_unlocked",
                "rooftops_unlocked", "cumulative_rooftops_unlocked", "arr_unlocked", "cumulative_arr_unlocked",
            ]]
            tranche_summary_df = _round_dollar_columns(tranche_summary_df)

            enterprise_unlock_view_df = build_enterprise_unlock_view(enterprise_unlock_df)
            step_to_segment = tranche_summary_df.drop_duplicates(subset=["step"], keep="first").set_index("step")["segment"]
            enterprise_unlock_view_df["segment"] = enterprise_unlock_view_df["Step Unlocked"].map(step_to_segment)
            enterprise_unlock_view_df["workflow"] = enterprise_unlock_view_df["function_completed_in_step"].astype(str).str.upper().map(workflow_map).fillna("")
            if df_enterprise_configs is not None and not df_enterprise_configs.empty:
                enterprise_unlock_view_df = enterprise_unlock_view_df.merge(df_enterprise_configs, on="ENTERPRISE ID", how="left")
            if df_enterprise_churn is not None and not df_enterprise_churn.empty:
                enterprise_unlock_view_df = enterprise_unlock_view_df.merge(df_enterprise_churn, on="ENTERPRISE ID", how="left")
            # Column order: ID, Name, Rooftops, ARR, then optional Configuration / Churn Propensity / Risk Category, then Step Unlocked, segment, function_completed_in_step, workflow
            cols_ent = ["ENTERPRISE ID", "Name", "Rooftops", "ARR"]
            for c in ["Configuration", "Churn Propensity", "Risk Category"]:
                if c in enterprise_unlock_view_df.columns:
                    cols_ent.append(c)
            cols_ent += ["Step Unlocked", "segment", "function_completed_in_step", "workflow"]
            cols_ent = [c for c in cols_ent if c in enterprise_unlock_view_df.columns]
            enterprise_unlock_view_df = enterprise_unlock_view_df[cols_ent]
            enterprise_unlock_view_df = _round_dollar_columns(enterprise_unlock_view_df)

            sheets.append(("TrancheSummary", _format_sheet_columns(tranche_summary_df)))
            segment_summary_df = build_segment_summary(tranche_summary_df, rooftop_segment_size=1000)
            if not segment_summary_df.empty:
                sheets.append(("SegmentSummary", segment_summary_df))
            sheets.append(("EnterpriseUnlockByStep", _format_sheet_columns(enterprise_unlock_view_df)))

        # Matrix from full mapping so all frequencies are shown (daily, weekly, monthly, quarterly, etc.)
        matrix_df = build_enterprise_function_usage_matrix(
            df_function_mapping,
            df_arr,
            workflow_map,
            completed_map=completed_map,
        )

        # Function summary from raw data (first tab); add per-scope "In scope (used as step)"
        if not function_frequency_df.empty:
            scope_freq = function_frequency_df.copy()
            selected_upper = {f.upper() for f in selected_functions}
            scope_freq["In scope (used as step)"] = scope_freq["Function"].apply(
                lambda f: "Yes" if (f.upper() in selected_upper) else "No"
            )
            cols = ["Function", "Workflow", "Complete", "In scope (used as step)"]
            cols += [c for c in scope_freq.columns if c not in cols]
            scope_freq = scope_freq[cols]
            sheets.insert(0, ("FunctionFrequencySummary", _format_sheet_columns(scope_freq)))

        # Matrix second in tab order; shows all frequencies
        if not matrix_df.empty:
            sheets.insert(1, ("EnterpriseFunctionUsageMatrix", matrix_df))

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
            unlock_df = _format_sheet_columns(enterprise_unlock_view_df)
            if args.dashboard_max_unlock_rows is not None:
                unlock_df = unlock_df.head(args.dashboard_max_unlock_rows)
            export_for_dashboard[scope_label] = {
                "enterpriseSummary": _dataframe_to_json_records(_format_sheet_columns(enterprise_summary_df)),
                "trancheSummary": _dataframe_to_json_records(_format_sheet_columns(tranche_summary_df)),
                "enterpriseUnlockByStep": _dataframe_to_json_records(unlock_df),
                # workflowMap stored only at top level to reduce file size
            }

    if args.export_json is not None and export_for_dashboard:
        out_json = Path(args.export_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {"scopes": export_for_dashboard, "workflowMap": workflow_map}
        indent = 2 if args.dashboard_pretty else None
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=indent, default=str)
        print(f"\nExported dashboard JSON: {out_json}")

    print(f"\nOutputs: " + ", ".join(f"{p.name} ({s})" for p, s in zip(saved_paths, ["Daily", "Daily+Weekly", "Daily+Weekly+Monthly"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
