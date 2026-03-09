# CDK Greedy Function Sequencing

Chooses which functions to modernize first so that enterprises (and ARR) are unlocked as quickly as possible.

## Setup

```bash
pip install -r requirements.txt
```

## How to build the input Excel(s)

### Option 1: One Excel file (two sheets)

Create a single `.xlsx` file with **two sheets**:

| Sheet name (default) | Purpose |
|----------------------|--------|
| **FunctionMapping**  | Which enterprises use which functions, and completion status |
| **EnterpriseARR**    | Enterprise ID and ARR (revenue) |

**Sheet 1 — FunctionMapping**  
Column headers must be exactly:

| Column           | Meaning |
|------------------|--------|
| ENTERPRISE ID    | Unique ID for the customer/enterprise (e.g. numeric or string). Must match the same column in the ARR sheet. |
| NAME             | Display name for the enterprise (can be empty). |
| FUNCTION         | Name of the function (e.g. "Payroll", "Inventory"). |
| COMPLETED        | One of: `Yes`, `Partial`, `No`, `Outside baseline`. Rows with "Outside baseline" are excluded. An (enterprise, function) pair counts as done only if all its rows are "Yes" (or "Partial" if you use that option). |
| USAGE_PATTERN    | One of: `1 - daily`, `2 - weekly`, `3 - monthly`. Other values are excluded. |

You can have multiple rows per (ENTERPRISE ID, FUNCTION) (e.g. different usage patterns); all must be completed for that pair to count as done.

**Sheet 2 — EnterpriseARR**  
Column headers must be exactly:

| Column        | Meaning |
|---------------|--------|
| ENTERPRISE ID | Same IDs as in FunctionMapping. |
| ARR           | Numeric annual recurring revenue. Non-numeric or blank is treated as 0. |

Only enterprises with ARR ≥ `--arr-threshold` (default 0) are included.

### Option 2: Two separate Excel files

- **Function mapping file:** One sheet with the same columns as the "FunctionMapping" sheet above. The script uses the first sheet.
- **Enterprise ARR file:** One sheet with the same columns as the "EnterpriseARR" sheet above. The script uses the first sheet.

Column names are case-sensitive and must match exactly.

## Running

**One Excel (two sheets):**

```bash
python FunctionSequencing.py --input-excel "CDK_Input.xlsx"
```

Custom sheet names:

```bash
python FunctionSequencing.py --input-excel "CDK_Input.xlsx" --sheet-function-mapping "Usage" --sheet-enterprise-arr "Revenue"
```

**Two Excel files:**

```bash
python FunctionSequencing.py --function-mapping "CDK_FunctionFrequencyMapping.xlsx" --enterprise-arr "CDK_ARRByEnterprise.xlsx"
```

**Common options:**

- `--k 150` — Max number of functions to select (default 150).
- `--arr-threshold 250000` — Minimum ARR for an enterprise to be included (default 0).
- `--objective logo_progress` or `--objective arr_progress` — Optimize for number of enterprises unlocked vs ARR unlocked (default: logo_progress).
- `--out-steps`, `--out-enterprises`, `--out-summary` — Output CSV paths.

Outputs: step summary, per-enterprise unlock detail, and a tranche (step) ARR summary.
