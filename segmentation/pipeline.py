"""
Customer-level configuration segmentation (Steps 4–10).

Column selections (from your Step 3 guidance; no additional columns used):
- Enterprise_Setup_Map.csv: Bucket (joined for reference on outputs only; excluded from clustering features)
- bobs_account_stuff: manufacturer_stmt (+ cnumber)
- 3PA_Internal_External_Usage Data: VENDOR, CNUM + Partner_Internal_External_Key
- 3PA_Titan_DataLake_Usage Data (optional): COMPONENT, CNUM — suffix after pip./dsapi. (case-
  insensitive), prefixed "Titan Data Lake - ", merged as INTERNAL supplement
- spooler_data: type, cnumber (bucketed: MICR / ADP_CDK / OTHER; UNSET dropped)
- form_data: formname, cnumber (canonical keys: trailing numeric suffix stripped)
- users_plus_profiles: profiles, cnumber (comma-split; trailing numeric suffix stripped like forms, then uppercased)
- Cnumber_to_Enterprise_Map: cnumber → Enterprise ID

Run from repo root:
  python -m segmentation.pipeline
"""
from __future__ import annotations

import codecs
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent

AGGREGATION_CHOICES = """
STEP 4 — AGGREGATION CHOICES (Enterprise ID is the unit of analysis)
======================================================================

Spine: Enterprise_Setup_Map.csv INNER restricted to Enterprise IDs that appear
at least once in Cnumber_to_Enterprise_Map.xlsx (Mapping). Enterprises with no
mapped cnumber are EXCLUDED from clustering outputs entirely.

cnumber → Enterprise ID: All pairs from Cnumber_to_Enterprise_Map.xlsx sheet
'Mapping' (after dropping null cnumber or Enterprise ID). If one cnumber maps
to multiple Enterprise IDs, the same cnumber-level facts are included in each
linked enterprise’s roll-up (union/max rules apply per enterprise’s edge list).

CSV text encoding (BOB, forms, spooler, users exports):
   - Read the first ~1MB of the file as bytes. If it decodes as UTF-8 (or
     UTF-8-BOM), use that. Otherwise assume Windows-1252 (cp1252), which matches
     typical US/Windows-sourced CSVs and avoids “always succeeds” mis-reads from
     latin-1 on ambiguous bytes.

1) Enterprise_Setup_Map (Bucket)
   - Joined from Enterprise_Setup_Map.csv for traceability on exports (customer_clusters, etc.).
   - Excluded from k-means / Jaccard — not a segmentation feature.

2) OEM / BOB (manufacturer_stmt)
   - Tokenize: split non-null values on a delimiter class [; , | / +] (one or
     more consecutive). Strip empties; uppercase tokens. In the current extract,
     empirical checks show semicolons only; other separators are supported for
     forward compatibility (slash could split composite labels—review if that
     appears in your source).
   - Enterprise metric: COUNT DISTINCT OEM tokens across ALL BOB rows whose
     cnumber maps to that enterprise (union across rooftops and cnubers).
   - flag_bobs_oem: 1 if enterprise has ≥1 mapped cnumber with any non-empty
     manufacturer_stmt after tokenization, else 0.

3) 3PA by partner class (Data.VENDOR, Data.CNUM + Partner_Internal_External_Key)
   - Classify VENDOR against Key.PARTNER ID (case-insensitive; with/without '3PA_' prefix).
   - Key.TYPE drives four buckets: INTERNAL, EXTERNAL, DATA YOUR WAY, and UNMAPPED
     (vendors not on the key).
   - Supplement: 3PA_Titan_DataLake_Usage.xlsx sheet Data (CNUM, COMPONENT) inner-joined to
     the mapping spine; COMPONENT yields an integration token = text after the first
     case-insensitive "pip." or "dsapi.", prefixed "Titan Data Lake - " with the trailing
     segment uppercased like other integration tokens. Every such row is INTERNAL and unioned
     with key-classified internals
     (also raises cnt_3pa_vendors_all and flag_3pa when present).
   - Enterprise: COUNT DISTINCT VENDOR (uppercased) per bucket over rows whose CNUM
     maps to the enterprise (union). cnt_3pa_vendors_all = distinct across all rows.
   - EXTERNAL bucket only: vendor 3PA_PRODUCTS is excluded from counts and from Jaccard
     / exact-signature external vendor sets (other buckets unchanged).
   - Set-overlap views (Jaccard, cohort coverage, exact-setup signatures): use INTERNAL and
     EXTERNAL 3PA only; DATA YOUR WAY and UNMAPPED are excluded there. Org setups from
     Enterprise_Setup_Map (`Bucket`) appear as dimension ``enterprise_setup`` (distinct labels
     in the spine, typically on the order of seven).

4) Spooler (type, cnumber)
   - Rows with type UNSET (case-insensitive) are dropped from the model.
   - Remaining types map to buckets: MICR if the code contains MICR; else ADP_CDK if
     the code starts with ADP or CDK; else OTHER.
   - Enterprise: COUNT DISTINCT bucket labels (not raw printer codes).

5) Forms (formname, cnumber)
   - Form names are canonicalized by stripping a trailing numeric suffix (optional
     separators before the digits) so codes that share the same letter prefix and
     differ only by that suffix count as one form (e.g. INV01 and INV-2 -> INV).
   - Enterprise: COUNT DISTINCT canonical form keys over mapped cnubers (union).

6) User profiles (users_plus_profiles.profiles, cnumber) — not Enterprise_Setup_Map
   - The `profiles` field carries **user profile** codes (often many distinct values per enterprise).
   - Canonicalization matches forms: optional spaces/hyphens/underscores before a trailing run of digits
     are stripped so versioned codes collapse (e.g. ROLE01 and ROLE-2 -> ROLE); then uppercased.
   - Enterprise: COUNT DISTINCT canonical profile keys over mapped cnubers (union).
   - profiles: split on commas (no spaces between tokens); each piece canonicalized as above.
   - flag_profiles: 1 if any non-empty canonical profile key on a mapped cnumber, else 0.
   - Enterprise setup type is **Enterprise_Setup_Map** (`Bucket`); it is not derived from `profiles`.

Missing cnubers: excluded at spine level (no row in outputs).

STEP 5–6: Numeric features = log1p(counts) + binary flags (Enterprise_Setup_Map / Bucket excluded).
StandardScaler fit on the full matrix used for clustering.

STEP 7: k-means (primary assignments), Ward hierarchical + Gaussian Mixture
(same scaled features) for validation / comparison, k ∈ {3,…,7}.
"""


def csv_text_encoding(path: Path, chunk_size: int = 1 << 20) -> str:
    """Use UTF-8 only if the *entire* file is valid UTF-8 (chunk-safe); else cp1252."""
    for enc in ("utf-8-sig", "utf-8"):
        decoder = codecs.getincrementaldecoder(enc)(errors="strict")
        try:
            with open(path, "rb") as fh:
                while True:
                    block = fh.read(chunk_size)
                    if not block:
                        break
                    decoder.decode(block, final=False)
                decoder.decode(b"", final=True)
            return enc
        except UnicodeDecodeError:
            continue
    return "cp1252"


def load_edges() -> pd.DataFrame:
    m = pd.read_excel(ROOT / "Cnumber_to_Enterprise_Map.xlsx", sheet_name="Mapping")
    m = m.dropna(subset=["cnumber", "Enterprise ID"])
    m["cnumber"] = m["cnumber"].astype(str).str.strip().str.upper()
    m["Enterprise ID"] = m["Enterprise ID"].astype(str).str.strip()
    return m[["cnumber", "Enterprise ID"]].drop_duplicates()


def load_spine(enterprise_ids_with_cnum: set[str]) -> pd.DataFrame:
    setup = pd.read_csv(ROOT / "Enterprise_Setup_Map.csv")
    setup["Enterprise ID"] = setup["Enterprise ID"].astype(str).str.strip()
    setup["Bucket"] = setup["Bucket"].astype(str).str.strip()
    out = setup[["Enterprise ID", "Enterprise Name", "Bucket"]].drop_duplicates("Enterprise ID")
    out = out[out["Enterprise ID"].isin(enterprise_ids_with_cnum)]
    return out


def enterprise_setup_frozenset(bucket_raw) -> frozenset[str]:
    """Enterprise_Setup_Map ``Bucket`` as a frozenset (0 or 1 label), uppercased for set overlap."""
    if pd.isna(bucket_raw):
        return frozenset()
    s = str(bucket_raw).strip()
    if not s:
        return frozenset()
    return frozenset({s.upper()})


def oem_tokens(cell) -> set[str]:
    if pd.isna(cell):
        return set()
    # Empirically ';' in current BOB extract; allow common list-style delimiters.
    parts = re.split(r"[;,\|/+]+", str(cell))
    return {p.strip().upper() for p in parts if p and str(p).strip()}


def build_bobs_features(edges: pd.DataFrame, encoding: str) -> tuple[pd.Series, pd.DataFrame]:
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
    cnt = d.groupby("Enterprise ID")["_oem"].nunique().rename("cnt_oem_codes")
    bb = bobs.dropna(subset=["manufacturer_stmt"]).copy()
    bb["_s"] = bb["manufacturer_stmt"].astype(str).str.strip()
    bb = bb[bb["_s"] != ""]
    has_ids = bb.merge(edges, on="cnumber", how="inner")["Enterprise ID"].drop_duplicates()
    flag = pd.Series(1, index=has_ids, name="flag_bobs_oem", dtype=int)
    return cnt, flag.to_frame()


# Excluded from EXTERNAL 3PA bucket only (counts + set-overlap / signatures).
EXCLUDED_EXTERNAL_3PA_VENDORS = frozenset({"3PA_PRODUCTS"})

TITAN_DATALAKE_USAGE_XLSX = ROOT / "3pa" / "3PA_Titan_DataLake_Usage.xlsx"
TITAN_DATALAKE_INTEGRATION_PREFIX = "Titan Data Lake - "


def titan_component_to_integration_key(raw) -> str | None:
    """
    From Titan Data.COMPONENT, take the substring after the first case-insensitive
    "pip." or "dsapi."; return TITAN_DATALAKE_INTEGRATION_PREFIX plus the suffix uppercased.
    """
    if pd.isna(raw):
        return None
    s = str(raw).strip()
    if not s:
        return None
    lower = s.lower()
    for needle in ("pip.", "dsapi."):
        i = lower.find(needle)
        if i != -1:
            rest = s[i + len(needle) :].strip()
            if not rest:
                return None
            return TITAN_DATALAKE_INTEGRATION_PREFIX + rest.upper()
    return None


def bucket_printer_type(raw) -> str | None:
    """
    Map raw spooler `type` to a coarse bucket for modeling. Returns None to exclude
    the row (unset).
    """
    if pd.isna(raw):
        return None
    s = str(raw).strip()
    if not s:
        return None
    if re.fullmatch(r"(?i)unset", s):
        return None
    u = s.upper()
    if "MICR" in u:
        return "MICR"
    if u.startswith("ADP") or u.startswith("CDK"):
        return "ADP_CDK"
    return "OTHER"


_formname_trailing_numeric = re.compile(r"[\s\-_]*\d+$", re.UNICODE)


def normalize_formname_token(raw) -> str | None:
    """
    Collapse form codes that share the same non-numeric prefix and differ only by a
    trailing numeric suffix.
    """
    if pd.isna(raw):
        return None
    s = str(raw).strip()
    if not s:
        return None
    core = _formname_trailing_numeric.sub("", s).strip()
    if not core:
        return None
    return core.upper()


def normalize_user_profile_token(raw) -> str | None:
    """
    Collapse profile codes that share the same non-numeric prefix and differ only by a
    trailing numeric suffix (same rule as normalize_formname_token).
    """
    if pd.isna(raw):
        return None
    s = str(raw).strip()
    if not s:
        return None
    core = _formname_trailing_numeric.sub("", s).strip()
    if not core:
        return None
    return core.upper()


def vendor_classifier(key: pd.DataFrame):
    id_to_type: dict[str, str] = {}
    for _, r in key.iterrows():
        pid = str(r["PARTNER ID"]).strip().upper()
        typ = str(r["TYPE"]).strip().upper()
        id_to_type[pid] = typ
        if pid.startswith("3PA_"):
            id_to_type[pid[4:]] = typ

    def cls(v: str) -> str:
        vu = str(v).strip().upper()
        if vu in id_to_type:
            return id_to_type[vu]
        if "3PA_" + vu in id_to_type:
            return id_to_type["3PA_" + vu]
        return "UNMAPPED"

    return cls


def titan_datalake_internal_usage_rows(edges: pd.DataFrame) -> pd.DataFrame:
    """
    Titan DataLake usage merged to Enterprise ID, all vendor_type INTERNAL.
    Reads COMPONENT (not VENDOR); integration id = suffix after pip./dsapi. with
    TITAN_DATALAKE_INTEGRATION_PREFIX, stored in VENDOR_U for pipeline compatibility.
    Missing file or unreadable sheet/columns yields an empty frame (optional input).
    """
    empty = pd.DataFrame(columns=["cnumber", "Enterprise ID", "VENDOR_U", "vendor_type"])
    if not TITAN_DATALAKE_USAGE_XLSX.exists():
        return empty
    try:
        t = pd.read_excel(
            TITAN_DATALAKE_USAGE_XLSX,
            sheet_name="Data",
            engine="openpyxl",
            usecols=["CNUM", "COMPONENT"],
        )
    except Exception:
        return empty
    t = t.dropna(subset=["CNUM", "COMPONENT"])
    t["cnumber"] = t["CNUM"].astype(str).str.strip().str.upper()
    t["VENDOR_U"] = t["COMPONENT"].map(titan_component_to_integration_key)
    t = t.dropna(subset=["VENDOR_U"])
    t = t.loc[t["cnumber"] != ""]
    t["vendor_type"] = "INTERNAL"
    m = t.merge(edges, on="cnumber", how="inner")
    return m[["cnumber", "Enterprise ID", "VENDOR_U", "vendor_type"]]


def three_pa_usage_classified(edges: pd.DataFrame) -> pd.DataFrame:
    """
    Usage rows joined to Enterprise ID with vendor_type from Partner_Internal_External_Key.TYPE:
    INTERNAL | EXTERNAL | DATA YOUR WAY | UNMAPPED.

    Titan DataLake usage (3PA_Titan_DataLake_Usage.xlsx, Data sheet) is concatenated
    as INTERNAL-only rows derived from COMPONENT (pip./dsapi. suffix + "Titan Data Lake - "
    prefix) so internal counts, all-vendor counts, and Jaccard internal sets union with
    internals from the partner key.
    """
    usage = pd.read_excel(
        ROOT / "3pa" / "3PA_Internal_External_Usage.xlsx",
        sheet_name="Data",
        engine="openpyxl",
    )
    key = pd.read_excel(
        ROOT / "3pa" / "3PA_Internal_External_Usage.xlsx",
        sheet_name="Partner_Internal_External_Key",
        engine="openpyxl",
    )
    classify = vendor_classifier(key)
    usage["CNUM"] = usage["CNUM"].astype(str).str.strip().str.upper()
    usage["VENDOR_U"] = usage["VENDOR"].astype(str).str.strip().str.upper()
    usage["vendor_type"] = usage["VENDOR"].map(classify)
    base = usage.rename(columns={"CNUM": "cnumber"}).merge(edges, on="cnumber", how="inner")
    tit = titan_datalake_internal_usage_rows(edges)
    if len(tit) == 0:
        return base
    if len(base) == 0:
        return tit
    return pd.concat([base, tit], ignore_index=True)


def build_3pa_features(edges: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = three_pa_usage_classified(edges)
    cols = [
        "cnt_3pa_vendors_all",
        "cnt_3pa_vendors_internal",
        "cnt_3pa_vendors_external",
        "cnt_3pa_vendors_data_your_way",
        "cnt_3pa_vendors_unmapped",
    ]
    if len(d) == 0:
        eidx = pd.Index(sorted(edges["Enterprise ID"].unique()))
        z = pd.DataFrame(0, index=eidx, columns=cols, dtype=int)
        return z, pd.DataFrame({"flag_3pa": 0}, index=eidx, dtype=int)

    out = pd.DataFrame()
    out["cnt_3pa_vendors_all"] = d.groupby("Enterprise ID")["VENDOR_U"].nunique()
    for label, sub in (
        ("internal", d["vendor_type"] == "INTERNAL"),
        (
            "external",
            (d["vendor_type"] == "EXTERNAL")
            & (~d["VENDOR_U"].isin(EXCLUDED_EXTERNAL_3PA_VENDORS)),
        ),
        ("data_your_way", d["vendor_type"] == "DATA YOUR WAY"),
        ("unmapped", d["vendor_type"] == "UNMAPPED"),
    ):
        out[f"cnt_3pa_vendors_{label}"] = d.loc[sub].groupby("Enterprise ID")["VENDOR_U"].nunique()
    out = out.fillna(0).astype(int)
    flag = (out["cnt_3pa_vendors_all"] > 0).astype(int).rename("flag_3pa").to_frame()
    return out, flag


def three_pa_vendor_frozensets_by_type(edges: pd.DataFrame) -> dict[str, pd.Series]:
    """
    Per enterprise, frozenset of distinct VENDOR_U for each Partner key TYPE bucket.
    Keys: internal, external, data_your_way, unmapped.
    Internal includes Titan DataLake COMPONENT-derived integrations (see three_pa_usage_classified).
    """
    d = three_pa_usage_classified(edges)
    mapping = {
        "internal": "INTERNAL",
        "external": "EXTERNAL",
        "data_your_way": "DATA YOUR WAY",
        "unmapped": "UNMAPPED",
    }
    out: dict[str, pd.Series] = {}
    for k, typ in mapping.items():
        if len(d) == 0:
            out[k] = pd.Series(dtype=object)
            continue
        sub = d.loc[d["vendor_type"] == typ]
        if k == "external":
            sub = sub.loc[~sub["VENDOR_U"].isin(EXCLUDED_EXTERNAL_3PA_VENDORS)]
        out[k] = (
            sub.groupby("Enterprise ID")["VENDOR_U"].agg(frozenset) if len(sub) else pd.Series(dtype=object)
        )
    return out


def rollup_union_long(
    edges: pd.DataFrame, path: Path, ccol: str, vcol: str, encoding: str
) -> pd.Series:
    kw = dict(chunksize=200_000, low_memory=False, encoding=encoding)
    chunks = []
    for ch in pd.read_csv(path, usecols=[ccol, vcol], **kw):
        ch[ccol] = ch[ccol].astype(str).str.strip().str.upper()
        d = ch[[ccol, vcol]].dropna(subset=[ccol, vcol])
        chunks.append(d.merge(edges, left_on=ccol, right_on="cnumber", how="inner"))
    if not chunks:
        return pd.Series(dtype=int)
    all_d = pd.concat(chunks, ignore_index=True)
    return all_d.groupby("Enterprise ID")[vcol].nunique()


def rollup_spooler_bucket_sets(edges: pd.DataFrame, encoding: str) -> pd.Series:
    """Per enterprise, frozenset of spooler bucket labels (UNSET rows excluded)."""
    path = ROOT / "spooler_data" / "spooler_data.csv"
    kw = dict(chunksize=200_000, low_memory=False, encoding=encoding)
    chunks = []
    for ch in pd.read_csv(path, usecols=["cnumber", "type"], **kw):
        ch["cnumber"] = ch["cnumber"].astype(str).str.strip().str.upper()
        ch["_bkt"] = ch["type"].map(bucket_printer_type)
        d = ch.dropna(subset=["cnumber", "_bkt"])
        chunks.append(d.merge(edges, on="cnumber", how="inner"))
    if not chunks:
        return pd.Series(dtype=object)
    all_d = pd.concat(chunks, ignore_index=True)
    return all_d.groupby("Enterprise ID")["_bkt"].agg(frozenset)


def rollup_form_canonical_sets(edges: pd.DataFrame, encoding: str) -> pd.Series:
    """Per enterprise, frozenset of canonical form keys (see normalize_formname_token)."""
    path = ROOT / "form_data" / "form_data.csv"
    kw = dict(chunksize=200_000, low_memory=False, encoding=encoding)
    chunks = []
    for ch in pd.read_csv(path, usecols=["cnumber", "formname"], **kw):
        ch["cnumber"] = ch["cnumber"].astype(str).str.strip().str.upper()
        ch["_fn"] = ch["formname"].map(normalize_formname_token)
        d = ch.dropna(subset=["cnumber", "_fn"])
        chunks.append(d.merge(edges, on="cnumber", how="inner"))
    if not chunks:
        return pd.Series(dtype=object)
    all_d = pd.concat(chunks, ignore_index=True)
    return all_d.groupby("Enterprise ID")["_fn"].agg(frozenset)


def build_spooler_features(edges: pd.DataFrame, encoding: str) -> tuple[pd.Series, pd.Series]:
    fs = rollup_spooler_bucket_sets(edges, encoding)
    eidx = pd.Index(sorted(edges["Enterprise ID"].unique()))
    if len(fs) == 0:
        cnt = pd.Series(0, index=eidx, name="cnt_spooler_types", dtype=int)
    else:
        cnt = fs.map(len).reindex(eidx, fill_value=0).astype(int).rename("cnt_spooler_types")
    flag = (cnt > 0).astype(int).rename("flag_spooler")
    return cnt, flag


def build_form_features(edges: pd.DataFrame, encoding: str) -> tuple[pd.Series, pd.Series]:
    fs = rollup_form_canonical_sets(edges, encoding)
    eidx = pd.Index(sorted(edges["Enterprise ID"].unique()))
    if len(fs) == 0:
        fn = pd.Series(0, index=eidx, name="cnt_formnames", dtype=int)
    else:
        fn = fs.map(len).reindex(eidx, fill_value=0).astype(int).rename("cnt_formnames")
    flag = (fn > 0).astype(int).rename("flag_forms")
    return fn, flag


def split_profile_tokens(x) -> set[str]:
    """Comma-split profiles; each token passed through normalize_user_profile_token."""
    if pd.isna(x):
        return set()
    out: set[str] = set()
    for part in str(x).split(","):
        t = normalize_user_profile_token(part)
        if t:
            out.add(t)
    return out


def build_profile_features(edges: pd.DataFrame, encoding: str) -> tuple[pd.Series, pd.DataFrame]:
    path = ROOT / "users_plus_profiles" / "users_plus_profiles.csv"
    kw = dict(chunksize=200_000, low_memory=False, encoding=encoding)
    eidx = pd.Index(sorted(edges["Enterprise ID"].unique()))

    chunks_p = []
    for ch in pd.read_csv(path, usecols=["cnumber", "profiles"], **kw):
        ch["cnumber"] = ch["cnumber"].astype(str).str.strip().str.upper()
        pe = ch.assign(_p=ch["profiles"].map(split_profile_tokens)).explode("_p")
        pe = pe.dropna(subset=["_p"])
        chunks_p.append(pe.merge(edges, on="cnumber", how="inner"))
    long_p = pd.concat(chunks_p, ignore_index=True) if chunks_p else pd.DataFrame()
    if len(long_p) == 0:
        cnt_p = pd.Series(0, index=eidx, name="cnt_profile_tokens", dtype=int)
        flag_df = pd.DataFrame({"flag_profiles": 0}, index=eidx)
        return cnt_p, flag_df
    cnt_p = long_p.groupby("Enterprise ID")["_p"].nunique().rename("cnt_profile_tokens")
    flag_df = (cnt_p > 0).astype(int).rename("flag_profiles").to_frame()
    return cnt_p, flag_df


def assemble_enterprise_table() -> tuple[pd.DataFrame, dict[str, str]]:
    edges = load_edges()
    ent_ok = set(edges["Enterprise ID"].unique())
    spine = load_spine(ent_ok)

    enc_meta = {
        "bobs": csv_text_encoding(ROOT / "bobs_account_stuff" / "bobs_account_stuff.csv"),
        "forms": csv_text_encoding(ROOT / "form_data" / "form_data.csv"),
        "spooler": csv_text_encoding(ROOT / "spooler_data" / "spooler_data.csv"),
        "users": csv_text_encoding(ROOT / "users_plus_profiles" / "users_plus_profiles.csv"),
    }

    b_cnt, b_flag = build_bobs_features(edges, enc_meta["bobs"])
    pa, fl_3pa = build_3pa_features(edges)
    sp_cnt, sp_flag = build_spooler_features(edges, enc_meta["spooler"])
    fn_cnt, ff_flag = build_form_features(edges, enc_meta["forms"])
    p_cnt, p_flag = build_profile_features(edges, enc_meta["users"])

    df = spine.set_index("Enterprise ID")
    for part in (
        b_cnt,
        b_flag,
        pa,
        fl_3pa,
        sp_cnt,
        sp_flag,
        fn_cnt,
        ff_flag,
        p_cnt,
        p_flag,
    ):
        df = df.join(part, how="left")

    fill0 = [
        "cnt_oem_codes",
        "cnt_3pa_vendors_all",
        "cnt_3pa_vendors_internal",
        "cnt_3pa_vendors_external",
        "cnt_3pa_vendors_data_your_way",
        "cnt_3pa_vendors_unmapped",
        "cnt_spooler_types",
        "cnt_formnames",
        "cnt_profile_tokens",
        "flag_bobs_oem",
        "flag_3pa",
        "flag_spooler",
        "flag_forms",
        "flag_profiles",
    ]
    for c in fill0:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype(float if c.startswith("cnt_") else int)

    return df.reset_index(), enc_meta


def build_model_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    count_cols = [
        "cnt_oem_codes",
        "cnt_3pa_vendors_all",
        "cnt_3pa_vendors_internal",
        "cnt_3pa_vendors_external",
        "cnt_3pa_vendors_data_your_way",
        "cnt_3pa_vendors_unmapped",
        "cnt_spooler_types",
        "cnt_formnames",
        "cnt_profile_tokens",
    ]
    flag_cols = [
        "flag_bobs_oem",
        "flag_3pa",
        "flag_spooler",
        "flag_forms",
        "flag_profiles",
    ]
    cnt_df = df[count_cols].apply(np.log1p)
    X_parts = [cnt_df.reset_index(drop=True)]
    X_parts.append(df[flag_cols].reset_index(drop=True))
    X = pd.concat(X_parts, axis=1)
    feature_names = list(X.columns)
    return X, feature_names, count_cols + flag_cols


def refine_labels(df_raw: pd.DataFrame) -> dict[int, str]:
    """Distinct, business-oriented names from cluster centroids (k-means ids are arbitrary)."""
    pop = df_raw.drop(columns=["cluster"]).mean(numeric_only=True)
    cent = df_raw.groupby("cluster").mean(numeric_only=True)
    sizes = df_raw["cluster"].value_counts()

    def name_for_row(cid: int, row: pd.Series) -> str:
        if row["cnt_formnames"] < 1.0 and row["cnt_profile_tokens"] > pop["cnt_profile_tokens"] * 1.2:
            return "User profile breadth without form/spooler visibility (check data coverage)"
        if (
            row["cnt_profile_tokens"] > pop["cnt_profile_tokens"] * 1.6
            and row["cnt_3pa_vendors_external"] > pop["cnt_3pa_vendors_external"] * 1.4
            and row["cnt_formnames"] > pop["cnt_formnames"] * 1.2
        ):
            return "Enterprise breadth — heavy forms, external 3PA, and user profile mix"
        if row["cnt_3pa_vendors_internal"] > pop["cnt_3pa_vendors_internal"] * 1.15 and row[
            "cnt_profile_tokens"
        ] > pop["cnt_profile_tokens"] * 1.1:
            return "Elevated user profiles with richer internal 3PA mix"
        # Below-average user profile breadth vs population.
        if row["cnt_profile_tokens"] < pop["cnt_profile_tokens"] * 0.45:
            if row["cnt_formnames"] >= pop["cnt_formnames"] * 0.75:
                return "Lower user profile breadth but stronger form-name diversity"
            if int(sizes[cid]) >= 500:
                return "Largest cohort — typical forms with modest user profile counts"
            if int(sizes[cid]) >= 200:
                return "Mid cohort — typical forms with modest user profile counts"
            return "Niche cohort — modest user profiles with mid-tier form counts"
        if row["cnt_profile_tokens"] < pop["cnt_profile_tokens"] * 0.75:
            return "Lean mainstream — moderate user profiles and forms"
        return "Core mainstream — typical user profiles, forms, and 3PA breadth"

    return {int(cid): name_for_row(int(cid), cent.loc[cid]) for cid in cent.index}


# Deck / PowerPoint exports (business column names; same numbers as model outputs)
DECK_SUMMARY_RENAMES: dict[str, str] = {
    "cluster": "Segment ID",
    "cluster_label": "Segment name",
    "size": "Enterprise count",
    "pct_of_customers": "Pct of enterprises (%)",
    "description": "Narrative (segment avg vs all enterprises)",
    "avg_cnt_oem_codes": "Avg distinct OEM codes (BOB footprint)",
    "avg_cnt_3pa_vendors_all": "Avg distinct 3PA vendors (all)",
    "avg_cnt_3pa_vendors_external": "Avg distinct 3PA vendors (external-classified)",
    "avg_cnt_3pa_vendors_internal": "Avg distinct 3PA vendors (internal-classified)",
    "avg_cnt_3pa_vendors_data_your_way": "Avg distinct 3PA vendors (Data Your Way)",
    "avg_cnt_3pa_vendors_unmapped": "Avg distinct 3PA vendors (unmapped key)",
    "avg_cnt_spooler_types": "Avg distinct spooler types",
    "avg_cnt_formnames": "Avg distinct form names",
    "avg_cnt_profile_tokens": "Avg distinct user profile codes",
    "avg_flag_3pa": "Share with any 3PA usage (0-1)",
    "avg_flag_forms": "Share with any forms data (0-1)",
    "avg_flag_profiles": "Share with any user profile codes (0-1)",
}

DECK_CUSTOMER_RENAMES: dict[str, str] = {
    "Enterprise ID": "Enterprise ID",
    "Bucket": "Enterprise_Setup_Map (reference only)",
    "cluster": "Segment ID",
    "cluster_label": "Segment name",
    "key_features": "Top deviations vs typical enterprise (σ)",
}


def feature_glossary_for_deck_rows() -> list[dict[str, str]]:
    """Static definitions aligned with segmentation features (for slides / appendix)."""
    return [
        {
            "model_column": "Bucket",
            "slide_name": "Enterprise_Setup_Map",
            "definition": "Bucket label from Enterprise_Setup_Map.csv; carried on outputs for traceability only — not used in clustering.",
        },
        {
            "model_column": "cnt_oem_codes",
            "slide_name": "Distinct OEM codes",
            "definition": "Count of distinct BOB manufacturer_stmt tokens after splitting on ; , | / +, uppercased, union across all mapped cnumbers for the enterprise.",
        },
        {
            "model_column": "cnt_3pa_vendors_all",
            "slide_name": "Distinct 3PA vendors (all)",
            "definition": "Distinct integration keys for mapped cnumbers: Internal/External usage Data plus optional Titan DataLake COMPONENT rows (suffix after pip./dsapi., prefixed Titan Data Lake - ), all partner classes combined.",
        },
        {
            "model_column": "cnt_3pa_vendors_external",
            "slide_name": "Distinct 3PA vendors (external)",
            "definition": "Distinct vendors whose Partner_Internal_External_Key.TYPE is EXTERNAL, excluding vendor 3PA_PRODUCTS.",
        },
        {
            "model_column": "cnt_3pa_vendors_internal",
            "slide_name": "Distinct 3PA vendors (internal)",
            "definition": "Distinct vendors whose Partner_Internal_External_Key.TYPE is INTERNAL, unioned with Titan DataLake COMPONENT integrations (text after pip. or dsapi., prefixed Titan Data Lake - ).",
        },
        {
            "model_column": "cnt_3pa_vendors_data_your_way",
            "slide_name": "Distinct 3PA vendors (Data Your Way)",
            "definition": "Distinct vendors whose Partner_Internal_External_Key.TYPE is DATA YOUR WAY.",
        },
        {
            "model_column": "cnt_3pa_vendors_unmapped",
            "slide_name": "Distinct 3PA vendors (unmapped)",
            "definition": "Distinct vendors not matched to the partner key table (retained for transparency; usually small).",
        },
        {
            "model_column": "cnt_spooler_types",
            "slide_name": "Distinct spooler types",
            "definition": "Count of distinct spooler buckets: MICR (code contains MICR), ADP_CDK (starts with ADP or CDK), OTHER; UNSET rows excluded.",
        },
        {
            "model_column": "cnt_formnames",
            "slide_name": "Distinct form names",
            "definition": "Count of distinct canonical form keys: trailing numeric suffix stripped so versioned codes collapse (e.g. INV01 and INV-2 -> INV).",
        },
        {
            "model_column": "cnt_profile_tokens",
            "slide_name": "Distinct user profile codes",
            "definition": "Count of distinct canonical user profile keys from users_plus_profiles.profiles: comma-split, then trailing numeric suffix stripped (same rule as forms), uppercased, union across mapped cnumbers. Enterprise_Setup_Map (Bucket) is separate.",
        },
        {
            "model_column": "flag_bobs_oem",
            "slide_name": "Flag: BOB OEM present",
            "definition": "1 if any mapped cnumber has non-empty manufacturer_stmt after tokenization.",
        },
        {
            "model_column": "flag_3pa",
            "slide_name": "Flag: 3PA present",
            "definition": "1 if enterprise has at least one 3PA integration row on a mapped cnumber (Internal/External usage and/or parsable Titan DataLake COMPONENT).",
        },
        {
            "model_column": "flag_spooler",
            "slide_name": "Flag: spooler present",
            "definition": "1 if any spooler row exists for a mapped cnumber.",
        },
        {
            "model_column": "flag_forms",
            "slide_name": "Flag: forms present",
            "definition": "1 if any form_data row exists for a mapped cnumber.",
        },
        {
            "model_column": "flag_profiles",
            "slide_name": "Flag: user profiles present",
            "definition": "1 if any non-empty canonical user profile key exists for a mapped cnumber (after comma-split and trailing-number collapse).",
        },
        {
            "model_column": "key_features (σ)",
            "slide_name": "Sigma deviations",
            "definition": "Per enterprise: which raw count metrics are most above/below the full population mean, in standard-deviation units (not the scaled k-means inputs).",
        },
    ]


def deck_slide_outline_rows(
    n_enterprises: int,
    chosen_k: int,
    silhouette: float,
    min_floor: int,
    summary_df: pd.DataFrame,
) -> list[dict[str, str | int]]:
    """Suggested slide order + copy derived from model outputs (edit freely for the room)."""
    rows: list[dict[str, str | int]] = [
        {
            "slide_order": 1,
            "suggested_title": "Objective",
            "content": (
                "Segment enterprises by configuration breadth (OEM, 3PA, spooler, forms, user profile codes) "
                "for prioritization and coverage conversations."
            ),
        },
        {
            "slide_order": 2,
            "suggested_title": "Scope and unit of analysis",
            "content": (
                f"{n_enterprises} enterprises with at least one mapped cnumber in Cnumber_to_Enterprise_Map. "
                "Metrics are counts of distinct codes/types unioned across mapped rooftops—not revenue or "
                "transaction volume. Definitions: feature_glossary_for_deck.csv."
            ),
        },
        {
            "slide_order": 3,
            "suggested_title": "Method — how segments were formed",
            "content": (
                f"k-means with k={chosen_k} on standardized features (log1p counts + binary flags; Enterprise_Setup_Map excluded). "
                f"k-means silhouette at chosen k ≈ {silhouette:.3f}. "
                f"When choosing k, segments smaller than max(25, 1.5% of base) were avoided (floor={min_floor}). "
                "Charts: charts/silhouette_vs_k.png, charts/cluster_distribution.png, "
                "charts/feature_comparison_by_cluster.png."
            ),
        },
    ]
    order = 4
    for _, r in summary_df.sort_values("cluster").iterrows():
        cid = int(r["cluster"])
        rows.append(
            {
                "slide_order": order,
                "suggested_title": f"Segment {cid}: {r['cluster_label']}",
                "content": (
                    f"{int(r['size'])} enterprises ({r['pct_of_customers']}% of base). {r['description']}"
                ),
            }
        )
        order += 1
    rows.append(
        {
            "slide_order": order,
            "suggested_title": "Optional — Jaccard / set-overlap view",
            "content": (
                "Alternative lens: similarity from overlapping distinct values per dimension. "
                "Run python -m segmentation.jaccard_similarity_clustering; charts in charts/jaccard_clusters/."
            ),
        }
    )
    rows.append(
        {
            "slide_order": order + 1,
            "suggested_title": "Optional — exact duplicate configurations",
            "content": (
                "Pairs sharing identical signatures: exact_setup_duplicate_groups.csv "
                "(python -m segmentation.exact_setup_analysis)."
            ),
        }
    )
    return rows


def write_deck_csvs(
    root: Path,
    summary_df: pd.DataFrame,
    out_clusters: pd.DataFrame,
    df_raw: pd.DataFrame,
    n_enterprises: int,
    chosen_k: int,
    min_floor: int,
    metrics_df: pd.DataFrame,
    enc_meta: dict[str, str],
) -> None:
    """Write business-labeled CSVs for decks (same values as cluster_summary / customer_clusters)."""
    deck_summary = summary_df.rename(columns=DECK_SUMMARY_RENAMES)
    col_order = [c for c in DECK_SUMMARY_RENAMES.values() if c in deck_summary.columns]
    deck_summary = deck_summary[col_order]
    deck_summary.to_csv(root / "cluster_summary_for_deck.csv", index=False)

    deck_cust = out_clusters.rename(columns=DECK_CUSTOMER_RENAMES)
    corder = [c for c in DECK_CUSTOMER_RENAMES.values() if c in deck_cust.columns]
    deck_cust = deck_cust[corder]
    deck_cust.to_csv(root / "customer_clusters_for_deck.csv", index=False)

    pd.DataFrame(feature_glossary_for_deck_rows()).to_csv(
        root / "feature_glossary_for_deck.csv", index=False
    )

    # One-row population averages (append below segment table in Excel for “vs all”)
    inv = {v: k for k, v in DECK_SUMMARY_RENAMES.items()}
    pop: dict[str, str | float | int] = {}
    for deck_col in col_order:
        tech = inv[deck_col]
        if tech == "cluster":
            pop[deck_col] = "POP"
        elif tech == "cluster_label":
            pop[deck_col] = "All enterprises (population)"
        elif tech == "size":
            pop[deck_col] = int(n_enterprises)
        elif tech == "pct_of_customers":
            pop[deck_col] = 100.0
        elif tech == "description":
            pop[deck_col] = (
                "Benchmark row: numeric columns are population means (same as ‘pop’ in segment narratives)."
            )
        elif tech.startswith("avg_"):
            base = tech[len("avg_") :]
            if base in df_raw.columns:
                pop[deck_col] = round(float(df_raw[base].mean()), 4)
            else:
                pop[deck_col] = np.nan
        else:
            pop[deck_col] = ""
    pd.DataFrame([pop]).to_csv(root / "population_benchmarks_for_deck.csv", index=False)

    km_k = metrics_df[(metrics_df["method"] == "kmeans") & (metrics_df["k"] == chosen_k)].iloc[0]
    methodology = pd.DataFrame(
        [
            {
                "enterprises_in_model": n_enterprises,
                "number_of_segments_k": chosen_k,
                "primary_algorithm": "k-means (standardized features: log1p counts + binary flags; Enterprise_Setup_Map excluded)",
                "silhouette_kmeans_at_chosen_k": round(float(km_k["silhouette"]), 6),
                "smallest_segment_enterprises": int(km_k["min_cluster_size"]),
                "largest_segment_enterprises": int(km_k["max_cluster_size"]),
                "min_cluster_size_floor_used": min_floor,
                "scope_exclusion": "Enterprises with zero mapped cnumbers in Cnumber_to_Enterprise_Map are excluded.",
                "aggregation_detail": "See segmentation/pipeline.py AGGREGATION_CHOICES",
                "csv_encoding_bobs": enc_meta.get("bobs", ""),
                "csv_encoding_forms": enc_meta.get("forms", ""),
                "csv_encoding_spooler": enc_meta.get("spooler", ""),
                "csv_encoding_users": enc_meta.get("users", ""),
            }
        ]
    )
    methodology.to_csv(root / "deck_methodology_snapshot.csv", index=False)

    pd.DataFrame(
        deck_slide_outline_rows(
            n_enterprises,
            chosen_k,
            float(km_k["silhouette"]),
            min_floor,
            summary_df,
        )
    ).to_csv(root / "deck_slide_outline.csv", index=False)


def key_feature_strings(df_raw: pd.DataFrame) -> list[str]:
    work = df_raw.copy()
    num_cols = [
        "cnt_oem_codes",
        "cnt_3pa_vendors_all",
        "cnt_3pa_vendors_internal",
        "cnt_3pa_vendors_external",
        "cnt_3pa_vendors_data_your_way",
        "cnt_3pa_vendors_unmapped",
        "cnt_spooler_types",
        "cnt_formnames",
        "cnt_profile_tokens",
    ]
    glob = work[num_cols].mean()
    glob_s = work[num_cols].std().replace(0, 1.0)
    out = []
    for _, row in work.iterrows():
        z = (row[num_cols] - glob) / glob_s
        top = z.abs().sort_values(ascending=False).head(4)
        parts = [f"{n}={'+' if z[n]>=0 else ''}{z[n]:.2f}σ" for n in top.index]
        out.append("; ".join(parts))
    return out


def main() -> None:
    sns.set_theme(style="whitegrid")
    charts_dir = ROOT / "charts"
    charts_dir.mkdir(exist_ok=True)

    print("Building enterprise table…")
    ent, enc_meta = assemble_enterprise_table()
    print("CSV encodings used:", enc_meta)

    X, feature_names, raw_numeric_cols = build_model_matrix(ent)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    ids = ent["Enterprise ID"].values

    k_min, k_max = 3, 7
    rows_metrics = []
    inertias = []
    last_km: dict[int, KMeans] = {}

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lab_km = km.fit_predict(Xs)
        sil_km = silhouette_score(Xs, lab_km)
        inertias.append(km.inertia_)
        last_km[k] = km

        h = AgglomerativeClustering(n_clusters=k, linkage="ward")
        lab_h = h.fit_predict(Xs)
        sil_h = silhouette_score(Xs, lab_h)

        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=42,
            reg_covar=1e-3,
            n_init=3,
        )
        lab_g = gmm.fit(Xs).predict(Xs)
        sil_g = silhouette_score(Xs, lab_g)

        vc = pd.Series(lab_km).value_counts().sort_index()
        rows_metrics.append(
            {
                "k": k,
                "method": "kmeans",
                "silhouette": sil_km,
                "inertia": km.inertia_,
                "min_cluster_size": int(vc.min()),
                "max_cluster_size": int(vc.max()),
            }
        )
        rows_metrics.append(
            {
                "k": k,
                "method": "hierarchical_ward",
                "silhouette": sil_h,
                "inertia": np.nan,
                "min_cluster_size": int(pd.Series(lab_h).value_counts().min()),
                "max_cluster_size": int(pd.Series(lab_h).value_counts().max()),
            }
        )
        rows_metrics.append(
            {
                "k": k,
                "method": "gaussian_mixture",
                "silhouette": sil_g,
                "inertia": np.nan,
                "min_cluster_size": int(pd.Series(lab_g).value_counts().min()),
                "max_cluster_size": int(pd.Series(lab_g).value_counts().max()),
            }
        )

    metrics_df = pd.DataFrame(rows_metrics)

    # Choose k: maximize k-means silhouette subject to no tiny fragments (<~1.5% of base).
    n = len(ent)
    min_floor = max(25, int(0.015 * n))
    km_sil = metrics_df[metrics_df["method"] == "kmeans"].copy()
    km_ok = km_sil[km_sil["min_cluster_size"] >= min_floor]
    if km_ok.empty:
        chosen_k = int(km_sil.loc[km_sil["silhouette"].idxmax(), "k"])
    else:
        chosen_k = int(km_ok.loc[km_ok["silhouette"].idxmax(), "k"])
        best_sil = km_ok["silhouette"].max()
        near = km_ok[np.isclose(km_ok["silhouette"], best_sil, atol=0.02)]
        # Prefer slightly simpler k when silhouette is within 0.02 (interpretability).
        for prefer in (5, 4, 6):
            if prefer in near["k"].values:
                chosen_k = prefer
                break

    km_final = KMeans(n_clusters=chosen_k, random_state=42, n_init=20)
    labels = km_final.fit_predict(Xs)

    df_raw = ent.copy()
    df_raw["cluster"] = labels
    friendly = refine_labels(df_raw)

    # Key drivers: ANOVA F or between/within variance ratio simplified as range of cluster means
    num_cols = [
        "cnt_oem_codes",
        "cnt_3pa_vendors_all",
        "cnt_3pa_vendors_internal",
        "cnt_3pa_vendors_external",
        "cnt_3pa_vendors_data_your_way",
        "cnt_3pa_vendors_unmapped",
        "cnt_spooler_types",
        "cnt_formnames",
        "cnt_profile_tokens",
    ]
    cm = df_raw.groupby("cluster")[num_cols].mean()
    overall = df_raw[num_cols].mean()
    between = (cm - overall).abs().mean(axis=0).sort_values(ascending=False)
    drivers = list(between.head(6).index)

    key_strs = key_feature_strings(df_raw)

    out_clusters = pd.DataFrame(
        {
            "Enterprise ID": ids,
            "Bucket": ent["Bucket"].fillna("").astype(str).str.strip().values,
            "cluster": labels,
            "cluster_label": [friendly[c] for c in labels],
            "key_features": key_strs,
        }
    )

    sizes = out_clusters["cluster"].value_counts().sort_index()
    summ_rows = []
    for c in sorted(out_clusters["cluster"].unique()):
        sub = df_raw[df_raw["cluster"] == c]
        row = {
            "cluster": c,
            "cluster_label": friendly[c],
            "size": int(len(sub)),
            "pct_of_customers": round(100.0 * len(sub) / n, 2),
            "description": friendly[c]
            + ". Avg vs population: "
            + ", ".join(
                f"{d}={sub[d].mean():.1f} (pop {overall[d]:.1f})" for d in drivers[:4]
            ),
        }
        for col in num_cols + ["flag_3pa", "flag_forms", "flag_profiles"]:
            row[f"avg_{col}"] = round(float(sub[col].mean()), 4) if col in sub.columns else np.nan
        summ_rows.append(row)
    summary_df = pd.DataFrame(summ_rows)

    metrics_df["chosen_k_kmeans"] = chosen_k
    metrics_df["aggregation_notes"] = "See segmentation/pipeline.py AGGREGATION_CHOICES"
    metrics_df["min_cluster_floor_used"] = min_floor
    metrics_df["csv_encoding_bobs"] = enc_meta.get("bobs", "")
    metrics_df["csv_encoding_forms"] = enc_meta.get("forms", "")
    metrics_df["csv_encoding_spooler"] = enc_meta.get("spooler", "")
    metrics_df["csv_encoding_users"] = enc_meta.get("users", "")

    out_clusters.to_csv(ROOT / "customer_clusters.csv", index=False)
    summary_df.to_csv(ROOT / "cluster_summary.csv", index=False)
    metrics_df.to_csv(ROOT / "cluster_quality_metrics.csv", index=False)

    write_deck_csvs(
        ROOT,
        summary_df,
        out_clusters,
        df_raw,
        n,
        chosen_k,
        min_floor,
        metrics_df,
        enc_meta,
    )

    # Charts
    plt.figure(figsize=(8, 4))
    sizes.plot(kind="bar", color="steelblue")
    plt.title(f"Customer distribution by cluster (k={chosen_k})")
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(charts_dir / "cluster_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(range(k_min, k_max + 1), inertias, marker="o")
    plt.title("K-means elbow (inertia vs k)")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.tight_layout()
    plt.savefig(charts_dir / "elbow_kmeans.png", dpi=150)
    plt.close()

    plot_df = df_raw.melt(
        id_vars=["cluster"],
        value_vars=num_cols,
        var_name="feature",
        value_name="value",
    )
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=plot_df, x="feature", y="value", hue="cluster", showfliers=False)
    plt.xticks(rotation=35, ha="right")
    plt.title("Feature comparison by cluster (raw counts; outliers hidden)")
    plt.tight_layout()
    plt.savefig(charts_dir / "feature_comparison_by_cluster.png", dpi=150)
    plt.close()

    # Silhouette chart
    plt.figure(figsize=(8, 4))
    km_m = metrics_df[metrics_df["method"] == "kmeans"]
    plt.plot(km_m["k"], km_m["silhouette"], marker="o", label="k-means")
    hm = metrics_df[metrics_df["method"] == "hierarchical_ward"]
    plt.plot(hm["k"], hm["silhouette"], marker="s", label="hierarchical")
    gm = metrics_df[metrics_df["method"] == "gaussian_mixture"]
    plt.plot(gm["k"], gm["silhouette"], marker="^", label="Gaussian mixture")
    plt.axvline(chosen_k, color="gray", linestyle="--", label=f"chosen k={chosen_k}")
    plt.xlabel("k")
    plt.ylabel("Silhouette")
    plt.legend()
    plt.title("Silhouette vs k")
    plt.tight_layout()
    plt.savefig(charts_dir / "silhouette_vs_k.png", dpi=150)
    plt.close()

    km_chosen = metrics_df[(metrics_df["method"] == "kmeans") & (metrics_df["k"] == chosen_k)].iloc[0]
    from segmentation.configuration_overlap_report import write_configuration_overlap_report

    write_configuration_overlap_report(
        ROOT,
        charts_dir,
        kmeans_chosen_k=chosen_k,
        kmeans_silhouette=float(km_chosen["silhouette"]),
    )

    from segmentation.enterprise_dimension_export import write_enterprise_dimension_workbook

    write_enterprise_dimension_workbook(ROOT)

    print("Chosen k (k-means):", chosen_k)
    print(metrics_df[metrics_df["method"] == "kmeans"][["k", "silhouette", "min_cluster_size"]])
    print(
        "Wrote CSVs and charts to project root and charts/. "
        "Deck-oriented CSVs: cluster_summary_for_deck.csv, customer_clusters_for_deck.csv, "
        "feature_glossary_for_deck.csv, population_benchmarks_for_deck.csv, "
        "deck_methodology_snapshot.csv, deck_slide_outline.csv. "
        "Overlap answer: configuration_overlap_report.csv, "
        "configuration_overlap_executive_summary.txt, "
        "charts/pairwise_jaccard_combined_histogram.png. "
        "Excel: enterprise_configuration_by_dimension.xlsx (dimension_coverage, restricted Jaccard overview, …)."
    )


if __name__ == "__main__":
    main()
