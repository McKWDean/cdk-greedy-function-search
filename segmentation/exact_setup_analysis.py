"""
Find Enterprise IDs with identical configuration *values* (not just counts).

Signature (all must match; aligned with Jaccard DIMENSION_KEYS):
  - Exact set of OEM tokens (BOB manufacturer_stmt, same tokenization as pipeline)
  - Exact sets of 3PA vendors: internal, external (excl. 3PA_PRODUCTS) — not Data Your Way / unmapped
  - Exact set of Enterprise_Setup_Map org-setup labels (``Bucket``, uppercased frozenset)
  - Exact set of spooler buckets (MICR / ADP_CDK / OTHER; UNSET excluded)
  - Exact set of canonical form keys (trailing numeric suffix stripped)
  - Exact set of canonical user profile keys (comma-split `profiles`, trailing suffix stripped like forms)

Same spine as pipeline: only enterprises with ≥1 mapped cnumber.

Run: python -m segmentation.exact_setup_analysis
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

from segmentation.pipeline import (
    ROOT,
    csv_text_encoding,
    enterprise_setup_frozenset,
    load_edges,
    load_spine,
    oem_tokens,
    rollup_form_canonical_sets,
    rollup_spooler_bucket_sets,
    split_profile_tokens,
    three_pa_vendor_frozensets_by_type,
)


def bobs_oem_sets(edges: pd.DataFrame, encoding: str) -> pd.Series:
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
    return d.groupby("Enterprise ID")["_oem"].agg(frozenset).rename("oem_set")


def profile_token_sets(edges: pd.DataFrame, encoding: str) -> pd.Series:
    path = ROOT / "users_plus_profiles" / "users_plus_profiles.csv"
    kw = dict(chunksize=200_000, low_memory=False, encoding=encoding)
    chunks_p = []
    for ch in pd.read_csv(path, usecols=["cnumber", "profiles"], **kw):
        ch["cnumber"] = ch["cnumber"].astype(str).str.strip().str.upper()
        pe = ch.assign(_p=ch["profiles"].map(split_profile_tokens)).explode("_p")
        pe = pe.dropna(subset=["_p"])
        chunks_p.append(pe.merge(edges, on="cnumber", how="inner"))
    long_p = pd.concat(chunks_p, ignore_index=True) if chunks_p else pd.DataFrame()
    if len(long_p) == 0:
        return pd.Series(dtype=object)
    return long_p.groupby("Enterprise ID")["_p"].agg(frozenset).rename("profile_token_set")


def empty_frozen() -> frozenset:
    return frozenset()


def _set_for(series: pd.Series, eid: str) -> frozenset:
    if eid in series.index:
        return series.loc[eid]
    return empty_frozen()


def enterprises_grouped_by_exact_signature() -> tuple[int, dict[tuple, list[tuple[str, str, str]]]]:
    """
    Full multi-dimensional configuration signature (same components as Jaccard DIMENSION_KEYS).
    Each member stores reference Bucket string for reporting (redundant with enterprise_setup in sig).
    Returns (n_enterprises, signature_tuple -> [(Enterprise ID, Enterprise Name, Bucket_ref), ...]).
    """
    edges = load_edges()
    ent_ok = set(edges["Enterprise ID"].unique())
    spine = load_spine(ent_ok).set_index("Enterprise ID")

    enc_bobs = csv_text_encoding(ROOT / "bobs_account_stuff" / "bobs_account_stuff.csv")
    enc_forms = csv_text_encoding(ROOT / "form_data" / "form_data.csv")
    enc_spool = csv_text_encoding(ROOT / "spooler_data" / "spooler_data.csv")
    enc_users = csv_text_encoding(ROOT / "users_plus_profiles" / "users_plus_profiles.csv")

    oem_s = bobs_oem_sets(edges, enc_bobs)
    v3 = three_pa_vendor_frozensets_by_type(edges)
    sp_s = rollup_spooler_bucket_sets(edges, enc_spool).rename("spooler_set")
    fn_s = rollup_form_canonical_sets(edges, enc_forms).rename("form_formname_set")
    prof_s = profile_token_sets(edges, enc_users)

    by_sig: dict[tuple, list[tuple[str, str, str]]] = defaultdict(list)
    for eid in spine.index:
        b_raw = spine.loc[eid, "Bucket"]
        bucket_ref = "" if pd.isna(b_raw) else str(b_raw).strip()
        sig = (
            _set_for(oem_s, eid),
            _set_for(v3["internal"], eid),
            _set_for(v3["external"], eid),
            enterprise_setup_frozenset(b_raw),
            _set_for(sp_s, eid),
            _set_for(fn_s, eid),
            _set_for(prof_s, eid),
        )
        by_sig[sig].append((eid, spine.loc[eid, "Enterprise Name"], bucket_ref))

    return len(spine), dict(by_sig)


def main() -> None:
    n_ent, by_sig = enterprises_grouped_by_exact_signature()

    dup_groups = {k: v for k, v in by_sig.items() if len(v) > 1}
    n_enterprises_in_dup_groups = sum(len(v) for v in dup_groups.values())
    n_signatures_solo = sum(1 for v in by_sig.values() if len(v) == 1)
    n_enterprises_solo = sum(len(v) for v in by_sig.values() if len(v) == 1)

    print("=== Exact multi-dimensional setup match ===")
    print("Enterprises analyzed:", n_ent)
    print("Configuration signatures held by exactly one enterprise:", n_signatures_solo)
    print("Enterprises with no exact peer (sole occupant of their signature):", n_enterprises_solo)
    print("Signatures shared by 2+ enterprises (duplicate groups):", len(dup_groups))
    print("Enterprises that belong to a duplicate group (have at least one exact peer):", n_enterprises_in_dup_groups)

    out_rows = []
    for gid, (sig, members) in enumerate(sorted(dup_groups.items(), key=lambda x: -len(x[1])), start=1):
        eids = [m[0] for m in members]
        names = [m[1] for m in members]
        oem, vi, ve, eset, sp, fn, prf = (
            sig[0],
            sig[1],
            sig[2],
            sig[3],
            sig[4],
            sig[5],
            sig[6],
        )
        ref_buckets = sorted({m[2] for m in members if m[2]})
        out_rows.append(
            {
                "exact_setup_group_id": gid,
                "n_enterprises": len(members),
                "reference_Buckets": "|".join(ref_buckets),
                "n_oem_tokens": len(oem),
                "n_3pa_vendors_internal": len(vi),
                "n_3pa_vendors_external": len(ve),
                "n_enterprise_setup_labels": len(eset),
                "n_spooler_types": len(sp),
                "n_formnames": len(fn),
                "n_profile_tokens": len(prf),
                "Enterprise_IDs": "|".join(eids),
                "Enterprise_Names": "|".join(names),
            }
        )

    out_path = ROOT / "exact_setup_duplicate_groups.csv"
    pd.DataFrame(out_rows).to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(out_rows)} groups with 2+ members).")

    if dup_groups:
        print("\nLargest groups (id, count, reference_Buckets):")
        for r in sorted(out_rows, key=lambda x: -x["n_enterprises"])[:15]:
            print(
                f"  group {r['exact_setup_group_id']}: n={r['n_enterprises']} | {r['reference_Buckets']!r} | "
                f"IDs: {r['Enterprise_IDs'][:120]}{'...' if len(r['Enterprise_IDs']) > 120 else ''}"
            )
    else:
        print("\nNo two enterprises share an identical full signature on these dimensions.")


if __name__ == "__main__":
    main()
