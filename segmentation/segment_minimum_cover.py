"""
Minimum subset of Segment 1 enterprises whose *combined* configuration atoms match
the full segment union.

Model (aligned with cohort coverage / ``all_dimensions_tagged_union``):
  - An **atom** is ``dimension|value`` for each non-empty token in each Jaccard dimension.
  - **Universe** for the segment = union of atoms over all segment enterprises that appear
    in the analysis spine.
  - **Goal**: smallest subset of those enterprises whose union of atoms equals the universe.
    (Set cover; exact minimum is NP-hard — we use the standard greedy approximation.)

**ID format:** Segment list uses 5-digit codes from your extract (e.g. ``21100``). In
``Enterprise_Setup_Map`` these match **E** + code × **10** (e.g. ``E211000``). Override
``segment_enterprise_ids()`` if your mapping differs.

Run from repo root:
  python -m segmentation.segment_minimum_cover

Outputs (repo root):
  - ``segment1_minimum_cover.csv`` — chosen enterprises in pick order
  - ``segment1_minimum_cover_summary.txt`` — counts, missing IDs, verification
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from segmentation.enterprise_dimension_export import _sanitize_cell_text
from segmentation.jaccard_similarity_clustering import (
    DIMENSION_KEYS,
    build_enterprise_dimension_sets,
)

ROOT = Path(__file__).resolve().parent.parent

# Segment 1: five-digit codes from source list (order preserved; duplicates collapsed).
_SEGMENT_1_SHORT_CODES: tuple[int, ...] = (
    26500,
    21100,
    28800,
    23800,
    27800,
    23500,
    24800,
    20200,
    26800,
    21800,
    24500,
    25800,
    28500,
    21500,
    20500,
    27500,
    25500,
    22500,
    22800,
    22600,
    23600,
    25600,
    20600,
    28600,
    21600,
    24600,
    27600,
    26600,
    22100,
    28100,
    24100,
    27100,
    25100,
    20100,
    26100,
    23100,
    21900,
    24900,
    20900,
    22900,
    23900,
    26900,
    25900,
    27900,
    28900,
    27200,
    22200,
    28200,
    21200,
    24200,
    26200,
    23200,
    25200,
    27300,
    22300,
    21300,
    24300,
    28300,
    26300,
    20300,
    25300,
    23300,
    20400,
    25400,
    23400,
    28400,
    26400,
    21400,
    24400,
    22400,
    27400,
    20700,
    24700,
    21700,
    27700,
    22700,
    28700,
    25700,
    23700,
    26700,
)


def segment_enterprise_ids(*, id_format: str = "times10") -> tuple[str, ...]:
    """
    Map short numeric codes to Enterprise ID strings in the spine.

    - ``times10``: ``E`` + (code * 10), e.g. ``21100`` -> ``E211000`` (matches many Setup Map rows).
    - ``literal``: ``E`` + code as digits, e.g. ``21100`` -> ``E21100``.
    """
    if id_format == "times10":
        return tuple(f"E{n * 10}" for n in _SEGMENT_1_SHORT_CODES)
    if id_format == "literal":
        return tuple(f"E{n}" for n in _SEGMENT_1_SHORT_CODES)
    raise ValueError(f"unknown id_format: {id_format!r}")


def tagged_atoms_for_enterprise(df_sets: pd.DataFrame, eid: str) -> set[str]:
    eid = str(eid).strip()
    if eid not in df_sets.index:
        return set()
    out: set[str] = set()
    for col in DIMENSION_KEYS:
        for t in df_sets.loc[eid, col]:
            tok = _sanitize_cell_text(str(t).strip())
            if tok:
                out.add(f"{col}|{tok}")
    return out


def segment_atom_universe(
    df_sets: pd.DataFrame,
    segment_ids: tuple[str, ...],
) -> tuple[set[str], list[str]]:
    """Union of tagged atoms over segment; list of segment IDs present in df_sets index."""
    u: set[str] = set()
    present: list[str] = []
    for eid in segment_ids:
        eid = str(eid).strip()
        if eid not in df_sets.index:
            continue
        present.append(eid)
        u |= tagged_atoms_for_enterprise(df_sets, eid)
    return u, present


def greedy_set_cover(
    universe: set[str],
    candidates: list[str],
    covers: dict[str, set[str]],
) -> tuple[list[str], list[tuple[str, int, int]]]:
    """
    Greedy: repeatedly pick the candidate covering the most still-uncovered atoms.
    Returns (ordered enterprise IDs, list of (eid, n_newly_covered, cumulative_covered)).
    """
    uncovered = set(universe)
    chosen: list[str] = []
    trace: list[tuple[str, int, int]] = []
    cum = 0

    while uncovered:
        best_eid: str | None = None
        best_gain = 0
        for eid in candidates:
            if eid in chosen:
                continue
            gain = len(covers[eid] & uncovered)
            if gain > best_gain or (
                gain == best_gain and gain > 0 and (best_eid is None or eid < best_eid)
            ):
                best_gain = gain
                best_eid = eid
        if best_eid is None or best_gain == 0:
            break
        new_cov = covers[best_eid] & uncovered
        chosen.append(best_eid)
        uncovered -= new_cov
        cum += len(new_cov)
        trace.append((best_eid, len(new_cov), cum))

    return chosen, trace


def run_segment1(
    root: Path | None = None,
    *,
    id_format: str = "times10",
) -> tuple[pd.DataFrame, str]:
    root = root or ROOT
    segment_ids = segment_enterprise_ids(id_format=id_format)
    df_sets, _enc = build_enterprise_dimension_sets()

    defined = set(segment_ids)
    missing = [e for e in segment_ids if e not in df_sets.index]
    universe, present = segment_atom_universe(df_sets, segment_ids)

    covers = {eid: tagged_atoms_for_enterprise(df_sets, eid) for eid in present}
    chosen, trace = greedy_set_cover(universe, present, covers)

    chosen_set = set(chosen)
    union_chosen: set[str] = set()
    for eid in chosen:
        union_chosen |= covers[eid]
    ok = union_chosen == universe

    names = {idx: str(df_sets.loc[idx, "Enterprise Name"]) for idx in df_sets.index}

    rows = []
    for i, (eid, n_new, cum) in enumerate(trace, start=1):
        rows.append(
            {
                "pick_order": i,
                "Enterprise ID": eid,
                "Enterprise Name": _sanitize_cell_text(names.get(eid, "")),
                "n_tagged_atoms_newly_covered": n_new,
                "n_tagged_atoms_cumulative": cum,
            }
        )
    out_df = pd.DataFrame(rows)

    summary_lines = [
        "Segment 1 - minimum cover (greedy set cover on tagged dimension|value atoms)",
        "=" * 72,
        f"ID mapping: {id_format} (see segment_enterprise_ids docstring)",
        "=" * 72,
        f"Segment list size (defined IDs): {len(segment_ids)}",
        f"In spine / model (have >=1 mapped cnumber): {len(present)}",
        f"Not in model scope: {len(missing)}",
    ]
    if missing:
        summary_lines.append("Missing Enterprise IDs: " + ", ".join(missing))
    summary_lines.extend(
        [
            f"|Universe| distinct tagged atoms across segment: {len(universe)}",
            f"Greedy cover size (enterprises): {len(chosen)}",
            f"Union of chosen covers universe: {ok}",
        ]
    )
    if not ok:
        leftover = universe - union_chosen
        summary_lines.append(f"Uncovered atoms (should be empty): {len(leftover)}")
    summary_lines.append("")
    summary_lines.append(
        "Note: Greedy gives a log(|U|)+1 approximation to set cover optimum; "
        "true minimum may be smaller. For an exact optimum you would need ILP / exhaustive search."
    )
    summary = "\n".join(summary_lines) + "\n"

    csv_path = root / "segment1_minimum_cover.csv"
    txt_path = root / "segment1_minimum_cover_summary.txt"
    out_df.to_csv(csv_path, index=False)
    txt_path.write_text(summary, encoding="utf-8")

    return out_df, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Greedy minimum enterprise cover for Segment 1")
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repo root (default: parent of segmentation/)",
    )
    parser.add_argument(
        "--id-format",
        choices=("times10", "literal"),
        default="times10",
        help="How to map 5-digit segment codes to Enterprise ID (default: times10 -> E211000)",
    )
    args = parser.parse_args()
    df, summary = run_segment1(args.root, id_format=args.id_format)
    print(summary)
    print(f"Wrote {args.root / 'segment1_minimum_cover.csv'}")
    print(f"Wrote {args.root / 'segment1_minimum_cover_summary.txt'}")
    print(f"Selected {len(df)} enterprises (see CSV for order).")


if __name__ == "__main__":
    main()
