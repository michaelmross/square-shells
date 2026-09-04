#!/usr/bin/env python3
"""
multiplication_coverage.py

Deterministic multiplication coverage of the square shell

    n^2 < m < (n+1)^2.

No primality test and no prime list are used.

There are exactly n odd integers in the shell. For every odd row label

    3 <= a <= n,

the script marks shell offsets k for which

    a | (n^2 + k),   1 <= k <= 2n,

and n^2+k is odd.

Every odd composite in the shell has an odd divisor a <= n. Hence every
odd composite is marked by at least one multiplication row. The final
uncovered positions are therefore exactly the prime positions, but they
are found here using multiplication/divisibility alone.

Rows are grouped into quotient bands

    q = floor(n/a),

equivalently

    n/(q+1) < a <= n/q.

For every n and every band the program records:
  * raw_edges            factor-pair hits, with multiplicity;
  * distinct_hits        distinct positions hit by that band;
  * newly_covered        positions not hit by any smaller row;
  * within_collisions    duplicate edges within the band;
  * prior_overlap        distinct band positions already covered earlier;
  * collision_excess     raw_edges - newly_covered;
  * covered_after        total covered odd positions after the band;
  * survivors_after      still-uncovered odd positions after the band.

For the whole shell it records the exact identity

    holes = collision_excess - edge_surplus,

where

    edge_surplus      = raw_edges - n,
    collision_excess  = raw_edges - distinct_covered.

Thus complete multiplication coverage is exactly holes == 0.

Examples
--------
Single shell with band detail:

    python multiplication_coverage.py --n 13 --print-bands

Range:

    python multiplication_coverage.py --start 2 --stop 2000 \
        --outdir coverage_out

Also save every final uncovered position:

    python multiplication_coverage.py --start 2 --stop 2000 \
        --outdir coverage_out --write-survivors

Also save row-by-row diagnostics (can be large):

    python multiplication_coverage.py --start 2 --stop 500 \
        --outdir coverage_out --write-rows

For long runs, print progress:

    python multiplication_coverage.py --start 2 --stop 10000 \
        --outdir coverage_out --progress-every 500
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class ShellSummary:
    n: int
    shell_lo: int
    shell_hi: int
    candidate_count: int
    raw_edges: int
    distinct_covered: int
    collision_excess: int
    edge_surplus: int
    holes: int
    identity_check: int
    active_rows: int
    total_rows: int
    band_count: int
    first_hole_offset: int | None
    first_hole_value: int | None


@dataclass
class BandStat:
    n: int
    band_order: int
    q: int
    a_min: int
    a_max: int
    row_count: int
    raw_edges: int
    distinct_hits: int
    newly_covered: int
    within_collisions: int
    prior_overlap: int
    collision_excess: int
    covered_after: int
    survivors_after: int
    min_hit_offset: int | None
    max_hit_offset: int | None
    min_new_offset: int | None
    max_new_offset: int | None


@dataclass
class RowStat:
    n: int
    a: int
    q: int
    r: int
    raw_edges: int
    newly_covered: int
    redundant_edges: int
    covered_after: int
    survivors_after: int
    min_hit_offset: int | None
    max_hit_offset: int | None


def odd_candidate_offsets(n: int) -> range:
    """Offsets k for which n^2+k is odd, 1 <= k <= 2n."""
    # n^2 has the same parity as n.
    first = 1 if n % 2 == 0 else 2
    return range(first, 2 * n + 1, 2)


def row_offsets(n: int, a: int) -> range:
    """
    Admissible offsets k hit by odd multiplication row a.

    Write n = q*a+r. Then

        a | n^2+k  <=>  a | r^2+k.

    So k lies in one residue class modulo a. Restricting n^2+k to odd
    numbers selects every other member of that class, hence step 2a.
    """
    if a < 3 or a > n or a % 2 == 0:
        raise ValueError("a must be odd with 3 <= a <= n")

    r = n % a
    residue = (-r * r) % a

    # Smallest positive k in the residue class.
    first = residue if residue != 0 else a

    # n^2+k must be odd, so k has parity opposite n.
    target_parity = 1 - (n & 1)
    if (first & 1) != target_parity:
        first += a  # a is odd, so this flips parity.

    return range(first, 2 * n + 1, 2 * a)


def range_last(r: range) -> int | None:
    if len(r) == 0:
        return None
    return r.start + (len(r) - 1) * r.step


def analyze_shell(
    n: int,
    collect_rows: bool = False,
) -> tuple[ShellSummary, list[BandStat], list[RowStat], list[int]]:
    if n < 2:
        raise ValueError("n must be >= 2")

    covered: set[int] = set()
    band_stats: list[BandStat] = []
    row_stats: list[RowStat] = []

    total_raw = 0
    active_rows = 0

    current_q: int | None = None
    band_order = 0
    band_a_min: int | None = None
    band_a_max: int | None = None
    band_row_count = 0
    band_raw = 0
    band_union: set[int] = set()
    band_new_count = 0
    band_min_new: int | None = None
    band_max_new: int | None = None

    def flush_band() -> None:
        nonlocal band_order, band_a_min, band_a_max
        nonlocal band_row_count, band_raw, band_union
        nonlocal band_new_count, band_min_new, band_max_new

        if current_q is None or band_a_min is None or band_a_max is None:
            return

        band_order += 1
        distinct_hits = len(band_union)
        newly_covered = band_new_count
        within_collisions = band_raw - distinct_hits
        prior_overlap = distinct_hits - newly_covered
        collision_excess = band_raw - newly_covered

        band_stats.append(
            BandStat(
                n=n,
                band_order=band_order,
                q=current_q,
                a_min=band_a_min,
                a_max=band_a_max,
                row_count=band_row_count,
                raw_edges=band_raw,
                distinct_hits=distinct_hits,
                newly_covered=newly_covered,
                within_collisions=within_collisions,
                prior_overlap=prior_overlap,
                collision_excess=collision_excess,
                covered_after=len(covered),
                survivors_after=n - len(covered),
                min_hit_offset=min(band_union) if band_union else None,
                max_hit_offset=max(band_union) if band_union else None,
                min_new_offset=band_min_new,
                max_new_offset=band_max_new,
            )
        )

        band_a_min = None
        band_a_max = None
        band_row_count = 0
        band_raw = 0
        band_union = set()
        band_new_count = 0
        band_min_new = None
        band_max_new = None

    for a in range(3, n + 1, 2):
        q = n // a

        if current_q is None:
            current_q = q
        elif q != current_q:
            flush_band()
            current_q = q

        hits = row_offsets(n, a)
        raw_count = len(hits)

        new_hits = [k for k in hits if k not in covered]
        new_count = len(new_hits)

        total_raw += raw_count
        if new_count:
            active_rows += 1

        band_a_min = a if band_a_min is None else min(band_a_min, a)
        band_a_max = a if band_a_max is None else max(band_a_max, a)
        band_row_count += 1
        band_raw += raw_count
        band_union.update(hits)
        band_new_count += new_count

        if new_hits:
            lo_new = new_hits[0]
            hi_new = new_hits[-1]
            band_min_new = lo_new if band_min_new is None else min(band_min_new, lo_new)
            band_max_new = hi_new if band_max_new is None else max(band_max_new, hi_new)

        covered.update(hits)

        if collect_rows:
            row_stats.append(
                RowStat(
                    n=n,
                    a=a,
                    q=q,
                    r=n % a,
                    raw_edges=raw_count,
                    newly_covered=new_count,
                    redundant_edges=raw_count - new_count,
                    covered_after=len(covered),
                    survivors_after=n - len(covered),
                    min_hit_offset=hits.start if raw_count else None,
                    max_hit_offset=range_last(hits),
                )
            )

    flush_band()

    holes = [k for k in odd_candidate_offsets(n) if k not in covered]
    distinct_covered = len(covered)
    collision_excess = total_raw - distinct_covered
    edge_surplus = total_raw - n

    summary = ShellSummary(
        n=n,
        shell_lo=n * n,
        shell_hi=(n + 1) * (n + 1),
        candidate_count=n,
        raw_edges=total_raw,
        distinct_covered=distinct_covered,
        collision_excess=collision_excess,
        edge_surplus=edge_surplus,
        holes=len(holes),
        identity_check=collision_excess - edge_surplus,
        active_rows=active_rows,
        total_rows=max(0, (n - 1) // 2),
        band_count=len(band_stats),
        first_hole_offset=holes[0] if holes else None,
        first_hole_value=n * n + holes[0] if holes else None,
    )

    if summary.identity_check != summary.holes:
        raise AssertionError("coverage identity failed")
    if distinct_covered + len(holes) != n:
        raise AssertionError("candidate partition failed")

    return summary, band_stats, row_stats, holes


def print_shell(
    summary: ShellSummary,
    bands: list[BandStat],
    holes: list[int],
    print_bands: bool = False,
) -> None:
    print(
        f"n={summary.n}  shell=({summary.shell_lo},{summary.shell_hi})  "
        f"odd_positions={summary.candidate_count}"
    )
    print(
        f"raw_edges={summary.raw_edges}  "
        f"covered={summary.distinct_covered}  "
        f"collisions={summary.collision_excess}  "
        f"edge_surplus={summary.edge_surplus}  "
        f"holes={summary.holes}"
    )
    print(
        "identity: holes = collisions - edge_surplus = "
        f"{summary.collision_excess} - ({summary.edge_surplus}) "
        f"= {summary.identity_check}"
    )

    if holes:
        values = [summary.n * summary.n + k for k in holes]
        print("hole offsets:", " ".join(map(str, holes)))
        print("hole values: ", " ".join(map(str, values)))
    else:
        print("*** COMPLETE COVERAGE FOUND ***")

    if print_bands:
        print()
        header = (
            "order  q    a-range   rows  raw  distinct  new  "
            "within-coll  prior-overlap  survivors"
        )
        print(header)
        print("-" * len(header))
        for b in bands:
            print(
                f"{b.band_order:5d}  {b.q:3d}  "
                f"{b.a_min:3d}-{b.a_max:<3d}  "
                f"{b.row_count:4d}  {b.raw_edges:3d}  "
                f"{b.distinct_hits:8d}  {b.newly_covered:3d}  "
                f"{b.within_collisions:11d}  {b.prior_overlap:13d}  "
                f"{b.survivors_after:9d}"
            )


def make_writer(path: Path, fieldnames: list[str]):
    f = path.open("w", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    return f, w


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Square-shell multiplication coverage with no primality tests."
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument("--n", type=int, help="analyze one shell n")
    g.add_argument("--start", type=int, default=2, help="first n in a range")
    p.add_argument("--stop", type=int, default=200, help="last n in a range (inclusive)")
    p.add_argument("--outdir", type=Path, default=Path("coverage_out"))
    p.add_argument(
        "--print-bands",
        action="store_true",
        help="print per-band diagnostics (most useful with --n)",
    )
    p.add_argument(
        "--write-survivors",
        action="store_true",
        help="write every final uncovered offset/value",
    )
    p.add_argument(
        "--write-rows",
        action="store_true",
        help="write row-by-row diagnostics; can produce a large file",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="print progress every this many shells (0 disables)",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="suppress console summary for a single n",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.n is not None:
        if args.n < 2:
            raise SystemExit("--n must be >= 2")
        start = stop = args.n
    else:
        start, stop = args.start, args.stop
        if start < 2 or stop < start:
            raise SystemExit("require 2 <= --start <= --stop")

    out = args.outdir
    out.mkdir(parents=True, exist_ok=True)

    summary_f, summary_w = make_writer(
        out / "shell_summary.csv",
        list(ShellSummary.__dataclass_fields__),
    )
    band_f, band_w = make_writer(
        out / "band_stats.csv",
        list(BandStat.__dataclass_fields__),
    )

    row_f = row_w = None
    if args.write_rows:
        row_f, row_w = make_writer(
            out / "row_stats.csv",
            list(RowStat.__dataclass_fields__),
        )

    surv_f = surv_w = None
    if args.write_survivors:
        surv_f, surv_w = make_writer(
            out / "survivors.csv",
            ["n", "offset", "value"],
        )

    complete_coverage: list[int] = []
    min_holes: int | None = None
    min_hole_ns: list[int] = []

    try:
        for count, n in enumerate(range(start, stop + 1), start=1):
            summary, bands, rows, holes = analyze_shell(
                n, collect_rows=args.write_rows
            )

            summary_w.writerow(asdict(summary))
            band_w.writerows(asdict(b) for b in bands)

            if row_w is not None:
                row_w.writerows(asdict(r) for r in rows)

            if surv_w is not None:
                surv_w.writerows(
                    {
                        "n": n,
                        "offset": k,
                        "value": n * n + k,
                    }
                    for k in holes
                )

            if summary.holes == 0:
                complete_coverage.append(n)

            if min_holes is None or summary.holes < min_holes:
                min_holes = summary.holes
                min_hole_ns = [n]
            elif summary.holes == min_holes:
                min_hole_ns.append(n)

            if args.n is not None and not args.quiet:
                print_shell(summary, bands, holes, args.print_bands)

            if args.progress_every and count % args.progress_every == 0:
                print(
                    f"processed through n={n}; "
                    f"current holes={summary.holes}; minimum={min_holes}"
                )

    finally:
        summary_f.close()
        band_f.close()
        if row_f is not None:
            row_f.close()
        if surv_f is not None:
            surv_f.close()

    if args.n is None:
        print(f"Analyzed n={start}..{stop}")
        shown = min_hole_ns[:20]
        suffix = " ..." if len(min_hole_ns) > 20 else ""
        print(f"minimum holes = {min_holes} at n={shown}{suffix}")
        if complete_coverage:
            print("COMPLETE COVERAGE at n =", complete_coverage)
        else:
            print("No complete coverage in the tested range.")
        print(f"Wrote {out / 'shell_summary.csv'}")
        print(f"Wrote {out / 'band_stats.csv'}")
        if args.write_rows:
            print(f"Wrote {out / 'row_stats.csv'}")
        if args.write_survivors:
            print(f"Wrote {out / 'survivors.csv'}")


if __name__ == "__main__":
    main()
