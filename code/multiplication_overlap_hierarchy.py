#!/usr/bin/env python3
"""
multiplication_overlap_hierarchy.py

Finite-depth collision hierarchy for deterministic multiplication coverage
of the square shell

    n^2 < m < (n+1)^2.

The computation uses no primality test.  It constructs the irreducible odd
multiplication rows by multiplication alone (an Eratosthenes-style marking
of reducible row labels), then counts how many irreducible rows hit each odd
position in the shell.

For an odd shell position m, define

    w(m) = number of irreducible odd rows a <= n with a | m.

Let

    S_j(n) = sum_m C(w(m), j).

For odd K, the Bonferroni quantity

    U_K = S_1 - S_2 + S_3 - ... + S_K

is an upper bound for the number of covered positions.  Since there are
exactly n odd positions in the shell,

    H_K = n - U_K

is a deterministic lower bound ("certified holes") for the number of
uncovered positions.

The exact number of uncovered positions is simply

    H_exact = #{m : w(m)=0}.

For odd K there is also the exact slack identity

    U_K - covered_exact
      = sum_{w >= K+1} count(w) * C(w-1, K),

so H_exact - H_K measures precisely how much is lost by truncating the
collision hierarchy at depth K.

Default output columns include S_1,...,S_9 and the odd-order bounds
K=1,3,5,7,9, plus exact holes and maximum multiplicative depth.

Examples
--------
Single shell:

    python multiplication_overlap_hierarchy.py --n 10000

Range:

    python multiplication_overlap_hierarchy.py \
        --start 2 --stop 10000 --outdir overlap_out \
        --write-histogram --progress-every 500

Verify exact hole counts against the earlier shell_summary.csv:

    python multiplication_overlap_hierarchy.py \
        --start 2 --stop 10000 \
        --verify-summary shell_summary.csv \
        --outdir overlap_out --write-histogram

Go deeper, e.g. through S_13:

    python multiplication_overlap_hierarchy.py --n 1000000 --max-order 13
"""

from __future__ import annotations

import argparse
import bisect
import csv
from collections import Counter
from math import comb
from pathlib import Path


def irreducible_odd_rows(limit: int) -> list[int]:
    """
    Construct odd irreducible multiplication-row labels <= limit.

    No primality test is called.  Start with all odd labels unmarked.
    Each first-unmarked row a is irreducible; its odd multiples a*a,
    a*(a+2), ... are marked reducible.
    """
    if limit < 3:
        return []

    reducible = bytearray(limit + 1)
    rows: list[int] = []

    for a in range(3, limit + 1, 2):
        if reducible[a]:
            continue
        rows.append(a)

        aa = a * a
        if aa <= limit:
            step = 2 * a
            # Slice marking is much faster than a Python loop.
            count = ((limit - aa) // step) + 1
            reducible[aa : limit + 1 : step] = b"\x01" * count

    return rows


def first_odd_in_shell(n: int) -> int:
    """Smallest odd integer strictly greater than n^2."""
    return n * n + (1 if n % 2 == 0 else 2)


def analyze_shell(
    n: int,
    rows: list[int],
    max_order: int,
) -> tuple[dict[str, int], Counter[int]]:
    """
    Return hierarchy statistics and the histogram of multiplicative depths.
    """
    if n < 2:
        raise ValueError("n must be >= 2")

    # There are exactly n odd positions in (n^2, (n+1)^2).
    depths = bytearray(n)
    lo_odd = first_odd_in_shell(n)
    lo = n * n
    hi = (n + 1) * (n + 1)

    upto = bisect.bisect_right(rows, n)

    for a in rows[:upto]:
        # First odd multiplier b such that a*b > n^2.
        b = lo // a + 1
        if b % 2 == 0:
            b += 1

        m0 = a * b
        step_m = 2 * a

        # All these m are odd and lie in the square shell.
        for m in range(m0, hi, step_m):
            idx = (m - lo_odd) // 2
            depths[idx] += 1

    hist: Counter[int] = Counter(depths)
    max_depth = max(hist) if hist else 0

    stats: dict[str, int] = {
        "n": n,
        "shell_lo": lo,
        "shell_hi": hi,
        "candidate_count": n,
        "irreducible_rows": upto,
        "exact_covered": n - hist.get(0, 0),
        "exact_holes": hist.get(0, 0),
        "max_depth": max_depth,
    }

    # S_j = sum C(w,j).
    S: dict[int, int] = {}
    for j in range(1, max_order + 1):
        sj = 0
        for w, count in hist.items():
            if w >= j:
                sj += count * comb(w, j)
        S[j] = sj
        stats[f"S{j}"] = sj

    # Odd Bonferroni upper bounds on coverage and corresponding
    # deterministic lower bounds on holes.
    running = 0
    for j in range(1, max_order + 1):
        running += S[j] if j % 2 else -S[j]

        if j % 2 == 1:
            U = running
            H = n - U
            slack = stats["exact_holes"] - H

            # Independent computation of the exact truncation slack:
            #
            #   U_j - exact_covered
            #     = sum_{w>=j+1} count(w) C(w-1,j)
            #
            # for odd j.
            tail_slack = 0
            for w, count in hist.items():
                if w >= j + 1:
                    tail_slack += count * comb(w - 1, j)

            if slack != tail_slack:
                raise AssertionError(
                    f"slack identity failed at n={n}, order={j}: "
                    f"{slack} != {tail_slack}"
                )

            stats[f"U{j}"] = U
            stats[f"certified_holes_{j}"] = H
            stats[f"slack_{j}"] = slack
            stats[f"certifies_hole_{j}"] = int(H > 0)

    return stats, hist


def load_verification(path: Path) -> dict[int, int]:
    """Load n -> exact holes from the earlier shell_summary.csv."""
    result: dict[int, int] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "n" not in (reader.fieldnames or []) or "holes" not in (reader.fieldnames or []):
            raise ValueError("verification CSV must contain columns 'n' and 'holes'")
        for row in reader:
            result[int(row["n"])] = int(row["holes"])
    return result


def fieldnames_for(max_order: int) -> list[str]:
    fields = [
        "n",
        "shell_lo",
        "shell_hi",
        "candidate_count",
        "irreducible_rows",
        "exact_covered",
        "exact_holes",
        "max_depth",
    ]
    fields.extend(f"S{j}" for j in range(1, max_order + 1))
    for j in range(1, max_order + 1, 2):
        fields.extend(
            [
                f"U{j}",
                f"certified_holes_{j}",
                f"slack_{j}",
                f"certifies_hole_{j}",
            ]
        )
    return fields


def print_single(stats: dict[str, int], hist: Counter[int], max_order: int) -> None:
    n = stats["n"]
    print(
        f"n={n}  shell=({stats['shell_lo']},{stats['shell_hi']})  "
        f"odd_positions={n}"
    )
    print(
        f"irreducible_rows={stats['irreducible_rows']}  "
        f"covered={stats['exact_covered']}  "
        f"exact_holes={stats['exact_holes']}  "
        f"max_depth={stats['max_depth']}"
    )

    print("\ndepth histogram: w -> positions")
    for w in sorted(hist):
        print(f"  {w:2d} -> {hist[w]}")

    print("\nintersection sums:")
    print("  " + "  ".join(f"S{j}={stats[f'S{j}']}" for j in range(1, max_order + 1)))

    print("\nodd Bonferroni hierarchy:")
    print(" order          U_K   certified holes   slack   certifies?")
    print(" ----------------------------------------------------------")
    for j in range(1, max_order + 1, 2):
        print(
            f" {j:5d}  "
            f"{stats[f'U{j}']:11d}  "
            f"{stats[f'certified_holes_{j}']:16d}  "
            f"{stats[f'slack_{j}']:6d}   "
            f"{'yes' if stats[f'certifies_hole_{j}'] else 'no'}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Finite-depth multiplication-collision hierarchy."
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument("--n", type=int, help="analyze a single shell")
    g.add_argument("--start", type=int, default=2, help="first n in range")
    p.add_argument("--stop", type=int, default=10000, help="last n in range")
    p.add_argument(
        "--max-order",
        type=int,
        default=9,
        help="largest S_j to compute (default 9)",
    )
    p.add_argument("--outdir", type=Path, default=Path("overlap_out"))
    p.add_argument(
        "--write-histogram",
        action="store_true",
        help="write depth_histogram.csv (one row per n and depth)",
    )
    p.add_argument(
        "--verify-summary",
        type=Path,
        help="optional earlier shell_summary.csv; exact holes are checked",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="print progress every this many shells",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.max_order < 1:
        raise SystemExit("--max-order must be >= 1")

    if args.n is not None:
        if args.n < 2:
            raise SystemExit("--n must be >= 2")
        start = stop = args.n
    else:
        start, stop = args.start, args.stop
        if start < 2 or stop < start:
            raise SystemExit("require 2 <= --start <= --stop")

    rows = irreducible_odd_rows(stop)

    verification = (
        load_verification(args.verify_summary)
        if args.verify_summary is not None
        else {}
    )

    if args.n is not None:
        stats, hist = analyze_shell(args.n, rows, args.max_order)
        if verification and args.n in verification:
            expected = verification[args.n]
            if stats["exact_holes"] != expected:
                raise AssertionError(
                    f"verification failed at n={args.n}: "
                    f"{stats['exact_holes']} != {expected}"
                )
        print_single(stats, hist, args.max_order)
        return

    args.outdir.mkdir(parents=True, exist_ok=True)
    summary_path = args.outdir / "overlap_summary.csv"

    summary_file = summary_path.open("w", newline="", encoding="utf-8")
    summary_writer = csv.DictWriter(
        summary_file, fieldnames=fieldnames_for(args.max_order)
    )
    summary_writer.writeheader()

    hist_file = hist_writer = None
    hist_path = args.outdir / "depth_histogram.csv"
    if args.write_histogram:
        hist_file = hist_path.open("w", newline="", encoding="utf-8")
        hist_writer = csv.DictWriter(
            hist_file, fieldnames=["n", "depth", "positions"]
        )
        hist_writer.writeheader()

    odd_orders = list(range(1, args.max_order + 1, 2))
    fail_count = {k: 0 for k in odd_orders}
    first_failure: dict[int, int | None] = {k: None for k in odd_orders}
    worst_certified: dict[int, tuple[int, int] | None] = {
        k: None for k in odd_orders
    }

    verified = 0

    try:
        for count, n in enumerate(range(start, stop + 1), start=1):
            stats, hist = analyze_shell(n, rows, args.max_order)
            summary_writer.writerow(stats)

            if hist_writer is not None:
                for depth in sorted(hist):
                    hist_writer.writerow(
                        {
                            "n": n,
                            "depth": depth,
                            "positions": hist[depth],
                        }
                    )

            if n in verification:
                expected = verification[n]
                if stats["exact_holes"] != expected:
                    raise AssertionError(
                        f"verification failed at n={n}: "
                        f"{stats['exact_holes']} != {expected}"
                    )
                verified += 1

            for k in odd_orders:
                h = stats[f"certified_holes_{k}"]
                if h <= 0:
                    fail_count[k] += 1
                    if first_failure[k] is None:
                        first_failure[k] = n

                current = worst_certified[k]
                if current is None or h < current[1]:
                    worst_certified[k] = (n, h)

            if args.progress_every and count % args.progress_every == 0:
                pieces = "  ".join(
                    f"H{k}={stats[f'certified_holes_{k}']}"
                    for k in odd_orders
                )
                print(
                    f"through n={n}: exact={stats['exact_holes']}  "
                    f"max_depth={stats['max_depth']}  {pieces}"
                )

    finally:
        summary_file.close()
        if hist_file is not None:
            hist_file.close()

    print(f"Analyzed n={start}..{stop}")
    if verification:
        print(f"Verified exact hole counts for {verified} shells.")

    print("\nCertification summary:")
    print(" order   failures   first failure   minimum certified holes (at n)")
    print(" -------------------------------------------------------------------")
    for k in odd_orders:
        worst_n, worst_h = worst_certified[k]  # type: ignore[misc]
        ff = first_failure[k]
        print(
            f" {k:5d}  {fail_count[k]:9d}  "
            f"{str(ff) if ff is not None else 'none':>13}  "
            f"{worst_h:23d} (n={worst_n})"
        )

    print(f"\nWrote {summary_path}")
    if args.write_histogram:
        print(f"Wrote {hist_path}")


if __name__ == "__main__":
    main()
