#!/usr/bin/env python3
"""
overnight_shifted_hierarchy.py

Dense scan of square shells for the shifted Bonferroni / multiplication
coverage inequalities.

For odd K and integer r >= 2 define

    P_{K,r}(w)
      = 1 + (w-1) * product_{j=0}^{K-2}(w-r-j)
                    / product_{j=0}^{K-2}(r+j).

For every integer w >= 1, P_{K,r}(w) >= 1, while P_{K,r}(0)=0.
Therefore, if w(m) is the number of irreducible odd multiplication rows
hitting an odd shell position m,

    covered <= sum_m P_{K,r}(w(m)).

A hole is certified whenever this upper bound is < n, since there are
exactly n odd positions in (n^2,(n+1)^2).

The program scans EVERY n in a requested range.  It reports:
  * first failure of the fixed r=3 inequality at each K;
  * number of failures;
  * worst ratio bound/n;
  * the same statistics when r is allowed to vary from 2..rmax;
  * sampled/event rows to a CSV for later inspection;
  * a compact CSV recording the initial best r and every best-r transition.

The computation uses multiplication only.  "Irreducible rows" are constructed
by Eratosthenes-style marking; no primality test is called.

Windows example (use all but one CPU core):

    python overnight_shifted_hierarchy.py ^
        --start 10001 --stop 1000000 ^
        --orders 3,5,7,9,11 ^
        --rmax 12 ^
        --workers 0 ^
        --write-every 1000 ^
        --out overnight_hierarchy.csv

For a shorter trial first:

    python overnight_shifted_hierarchy.py --start 10001 --stop 50000 --workers 0
"""

from __future__ import annotations

import argparse
import bisect
import csv
import math
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROWS: list[int] = []
ORDERS: tuple[int, ...] = ()
RMAX: int = 12


def irreducible_odd_rows(limit: int) -> list[int]:
    """Construct odd irreducible multiplication rows <= limit by marking."""
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
            count = ((limit - aa) // step) + 1
            reducible[aa : limit + 1 : step] = b"\x01" * count
    return rows


def init_worker(stop: int, orders: tuple[int, ...], rmax: int) -> None:
    global ROWS, ORDERS, RMAX
    ROWS = irreducible_odd_rows(stop)
    ORDERS = orders
    RMAX = rmax


def first_odd_in_shell(n: int) -> int:
    return n * n + (1 if n % 2 == 0 else 2)


def depth_histogram(n: int) -> Counter[int]:
    """
    Histogram of w(m), the number of irreducible odd rows a<=n dividing m,
    over the n odd positions in n^2 < m < (n+1)^2.
    """
    depths = bytearray(n)
    lo = n * n
    hi = (n + 1) * (n + 1)
    lo_odd = first_odd_in_shell(n)

    upto = bisect.bisect_right(ROWS, n)
    for a in ROWS[:upto]:
        # First odd multiplier b with a*b > n^2.
        b = lo // a + 1
        if b % 2 == 0:
            b += 1

        m0 = a * b
        step = 2 * a
        for m in range(m0, hi, step):
            depths[(m - lo_odd) // 2] += 1

    return Counter(depths)


def denominator(K: int, r: int) -> int:
    d = 1
    for j in range(K - 1):
        d *= r + j
    return d


def scaled_bound_and_margin(
    hist: Counter[int], n: int, K: int, r: int
) -> tuple[int, int, int]:
    """
    Return (bound_numerator, denominator, margin_numerator), exactly.

    bound = bound_numerator / denominator
    margin = n - bound = margin_numerator / denominator
    """
    D = denominator(K, r)
    B = 0
    for w, count in hist.items():
        if w == 0:
            continue

        prod = w - 1
        for j in range(K - 1):
            prod *= w - r - j

        term = D + prod  # D * P_{K,r}(w)

        # Universal majorant sanity check at every observed multiplicity.
        if term < D:
            raise AssertionError(
                f"pointwise majorant failed: K={K}, r={r}, w={w}"
            )
        B += count * term

    M = n * D - B
    return B, D, M


def analyze_n(n: int) -> tuple:
    hist = depth_histogram(n)
    exact_holes = hist.get(0, 0)
    max_depth = max(hist) if hist else 0

    results = []
    for K in ORDERS:
        # Fixed r=3, the first shifted inequality discovered by the LP.
        B3, D3, M3 = scaled_bound_and_margin(hist, n, K, 3)
        fixed_ratio = B3 / (D3 * n)
        fixed_margin = M3 / D3

        # Ordinary Bonferroni is r=2.
        B2, D2, M2 = scaled_bound_and_margin(hist, n, K, 2)
        bonf_ratio = B2 / (D2 * n)
        bonf_margin = M2 / D2

        # Best shifted root location in the requested discrete family.
        best_r = 2
        best_margin_num = M2
        best_D = D2
        best_ratio = bonf_ratio

        for r in range(3, RMAX + 1):
            B, D, M = scaled_bound_and_margin(hist, n, K, r)
            ratio = B / (D * n)

            # Compare margins M/D exactly by cross multiplication.
            if M * best_D > best_margin_num * D:
                best_r = r
                best_margin_num = M
                best_D = D
                best_ratio = ratio

        best_margin = best_margin_num / best_D

        results.append(
            (
                K,
                fixed_ratio,
                fixed_margin,
                bonf_ratio,
                bonf_margin,
                best_r,
                best_ratio,
                best_margin,
            )
        )

    return n, exact_holes, max_depth, tuple(results)


@dataclass
class Tracker:
    first_fixed_failure: int | None = None
    fixed_failures: int = 0
    worst_fixed_ratio: float = -1.0
    worst_fixed_n: int | None = None

    first_best_failure: int | None = None
    best_failures: int = 0
    worst_best_ratio: float = -1.0
    worst_best_n: int | None = None
    worst_best_r: int | None = None

    max_best_r: int = 0
    best_r_changes: int = 0
    previous_best_r: int | None = None


def parse_orders(s: str) -> tuple[int, ...]:
    vals = tuple(int(x.strip()) for x in s.split(",") if x.strip())
    if not vals:
        raise argparse.ArgumentTypeError("empty order list")
    if any(k < 1 or k % 2 == 0 for k in vals):
        raise argparse.ArgumentTypeError("orders must be positive odd integers")
    return vals


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Dense shifted-Bonferroni hierarchy scan."
    )
    ap.add_argument("--start", type=int, default=10001)
    ap.add_argument("--stop", type=int, default=1000000)
    ap.add_argument("--orders", type=parse_orders, default=(3, 5, 7, 9, 11))
    ap.add_argument("--rmax", type=int, default=12)
    ap.add_argument(
        "--workers",
        type=int,
        default=0,
        help="0 = all but one CPU core; 1 = serial",
    )
    ap.add_argument(
        "--chunksize",
        type=int,
        default=64,
        help="multiprocessing map chunk size",
    )
    ap.add_argument(
        "--write-every",
        type=int,
        default=1000,
        help="write one sampled CSV row every this many n (failures also written)",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=10000,
        help="print progress every this many completed n",
    )
    ap.add_argument("--out", type=Path, default=Path("overnight_hierarchy.csv"))
    ap.add_argument(
        "--transitions-out",
        type=Path,
        default=None,
        help=(
            "compact CSV of initial/best-r transitions; default is "
            "<out stem>_transitions.csv beside --out"
        ),
    )
    ap.add_argument(
        "--print-transitions",
        action="store_true",
        help="also print every best-r change to the console",
    )
    args = ap.parse_args()

    if args.start < 2 or args.stop < args.start:
        raise SystemExit("require 2 <= start <= stop")
    if args.rmax < 3:
        raise SystemExit("--rmax must be at least 3")

    cpu = os.cpu_count() or 2
    workers = args.workers if args.workers > 0 else max(1, cpu - 1)

    print("Dense shifted-hierarchy scan")
    print(f"  n: {args.start:,} .. {args.stop:,}  (EVERY shell)")
    print(f"  orders: {','.join(map(str, args.orders))}")
    print(f"  shifted roots: r=2..{args.rmax}  (r=2 is Bonferroni)")
    print(f"  fixed comparison: r=3")
    print(f"  workers: {workers}")
    if args.transitions_out is None:
        args.transitions_out = args.out.with_name(
            args.out.stem + "_transitions.csv"
        )

    print(f"  output: {args.out}")
    print(f"  transitions: {args.transitions_out}")
    print()

    trackers = {K: Tracker() for K in args.orders}
    total = args.stop - args.start + 1
    t0 = time.time()

    fieldnames = ["n", "exact_holes", "max_depth"]
    for K in args.orders:
        fieldnames += [
            f"K{K}_r3_ratio",
            f"K{K}_r3_margin",
            f"K{K}_bonf_ratio",
            f"K{K}_bonf_margin",
            f"K{K}_best_r",
            f"K{K}_best_ratio",
            f"K{K}_best_margin",
        ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.transitions_out.parent.mkdir(parents=True, exist_ok=True)

    transition_fields = [
        "n",
        "K",
        "event",
        "old_best_r",
        "new_best_r",
        "best_ratio",
        "best_margin",
        "exact_holes",
        "max_depth",
    ]

    # Main CSV: regular samples plus all structurally interesting event shells.
    # Transition CSV: one compact row per initial selection / best-r change.
    with (
        args.out.open("w", newline="", encoding="utf-8") as f,
        args.transitions_out.open("w", newline="", encoding="utf-8") as tf,
    ):
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        transition_writer = csv.DictWriter(tf, fieldnames=transition_fields)
        transition_writer.writeheader()

        def consume(records: Iterable[tuple]) -> None:
            for done, rec in enumerate(records, start=1):
                n, exact_holes, max_depth, kres = rec
                row = {
                    "n": n,
                    "exact_holes": exact_holes,
                    "max_depth": max_depth,
                }
                write_this = (
                    n == args.start
                    or n == args.stop
                    or (args.write_every > 0 and n % args.write_every == 0)
                )

                for (
                    K,
                    fixed_ratio,
                    fixed_margin,
                    bonf_ratio,
                    bonf_margin,
                    best_r,
                    best_ratio,
                    best_margin,
                ) in kres:
                    tr = trackers[K]

                    # Record the initial winning shift and every later change.
                    if tr.previous_best_r is None:
                        tr.previous_best_r = best_r
                        transition_writer.writerow(
                            {
                                "n": n,
                                "K": K,
                                "event": "initial_best_r",
                                "old_best_r": "",
                                "new_best_r": best_r,
                                "best_ratio": f"{best_ratio:.12g}",
                                "best_margin": f"{best_margin:.12g}",
                                "exact_holes": exact_holes,
                                "max_depth": max_depth,
                            }
                        )
                        write_this = True
                    elif best_r != tr.previous_best_r:
                        old_best_r = tr.previous_best_r
                        tr.previous_best_r = best_r
                        tr.best_r_changes += 1
                        transition_writer.writerow(
                            {
                                "n": n,
                                "K": K,
                                "event": "best_r_change",
                                "old_best_r": old_best_r,
                                "new_best_r": best_r,
                                "best_ratio": f"{best_ratio:.12g}",
                                "best_margin": f"{best_margin:.12g}",
                                "exact_holes": exact_holes,
                                "max_depth": max_depth,
                            }
                        )
                        write_this = True
                        if args.print_transitions:
                            print(
                                f"BEST-r CHANGE: K={K}, n={n:,}, "
                                f"{old_best_r} -> {best_r}, "
                                f"ratio={best_ratio:.9f}"
                            )

                    if fixed_ratio >= 1.0:
                        tr.fixed_failures += 1
                        if tr.first_fixed_failure is None:
                            tr.first_fixed_failure = n
                            write_this = True
                            print(
                                f"FIRST r=3 FAILURE: K={K}, "
                                f"n={n:,}, ratio={fixed_ratio:.9f}"
                            )

                    if fixed_ratio > tr.worst_fixed_ratio:
                        tr.worst_fixed_ratio = fixed_ratio
                        tr.worst_fixed_n = n

                    if best_ratio >= 1.0:
                        tr.best_failures += 1
                        if tr.first_best_failure is None:
                            tr.first_best_failure = n
                            write_this = True
                            print(
                                f"FIRST best-r FAILURE: K={K}, "
                                f"n={n:,}, best r={best_r}, "
                                f"ratio={best_ratio:.9f}"
                            )

                    if best_ratio > tr.worst_best_ratio:
                        tr.worst_best_ratio = best_ratio
                        tr.worst_best_n = n
                        tr.worst_best_r = best_r

                    tr.max_best_r = max(tr.max_best_r, best_r)

                    row.update(
                        {
                            f"K{K}_r3_ratio": f"{fixed_ratio:.12g}",
                            f"K{K}_r3_margin": f"{fixed_margin:.12g}",
                            f"K{K}_bonf_ratio": f"{bonf_ratio:.12g}",
                            f"K{K}_bonf_margin": f"{bonf_margin:.12g}",
                            f"K{K}_best_r": best_r,
                            f"K{K}_best_ratio": f"{best_ratio:.12g}",
                            f"K{K}_best_margin": f"{best_margin:.12g}",
                        }
                    )

                if write_this:
                    writer.writerow(row)
                if args.progress_every and done % args.progress_every == 0:
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed else 0.0
                    remain = (total - done) / rate if rate else float("inf")
                    print(
                        f"progress: {done:,}/{total:,} "
                        f"(n={n:,})  "
                        f"{rate:.1f} shells/s  "
                        f"ETA {remain/3600:.2f} h"
                    )
                    f.flush()
                    tf.flush()

        if workers == 1:
            init_worker(args.stop, args.orders, args.rmax)
            consume(analyze_n(n) for n in range(args.start, args.stop + 1))
        else:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=init_worker,
                initargs=(args.stop, args.orders, args.rmax),
            ) as ex:
                records = ex.map(
                    analyze_n,
                    range(args.start, args.stop + 1),
                    chunksize=args.chunksize,
                )
                consume(records)

    elapsed = time.time() - t0

    print()
    print("=" * 76)
    print("FINAL SUMMARY")
    print("=" * 76)
    print(
        f"Scanned every n from {args.start:,} through {args.stop:,} "
        f"({total:,} shells) in {elapsed/3600:.2f} h."
    )
    print()
    for K in args.orders:
        tr = trackers[K]
        print(f"K={K}")
        print(
            "  fixed r=3: "
            f"first failure = "
            f"{tr.first_fixed_failure if tr.first_fixed_failure is not None else 'NONE'}, "
            f"failures = {tr.fixed_failures:,}/{total:,}, "
            f"worst ratio = {tr.worst_fixed_ratio:.9f} "
            f"at n={tr.worst_fixed_n}"
        )
        print(
            f"  best r in 2..{args.rmax}: "
            f"first failure = "
            f"{tr.first_best_failure if tr.first_best_failure is not None else 'NONE'}, "
            f"failures = {tr.best_failures:,}/{total:,}, "
            f"worst ratio = {tr.worst_best_ratio:.9f} "
            f"at n={tr.worst_best_n} "
            f"(best r there={tr.worst_best_r})"
        )
        print(f"  largest best-r selected anywhere: {tr.max_best_r}")
        print(f"  number of best-r changes: {tr.best_r_changes:,}")
        print()

    print(f"Sampled/event detail written to: {args.out}")
    print(f"Best-r transition log written to: {args.transitions_out}")


if __name__ == "__main__":
    main()
