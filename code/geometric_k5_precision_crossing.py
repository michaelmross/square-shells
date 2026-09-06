#!/usr/bin/env python3
"""
geometric_k5_precision_crossing.py

Precision wrapper for geometric_k5_hybrid.py.

Purpose
-------
Automatically refine the crossing (v2: robust per-cutoff bracketing)

    M^geom_{5,3}(n) = 1

using:
  * a coarse logarithmic bisection;
  * two FFT grids G and 2G;
  * first-order Richardson extrapolation in 1/G;
  * two exact/PNT cutoffs C and 2C (or user supplied values)
    to estimate sensitivity to the hybrid prime tail.

The script is intended to turn the "how precisely does it cross?" exercise
into one command.

It expects geometric_k5_hybrid.py in the same directory.

Example
-------
    python geometric_k5_precision_crossing.py \
      --bracket 1e11,2e11 \
      --cutoffs 2e8,4e8 \
      --grids 262144,524288 \
      --out k5_precision_crossing.csv

Typical interpretation of the output:
  * root_richardson is the best crossing estimate for each cutoff;
  * the difference between the two cutoff estimates is a practical
    hybrid-tail sensitivity estimate;
  * the final summary prints a central estimate and a conservative
    numerical uncertainty scale.

This remains a numerical asymptotic model because the prime tail above the
exact cutoff is replaced by dt/log t.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

# Import the computation engine from the companion script.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from geometric_k5_hybrid import (  # noqa: E402
    analyze_target,
    parse_int,
    parse_int_list,
    segmented_primes,
)


def richardson(m_g: float, m_2g: float) -> float:
    """
    First-order Richardson extrapolation assuming
        M(G) = M(infty) + c/G + O(G^-2).
    """
    return 2.0 * m_2g - m_g


def secant_root(n1: int, f1: float, n2: int, f2: float) -> float:
    """Linear interpolation root in n."""
    if f2 == f1:
        return 0.5 * (n1 + n2)
    return n1 - f1 * (n2 - n1) / (f2 - f1)


def eval_m(
    n: int,
    primes_all: np.ndarray,
    cutoff: int,
    grid: int,
) -> float:
    stop = np.searchsorted(primes_all, cutoff, side="right")
    primes = primes_all[:stop]
    row = analyze_target(n, primes, cutoff, grid)
    return float(row["M53_geom"])


def coarse_bisect(
    lo: int,
    hi: int,
    primes_all: np.ndarray,
    cutoff: int,
    grid: int,
    target_width: float,
    max_iter: int = 30,
) -> tuple[int, int]:
    """
    Coarse bracket refinement using one cutoff/grid.
    Stops when relative bracket width is below target_width.
    """
    mlo = eval_m(lo, primes_all, cutoff, grid)
    mhi = eval_m(hi, primes_all, cutoff, grid)

    flo = mlo - 1.0
    fhi = mhi - 1.0
    if flo * fhi > 0:
        raise SystemExit(
            f"Initial bracket does not straddle 1: "
            f"M({lo})={mlo:.9f}, M({hi})={mhi:.9f}"
        )

    for _ in range(max_iter):
        if (hi - lo) / math.sqrt(lo * hi) <= target_width:
            break

        mid = int(round(math.sqrt(lo * hi)))
        if mid <= lo or mid >= hi:
            break

        mmid = eval_m(mid, primes_all, cutoff, grid)
        fmid = mmid - 1.0

        if flo * fmid <= 0:
            hi, mhi, fhi = mid, mmid, fmid
        else:
            lo, mlo, flo = mid, mmid, fmid

    return lo, hi


def precision_root_for_cutoff(
    lo: int,
    hi: int,
    outer_lo: int,
    outer_hi: int,
    primes_all: np.ndarray,
    cutoff: int,
    g1: int,
    g2: int,
    iterations: int,
) -> tuple[float, list[dict[str, float | int]]]:
    """
    Refine a crossing using Richardson-extrapolated values at two grids.

    Each iteration evaluates the current endpoints using both grids,
    forms M_inf ~= 2*M(g2)-M(g1), and secant-interpolates a new n.
    """
    cache: dict[tuple[int, int], float] = {}
    rows: list[dict[str, float | int]] = []

    def M(n: int, grid: int) -> float:
        key = (n, grid)
        if key not in cache:
            cache[key] = eval_m(n, primes_all, cutoff, grid)
        return cache[key]

    def Minf(n: int) -> tuple[float, float, float]:
        m1 = M(n, g1)
        m2 = M(n, g2)
        mr = richardson(m1, m2)
        return m1, m2, mr

    # The cheap coarse bracket is formed with one cutoff/grid.  At the
    # precision stage, changing either the cutoff or the Richardson grid
    # pair shifts the numerical root slightly.  Therefore the same narrow
    # coarse bracket is NOT guaranteed to straddle M=1 for every precision
    # setting.  Repair it against the user's original outer bracket before
    # starting the expensive refinement.
    lo_r = Minf(lo)[2]
    hi_r = Minf(hi)[2]
    flo = lo_r - 1.0
    fhi = hi_r - 1.0

    if flo * fhi > 0:
        if flo < 0.0 and fhi < 0.0:
            print(
                f"  cutoff={cutoff:,}: precision root lies above the "
                f"coarse bracket; extending upper end to {outer_hi:,}",
                flush=True,
            )
            hi = outer_hi
            hi_r = Minf(hi)[2]
            fhi = hi_r - 1.0
        elif flo > 0.0 and fhi > 0.0:
            print(
                f"  cutoff={cutoff:,}: precision root lies below the "
                f"coarse bracket; extending lower end to {outer_lo:,}",
                flush=True,
            )
            lo = outer_lo
            lo_r = Minf(lo)[2]
            flo = lo_r - 1.0

    if flo * fhi > 0:
        raise RuntimeError(
            f"Even the original bracket does not straddle the Richardson "
            f"crossing at cutoff={cutoff}: "
            f"M({lo})={lo_r:.12f}, M({hi})={hi_r:.12f}. "
            f"Use a wider --bracket."
        )

    print(
        f"  cutoff={cutoff:,}: precision bracket "
        f"[{lo:,}, {hi:,}] "
        f"(M={lo_r:.9f} .. {hi_r:.9f})",
        flush=True,
    )

    for it in range(iterations):
        lo_m1, lo_m2, lo_r = Minf(lo)
        hi_m1, hi_m2, hi_r = Minf(hi)

        rows.extend([
            {
                "cutoff": cutoff,
                "iteration": it,
                "n": lo,
                "grid_lo": g1,
                "grid_hi": g2,
                "M_grid_lo": lo_m1,
                "M_grid_hi": lo_m2,
                "M_richardson": lo_r,
                "richardson_margin": 1.0 - lo_r,
            },
            {
                "cutoff": cutoff,
                "iteration": it,
                "n": hi,
                "grid_lo": g1,
                "grid_hi": g2,
                "M_grid_lo": hi_m1,
                "M_grid_hi": hi_m2,
                "M_richardson": hi_r,
                "richardson_margin": 1.0 - hi_r,
            },
        ])

        flo = lo_r - 1.0
        fhi = hi_r - 1.0

        if flo * fhi > 0:
            raise RuntimeError(
                f"Richardson bracket unexpectedly lost during refinement "
                f"at cutoff={cutoff}: lo_M={lo_r:.12f}, hi_M={hi_r:.12f}. "
                f"This is no longer the coarse-bracket issue; inspect the "
                f"grid/cutoff stability."
            )

        root = secant_root(lo, flo, hi, fhi)
        nmid = int(round(root))
        nmid = max(lo + 1, min(hi - 1, nmid))

        mid_m1, mid_m2, mid_r = Minf(nmid)
        rows.append(
            {
                "cutoff": cutoff,
                "iteration": it,
                "n": nmid,
                "grid_lo": g1,
                "grid_hi": g2,
                "M_grid_lo": mid_m1,
                "M_grid_hi": mid_m2,
                "M_richardson": mid_r,
                "richardson_margin": 1.0 - mid_r,
            }
        )

        fmid = mid_r - 1.0
        if abs(fmid) < 1e-10:
            return float(nmid), rows

        if flo * fmid <= 0:
            hi = nmid
        else:
            lo = nmid

        if hi - lo <= 1000:
            flo_now = Minf(lo)[2] - 1.0
            fhi_now = Minf(hi)[2] - 1.0
            return secant_root(lo, flo_now, hi, fhi_now), rows

    lo_r = Minf(lo)[2]
    hi_r = Minf(hi)[2]
    root = secant_root(lo, lo_r - 1.0, hi, hi_r - 1.0)
    return root, rows


def write_rows(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Precision crossing finder for M^geom_{5,3}(n)=1."
    )
    ap.add_argument(
        "--bracket",
        default="1e11,2e11",
        help="initial crossing bracket lo,hi",
    )
    ap.add_argument(
        "--cutoffs",
        default="2e8,4e8",
        help="two exact-prime cutoffs, e.g. 2e8,4e8",
    )
    ap.add_argument(
        "--grids",
        default="262144,524288",
        help="two FFT grids G,2G",
    )
    ap.add_argument(
        "--coarse-grid",
        type=int,
        default=131072,
        help="grid used for initial bracket refinement",
    )
    ap.add_argument(
        "--coarse-relative-width",
        type=float,
        default=5e-4,
        help="relative width of coarse bracket before precision stage",
    )
    ap.add_argument(
        "--iterations",
        type=int,
        default=4,
        help="precision secant/bracketing iterations per cutoff",
    )
    ap.add_argument(
        "--segment-odds",
        type=int,
        default=2_000_000,
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("geometric_k5_precision_crossing.csv"),
    )
    args = ap.parse_args()

    lo, hi = parse_int_list(args.bracket)
    cutoffs = parse_int_list(args.cutoffs)
    grids = parse_int_list(args.grids)

    if len(cutoffs) != 2:
        raise SystemExit("--cutoffs must contain exactly two values")
    if len(grids) != 2:
        raise SystemExit("--grids must contain exactly two values")

    c1, c2 = sorted(cutoffs)
    g1, g2 = sorted(grids)

    if g2 != 2 * g1:
        print(
            "Warning: Richardson formula assumes second grid is twice first.",
            flush=True,
        )

    max_cutoff = c2
    print(f"Sieve exact low primes through {max_cutoff:,} ...", flush=True)
    primes_all = segmented_primes(max_cutoff, args.segment_odds)
    print(f"Stored {len(primes_all):,} odd primes.", flush=True)

    # Refine the initial bracket cheaply using the larger cutoff.
    clo, chi = coarse_bisect(
        lo,
        hi,
        primes_all,
        c2,
        args.coarse_grid,
        args.coarse_relative_width,
    )
    print(f"Coarse bracket: [{clo:,}, {chi:,}]", flush=True)

    all_rows: list[dict[str, float | int]] = []
    roots: list[tuple[int, float]] = []

    for cutoff in (c1, c2):
        root, rows = precision_root_for_cutoff(
            clo,
            chi,
            lo,
            hi,
            primes_all,
            cutoff,
            g1,
            g2,
            args.iterations,
        )
        roots.append((cutoff, root))
        all_rows.extend(rows)
        print(
            f"cutoff={cutoff:,}: Richardson crossing "
            f"n ~= {root:,.0f}",
            flush=True,
        )

    write_rows(args.out, all_rows)

    r1 = roots[0][1]
    r2 = roots[1][1]
    central = r2
    cutoff_shift = abs(r2 - r1)

    print("\nSummary")
    print("-------")
    print(f"best estimate (larger cutoff): n ~= {central:,.0f}")
    print(f"cutoff sensitivity:           {cutoff_shift:,.0f}")
    print(
        "Suggested numerical report:     "
        f"n ~= {central:,.0f} "
        f"(cutoff shift about {cutoff_shift:,.0f})"
    )
    print(
        "\nThis uncertainty is numerical/model sensitivity only; "
        "it is not a rigorous error bound."
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
