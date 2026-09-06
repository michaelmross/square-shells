#!/usr/bin/env python3
"""
geometric_k5_precision_crossing.py  (v3)

Precision wrapper for geometric_k5_hybrid.py.

This version accepts either TWO or THREE FFT grids.

With two grids G1,G2 it uses a linear extrapolation in h=1/G:

    M(G) = M_inf + a/G + O(G^-2).

With three grids G1,G2,G3 it fits

    M(G) = M_inf + a/G + b/G^2,

and uses the fitted M_inf to locate the crossing

    M^geom_{5,3}(n) = 1.

The three-grid mode is recommended for the final crossing estimate.

The script also:
  * forms a cheap coarse bracket first;
  * RECHECKS/REPAIRS that bracket separately for each precision cutoff,
    because the crossing moves slightly when cutoff/grid settings change;
  * compares two exact/PNT cutoffs to expose hybrid-tail sensitivity;
  * writes every evaluated precision point to CSV.

It expects geometric_k5_hybrid.py in the same directory.

Recommended run
---------------
    python geometric_k5_precision_crossing.py \
      --bracket 1e11,2e11 \
      --cutoffs 2e8,4e8 \
      --grids 262144,524288,1048576 \
      --out k5_precision_crossing_3grid.csv

The calculation remains a numerical asymptotic model: primes above the
exact cutoff are represented by the PNT density dt/log t.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from geometric_k5_hybrid import (  # noqa: E402
    analyze_target,
    parse_int_list,
    segmented_primes,
)


def secant_root(n1: int, f1: float, n2: int, f2: float) -> float:
    """Linear interpolation of a zero between two nearby n-values."""
    if f2 == f1:
        return 0.5 * (n1 + n2)
    return n1 - f1 * (n2 - n1) / (f2 - f1)


def extrapolate_infinite_grid(
    grids: list[int],
    values: list[float],
) -> tuple[float, dict[str, float]]:
    """
    Extrapolate M(G) to G=infinity.

    Two grids:
        fit M = M_inf + a/G.

    Three grids:
        fit M = M_inf + a/G + b/G^2.

    Returns (M_inf, diagnostics).
    """
    if len(grids) != len(values):
        raise ValueError("grids and values must have the same length")
    if len(grids) not in (2, 3):
        raise ValueError("need exactly two or three grids")

    h = 1.0 / np.asarray(grids, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)

    if len(grids) == 2:
        # Columns: 1, h
        A = np.column_stack([np.ones(2), h])
        coeff = np.linalg.solve(A, y)
        m_inf, a = map(float, coeff)
        return m_inf, {
            "fit_a_over_G": a,
            "fit_b_over_G2": float("nan"),
            "largest_grid_shift": m_inf - float(y[-1]),
        }

    # Columns: 1, h, h^2
    A = np.column_stack([np.ones(3), h, h * h])
    coeff = np.linalg.solve(A, y)
    m_inf, a, b = map(float, coeff)

    diagnostics = {
        "fit_a_over_G": a,
        "fit_b_over_G2": b,
        "largest_grid_shift": m_inf - float(y[-1]),
    }

    # If the grids happen to double, also report the two adjacent
    # first-order Richardson extrapolants.  Their convergence is useful
    # diagnostic information but is not used as the final estimate.
    if grids[1] == 2 * grids[0] and grids[2] == 2 * grids[1]:
        r12 = 2.0 * values[1] - values[0]
        r23 = 2.0 * values[2] - values[1]
        diagnostics["richardson_12"] = float(r12)
        diagnostics["richardson_23"] = float(r23)
        diagnostics["richardson_change"] = float(r23 - r12)
    else:
        diagnostics["richardson_12"] = float("nan")
        diagnostics["richardson_23"] = float("nan")
        diagnostics["richardson_change"] = float("nan")

    return m_inf, diagnostics


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
    Cheap initial bracket refinement using one cutoff and one grid.
    """
    mlo = eval_m(lo, primes_all, cutoff, grid)
    mhi = eval_m(hi, primes_all, cutoff, grid)

    flo = mlo - 1.0
    fhi = mhi - 1.0
    if flo * fhi > 0:
        raise SystemExit(
            f"Initial --bracket does not straddle 1 at the coarse settings: "
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
    coarse_lo: int,
    coarse_hi: int,
    outer_lo: int,
    outer_hi: int,
    primes_all: np.ndarray,
    cutoff: int,
    grids: list[int],
    iterations: int,
) -> tuple[float, list[dict[str, float | int]]]:
    """
    Refine the crossing for one exact/PNT cutoff using 2- or 3-grid
    extrapolated values.

    Crucially, the coarse bracket is revalidated at the precision settings.
    If the precision crossing has shifted outside that narrow bracket, the
    appropriate endpoint is extended back to the user's original bracket.
    """
    cache: dict[tuple[int, int], float] = {}
    rows: list[dict[str, float | int]] = []

    def M(n: int, grid: int) -> float:
        key = (n, grid)
        if key not in cache:
            cache[key] = eval_m(n, primes_all, cutoff, grid)
        return cache[key]

    def Minf(n: int) -> tuple[list[float], float, dict[str, float]]:
        vals = [M(n, g) for g in grids]
        m_inf, diag = extrapolate_infinite_grid(grids, vals)
        return vals, m_inf, diag

    def make_row(
        iteration: int,
        role: str,
        n: int,
        vals: list[float],
        m_inf: float,
        diag: dict[str, float],
    ) -> dict[str, float | int | str]:
        row: dict[str, float | int | str] = {
            "cutoff": cutoff,
            "iteration": iteration,
            "role": role,
            "n": n,
            "grid_count": len(grids),
            "M_extrapolated": m_inf,
            "extrapolated_margin": 1.0 - m_inf,
            "fit_a_over_G": diag["fit_a_over_G"],
            "fit_b_over_G2": diag["fit_b_over_G2"],
            "largest_grid_shift": diag["largest_grid_shift"],
            "richardson_12": diag.get("richardson_12", float("nan")),
            "richardson_23": diag.get("richardson_23", float("nan")),
            "richardson_change": diag.get("richardson_change", float("nan")),
        }
        for i, (g, v) in enumerate(zip(grids, vals), start=1):
            row[f"grid_{i}"] = g
            row[f"M_grid_{i}"] = v
        # Make CSV columns stable between 2-grid and 3-grid mode.
        if len(grids) == 2:
            row["grid_3"] = ""
            row["M_grid_3"] = ""
        return row

    lo, hi = coarse_lo, coarse_hi
    lo_vals, lo_inf, lo_diag = Minf(lo)
    hi_vals, hi_inf, hi_diag = Minf(hi)
    flo = lo_inf - 1.0
    fhi = hi_inf - 1.0

    # Repair the narrow coarse bracket at this cutoff/grid extrapolation.
    if flo * fhi > 0:
        if flo < 0.0 and fhi < 0.0:
            print(
                f"  cutoff={cutoff:,}: precision root lies above the "
                f"coarse bracket; extending upper end to {outer_hi:,}",
                flush=True,
            )
            hi = outer_hi
            hi_vals, hi_inf, hi_diag = Minf(hi)
            fhi = hi_inf - 1.0
        elif flo > 0.0 and fhi > 0.0:
            print(
                f"  cutoff={cutoff:,}: precision root lies below the "
                f"coarse bracket; extending lower end to {outer_lo:,}",
                flush=True,
            )
            lo = outer_lo
            lo_vals, lo_inf, lo_diag = Minf(lo)
            flo = lo_inf - 1.0

    if flo * fhi > 0:
        raise RuntimeError(
            f"Even the original bracket does not straddle the extrapolated "
            f"crossing at cutoff={cutoff}: "
            f"M({lo})={lo_inf:.12f}, M({hi})={hi_inf:.12f}. "
            f"Use a wider --bracket."
        )

    mode_name = "quadratic 3-grid" if len(grids) == 3 else "linear 2-grid"
    print(
        f"  cutoff={cutoff:,}: precision bracket "
        f"[{lo:,}, {hi:,}] "
        f"(M={lo_inf:.9f} .. {hi_inf:.9f}; {mode_name})",
        flush=True,
    )

    for it in range(iterations):
        # Re-evaluate from cache so endpoint diagnostics correspond exactly
        # to the current bracket.
        lo_vals, lo_inf, lo_diag = Minf(lo)
        hi_vals, hi_inf, hi_diag = Minf(hi)
        flo = lo_inf - 1.0
        fhi = hi_inf - 1.0

        rows.append(make_row(it, "lo", lo, lo_vals, lo_inf, lo_diag))
        rows.append(make_row(it, "hi", hi, hi_vals, hi_inf, hi_diag))

        if flo * fhi > 0:
            raise RuntimeError(
                f"Extrapolated bracket unexpectedly lost during refinement "
                f"at cutoff={cutoff}: "
                f"lo_M={lo_inf:.12f}, hi_M={hi_inf:.12f}."
            )

        root = secant_root(lo, flo, hi, fhi)
        nmid = int(round(root))
        nmid = max(lo + 1, min(hi - 1, nmid))

        mid_vals, mid_inf, mid_diag = Minf(nmid)
        fmid = mid_inf - 1.0
        rows.append(make_row(it, "mid", nmid, mid_vals, mid_inf, mid_diag))

        if abs(fmid) < 1e-11:
            return float(nmid), rows

        if flo * fmid <= 0:
            hi = nmid
        else:
            lo = nmid

        if hi - lo <= 1000:
            lo_inf_now = Minf(lo)[1]
            hi_inf_now = Minf(hi)[1]
            return (
                secant_root(
                    lo, lo_inf_now - 1.0,
                    hi, hi_inf_now - 1.0,
                ),
                rows,
            )

    lo_inf = Minf(lo)[1]
    hi_inf = Minf(hi)[1]
    root = secant_root(lo, lo_inf - 1.0, hi, hi_inf - 1.0)
    return root, rows


def write_rows(
    path: Path,
    rows: list[dict[str, float | int | str]],
) -> None:
    if not rows:
        return

    # Union of keys, preserving first-seen order.
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Precision crossing finder for M^geom_{5,3}(n)=1 "
            "with 2- or 3-grid extrapolation."
        )
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
        default="262144,524288,1048576",
        help=(
            "two or three FFT grids. Recommended three-grid choice: "
            "262144,524288,1048576"
        ),
    )
    ap.add_argument(
        "--coarse-grid",
        type=int,
        default=131072,
        help="grid used for initial cheap bracket refinement",
    )
    ap.add_argument(
        "--coarse-relative-width",
        type=float,
        default=5e-4,
        help="relative width of the cheap coarse bracket",
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

    bracket = parse_int_list(args.bracket)
    cutoffs = parse_int_list(args.cutoffs)
    grids = parse_int_list(args.grids)

    if len(bracket) != 2:
        raise SystemExit("--bracket must contain exactly two values")
    if len(cutoffs) != 2:
        raise SystemExit("--cutoffs must contain exactly two values")
    if len(grids) not in (2, 3):
        raise SystemExit("--grids must contain exactly two or three values")

    lo, hi = sorted(bracket)
    c1, c2 = sorted(cutoffs)
    grids = sorted(grids)

    if len(set(grids)) != len(grids):
        raise SystemExit("--grids values must be distinct")
    if any(g <= 0 for g in grids):
        raise SystemExit("--grids values must be positive")

    if len(grids) == 3:
        if not (grids[1] == 2 * grids[0] and grids[2] == 2 * grids[1]):
            print(
                "Note: grids do not double. The quadratic fit still works; "
                "adjacent Richardson diagnostics will be omitted.",
                flush=True,
            )
    elif grids[1] != 2 * grids[0]:
        print(
            "Note: grids do not double. A general linear fit in 1/G "
            "will be used.",
            flush=True,
        )

    max_cutoff = c2
    print(f"Sieve exact low primes through {max_cutoff:,} ...", flush=True)
    primes_all = segmented_primes(max_cutoff, args.segment_odds)
    print(f"Stored {len(primes_all):,} odd primes.", flush=True)

    clo, chi = coarse_bisect(
        lo,
        hi,
        primes_all,
        c2,
        args.coarse_grid,
        args.coarse_relative_width,
    )
    print(f"Coarse bracket: [{clo:,}, {chi:,}]", flush=True)

    all_rows: list[dict[str, float | int | str]] = []
    roots: list[tuple[int, float]] = []

    for cutoff in (c1, c2):
        root, rows = precision_root_for_cutoff(
            clo,
            chi,
            lo,
            hi,
            primes_all,
            cutoff,
            grids,
            args.iterations,
        )
        roots.append((cutoff, root))
        all_rows.extend(rows)

        label = (
            "quadratic 3-grid crossing"
            if len(grids) == 3
            else "linear 2-grid crossing"
        )
        print(
            f"cutoff={cutoff:,}: {label} "
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
