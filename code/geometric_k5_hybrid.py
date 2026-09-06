#!/usr/bin/env python3
"""
geometric_k5_hybrid.py

Hybrid large-n computation of the product-truncated quintic main term

    M^geom_{5,3}(n)
      = E1 - (14/15)E2 + (4/5)E3 - (3/5)E4 + (1/3)E5,

where

    E_j = sum_{3 <= p1 < ... < pj <= n, p1...pj < (n+1)^2}
              1/(p1...pj).

This version is intended for n far beyond the range where enumerating all
primes <= n is attractive (e.g. 10^12 and above).

Method
------
1. Sieve primes exactly only up to a user-chosen cutoff C.
2. Above C, replace the prime measure by the PNT density dt/log t.
3. Put prime reciprocal-power measures onto the normalized logarithmic axis

       y = r log p / log X,   X=(n+1)^2,

   so the product cutoff is simply "total y < 1".
4. Recover E3,E4,E5 from measure-valued Newton identities and FFT
   convolutions.

For r >= 1, the PNT tail in a prime bin [a,b] uses

    integral_a^b t^{-r} d pi(t)  ~  integral_a^b t^{-r} dt/log t.

The antiderivative is

    log log t                         (r=1),
    Ei(-(r-1) log t)                  (r>=2).

IMPORTANT
---------
This is a high-accuracy NUMERICAL MODEL, not a rigorous large-n evaluation.
Its two controllable numerical approximations are:

* the exact/PNT split at --cutoff;
* logarithmic binning at --grid.

Use --stability-cutoffs and/or several grids before treating a crossing
location as numerically settled.

At n=10^9, cutoff=10^7 and grid=131072 reproduces the fully enumerated
result to a few parts in 10^6; cutoff=10^8 improves this further.

Examples
--------
Milestones through 10^12:

    python geometric_k5_hybrid.py \
      --n 1e9,1e10,1e11,2e11,5e11,1e12 \
      --cutoff 1e7 --grid 131072 \
      --out k5_hybrid_to_1e12.csv

Cutoff-stability check at 10^12:

    python geometric_k5_hybrid.py \
      --n 1e12 \
      --stability-cutoffs 1e6,1e7,1e8 \
      --grid 131072 \
      --out k5_cutoff_stability_1e12.csv

Locate the M^geom_{5,3}=1 crossing by logarithmic bisection:

    python geometric_k5_hybrid.py \
      --crossing 1e11,2e11 \
      --cutoff 1e7 --grid 131072 \
      --crossing-tol 1e-4 \
      --out k5_crossing.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
from scipy.special import expi


COEFF = np.array(
    [0.0, 1.0, -14.0 / 15.0, 4.0 / 5.0, -3.0 / 5.0, 1.0 / 3.0],
    dtype=np.float64,
)


def parse_int(s: str) -> int:
    """Accept integers or scientific notation such as 1e12."""
    s = s.strip()
    if not s:
        raise ValueError("empty integer")
    if any(ch in s.lower() for ch in (".", "e")):
        return int(round(float(s)))
    return int(s)


def parse_int_list(s: str) -> list[int]:
    return [parse_int(x) for x in s.split(",") if x.strip()]


def simple_primes(limit: int) -> np.ndarray:
    """Small base-prime sieve."""
    if limit < 2:
        return np.empty(0, dtype=np.int64)

    sieve = np.ones(limit + 1, dtype=bool)
    sieve[:2] = False
    if limit >= 4:
        sieve[4::2] = False

    for p in range(3, math.isqrt(limit) + 1, 2):
        if sieve[p]:
            sieve[p * p :: 2 * p] = False

    return np.flatnonzero(sieve).astype(np.int64)


def segmented_primes(limit: int, segment_odds: int = 2_000_000) -> np.ndarray:
    """
    Return odd primes <= limit.

    This stores only primes up to the exact cutoff, not primes up to target n.
    """
    if limit < 3:
        return np.empty(0, dtype=np.int64)

    base = simple_primes(math.isqrt(limit))
    chunks: list[np.ndarray] = []

    for low in range(3, limit + 1, 2 * segment_odds):
        high = min(limit, low + 2 * segment_odds - 2)
        if low % 2 == 0:
            low += 1

        size = ((high - low) // 2) + 1
        seg = np.ones(size, dtype=bool)

        # The segment contains odd integers, so ignore base prime 2.
        for p0 in base:
            p = int(p0)
            if p == 2:
                continue
            pp = p * p
            if pp > high:
                break

            start = max(pp, ((low + p - 1) // p) * p)
            if start % 2 == 0:
                start += p

            seg[(start - low) // 2 :: p] = False

        chunks.append((low + 2 * np.flatnonzero(seg)).astype(np.int64))

    if not chunks:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(chunks)


def pnt_weight_integral(a: np.ndarray, b: np.ndarray, r: int) -> np.ndarray:
    """
    Vectorized integral of t^{-r}/log(t) from a to b.

    Caller must supply 1 < a < b.
    """
    la = np.log(a)
    lb = np.log(b)

    if r == 1:
        return np.log(lb / la)

    c = float(r - 1)
    return expi(-c * lb) - expi(-c * la)


def scalar_pnt_tail(cutoff: float, n: float, r: int) -> float:
    if n <= cutoff:
        return 0.0

    a = np.array([cutoff], dtype=np.float64)
    b = np.array([n], dtype=np.float64)
    return float(pnt_weight_integral(a, b, r)[0])


def exact_low_measures(
    primes: np.ndarray,
    target_n: int,
    grid: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Exact prime measures from the stored low primes, expressed in the
    target n's logarithmic coordinate.
    """
    X = float((target_n + 1) ** 2)
    logX = math.log(X)

    use = primes[primes <= target_n]
    pf = use.astype(np.float64)

    hist = [np.zeros(grid, dtype=np.float64) for _ in range(5)]
    power = np.zeros(6, dtype=np.float64)

    if pf.size == 0:
        return hist, power

    inv = 1.0 / pf
    log_ratio = np.log(pf) / logX
    invpow = np.ones_like(inv)

    for r in range(1, 6):
        invpow *= inv
        power[r] = float(invpow.sum())

        pos = r * log_ratio
        mask = pos < 1.0
        if np.any(mask):
            idx = np.floor(pos[mask] * grid).astype(np.int64)
            # Numerical guard for a value rounded exactly to grid.
            idx = np.minimum(idx, grid - 1)
            hist[r - 1] += np.bincount(
                idx, weights=invpow[mask], minlength=grid
            )

    return hist, power


def add_pnt_tail(
    hist: list[np.ndarray],
    power: np.ndarray,
    target_n: int,
    cutoff: int,
    grid: int,
) -> None:
    """
    Add PNT-density contributions from cutoff < p <= target_n.
    """
    if target_n <= cutoff:
        return

    X = float((target_n + 1) ** 2)
    logX = math.log(X)

    # Scalar unrestricted power sums.
    for r in range(1, 6):
        power[r] += scalar_pnt_tail(float(cutoff), float(target_n), r)

    # Product-truncated log-axis measures.
    j = np.arange(grid, dtype=np.float64)
    y0 = j / grid
    y1 = (j + 1.0) / grid

    for r in range(1, 6):
        lo = np.exp(logX * y0 / r)
        hi = np.exp(logX * y1 / r)

        lo = np.maximum(lo, float(cutoff))
        hi = np.minimum(hi, float(target_n))

        valid = hi > lo
        if not np.any(valid):
            continue

        vals = np.zeros(grid, dtype=np.float64)
        vals[valid] = pnt_weight_integral(lo[valid], hi[valid], r)
        hist[r - 1] += vals


def unrestricted_elementary(power: np.ndarray) -> np.ndarray:
    """e_0,...,e_5 from scalar Newton identities."""
    e = np.zeros(6, dtype=np.float64)
    e[0] = 1.0

    for k in range(1, 6):
        e[k] = (
            sum(
                ((-1) ** (i - 1)) * e[k - i] * power[i]
                for i in range(1, k + 1)
            )
            / k
        )
    return e


def geometric_e3_e5(hist: list[np.ndarray]) -> tuple[float, float, float]:
    """
    E3,E4,E5 from measure-valued Newton identities.

    Histogram index addition models addition of normalized logarithms, and
    summing convolution coefficients with index < grid enforces product < X.
    """
    grid = len(hist[0])

    fft_len = 1
    while fft_len < 5 * grid:
        fft_len <<= 1

    F: list[np.ndarray | None] = [None]
    for a in hist:
        padded = np.zeros(fft_len, dtype=np.float64)
        padded[:grid] = a
        F.append(np.fft.rfft(padded))

    def cutoff_mass(powers: dict[int, int]) -> float:
        z = np.ones_like(F[1])
        for r, exponent in powers.items():
            z *= F[r] ** exponent
        conv = np.fft.irfft(z, n=fft_len)
        return float(conv[:grid].sum())

    # e3 = (p1^3 - 3 p1 p2 + 2 p3)/6
    a111 = cutoff_mass({1: 3})
    a12 = cutoff_mass({1: 1, 2: 1})
    a3 = float(hist[2].sum())
    E3 = (a111 - 3.0 * a12 + 2.0 * a3) / 6.0

    # e4 = (p1^4 - 6 p1^2 p2 + 3 p2^2 + 8 p1 p3 - 6 p4)/24
    a1111 = cutoff_mass({1: 4})
    a112 = cutoff_mass({1: 2, 2: 1})
    a22 = cutoff_mass({2: 2})
    a13 = cutoff_mass({1: 1, 3: 1})
    a4 = float(hist[3].sum())
    E4 = (
        a1111
        - 6.0 * a112
        + 3.0 * a22
        + 8.0 * a13
        - 6.0 * a4
    ) / 24.0

    # e5 = (p1^5 - 10 p1^3 p2 + 15 p1 p2^2 + 20 p1^2 p3
    #       - 20 p2 p3 - 30 p1 p4 + 24 p5)/120
    a11111 = cutoff_mass({1: 5})
    a1112 = cutoff_mass({1: 3, 2: 1})
    a122 = cutoff_mass({1: 1, 2: 2})
    a113 = cutoff_mass({1: 2, 3: 1})
    a23 = cutoff_mass({2: 1, 3: 1})
    a14 = cutoff_mass({1: 1, 4: 1})
    a5 = float(hist[4].sum())

    E5 = (
        a11111
        - 10.0 * a1112
        + 15.0 * a122
        + 20.0 * a113
        - 20.0 * a23
        - 30.0 * a14
        + 24.0 * a5
    ) / 120.0

    return E3, E4, E5


def analyze_target(
    n: int,
    primes: np.ndarray,
    cutoff: int,
    grid: int,
) -> dict[str, float | int]:
    t0 = time.perf_counter()

    hist, power = exact_low_measures(primes, n, grid)
    add_pnt_tail(hist, power, n, cutoff, grid)

    e = unrestricted_elementary(power)
    E3, E4, E5 = geometric_e3_e5(hist)

    E = np.array([1.0, e[1], e[2], E3, E4, E5], dtype=np.float64)

    Mgeom = float(np.dot(COEFF[1:], E[1:]))
    Munr = float(np.dot(COEFF[1:], e[1:]))

    return {
        "n": n,
        "X_exclusive": (n + 1) ** 2,
        "cutoff": cutoff,
        "grid": grid,
        "low_prime_count": int(np.searchsorted(primes, n, side="right")),
        "E1_geom": E[1],
        "E2_geom": E[2],
        "E3_geom": E[3],
        "E4_geom": E[4],
        "E5_geom": E[5],
        "M53_geom": Mgeom,
        "geom_margin_1_minus_M": 1.0 - Mgeom,
        "M53_unrestricted": Munr,
        "unrestricted_margin_1_minus_M": 1.0 - Munr,
        "removed_high_product_mass": Munr - Mgeom,
        "seconds": time.perf_counter() - t0,
    }


def write_csv(rows: list[dict[str, float | int]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_targets(
    targets: list[int],
    cutoff: int,
    grid: int,
    segment_odds: int,
) -> list[dict[str, float | int]]:
    actual_cutoff = min(cutoff, max(targets))
    t0 = time.perf_counter()
    print(f"Sieve exact low primes through {actual_cutoff:,} ...", flush=True)
    primes = segmented_primes(actual_cutoff, segment_odds)
    print(
        f"  stored {len(primes):,} odd primes in "
        f"{time.perf_counter() - t0:.2f}s",
        flush=True,
    )

    rows: list[dict[str, float | int]] = []
    for n in targets:
        row = analyze_target(n, primes, actual_cutoff, grid)
        rows.append(row)
        print(
            f"n={n:,}  "
            f"Mgeom={row['M53_geom']:.12f}  "
            f"1-M={row['geom_margin_1_minus_M']:+.12f}  "
            f"Munrestricted={row['M53_unrestricted']:.12f}  "
            f"time={row['seconds']:.2f}s",
            flush=True,
        )
    return rows


def run_stability(
    target: int,
    cutoffs: list[int],
    grid: int,
    segment_odds: int,
) -> list[dict[str, float | int]]:
    max_cutoff = min(max(cutoffs), target)

    print(f"Sieve exact low primes through {max_cutoff:,} ...", flush=True)
    primes_all = segmented_primes(max_cutoff, segment_odds)

    rows: list[dict[str, float | int]] = []
    for cutoff in cutoffs:
        cutoff = min(cutoff, target)
        stop = np.searchsorted(primes_all, cutoff, side="right")
        primes = primes_all[:stop]
        row = analyze_target(target, primes, cutoff, grid)
        rows.append(row)
        print(
            f"cutoff={cutoff:,}  "
            f"Mgeom={row['M53_geom']:.12f}  "
            f"1-M={row['geom_margin_1_minus_M']:+.12f}",
            flush=True,
        )

    if len(rows) >= 2:
        vals = [float(r["M53_geom"]) for r in rows]
        print(
            "cutoff spread in Mgeom: "
            f"{max(vals) - min(vals):.3e}",
            flush=True,
        )
    return rows


def crossing_search(
    lo: int,
    hi: int,
    cutoff: int,
    grid: int,
    segment_odds: int,
    tol: float,
    max_iter: int,
) -> list[dict[str, float | int]]:
    """
    Logarithmic bisection for Mgeom(n)=1.

    Stops when |M-1| <= tol or after max_iter iterations.
    """
    if lo >= hi:
        raise ValueError("crossing bracket must satisfy lo < hi")

    actual_cutoff = min(cutoff, hi)
    print(f"Sieve exact low primes through {actual_cutoff:,} ...", flush=True)
    primes = segmented_primes(actual_cutoff, segment_odds)

    cache: dict[int, dict[str, float | int]] = {}

    def ev(n: int) -> dict[str, float | int]:
        if n not in cache:
            cache[n] = analyze_target(n, primes, actual_cutoff, grid)
            print(
                f"n={n:,}  Mgeom={cache[n]['M53_geom']:.12f}  "
                f"1-M={cache[n]['geom_margin_1_minus_M']:+.12f}",
                flush=True,
            )
        return cache[n]

    rlo = ev(lo)
    rhi = ev(hi)
    flo = float(rlo["M53_geom"]) - 1.0
    fhi = float(rhi["M53_geom"]) - 1.0

    if flo * fhi > 0:
        raise SystemExit(
            "Bracket does not straddle Mgeom=1. "
            f"M({lo})={float(rlo['M53_geom']):.9f}, "
            f"M({hi})={float(rhi['M53_geom']):.9f}"
        )

    for _ in range(max_iter):
        # Geometric midpoint is natural for a many-decade scale.
        mid = int(round(math.sqrt(lo * hi)))
        if mid <= lo or mid >= hi:
            break

        rm = ev(mid)
        fm = float(rm["M53_geom"]) - 1.0

        if abs(fm) <= tol:
            break

        if flo * fm <= 0:
            hi, rhi, fhi = mid, rm, fm
        else:
            lo, rlo, flo = mid, rm, fm

    rows = [cache[n] for n in sorted(cache)]
    best = min(rows, key=lambda r: abs(float(r["M53_geom"]) - 1.0))
    print(
        "\nNearest evaluated point to the crossing: "
        f"n={int(best['n']):,}, Mgeom={float(best['M53_geom']):.12f}",
        flush=True,
    )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Hybrid large-n K=5,r=3 geometric main-term computation."
    )

    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--n",
        help="comma-separated target n values; accepts scientific notation",
    )
    mode.add_argument(
        "--crossing",
        help="lo,hi bracket for logarithmic search of Mgeom=1",
    )

    ap.add_argument(
        "--cutoff",
        type=parse_int,
        default=10_000_000,
        help="exact-prime cutoff (default 1e7)",
    )
    ap.add_argument(
        "--stability-cutoffs",
        help="comma-separated exact cutoffs; requires a single --n target",
    )
    ap.add_argument(
        "--grid",
        type=int,
        default=131072,
        help="logarithmic FFT bins (default 131072)",
    )
    ap.add_argument(
        "--segment-odds",
        type=int,
        default=2_000_000,
        help="odd integers per sieve segment",
    )
    ap.add_argument(
        "--crossing-tol",
        type=float,
        default=1e-4,
        help="stop crossing search when |Mgeom-1| <= this",
    )
    ap.add_argument(
        "--crossing-max-iter",
        type=int,
        default=20,
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("geometric_k5_hybrid_results.csv"),
    )

    args = ap.parse_args()

    if args.grid < 1024:
        raise SystemExit("--grid is implausibly small")
    if args.cutoff < 3:
        raise SystemExit("--cutoff must be >= 3")

    if args.crossing:
        lo, hi = parse_int_list(args.crossing)
        rows = crossing_search(
            lo,
            hi,
            args.cutoff,
            args.grid,
            args.segment_odds,
            args.crossing_tol,
            args.crossing_max_iter,
        )

    else:
        targets = parse_int_list(
            args.n or "1e9,1e10,1e11,2e11,5e11,1e12"
        )

        if args.stability_cutoffs:
            if len(targets) != 1:
                raise SystemExit(
                    "--stability-cutoffs requires exactly one --n target"
                )
            cutoffs = parse_int_list(args.stability_cutoffs)
            rows = run_stability(
                targets[0], cutoffs, args.grid, args.segment_odds
            )
        else:
            rows = run_targets(
                targets, args.cutoff, args.grid, args.segment_odds
            )

    write_csv(rows, args.out)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
