#!/usr/bin/env python3
"""
Scalable K=5 geometric main-term computation for square shells.

For n >= 3, put X=(n+1)^2 and

    E_j(n) = sum 1/(p1...pj)

over odd primes 3 <= p1 < ... < pj <= n satisfying p1...pj < X.

The target certificate is the shifted quintic (K,r)=(5,3):

    M_5,3^geom
      = E1 - (14/15)E2 + (4/5)E3 - (3/5)E4 + (1/3)E5.

Why this version scales
-----------------------
The original exact recursion explicitly enumerated many prime-product
prefixes.  This version instead bins the prime measures

    mu_r = sum_{p<=n, p odd} p^(-r) delta_{r log p / log X}

on the normalized logarithmic product axis.  Product truncation is simply
"total coordinate < 1".  Newton's identities lift to convolution identities:

 E3 = (mu1^3 - 3 mu1*mu2 + 2 mu3)/6,
 E4 = (mu1^4 - 6 mu1^2*mu2 + 3 mu2^2
       + 8 mu1*mu3 - 6 mu4)/24,
 E5 = (mu1^5 - 10 mu1^3*mu2 + 15 mu1*mu2^2
       + 20 mu1^2*mu3 - 20 mu2*mu3
       - 30 mu1*mu4 + 24 mu5)/120.

Convolutions are evaluated by FFT.  Prime measures are accumulated with a
NumPy segmented sieve, so primes are never stored globally.

E1 and E2 are exact (up to floating-point summation): every product of two
distinct primes <= n is < (n+1)^2.  E3..E5 have only the logarithmic binning
error.  With --grid 131072, the n=10^6 value of M_5,3^geom agrees with the
previous exact recursive calculation to about 3e-8.

Examples
--------
    python geometric_k5_scalable.py --n 1000000
    python geometric_k5_scalable.py --n 1000000,10000000,100000000,1000000000
    python geometric_k5_scalable.py --n 1000000000 --grid 262144 --out k5_1e9.csv
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd


def simple_primes(limit: int) -> np.ndarray:
    """Primes <= limit, used only as base primes for the segmented sieve."""
    if limit < 2:
        return np.empty(0, dtype=np.int64)
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[:2] = False
    if limit >= 4:
        sieve[4::2] = False
    for p in range(3, math.isqrt(limit) + 1, 2):
        if sieve[p]:
            sieve[p * p :: 2 * p] = False
    return np.flatnonzero(sieve)


def prime_histograms(
    n: int,
    grid: int,
    segment_odds: int = 2_000_000,
) -> tuple[list[np.ndarray], np.ndarray, int, float]:
    """
    Return logarithmic prime-measure histograms mu_1,...,mu_5,
    scalar power sums sum p^{-r}, the odd-prime count, and log X.

    Histogram coordinate is y = r log p / log X; only y<1 is retained.
    """
    log_x = 2.0 * math.log(n + 1.0)
    base = simple_primes(math.isqrt(n))

    hist = [np.zeros(grid, dtype=np.float64) for _ in range(5)]
    power_sums = np.zeros(6, dtype=np.float64)
    odd_prime_count = 0

    for low in range(3, n + 1, 2 * segment_odds):
        high = min(n, low + 2 * segment_odds - 2)
        if low % 2 == 0:
            low += 1
        size = ((high - low) // 2) + 1
        seg = np.ones(size, dtype=bool)

        # base[0] is 2; the segment contains only odd integers.
        for p0 in base[1:]:
            p = int(p0)
            pp = p * p
            if pp > high:
                break
            start = max(pp, ((low + p - 1) // p) * p)
            if start % 2 == 0:
                start += p
            seg[(start - low) // 2 :: p] = False

        primes = low + 2 * np.flatnonzero(seg)
        odd_prime_count += int(primes.size)

        pf = primes.astype(np.float64)
        inv = 1.0 / pf
        logp_over_logx = np.log(pf) / log_x

        invpow = np.ones_like(inv)
        for r in range(1, 6):
            invpow *= inv
            power_sums[r] += invpow.sum()

            pos = r * logp_over_logx
            mask = pos < 1.0
            if np.any(mask):
                idx = np.floor(pos[mask] * grid).astype(np.int64)
                hist[r - 1] += np.bincount(
                    idx, weights=invpow[mask], minlength=grid
                )

    return hist, power_sums, odd_prime_count, log_x


def unrestricted_elementary(power_sums: np.ndarray) -> np.ndarray:
    """e_0,...,e_5 from scalar Newton identities."""
    e = np.zeros(6, dtype=np.float64)
    e[0] = 1.0
    for k in range(1, 6):
        e[k] = sum(
            ((-1) ** (i - 1)) * e[k - i] * power_sums[i]
            for i in range(1, k + 1)
        ) / k
    return e


def geometric_e3_e5(hist: list[np.ndarray]) -> tuple[float, float, float]:
    """
    Product-truncated E3,E4,E5 from measure-valued Newton identities.
    """
    grid = len(hist[0])

    # Need enough zero padding for the fivefold convolution of arrays
    # supported on 0,...,grid-1.
    fft_len = 1
    while fft_len < 5 * grid:
        fft_len <<= 1

    F = [None]
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

    # j=3
    a111 = cutoff_mass({1: 3})
    a12 = cutoff_mass({1: 1, 2: 1})
    a3 = float(hist[2].sum())
    E3 = (a111 - 3.0 * a12 + 2.0 * a3) / 6.0

    # j=4
    a1111 = cutoff_mass({1: 4})
    a112 = cutoff_mass({1: 2, 2: 1})
    a22 = cutoff_mass({2: 2})
    a13 = cutoff_mass({1: 1, 3: 1})
    a4 = float(hist[3].sum())
    E4 = (
        a1111 - 6.0 * a112 + 3.0 * a22 + 8.0 * a13 - 6.0 * a4
    ) / 24.0

    # j=5
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


def analyze_n(n: int, grid: int, segment_odds: int) -> dict[str, float | int]:
    t0 = time.perf_counter()
    hist, power_sums, prime_count, log_x = prime_histograms(
        n, grid, segment_odds
    )
    sieve_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    e = unrestricted_elementary(power_sums)
    E3, E4, E5 = geometric_e3_e5(hist)
    fft_seconds = time.perf_counter() - t1

    E = np.zeros(6, dtype=np.float64)
    E[0] = 1.0
    E[1] = e[1]
    E[2] = e[2]
    E[3], E[4], E[5] = E3, E4, E5

    c = np.array([0.0, 1.0, -14.0 / 15.0, 4.0 / 5.0, -3.0 / 5.0, 1.0 / 3.0])
    M_geom = float(np.dot(c[1:], E[1:]))
    M_unrestricted = float(np.dot(c[1:], e[1:]))

    row: dict[str, float | int] = {
        "n": n,
        "X_exclusive": (n + 1) ** 2,
        "odd_prime_count": prime_count,
        "grid": grid,
        "E1_geom": E[1],
        "E2_geom": E[2],
        "E3_geom": E[3],
        "E4_geom": E[4],
        "E5_geom": E[5],
        "e1_unrestricted": e[1],
        "e2_unrestricted": e[2],
        "e3_unrestricted": e[3],
        "e4_unrestricted": e[4],
        "e5_unrestricted": e[5],
        "M53_geom": M_geom,
        "M53_unrestricted": M_unrestricted,
        "geom_margin_1_minus_M": 1.0 - M_geom,
        "unrestricted_margin_1_minus_M": 1.0 - M_unrestricted,
        "removed_high_product_mass": M_unrestricted - M_geom,
        "sieve_seconds": sieve_seconds,
        "fft_seconds": fft_seconds,
        "total_seconds": time.perf_counter() - t0,
    }
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--n",
        default="1000000,10000000,100000000,1000000000",
        help="comma-separated shell indices",
    )
    ap.add_argument(
        "--grid",
        type=int,
        default=131072,
        help="number of logarithmic bins (default 131072)",
    )
    ap.add_argument(
        "--segment-odds",
        type=int,
        default=2_000_000,
        help="odd integers per sieve segment",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("geometric_k5_scalable_results.csv"),
    )
    args = ap.parse_args()

    ns = [int(s.strip()) for s in args.n.split(",") if s.strip()]
    rows = []

    for n in ns:
        print(f"\nComputing n={n:,} ...", flush=True)
        row = analyze_n(n, args.grid, args.segment_odds)
        rows.append(row)
        print(
            f"  M53_geom={row['M53_geom']:.12f}  "
            f"margin={row['geom_margin_1_minus_M']:.12f}  "
            f"M53_unrestricted={row['M53_unrestricted']:.12f}  "
            f"time={row['total_seconds']:.2f}s"
        )

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
