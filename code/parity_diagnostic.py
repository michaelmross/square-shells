#!/usr/bin/env python3
"""
parity_diagnostic.py

Diagnose where cancellation occurs in the shifted multiplication-overlap
certificate, with special attention to parity-like odd/even cancellation.

Default experiment:
    n = 1,000,000
    K = 5, r = 3

The script:
  1. constructs odd irreducible multiplication rows p <= n;
  2. computes the exact multiplicative depth w(m) at every odd position in
     n^2 < m < (n+1)^2;
  3. finds the shortest odd-window length L for which every L-position window
     has positive K,r certificate;
  4. selects:
       - worst passing window of length L,
       - worst failing window of length L-1,
       - median passing window of length L;
  5. decomposes R exactly by intersection order j;
  6. decomposes R exactly by the largest multiplication row p, aggregated into
     quotient bands q=floor(n/p).

The largest-row decomposition is exact because every j-fold subset has a
unique largest prime p. Its expected reciprocal mass at p is
    (1/p) * e_{j-1}(primes < p).
"""

from __future__ import annotations

import argparse
import csv
from fractions import Fraction
from math import comb, gcd
from pathlib import Path

import numpy as np


def lcm(a: int, b: int) -> int:
    return a // gcd(a, b) * b


def odd_primes_upto(n: int) -> np.ndarray:
    sieve = np.ones(n + 1, dtype=np.bool_)
    sieve[:2] = False
    lim = int(n**0.5)
    for p in range(2, lim + 1):
        if sieve[p]:
            sieve[p * p : n + 1 : p] = False
    return np.nonzero(sieve)[0][1:]  # omit 2


def binomial_coefficients_P(K: int, r: int) -> list[Fraction]:
    """
    P_{K,r}(w)=1+(w-1)prod_{j=0}^{K-2}(w-r-j)/prod_{j=0}^{K-2}(r+j)
    expanded as sum c_j C(w,j), j=0..K.
    """
    D = 1
    for j in range(K - 1):
        D *= r + j

    vals = []
    for w in range(K + 1):
        z = w - 1
        for j in range(K - 1):
            z *= w - r - j
        vals.append(Fraction(1, 1) + Fraction(z, D))

    coeffs = []
    cur = vals
    for _ in range(K + 1):
        coeffs.append(cur[0])
        cur = [cur[i + 1] - cur[i] for i in range(len(cur) - 1)]
    return coeffs


def quotient_group(q: int) -> str:
    if q <= 20:
        return str(q)
    cuts = [50, 100, 200, 500, 1000, 2000, 5000, 10000,
            20000, 50000, 100000, 200000, 500000]
    lo = 21
    for hi in cuts:
        if q <= hi:
            return f"{lo}-{hi}"
        lo = hi + 1
    return f"{lo}+"


def group_sort_key(g: str) -> int:
    if "-" in g:
        return int(g.split("-")[0])
    if g.endswith("+"):
        return int(g[:-1])
    return int(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--r", type=int, default=3)
    ap.add_argument("--search-max", type=int, default=2000)
    ap.add_argument("--summary-out", type=Path,
                    default=Path("parity_diagnostic_summary.csv"))
    ap.add_argument("--bands-out", type=Path,
                    default=Path("parity_diagnostic_bands.csv"))
    args = ap.parse_args()

    n, K, r = args.n, args.K, args.r
    if K % 2 == 0:
        raise SystemExit("K should be odd for this majorant family.")

    coeff = binomial_coefficients_P(K, r)
    # P has no constant term.
    assert coeff[0] == 0

    scale = 1
    for c in coeff[1:]:
        scale = lcm(scale, c.denominator)

    cfloat = np.array([float(c) for c in coeff], dtype=float)

    primes = odd_primes_upto(n)
    lo_odd = n * n + (1 if n % 2 == 0 else 2)

    # Exact depth array over the n odd positions.
    depth = np.zeros(n, dtype=np.int16)
    residues = np.zeros(len(primes), dtype=np.int64)
    for i, p0 in enumerate(primes):
        p = int(p0)
        inv2 = (p + 1) // 2
        residue = (-(lo_odd % p) * inv2) % p
        residues[i] = residue
        depth[residue::p] += 1

    maxw = int(depth.max())

    # Integer-scaled point score scale*(1-P(w)).
    point_score = np.zeros(maxw + 1, dtype=np.int64)
    for w in range(maxw + 1):
        Pw = sum(coeff[j] * comb(w, j)
                 for j in range(1, K + 1) if w >= j)
        point_score[w] = int((Fraction(1, 1) - Pw) * scale)

    scores = point_score[depth]
    prefix = np.concatenate(([0], np.cumsum(scores, dtype=np.int64)))

    # Find first length L with positive score in every L-position window.
    threshold = None
    for L in range(1, min(args.search_max, n) + 1):
        ws = prefix[L:] - prefix[:-L]
        if int(ws.min()) > 0:
            threshold = L
            break
    if threshold is None:
        raise SystemExit("No uniformly certified length found within --search-max.")

    L = threshold
    ws = prefix[L:] - prefix[:-L]
    worst_pass_start = int(ws.argmin())
    median_value = float(np.median(ws))
    median_start = int(np.argmin(np.abs(ws - median_value)))

    if L > 1:
        wsf = prefix[L - 1:] - prefix[:-(L - 1)]
        worst_fail_start = int(wsf.argmin())
    else:
        wsf = None
        worst_fail_start = None

    # Exact elementary reciprocal sums and expected increments assigned by
    # largest prime p.
    e = np.zeros(K + 1, dtype=float)
    e[0] = 1.0
    expected_inc = np.zeros((len(primes), K + 1), dtype=float)

    for i, p0 in enumerate(primes):
        p = float(p0)
        z = 1.0 / p
        for j in range(1, K + 1):
            expected_inc[i, j] = z * e[j - 1]
        for j in range(K, 0, -1):
            e[j] += z * e[j - 1]

    M = float(sum(cfloat[j] * e[j] for j in range(1, K + 1)))

    combtab = np.zeros((maxw + 1, K + 1), dtype=np.int64)
    for w in range(maxw + 1):
        for j in range(min(w, K) + 1):
            combtab[w, j] = comb(w, j)

    def decompose(label: str, start: int, length: int):
        before = np.zeros(length, dtype=np.int16)
        actual_by_p = np.zeros((len(primes), K + 1), dtype=float)

        for i, p0 in enumerate(primes):
            p = int(p0)
            rr = int(residues[i])

            if rr < start:
                first = rr + ((start - rr + p - 1) // p) * p
            else:
                first = rr
            if first >= start + length:
                continue

            loc = np.arange(first, start + length, p, dtype=np.int64) - start
            b = before[loc]

            # Every j-subset whose largest prime is p is obtained by selecting
            # j-1 of the smaller prime factors already present.
            for j in range(1, K + 1):
                actual_by_p[i, j] = combtab[b, j - 1].sum()

            before[loc] += 1

        assert np.array_equal(before, depth[start:start + length])

        S = actual_by_p[:, 1:].sum(axis=0)
        main = length * e[1:]
        R = S - main
        weighted = np.array([cfloat[j] * R[j - 1]
                             for j in range(1, K + 1)])
        Rweighted = float(weighted.sum())
        main_margin = length * (1.0 - M)
        certificate = main_margin - Rweighted

        local_depth = depth[start:start + length]
        hist = np.bincount(local_depth, minlength=maxw + 1)
        exact_primes = int(hist[0])

        odd_R = float(sum(weighted[j - 1] for j in range(1, K + 1, 2)))
        even_R = float(sum(weighted[j - 1] for j in range(2, K + 1, 2)))

        # Largest-prime quotient bands.
        bands = {}
        for i, p0 in enumerate(primes):
            p = int(p0)
            g = quotient_group(n // p)
            rec = bands.setdefault(
                g,
                {
                    "actual": np.zeros(K, dtype=float),
                    "main": np.zeros(K, dtype=float),
                },
            )
            rec["actual"] += actual_by_p[i, 1:]
            rec["main"] += length * expected_inc[i, 1:]

        band_rows = []
        for g, rec in bands.items():
            RR = rec["actual"] - rec["main"]
            WW = np.array([cfloat[j] * RR[j - 1]
                           for j in range(1, K + 1)])
            band_rows.append(
                {
                    "window": label,
                    "q_band": g,
                    "weighted_R_total": float(WW.sum()),
                    **{f"R_j{j}": float(RR[j - 1])
                       for j in range(1, K + 1)},
                    **{f"weighted_R_j{j}": float(WW[j - 1])
                       for j in range(1, K + 1)},
                }
            )

        summary = {
            "window": label,
            "start_index": start,
            "odd_positions": length,
            "start_value": lo_odd + 2 * start,
            "end_value": lo_odd + 2 * (start + length - 1),
            "exact_primes": exact_primes,
            "max_depth": int(local_depth.max()),
            "main_term_M": M,
            "main_margin_N_times_1_minus_M": main_margin,
            "weighted_R": Rweighted,
            "certificate_holes_lower_bound": certificate,
            "odd_j_weighted_R": odd_R,
            "even_j_weighted_R": even_R,
            **{f"S{j}": float(S[j - 1]) for j in range(1, K + 1)},
            **{f"main_j{j}": float(main[j - 1]) for j in range(1, K + 1)},
            **{f"R_j{j}": float(R[j - 1]) for j in range(1, K + 1)},
            **{f"weighted_R_j{j}": float(weighted[j - 1])
               for j in range(1, K + 1)},
        }
        return summary, band_rows

    summaries = []
    all_bands = []

    s, b = decompose("worst_passing", worst_pass_start, L)
    summaries.append(s); all_bands += b

    if L > 1 and worst_fail_start is not None:
        s, b = decompose("worst_failing_L_minus_1", worst_fail_start, L - 1)
        summaries.append(s); all_bands += b

    s, b = decompose("median_passing", median_start, L)
    summaries.append(s); all_bands += b

    # Distribution of R across all threshold-length windows.
    cert_all = ws.astype(float) / scale
    R_all = L * (1.0 - M) - cert_all
    qvals = np.quantile(R_all, [0, .01, .10, .50, .90, .99, .999, 1.0])

    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    # Sort quotient bands numerically inside each window.
    order = {s["window"]: i for i, s in enumerate(summaries)}
    all_bands.sort(key=lambda x: (order[x["window"]], group_sort_key(x["q_band"])))

    with args.bands_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_bands[0].keys()))
        writer.writeheader()
        writer.writerows(all_bands)

    print(f"n={n:,}, K={K}, r={r}")
    print("coefficients:",
          ", ".join(f"c{j}={coeff[j]}" for j in range(1, K + 1)))
    print(f"odd irreducible rows: {len(primes):,}")
    print(f"maximum multiplicative depth: {maxw}")
    print(f"main term M_{{K,r}} = {M:.12f}")
    print(f"main margin density 1-M = {1-M:.12f}")
    print()
    print(f"shortest uniformly certified odd-window length: {L}")
    print(f"corresponding ordinary-integer length: {2*L}")
    print()

    for s in summaries:
        print(s["window"])
        print(
            f"  values {int(s['start_value'])} .. {int(s['end_value'])}, "
            f"odd positions={int(s['odd_positions'])}, "
            f"exact primes={int(s['exact_primes'])}"
        )
        print(
            f"  main margin={s['main_margin_N_times_1_minus_M']:.6f}, "
            f"R={s['weighted_R']:.6f}, "
            f"certificate={s['certificate_holes_lower_bound']:.6f}"
        )
        print(
            f"  odd-j weighted R={s['odd_j_weighted_R']:.6f}, "
            f"even-j weighted R={s['even_j_weighted_R']:.6f}"
        )
        print(
            "  weighted R by j: "
            + ", ".join(
                f"j{j}={s[f'weighted_R_j{j}']:.6f}"
                for j in range(1, K + 1)
            )
        )
        print()

    print("R distribution over all threshold-length windows:")
    for name, val in zip(
        ["min", "1%", "10%", "median", "90%", "99%", "99.9%", "max"], qvals
    ):
        print(f"  {name:>6}: {val:.6f}")
    print()
    print(f"summary CSV: {args.summary_out}")
    print(f"quotient-band CSV: {args.bands_out}")


if __name__ == "__main__":
    main()
