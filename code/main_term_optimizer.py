#!/usr/bin/env python3
"""
main_term_optimizer.py

Exact finite-n optimizer for the shifted multiplication-overlap main term.

For odd K and integer r >= 2,
    P_{K,r}(w) = 1 + (w-1) prod_{j=0}^{K-2}(w-r-j)
                       / prod_{j=0}^{K-2}(r+j).

Write P in the binomial basis
    P(w) = sum_{j=1}^K c_j C(w,j).

The density/main term is
    M_{K,r}(n) = sum_{j=1}^K c_j e_j,
where
    e_j = sum_{p1<...<pj<=n, odd primes} 1/(p1...pj).

This script computes the e_j exactly to floating precision by dynamic
elementary-symmetric updates, then searches K,r.

Examples:
    python main_term_optimizer.py --n 1000000
    python main_term_optimizer.py --n 1000000 --kmax 15 --rmax 8
"""

from __future__ import annotations

import argparse
from fractions import Fraction
from math import prod


def odd_primes_upto(n: int) -> list[int]:
    if n < 3:
        return []
    sieve = bytearray(b"\x01") * (n + 1)
    sieve[0:2] = b"\x00\x00"
    limit = int(n**0.5)
    for p in range(2, limit + 1):
        if sieve[p]:
            start = p * p
            count = ((n - start) // p) + 1
            sieve[start : n + 1 : p] = b"\x00" * count
    return [p for p in range(3, n + 1, 2) if sieve[p]]


def binomial_coefficients_P(K: int, r: int) -> list[Fraction]:
    """Return c_j with P(w)=sum_j c_j C(w,j), via forward differences."""
    D = prod(range(r, r + K - 1)) if K > 1 else 1
    values: list[Fraction] = []

    for w in range(K + 1):
        numer = w - 1
        for j in range(K - 1):
            numer *= w - r - j
        P = Fraction(1, 1) + Fraction(numer, D)
        values.append(P)

    coeffs: list[Fraction] = []
    current = values
    for _ in range(K + 1):
        coeffs.append(current[0])
        current = [current[i + 1] - current[i] for i in range(len(current) - 1)]
    return coeffs


def elementary_reciprocals(primes: list[int], kmax: int) -> tuple[float, list[float]]:
    e = [0.0] * (kmax + 1)
    e[0] = 1.0
    T = 0.0
    for p in primes:
        q = 1.0 / p
        T += q
        for j in range(kmax, 0, -1):
            e[j] += q * e[j - 1]
    return T, e


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--kmax", type=int, default=21)
    ap.add_argument("--rmax", type=int, default=8)
    args = ap.parse_args()

    if args.n < 3:
        raise SystemExit("--n must be at least 3")
    if args.kmax < 1:
        raise SystemExit("--kmax must be positive")

    primes = odd_primes_upto(args.n)
    T, e = elementary_reciprocals(primes, args.kmax)

    print(f"n={args.n:,}")
    print(f"odd irreducible rows={len(primes):,}")
    print(f"T=sum(1/p), odd p<=n = {T:.12f}")
    print()

    feasible = []
    all_rows = []

    for K in range(1, args.kmax + 1, 2):
        for r in range(2, args.rmax + 1):
            coeffs = binomial_coefficients_P(K, r)
            M = sum(float(coeffs[j]) * e[j] for j in range(1, K + 1))
            all_rows.append((K, r, M))
            if M < 1.0:
                feasible.append((K, r, M))

    if not feasible:
        print("No M_{K,r}<1 in requested search range.")
    else:
        minK = min(K for K, r, M in feasible)
        candidates = sorted(
            [(M, r) for K, r, M in feasible if K == minK]
        )
        bestM, bestR = candidates[0]
        print(f"smallest successful odd K: {minK}")
        print(f"best r at that K: {bestR}")
        print(f"M = {bestM:.12f}")
        print(f"main-term margin 1-M = {1.0-bestM:.12f}")
        print()

    print("best r for each odd K:")
    for K in range(1, args.kmax + 1, 2):
        rows = [(M, r) for k, r, M in all_rows if k == K]
        M, r = min(rows)
        flag = "YES" if M < 1.0 else "no"
        print(f"  K={K:2d}: r={r:2d}, M={M:.12f}, M<1? {flag}")


if __name__ == "__main__":
    main()
