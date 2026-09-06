#!/usr/bin/env python3
"""
Dyadic Liouville parity diagnostic for multiplication coverage in a square shell.

For the first Y odd positions J in

    n^2 < m < (n+1)^2,

the script factors each m by the odd prime rows p <= n, forms the row radical
R_n(m), and evaluates

    T(D) = sum_{D <= d < 2D} mu(d)^2 1_{P+(d)<=n}
           sum_{q: dq in J} lambda(q)

dyadically.  The exact checksum is

    sum_D T(D) = -H(J),

where H(J) is the number of prime (uncovered) odd positions in J.

The default experiment is n=10^6 and Y=floor((n^2)^(4/9)).
No primality test on the shell positions is needed: H(J) is detected by absence
of a row factor p <= n.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def primes_upto(n: int) -> list[int]:
    sieve = bytearray(b"\x01") * (n + 1)
    sieve[:2] = b"\x00\x00"
    for p in range(2, int(n**0.5) + 1):
        if sieve[p]:
            start = p * p
            sieve[start : n + 1 : p] = b"\x00" * (((n - start) // p) + 1)
    return [i for i in range(2, n + 1) if sieve[i]]


def integer_kth_root(a: int, k: int) -> int:
    """Return floor(a^(1/k)) using integer arithmetic."""
    if a < 0 or k < 1:
        raise ValueError("require a >= 0 and k >= 1")
    if a < 2:
        return a
    lo, hi = 1, 1 << ((a.bit_length() + k - 1) // k)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if mid**k <= a:
            lo = mid
        else:
            hi = mid
    return lo


def default_Y(n: int) -> int:
    x = n * n
    # floor(x^(4/9)) = floor((x^4)^(1/9)) exactly.
    return integer_kth_root(x**4, 9)


def run(n: int, Y: int, outcsv: Path) -> dict[str, object]:
    x = n * n
    m0 = x + 1 if x % 2 == 0 else x + 2
    if m0 % 2 == 0:
        m0 += 1

    residual = [m0 + 2 * i for i in range(Y)]
    factors: list[list[int]] = [[] for _ in range(Y)]
    omega_parity = bytearray(Y)  # Omega(m) modulo 2, hence Liouville sign.

    for p in primes_upto(n):
        if p == 2:
            continue
        # Solve m0 + 2*i == 0 mod p.
        inv2 = (p + 1) // 2
        first = ((-m0) % p * inv2) % p
        for i in range(first, Y, p):
            r = residual[i]
            if r % p:
                continue
            factors[i].append(p)
            while r % p == 0:
                r //= p
                omega_parity[i] ^= 1
            residual[i] = r

    # In a square shell, after all p <= n have been removed, the leftover is
    # either 1 or one prime > n: two factors > n would have product >= (n+1)^2.
    for i, r in enumerate(residual):
        if r > 1:
            omega_parity[i] ^= 1

    maxk = x.bit_length()
    bins = [0] * (maxk + 1)
    counts = [0] * (maxk + 1)
    threshold = x / Y
    below_exact = 0
    above_exact = 0
    holes = 0

    for i, ps in enumerate(factors):
        if not ps:
            holes += 1

        lambda_m = -1 if omega_parity[i] else 1
        divisors = [(1, 1)]  # (d, mu(d)) for d | R_n(m).
        for p in ps:
            divisors += [(d * p, -mu_d) for d, mu_d in divisors]

        for d, mu_d in divisors:
            contribution = lambda_m * mu_d  # lambda(m/d) = lambda(m) lambda(d).
            k = d.bit_length() - 1
            bins[k] += contribution
            counts[k] += 1
            if d < threshold:
                below_exact += contribution
            else:
                above_exact += contribution

    total = sum(bins)
    if total != -holes:
        raise AssertionError(f"checksum failed: sum T(D)={total}, -H(J)={-holes}")
    if below_exact + above_exact != total:
        raise AssertionError("threshold split does not reproduce total")

    rows: list[dict[str, int]] = []
    cumulative = 0
    for k, value in enumerate(bins):
        lo = 1 << k
        hi = 1 << (k + 1)
        if lo > x:
            break
        cumulative += value
        rows.append(
            {
                "k": k,
                "D_lo": lo,
                "D_hi": hi,
                "T_D": value,
                "cumulative_T": cumulative,
                "divisor_terms": counts[k],
                "below_x_over_Y": int(hi <= threshold),
            }
        )

    outcsv.parent.mkdir(parents=True, exist_ok=True)
    with outcsv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    return {
        "n": n,
        "x": x,
        "Y_odd_positions": Y,
        "m0": m0,
        "m_last": m0 + 2 * (Y - 1),
        "holes": holes,
        "sum_T": total,
        "x_over_Y": threshold,
        "T_below_x_over_Y": below_exact,
        "T_at_or_above_x_over_Y": above_exact,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--Y", type=int, default=None, help="number of odd positions")
    ap.add_argument("--out", type=Path, default=Path("liouville_parity_bins.csv"))
    args = ap.parse_args()

    Y = args.Y if args.Y is not None else default_Y(args.n)
    info = run(args.n, Y, args.out)
    for key, value in info.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
