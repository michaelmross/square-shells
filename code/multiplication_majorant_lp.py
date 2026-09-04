#!/usr/bin/env python3
"""
multiplication_majorant_lp.py

Find a low-order pointwise majorant for multiplication coverage.

Input CSV(s) must contain columns:
    n, S1, S2, ..., SK

We seek coefficients c_1,...,c_K minimizing the worst normalized bound

    t = max_n (c_1 S1(n) + ... + c_K SK(n))/n

subject to the pointwise constraints

    c_1 C(w,1) + ... + c_K C(w,K) >= 1

for integer multiplicities 1 <= w <= W.

After solving numerically, coefficients are rationalized and the resulting
polynomial is factored.  If the factorization proves the inequality for all
integer w >= 1, the finite-W LP has discovered a genuine universal majorant.

Example:
    python multiplication_majorant_lp.py overlap_summary.csv --order 3 --max-w 100
"""

from __future__ import annotations

import argparse
from fractions import Fraction
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

try:
    import sympy as sp
except ImportError:
    sp = None


def solve_lp(df: pd.DataFrame, K: int, W: int):
    scol = [f"S{j}" for j in range(1, K + 1)]
    missing = [c for c in ["n", *scol] if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    # Variables: c_1,...,c_K,t. Objective: minimize t.
    A_ub = []
    b_ub = []

    # Pointwise majorant constraints:
    #   sum_j c_j C(w,j) >= 1
    # -> -sum_j c_j C(w,j) <= -1.
    for w in range(1, W + 1):
        row = [-(comb(w, j) if j <= w else 0) for j in range(1, K + 1)]
        A_ub.append(row + [0.0])
        b_ub.append(-1.0)

    # Dataset minimax constraints:
    #   sum_j c_j S_j(n) <= t n.
    S = df[scol].to_numpy(dtype=float)
    ns = df["n"].to_numpy(dtype=float)
    for srow, n in zip(S, ns):
        A_ub.append(list(srow) + [-float(n)])
        b_ub.append(0.0)

    result = linprog(
        c=[0.0] * K + [1.0],
        A_ub=np.asarray(A_ub, dtype=float),
        b_ub=np.asarray(b_ub, dtype=float),
        bounds=[(None, None)] * K + [(None, None)],
        method="highs",
    )
    if not result.success:
        raise RuntimeError(result.message)

    coeffs = result.x[:K]
    t = result.x[K]
    bounds = S @ coeffs
    return coeffs, t, bounds


def rationalize(coeffs, max_denominator: int):
    return [Fraction(float(x)).limit_denominator(max_denominator) for x in coeffs]


def point_values(coeffs, W: int):
    vals = []
    for w in range(1, W + 1):
        vals.append(
            sum(float(coeffs[j - 1]) * comb(w, j) for j in range(1, len(coeffs) + 1))
        )
    return vals


def factor_polynomial(fracs):
    if sp is None:
        return None
    w = sp.symbols("w", integer=True)
    P = 0
    for j, c in enumerate(fracs, start=1):
        P += sp.Rational(c.numerator, c.denominator) * sp.binomial(w, j)
    return sp.factor(sp.expand_func(P - 1))


def bonferroni_coeffs(K: int):
    return np.asarray([1.0 if j % 2 else -1.0 for j in range(1, K + 1)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+", type=Path,
                    help="one or more overlap-summary CSV files")
    ap.add_argument("--order", "-K", type=int, default=5)
    ap.add_argument("--max-w", type=int, default=100,
                    help="largest multiplicity imposed in the numerical LP")
    ap.add_argument("--max-denominator", type=int, default=100000)
    args = ap.parse_args()

    if args.order < 1:
        raise SystemExit("--order must be positive")

    frames = [pd.read_csv(p) for p in args.csv]
    df = pd.concat(frames, ignore_index=True)

    K = args.order
    coeffs, t, lp_bounds = solve_lp(df, K, args.max_w)
    fracs = rationalize(coeffs, args.max_denominator)

    ns = df["n"].to_numpy(dtype=float)
    ratios = lp_bounds / ns
    worst_i = int(np.argmax(ratios))

    print(f"rows: {len(df)}")
    print(f"K={K}, pointwise constraints w=1..{args.max_w}")
    print(f"optimal worst ratio t = {t:.12g}")
    print(f"worst row: n={int(df.iloc[worst_i]['n'])}")
    print()
    print("coefficients:")
    for j, (x, q) in enumerate(zip(coeffs, fracs), start=1):
        print(f"  c{j} = {x:.12g}   ~   {q}")

    fact = factor_polynomial(fracs)
    if fact is not None:
        print()
        print("factorization of P(w)-1 after rationalization:")
        print(f"  {fact}")

    # Re-evaluate using rationalized coefficients.
    rcoeffs = np.asarray([float(q) for q in fracs])
    S = df[[f"S{j}" for j in range(1, K + 1)]].to_numpy(dtype=float)
    rbounds = S @ rcoeffs
    cert = rbounds < ns
    rratios = rbounds / ns
    ri = int(np.argmax(rratios))

    print()
    print("rationalized-coefficient check:")
    print(f"  certified rows: {int(cert.sum())}/{len(df)}")
    print(f"  worst ratio: {rratios[ri]:.12g} at n={int(df.iloc[ri]['n'])}")
    print(f"  margin there: {ns[ri] - rbounds[ri]:.12g}")

    pv = point_values(rcoeffs, min(args.max_w, 20))
    print()
    print("P(w) for first multiplicities:")
    for w, v in enumerate(pv, start=1):
        print(f"  w={w:2d}: {v:.12g}")

    # Compare with ordinary odd-order Bonferroni when K is odd.
    if K % 2 == 1:
        bc = bonferroni_coeffs(K)
        bbounds = S @ bc
        bcert = bbounds < ns
        bratios = bbounds / ns
        bi = int(np.argmax(bratios))
        print()
        print("ordinary Bonferroni comparison:")
        print(f"  certified rows: {int(bcert.sum())}/{len(df)}")
        print(f"  worst ratio: {bratios[bi]:.12g} at n={int(df.iloc[bi]['n'])}")


if __name__ == "__main__":
    main()
