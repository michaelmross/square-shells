#!/usr/bin/env python3
"""
q1_cell_check.py -- the q = 1 cofactor cell of a square shell and the
mean-density bookkeeping behind Remark A.7.

Part 1 (default n = 10^6).  For the shell J_n = {odd m : n^2 < m < (n+1)^2},
a segmented factorization over the odd primes p <= n gives, for every m,
its multiplication depth w_n(m), whether it is squarefree, and whether it is
n-smooth (the residual after removing all p <= n is 1; otherwise it is a
single prime > n).  From that the script forms

    H(n) = #{m in J_n : w_n(m) = 0}                      primes in the shell
    M_J  = sum_{m in J_n} mu(m)
    T1   = sum_{m in J_n, squarefree, n-smooth} mu(m)    the q = 1 cell
    W    = sum_{3 <= m' <= n, m' odd} mu(m') pi(J_n/m')

and checks the identity  T1 = M_J + H(n) + W.  (The identity is algebraic:
a non-smooth squarefree m is p*m' with a unique prime p > n and
mu(m) = -mu(m'); the m' = 1 term is H(n).)  The content is the comparison
of T1 with its mean-density value

    2n * rho(x),  rho(t) = -(2/log^2 t) [1 + 2 b1/log t + 6 b2/log^2 t + 24 b3/log^3 t + ...],

which is what T1 equals when M_J, F(n) = H(n) - 2n/L, and every short-interval
prime count pi(J_n/m') are replaced by their expectations.  The same expansion
follows from W's expectation using sum_{m odd} mu(m)/m = 0 and
sum_{m odd} mu(m) log^k m / m = -2 k! b_{k-1}.

Part 2 (--density).  For small n it checks the global asymptotic

    C(t) := sum_{d <= t, d odd, P+(d) <= n} mu(d),   t = n^2,
    C(t) = M_odd(t) + sum_{m <= n, m odd} mu(m) (pi(t/m) - pi(n))          [exact]
         ~ -(2t/log^2 t) [1 + 2(1+b1)/log t + 6(1+b1+b2)/log^2 t + ...],

printing the direct count, the exact identity, the li-smoothed version, and
the asymptotic expansion, so that prime fluctuations and truncation error are
visible separately.

Requires numpy.  Runtime for the default: about a minute; memory ~ 100 MB.
"""

from __future__ import annotations

import argparse
import math

import numpy as np

GAMMA = 0.5772156649015329
LOG2 = math.log(2.0)
B1 = GAMMA + LOG2            # beta_1
B2 = 1.3811370               # beta_2 (Lemma A.1)
B3 = 1.4214196               # beta_3


def primes_upto(N: int) -> np.ndarray:
    s = np.ones(N + 1, dtype=bool)
    s[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return np.nonzero(s)[0]


def li_diff(a: float, b: float, steps: int = 4000) -> float:
    """int_a^b dt/log t, via Simpson's rule in w = log t."""
    wa, wb = math.log(a), math.log(b)
    if wb <= wa:
        return 0.0
    w = np.linspace(wa, wb, 2 * steps + 1)
    f = np.exp(w) / w
    h = (wb - wa) / (2 * steps)
    return float(h / 3 * (f[0] + f[-1] + 4 * f[1:-1:2].sum() + 2 * f[2:-1:2].sum()))


# ----------------------------------------------------------------------
# Part 1: the q = 1 cell of the shell at n
# ----------------------------------------------------------------------
def shell_cell(n: int) -> None:
    x = n * n
    m0 = x + 1                      # first odd position (x is even iff n is)
    if m0 % 2 == 0:
        m0 += 1
    L = 2 * math.log(n + 1)
    pr = [int(p) for p in primes_upto(n) if p >= 3]

    w = np.zeros(n, dtype=np.int16)        # multiplication depth
    sqf = np.ones(n, dtype=bool)           # squarefree flag
    rem = m0 + 2 * np.arange(n, dtype=np.int64)   # residual after removing p <= n

    for p in pr:
        inv2 = (p + 1) // 2
        i0 = ((-m0) % p * inv2) % p         # first index with m == 0 (mod p)
        idx = np.arange(i0, n, p)
        w[idx] += 1
        r = rem[idx] // p
        while True:
            mk = (r % p == 0)
            if not mk.any():
                break
            r[mk] //= p
        rem[idx] = r
        p2 = p * p
        j0 = ((-m0) % p2 * ((p2 + 1) // 2)) % p2
        if j0 < n:
            sqf[j0::p2] = False

    smooth = rem == 1
    H = int((w == 0).sum())
    # mu(m): (-1)^w for smooth squarefree m, -(-1)^w when one prime > n remains
    mu = np.where(sqf, np.where(smooth, (-1.0) ** w, -((-1.0) ** w)), 0.0)
    T1 = int(mu[smooth].sum())
    MJ = int(mu.sum())
    W = T1 - MJ - H                         # = sum_{m'>=3} mu(m') pi(J_n/m')
    # direct recount: each non-smooth squarefree composite m = p*m' contributes mu(m') = (-1)^w
    W_direct = int(((-1.0) ** w[sqf & ~smooth & (w >= 1)]).sum())

    F = H - 2 * n / L
    two_n_over_L = 2 * n / L

    terms = [-4 * n / L ** 2,
             -8 * B1 * n / L ** 3,
             -24 * B2 * n / L ** 4,
             -96 * B3 * n / L ** 5]
    cum = np.cumsum(terms)

    print(f"n = {n},  L = 2 log(n+1) = {L:.6f},  2n/L = {two_n_over_L:.1f}")
    print(f"H(n) = {H}     F(n) = H - 2n/L = {F:+.1f}")
    print(f"M_J  = sum_(m in J) mu(m) = {MJ}")
    print(f"T1   = q = 1 cell = sum_(m in J, sqfree, n-smooth) mu(m) = {T1}")
    print(f"W    = sum_(m'>=3) mu(m') pi(J/m')  = {W}   (direct recount: {W_direct})")
    print(f"identity T1 = M_J + H + W : {T1} = {MJ} + {H} + {W} -> {'OK' if T1 == MJ + H + W else 'FAIL'}")
    print()
    print("mean-density value of the cell, 2n*rho(x), cumulative through k terms:")
    for k, c in enumerate(cum, start=2):
        print(f"   through 1/L^{k}: {c:10.1f}")
    print(f"T1 - (M_J + F)  = {T1 - MJ - F:10.1f}   <- what the cell is once M_J and the")
    print(f"                                   prime-count deviation F are removed")
    print(f"difference from 4-term mean density: {T1 - MJ - F - cum[-1]:+.1f}")
    print(f"expected size of that difference (fluctuation of the weighted prime counts,")
    print(f"   sqrt( (2n/L) * (1/2) log n ) ):  ~{math.sqrt(two_n_over_L * 0.5 * math.log(n)):.0f}")
    print(f"W expected = -2n/L + (mean density) = {-two_n_over_L + cum[-1]:.1f}   actual W = {W}")


# ----------------------------------------------------------------------
# Part 2: global density of odd n-smooth Moebius values near t = n^2
# ----------------------------------------------------------------------
def density_check(n: int) -> None:
    t = n * n
    pr_all = primes_upto(t)
    mu = np.ones(t + 1, dtype=np.int8)
    for p in pr_all:
        mu[p::p] *= -1
        q = p * p
        if q <= t:
            mu[q::q] = 0
    d = np.arange(t + 1, dtype=np.int64)
    rem = d.copy()
    for p in pr_all[pr_all <= n]:
        pk = int(p)
        while pk <= t:
            rem[pk::pk] //= int(p)
            pk *= int(p)
    odd = d % 2 == 1
    smooth = rem == 1
    C_direct = int(mu[odd & smooth & (mu != 0)].sum())

    # exact identity: C(t) = M_odd(t) + sum_{m<=n odd} mu(m) (pi(t/m) - pi(n))
    is_p = np.zeros(t + 1, dtype=bool)
    is_p[pr_all] = True
    pi = np.cumsum(is_p)
    M_odd_t = int(mu[odd & (mu != 0)].sum())
    m_vals = np.arange(1, n, 2)                     # odd m < n
    mum = mu[m_vals].astype(np.int64)
    pi_n = int(pi[n])
    C_identity = M_odd_t + int((mum * (pi[t // m_vals] - pi_n)).sum())
    # li-smoothed version (prime counts replaced by li), M_odd(t) dropped
    C_li = float(sum(int(mm) * li_diff(n, t / m) for m, mm in zip(m_vals, mum) if mm != 0))
    lt = math.log(t)
    C_asym2 = -2 * t / lt ** 2 * (1 + 2 * (1 + B1) / lt)
    C_asym3 = -2 * t / lt ** 2 * (1 + 2 * (1 + B1) / lt + 6 * (1 + B1 + B2) / lt ** 2)

    print(f"n = {n}, t = n^2 = {t}, log t = {lt:.4f}")
    print(f"  direct count  C(t) = {C_direct}")
    print(f"  exact identity     = {C_identity}   ({'OK' if C_identity == C_direct else 'FAIL'})")
    print(f"  li-smoothed        = {C_li:.0f}      (prime fluctuations removed)")
    print(f"  asymptotic, 2 terms= {C_asym2:.0f}")
    print(f"  asymptotic, 3 terms= {C_asym3:.0f}      (leading -2t/log^2 t = {-2 * t / lt ** 2:.0f})")
    print("  note: the expansion has coefficients ~ k! (b_0+...+b_{k-1}); at log t of 14-16 only")
    print("  the sign, the leading -2/log^2 t, and the first correction are meaningfully tested.")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=10 ** 6, help="shell index for Part 1")
    ap.add_argument("--density", nargs="*", type=int,
                    help="run Part 2 for these n (e.g. --density 1000 2000 3000)")
    args = ap.parse_args()
    if args.density is not None:
        for n in (args.density or [1000, 2000, 3000]):
            density_check(n)
    else:
        shell_cell(args.n)


if __name__ == "__main__":
    main()
