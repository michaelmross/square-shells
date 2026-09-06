#!/usr/bin/env python3
"""
geometric_k7_precision_crossing.py

Locate the numerical crossing of the ordinary seventh-order geometric
Bonferroni main term

    M^geom_{7,2}(n) = E1 - E2 + E3 - E4 + E5 - E6 + E7,

where

    E_j = sum_{3 <= p1 < ... < pj <= n, p1...pj < (n+1)^2}
              1/(p1...pj).

This is the K=7, r=2 member of the shifted-majorant hierarchy.

Method
------
* primes are exact up to each requested cutoff C;
* above C, prime measure is replaced by dt/log t;
* the seven weighted prime measures are binned on a fine normalized
  logarithmic grid;
* lower grids are obtained exactly by coarsening the fine histogram;
* measure-valued Newton identities are evaluated by FFT;
* M(G) is extrapolated with

      M(G) = M_inf + a/G + b/G^2

  when three grids are supplied;
* the crossing M_inf(n)=1 is refined by secant/bracketing.

Recommended run
---------------
python geometric_k7_precision_crossing.py \
  --bracket 3.9e13,4.1e13 \
  --cutoffs 1e8,2e8,4e8 \
  --grids 131072,262144,524288 \
  --out k7_precision_crossing.csv

Interpretation
--------------
This is a hybrid numerical model, not a rigorous evaluation, because the
prime tail above the exact cutoff is represented by dt/log t.  The cutoff
convergence is printed explicitly.

Requires: numpy, scipy.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from decimal import Decimal
from pathlib import Path

import numpy as np
from scipy.special import expi

K = 7


def parse_int(s: str) -> int:
    s = s.strip()
    return int(Decimal(s))


def parse_int_list(s: str) -> list[int]:
    return [parse_int(x) for x in s.split(",") if x.strip()]


def simple_primes(limit: int) -> np.ndarray:
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
    """Return odd primes <= limit."""
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

    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)


def pnt_bin_integral_from_logs(la: np.ndarray, lb: np.ndarray, r: int) -> np.ndarray:
    """
    Integral_a^b t^{-r}/log(t) dt, supplied through log(a), log(b).
    """
    if r == 1:
        return np.log(lb / la)

    c = float(r - 1)
    return expi(-c * lb) - expi(-c * la)


def build_fine_hist(
    n: int,
    primes_all: np.ndarray,
    cutoff: int,
    fine_grid: int,
) -> np.ndarray:
    """
    Build the seven prime measures on the finest grid.

    Exact prime contribution: p <= cutoff.
    Hybrid PNT contribution:  cutoff < p <= n.
    """
    logn = math.log(n)
    logX = 2.0 * math.log(n + 1)

    stop = np.searchsorted(primes_all, min(cutoff, n), side="right")
    pf = primes_all[:stop].astype(np.float64)

    hist = np.zeros((K, fine_grid), dtype=np.float64)

    if pf.size:
        inv = 1.0 / pf
        log_ratio = np.log(pf) / logX
        invpow = np.ones_like(inv)

        for r in range(1, K + 1):
            invpow *= inv
            pos = r * log_ratio
            mask = pos < 1.0
            if np.any(mask):
                idx = np.floor(pos[mask] * fine_grid).astype(np.int64)
                idx = np.minimum(idx, fine_grid - 1)
                hist[r - 1] += np.bincount(
                    idx, weights=invpow[mask], minlength=fine_grid
                )

    if n <= cutoff:
        return hist

    j = np.arange(fine_grid, dtype=np.float64)
    y0 = j / fine_grid
    y1 = (j + 1.0) / fine_grid

    for r in range(1, K + 1):
        # Bin endpoint p = exp(logX * y/r), capped at n before exponentiating.
        z0 = np.minimum(logX * y0 / r, logn)
        z1 = np.minimum(logX * y1 / r, logn)

        lo = np.exp(z0)
        hi = np.exp(z1)

        lo = np.maximum(lo, float(cutoff))
        hi = np.minimum(hi, float(n))

        valid = hi > lo
        if not np.any(valid):
            continue

        vals = np.zeros(fine_grid, dtype=np.float64)
        la = np.log(lo[valid])
        lb = np.log(hi[valid])
        vals[valid] = pnt_bin_integral_from_logs(la, lb, r)
        hist[r - 1] += vals

    return hist


def coarsen_hist(hist_fine: np.ndarray, grid: int) -> np.ndarray:
    fine_grid = hist_fine.shape[1]
    if fine_grid % grid != 0:
        raise ValueError(f"fine grid {fine_grid} is not divisible by {grid}")

    factor = fine_grid // grid
    if factor == 1:
        return hist_fine
    return hist_fine.reshape(K, grid, factor).sum(axis=2)


def geometric_moments(hist: np.ndarray) -> np.ndarray:
    """
    Compute E_1,...,E_7 from the binned prime measures using the
    measure-valued Newton recurrence in Fourier space:

        k nu_k = sum_{i=1}^k (-1)^(i-1) nu_{k-i} * mu_i,

    with nu_0 = delta_0, and E_k = mass(nu_k on [0,1)).
    """
    grid = hist.shape[1]

    fft_len = 1
    while fft_len < (K + 1) * grid:
        fft_len <<= 1

    F: list[np.ndarray | None] = [None]
    for r in range(K):
        padded = np.zeros(fft_len, dtype=np.float64)
        padded[:grid] = hist[r]
        F.append(np.fft.rfft(padded))

    # Fourier transform of delta_0 is identically 1.
    N: list[np.ndarray | None] = [np.ones_like(F[1], dtype=np.complex128)]
    E = np.zeros(K + 1, dtype=np.float64)
    E[0] = 1.0

    for k in range(1, K + 1):
        z = np.zeros_like(F[1], dtype=np.complex128)
        for i in range(1, k + 1):
            z += ((-1) ** (i - 1)) * N[k - i] * F[i]
        Nk = z / k
        N.append(Nk)

        arr = np.fft.irfft(Nk, n=fft_len)
        E[k] = float(arr[:grid].sum())

    return E


def M7_from_hist(hist: np.ndarray) -> tuple[float, np.ndarray]:
    E = geometric_moments(hist)
    M = sum((1.0 if j % 2 else -1.0) * E[j] for j in range(1, K + 1))
    return float(M), E


def extrapolate(grids: list[int], values: list[float]) -> tuple[float, dict[str, float]]:
    h = 1.0 / np.asarray(grids, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)

    if len(grids) == 2:
        A = np.column_stack([np.ones(2), h])
        c = np.linalg.solve(A, y)
        return float(c[0]), {
            "fit_a": float(c[1]),
            "fit_b": float("nan"),
            "richardson_12": float("nan"),
            "richardson_23": float("nan"),
        }

    if len(grids) == 3:
        A = np.column_stack([np.ones(3), h, h * h])
        c = np.linalg.solve(A, y)
        d = {
            "fit_a": float(c[1]),
            "fit_b": float(c[2]),
            "richardson_12": float("nan"),
            "richardson_23": float("nan"),
        }
        if grids[1] == 2 * grids[0] and grids[2] == 2 * grids[1]:
            d["richardson_12"] = 2.0 * values[1] - values[0]
            d["richardson_23"] = 2.0 * values[2] - values[1]
        return float(c[0]), d

    raise ValueError("Use exactly two or three grids")


class Evaluator:
    def __init__(
        self,
        primes_all: np.ndarray,
        cutoff: int,
        grids: list[int],
    ):
        self.primes_all = primes_all
        self.cutoff = cutoff
        self.grids = grids
        self.fine_grid = max(grids)
        self.cache: dict[int, dict] = {}

    def evaluate(self, n: int) -> dict:
        if n in self.cache:
            return self.cache[n]

        t0 = time.perf_counter()
        fine = build_fine_hist(n, self.primes_all, self.cutoff, self.fine_grid)

        Ms: list[float] = []
        moments: list[np.ndarray] = []

        for G in self.grids:
            h = coarsen_hist(fine, G)
            M, E = M7_from_hist(h)
            Ms.append(M)
            moments.append(E)

        Minf, diag = extrapolate(self.grids, Ms)

        row = {
            "n": n,
            "cutoff": self.cutoff,
            "M_extrapolated": Minf,
            "margin_1_minus_M": 1.0 - Minf,
            "fit_a_over_G": diag["fit_a"],
            "fit_b_over_G2": diag["fit_b"],
            "richardson_12": diag["richardson_12"],
            "richardson_23": diag["richardson_23"],
            "seconds": time.perf_counter() - t0,
        }

        for i, (G, M) in enumerate(zip(self.grids, Ms), start=1):
            row[f"grid_{i}"] = G
            row[f"M_grid_{i}"] = M

        # Also retain extrapolated moments by fitting each E_j independently.
        for j in range(1, K + 1):
            vals = [float(E[j]) for E in moments]
            Einf, _ = extrapolate(self.grids, vals)
            row[f"E{j}_extrapolated"] = Einf

        self.cache[n] = row
        return row


def secant_n(n1: int, f1: float, n2: int, f2: float) -> int:
    if f2 == f1:
        return (n1 + n2) // 2
    x = n1 - f1 * (n2 - n1) / (f2 - f1)
    return int(round(x))


def refine_crossing(
    evaluator: Evaluator,
    lo: int,
    hi: int,
    iterations: int,
) -> tuple[float, list[dict]]:
    rows: list[dict] = []

    rlo = evaluator.evaluate(lo)
    rhi = evaluator.evaluate(hi)
    flo = rlo["M_extrapolated"] - 1.0
    fhi = rhi["M_extrapolated"] - 1.0

    if flo * fhi > 0:
        raise RuntimeError(
            f"Bracket does not straddle M=1 at cutoff={evaluator.cutoff}: "
            f"M({lo})={rlo['M_extrapolated']:.12f}, "
            f"M({hi})={rhi['M_extrapolated']:.12f}"
        )

    for it in range(iterations):
        rows.extend([dict(rlo, iteration=it, role="lo"),
                     dict(rhi, iteration=it, role="hi")])

        mid = secant_n(lo, flo, hi, fhi)
        mid = max(lo + 1, min(hi - 1, mid))

        rm = evaluator.evaluate(mid)
        fm = rm["M_extrapolated"] - 1.0
        rows.append(dict(rm, iteration=it, role="mid"))

        print(
            f"  cutoff={evaluator.cutoff:,}  "
            f"n={mid:,}  M={rm['M_extrapolated']:.12f}  "
            f"1-M={1-rm['M_extrapolated']:+.3e}",
            flush=True,
        )

        if abs(fm) < 1e-10:
            return float(mid), rows

        if flo * fm <= 0:
            hi, rhi, fhi = mid, rm, fm
        else:
            lo, rlo, flo = mid, rm, fm

        if hi - lo <= 1000:
            break

    root = lo - flo * (hi - lo) / (fhi - flo)
    return float(root), rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return

    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bracket",
        default="3.9e13,4.1e13",
        help="crossing bracket lo,hi",
    )
    ap.add_argument(
        "--cutoffs",
        default="1e8,2e8,4e8",
        help="two or more exact-prime cutoffs",
    )
    ap.add_argument(
        "--grids",
        default="131072,262144,524288",
        help="two or three nested grids",
    )
    ap.add_argument("--iterations", type=int, default=4)
    ap.add_argument("--segment-odds", type=int, default=2_000_000)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("k7_precision_crossing.csv"),
    )
    args = ap.parse_args()

    bracket = sorted(parse_int_list(args.bracket))
    cutoffs = sorted(set(parse_int_list(args.cutoffs)))
    grids = sorted(parse_int_list(args.grids))

    if len(bracket) != 2:
        raise SystemExit("--bracket requires exactly two values")
    if len(cutoffs) < 2:
        raise SystemExit("--cutoffs requires at least two values")
    if len(grids) not in (2, 3):
        raise SystemExit("--grids requires two or three values")
    if max(grids) % min(grids) != 0:
        raise SystemExit("grids must be nested divisors of the largest grid")

    lo, hi = bracket
    max_cutoff = max(cutoffs)

    print(f"Sieve exact odd primes through {max_cutoff:,} ...", flush=True)
    t0 = time.perf_counter()
    primes_all = segmented_primes(max_cutoff, args.segment_odds)
    print(
        f"Stored {len(primes_all):,} odd primes in "
        f"{time.perf_counter()-t0:.2f}s",
        flush=True,
    )

    all_rows: list[dict] = []
    roots: list[tuple[int, float]] = []

    for cutoff in cutoffs:
        print(f"\nRefining cutoff {cutoff:,} ...", flush=True)
        ev = Evaluator(primes_all, cutoff, grids)
        root, rows = refine_crossing(ev, lo, hi, args.iterations)
        roots.append((cutoff, root))
        all_rows.extend(rows)
        print(f"  crossing n ~= {root:,.0f}", flush=True)

    write_csv(args.out, all_rows)

    print("\nSummary")
    print("-------")
    for cutoff, root in roots:
        print(f"cutoff={cutoff:,}: n* ~= {root:,.0f}")

    if len(roots) >= 2:
        print("\nCutoff shifts:")
        for (c1, r1), (c2, r2) in zip(roots, roots[1:]):
            print(f"  {c1:,} -> {c2:,}: {r2-r1:+,.0f}")

    print(
        f"\nBest current hybrid estimate (largest cutoff): "
        f"n* ~= {roots[-1][1]:,.0f}"
    )
    print(
        "This is a numerical hybrid-model crossing, not a rigorous error bound."
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
