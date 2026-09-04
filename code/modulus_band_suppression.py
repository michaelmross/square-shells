#!/usr/bin/env python3
"""
modulus_band_suppression.py

Locate which modulus/product scales account for the suppression of the
dangerous remainder R in genuine square shells versus phase-scrambled controls.

Default:
    n=1,000,000, K=5, r=3, L=391 odd positions.

For every phase realization:
  1. Find the window of L odd positions with maximal weighted remainder R.
  2. In that worst window, enumerate every j-fold row intersection, j<=K.
     Each intersection has modulus d = product of its distinct row primes.
  3. Decompose the exact weighted remainder by:
       - d <= n, further grouped by q=floor(n/d);
       - d > n (one overflow band).

The MAIN term for every d<=n band is exact:
we enumerate all odd squarefree d<=n, classify omega(d), and add 1/d.
The d>n main term is exact by subtraction from the full elementary symmetric
sum e_j.

Models:
  genuine
  uniform
  independent_square

Example:
  python modulus_band_suppression.py --trials 100 --workers 2

Outputs:
  modulus_band_trials.csv
  modulus_band_details.csv
  modulus_band_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from math import comb
from pathlib import Path

import numpy as np

G = {}


def primes_upto(n: int) -> np.ndarray:
    s = np.ones(n + 1, dtype=np.bool_)
    s[:2] = False
    for p in range(2, int(n**0.5) + 1):
        if s[p]:
            s[p*p:n+1:p] = False
    a = np.nonzero(s)[0]
    return a[a >= 3]


def lcm(a, b):
    return a // math.gcd(a, b) * b


def coeffs_P(K, r):
    D = 1
    for j in range(K - 1):
        D *= r + j
    vals = []
    for w in range(K + 1):
        z = w - 1
        for j in range(K - 1):
            z *= w - r - j
        vals.append(Fraction(1, 1) + Fraction(z, D))
    out = []
    cur = vals
    for _ in range(K + 1):
        out.append(cur[0])
        cur = [cur[i+1] - cur[i] for i in range(len(cur)-1)]
    return out


def elementary(primes, K):
    e = np.zeros(K + 1, float)
    e[0] = 1.0
    for p0 in primes:
        z = 1.0 / float(p0)
        for j in range(K, 0, -1):
            e[j] += z * e[j-1]
    return e


def genuine_residues(n, primes):
    delta = 1 if n % 2 == 0 else 2
    lo = n*n + delta
    out = np.empty(len(primes), dtype=np.int64)
    for i, p0 in enumerate(primes):
        p = int(p0)
        out[i] = (-(lo % p) * ((p + 1)//2)) % p
    return out


def random_residues(mode, n, primes, rng):
    if mode == "uniform":
        return np.array([rng.integers(0, int(p)) for p in primes], dtype=np.int64)
    if mode == "independent_square":
        delta = 1 if n % 2 == 0 else 2
        out = np.empty(len(primes), dtype=np.int64)
        for i, p0 in enumerate(primes):
            p = int(p0)
            u = int(rng.integers(0, p))
            out[i] = (-((u*u + delta) % p) * ((p + 1)//2)) % p
        return out
    raise ValueError(mode)


def quotient_band_from_d(n: int, d: int) -> str:
    if d > n:
        return "d>n"
    q = n // d
    if q >= 100000:
        return "q>=100000"
    if q >= 10000:
        return "q=10000..99999"
    if q >= 1000:
        return "q=1000..9999"
    if q >= 100:
        return "q=100..999"
    if q >= 20:
        return "q=20..99"
    if q >= 10:
        return "q=10..19"
    if q >= 5:
        return "q=5..9"
    if q >= 2:
        return "q=2..4"
    return "q=1"


BAND_ORDER = [
    "q>=100000",
    "q=10000..99999",
    "q=1000..9999",
    "q=100..999",
    "q=20..99",
    "q=10..19",
    "q=5..9",
    "q=2..4",
    "q=1",
    "d>n",
]


def squarefree_main_bands(n: int, K: int, primes: np.ndarray, e: np.ndarray):
    """
    Exact sum_{d in band, omega(d)=j} 1/d for odd squarefree d<=n.
    """
    omega = np.zeros(n + 1, dtype=np.uint8)
    sqfree = np.ones(n + 1, dtype=np.bool_)
    sqfree[0] = False

    for p0 in primes:
        p = int(p0)
        omega[p:n+1:p] += 1
        pp = p * p
        if pp <= n:
            sqfree[pp:n+1:pp] = False

    bands = {b: np.zeros(K + 1, float) for b in BAND_ORDER}

    for d in range(3, n + 1, 2):
        if not sqfree[d]:
            continue
        j = int(omega[d])
        if 1 <= j <= K:
            bands[quotient_band_from_d(n, d)][j] += 1.0 / d

    for j in range(1, K + 1):
        used = sum(bands[b][j] for b in BAND_ORDER if b != "d>n")
        bands["d>n"][j] = e[j] - used

    return bands


def setup(n, K, r, L):
    primes = primes_upto(n)
    coeff = coeffs_P(K, r)
    scale = 1
    for c in coeff[1:]:
        scale = lcm(scale, c.denominator)

    point = np.zeros(65, dtype=np.int64)
    for w in range(65):
        Pw = sum(coeff[j] * comb(w, j)
                 for j in range(1, K + 1) if w >= j)
        point[w] = int((Fraction(1, 1) - Pw) * scale)

    e = elementary(primes, K)
    cf = np.array([float(c) for c in coeff])
    M = float(sum(cf[j] * e[j] for j in range(1, K + 1)))
    main_margin = L * (1.0 - M)
    main_bands = squarefree_main_bands(n, K, primes, e)

    return primes, coeff, cf, scale, point, e, M, main_margin, main_bands


def depth_from_residues(n, primes, residues):
    depth = np.zeros(n, dtype=np.int16)
    for p0, r0 in zip(primes, residues):
        depth[int(r0)::int(p0)] += 1
    return depth


def worst_window(depth, L, point, scale, main_margin):
    score = point[depth]
    pref = np.concatenate(([0], np.cumsum(score, dtype=np.int64)))
    cert_scaled = pref[L:] - pref[:-L]
    cert = cert_scaled.astype(float) / scale
    R = main_margin - cert
    i = int(np.argmax(R))
    return {
        "worst_start": i,
        "max_R": float(R[i]),
        "min_certificate": float(cert[i]),
        "fail_windows": int(np.count_nonzero(cert_scaled <= 0)),
        "total_windows": len(cert_scaled),
    }


def hit_primes_in_window(L, primes, residues, start):
    hits = [[] for _ in range(L)]
    end = start + L
    for p0, r0 in zip(primes, residues):
        p = int(p0)
        rr = int(r0)
        if rr < start:
            first = rr + ((start - rr + p - 1)//p)*p
        else:
            first = rr
        if first >= end:
            continue
        for t in range(first, end, p):
            hits[t-start].append(p)
    return hits


def actual_band_counts(n, K, hits):
    actual = {b: np.zeros(K + 1, dtype=np.int64) for b in BAND_ORDER}
    for plist in hits:
        w = len(plist)
        for j in range(1, min(K, w) + 1):
            for tup in itertools.combinations(plist, j):
                d = math.prod(tup)
                actual[quotient_band_from_d(n, d)][j] += 1
    return actual


def decompose(n, K, L, cf, main_bands, actual):
    rows = []
    total_R = 0.0
    low_R = 0.0

    for b in BAND_ORDER:
        Rj = np.zeros(K + 1, float)
        Wj = np.zeros(K + 1, float)

        for j in range(1, K + 1):
            main = L * main_bands[b][j]
            Rj[j] = float(actual[b][j]) - main
            Wj[j] = cf[j] * Rj[j]

        wr = float(Wj[1:].sum())
        total_R += wr
        if b != "d>n":
            low_R += wr

        row = {
            "band": b,
            "weighted_R": wr,
            "odd_weighted_R": float(sum(Wj[j] for j in range(1, K+1, 2))),
            "even_weighted_R": float(sum(Wj[j] for j in range(2, K+1, 2))),
        }
        for j in range(1, K + 1):
            row[f"actual_j{j}"] = int(actual[b][j])
            row[f"main_j{j}"] = L * main_bands[b][j]
            row[f"R_j{j}"] = Rj[j]
            row[f"weighted_R_j{j}"] = Wj[j]
        rows.append(row)

    return total_R, low_R, total_R - low_R, rows


def init_worker(n, K, r, L):
    primes, coeff, cf, scale, point, e, M, main_margin, main_bands = setup(n, K, r, L)
    G.update(
        n=n, K=K, r=r, L=L, primes=primes, coeff=coeff, cf=cf,
        scale=scale, point=point, e=e, M=M,
        main_margin=main_margin, main_bands=main_bands
    )


def analyze_realization(model, trial, seed):
    if model == "genuine":
        residues = genuine_residues(G["n"], G["primes"])
    else:
        rng = np.random.default_rng(seed)
        residues = random_residues(model, G["n"], G["primes"], rng)

    depth = depth_from_residues(G["n"], G["primes"], residues)
    ws = worst_window(depth, G["L"], G["point"], G["scale"], G["main_margin"])
    hits = hit_primes_in_window(G["L"], G["primes"], residues, ws["worst_start"])
    actual = actual_band_counts(G["n"], G["K"], hits)
    total_R, low_R, high_R, bands = decompose(
        G["n"], G["K"], G["L"], G["cf"], G["main_bands"], actual
    )

    if abs(total_R - ws["max_R"]) > 1e-7:
        raise AssertionError(f"band mismatch: {total_R} != {ws['max_R']}")

    summary = {
        "model": model,
        "trial": trial,
        "seed": int(seed),
        **ws,
        "R_d_le_n": low_R,
        "R_d_gt_n": high_R,
        "fraction_R_d_le_n": low_R / total_R if total_R else float("nan"),
    }

    detail = []
    for row in bands:
        detail.append({
            "model": model,
            "trial": trial,
            "seed": int(seed),
            "worst_start": ws["worst_start"],
            "max_R": ws["max_R"],
            **row,
        })
    return summary, detail


def worker(payload):
    return analyze_realization(*payload)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--r", type=int, default=3)
    ap.add_argument("--window", type=int, default=391)
    ap.add_argument("--trials", type=int, default=100,
                    help="trials per scrambled model")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260831)
    ap.add_argument("--trials-out", type=Path, default=Path("modulus_band_trials.csv"))
    ap.add_argument("--details-out", type=Path, default=Path("modulus_band_details.csv"))
    ap.add_argument("--summary-out", type=Path, default=Path("modulus_band_summary.csv"))
    args = ap.parse_args()

    primes, coeff, cf, scale, point, e, M, main_margin, main_bands = setup(
        args.n, args.K, args.r, args.window
    )

    ss = np.random.SeedSequence(args.seed)
    kids = ss.spawn(2 * args.trials)
    payloads = [("genuine", 0, 0)]
    z = 0
    for model in ["uniform", "independent_square"]:
        for trial in range(1, args.trials + 1):
            seed = int(kids[z].generate_state(1, dtype=np.uint64)[0])
            z += 1
            payloads.append((model, trial, seed))

    if args.workers == 1:
        init_worker(args.n, args.K, args.r, args.window)
        results = [worker(x) for x in payloads]
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=init_worker,
            initargs=(args.n, args.K, args.r, args.window),
        ) as ex:
            results = list(ex.map(worker, payloads, chunksize=1))

    trial_rows = [x[0] for x in results]
    detail_rows = [row for x in results for row in x[1]]

    args.trials_out.parent.mkdir(parents=True, exist_ok=True)
    with args.trials_out.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(trial_rows[0].keys()))
        wr.writeheader()
        wr.writerows(trial_rows)

    with args.details_out.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        wr.writeheader()
        wr.writerows(detail_rows)

    summary_rows = []
    for model in ["genuine", "uniform", "independent_square"]:
        zrows = [x for x in detail_rows if x["model"] == model]
        for b in BAND_ORDER:
            br = [x for x in zrows if x["band"] == b]
            a = np.array([x["weighted_R"] for x in br], float)
            o = np.array([x["odd_weighted_R"] for x in br], float)
            ev = np.array([x["even_weighted_R"] for x in br], float)
            summary_rows.append({
                "model": model,
                "band": b,
                "count": len(br),
                "mean_weighted_R": float(a.mean()),
                "median_weighted_R": float(np.median(a)),
                "q05_weighted_R": float(np.percentile(a, 5)),
                "q95_weighted_R": float(np.percentile(a, 95)),
                "mean_odd_weighted_R": float(o.mean()),
                "mean_even_weighted_R": float(ev.mean()),
            })

    with args.summary_out.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        wr.writeheader()
        wr.writerows(summary_rows)

    genuine = trial_rows[0]
    print(f"n={args.n:,}, K={args.K}, r={args.r}, L={args.window}")
    print(f"M={M:.12f}, main margin={main_margin:.12f}")
    print()
    print("GENUINE WORST WINDOW")
    print(f"  max R = {genuine['max_R']:.12f}")
    print(f"  R from d<=n = {genuine['R_d_le_n']:.12f}")
    print(f"  R from d>n  = {genuine['R_d_gt_n']:.12f}")
    print()

    for model in ["uniform", "independent_square"]:
        rr = [x for x in trial_rows if x["model"] == model]
        total = np.array([x["max_R"] for x in rr])
        low = np.array([x["R_d_le_n"] for x in rr])
        high = np.array([x["R_d_gt_n"] for x in rr])
        print(model.upper())
        print(f"  mean max R = {total.mean():.12f}")
        print(f"  mean R d<=n = {low.mean():.12f}")
        print(f"  mean R d>n  = {high.mean():.12f}")
        print()

    print("GENUINE BAND BREAKDOWN")
    for row in [x for x in detail_rows if x["model"] == "genuine"]:
        print(f"  {row['band']:>15}: {row['weighted_R']:+.9f}")
    print()
    print(f"trials:  {args.trials_out}")
    print(f"details: {args.details_out}")
    print(f"summary: {args.summary_out}")


if __name__ == "__main__":
    main()
