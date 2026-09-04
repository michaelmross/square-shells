#!/usr/bin/env python3
"""
nearby_square_ensemble.py

Control experiment complementary to phase_scramble_test.py.

For many genuine square shells n near a target n0, compute the same fixed
K,r local-window statistic:
    max R over every window of L odd positions,
    min certificate,
    number of failing windows.

This distinguishes:
  * a property specific to the single shell n0,
from
  * a stable property of genuine square-shell phase vectors in that scale.

Example:
    python nearby_square_ensemble.py ^
        --center 1000000 --radius 5000 --samples 101 ^
        --K 5 --r 3 --window 391 --workers 2
"""

from __future__ import annotations

import argparse
import bisect
import csv
import math
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from math import comb
from pathlib import Path

import numpy as np

G_PRIMES = None
G_K = 5
G_R = 3
G_L = 391


def primes_upto(n: int) -> np.ndarray:
    s = np.ones(n+1, dtype=np.bool_)
    s[:2] = False
    lim = int(n**0.5)
    for p in range(2, lim+1):
        if s[p]:
            s[p*p:n+1:p] = False
    a = np.nonzero(s)[0]
    return a[a >= 3]


def lcm(a,b):
    return a // math.gcd(a,b) * b


def coeffs_P(K, r):
    D=1
    for j in range(K-1):
        D *= r+j
    vals=[]
    for w in range(K+1):
        z=w-1
        for j in range(K-1):
            z *= w-r-j
        vals.append(Fraction(1,1)+Fraction(z,D))
    out=[]
    cur=vals
    for _ in range(K+1):
        out.append(cur[0])
        cur=[cur[i+1]-cur[i] for i in range(len(cur)-1)]
    return out


def elementary(primes, K):
    e=np.zeros(K+1,float); e[0]=1.
    for p0 in primes:
        z=1./float(p0)
        for j in range(K,0,-1):
            e[j] += z*e[j-1]
    return e


def init_worker(maxn, K, r, L):
    global G_PRIMES,G_K,G_R,G_L
    G_PRIMES=primes_upto(maxn)
    G_K,G_R,G_L=K,r,L


def one_n(n):
    K,R,L=G_K,G_R,G_L
    upto=np.searchsorted(G_PRIMES,n,side="right")
    primes=G_PRIMES[:upto]

    coeff=coeffs_P(K,R)
    scale=1
    for c in coeff[1:]:
        scale=lcm(scale,c.denominator)

    e=elementary(primes,K)
    cf=np.array([float(c) for c in coeff])
    M=float(sum(cf[j]*e[j] for j in range(1,K+1)))
    main_margin=L*(1-M)

    point=np.zeros(65,dtype=np.int64)
    for w in range(65):
        Pw=sum(coeff[j]*comb(w,j) for j in range(1,K+1) if w>=j)
        point[w]=int((Fraction(1,1)-Pw)*scale)

    lo_odd=n*n+(1 if n%2==0 else 2)
    depth=np.zeros(n,dtype=np.int16)
    for p0 in primes:
        p=int(p0)
        inv2=(p+1)//2
        rr=(-(lo_odd%p)*inv2)%p
        depth[rr::p]+=1

    score=point[depth]
    pref=np.concatenate(([0],np.cumsum(score,dtype=np.int64)))
    cert_scaled=pref[L:]-pref[:-L]
    cert=cert_scaled.astype(float)/scale
    Rvals=main_margin-cert

    return {
        "n": n,
        "M": M,
        "main_margin": main_margin,
        "max_R": float(Rvals.max()),
        "min_certificate": float(cert.min()),
        "fail_windows": int(np.count_nonzero(cert_scaled<=0)),
        "total_windows": len(cert_scaled),
        "max_depth": int(depth.max()),
        "R_median": float(np.median(Rvals)),
        "R_q99": float(np.quantile(Rvals,.99)),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--center",type=int,default=1_000_000)
    ap.add_argument("--radius",type=int,default=5000)
    ap.add_argument("--samples",type=int,default=101)
    ap.add_argument("--K",type=int,default=5)
    ap.add_argument("--r",type=int,default=3)
    ap.add_argument("--window",type=int,default=391)
    ap.add_argument("--workers",type=int,default=2)
    ap.add_argument("--out",type=Path,default=Path("nearby_square_ensemble.csv"))
    args=ap.parse_args()

    lo=max(3,args.center-args.radius)
    hi=args.center+args.radius
    ns=np.unique(np.rint(np.linspace(lo,hi,args.samples)).astype(int))
    if args.center not in ns:
        ns=np.sort(np.append(ns,args.center))

    if args.workers==1:
        init_worker(int(ns.max()),args.K,args.r,args.window)
        rows=[one_n(int(n)) for n in ns]
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=init_worker,
            initargs=(int(ns.max()),args.K,args.r,args.window)
        ) as ex:
            rows=list(ex.map(one_n,[int(n) for n in ns],chunksize=1))

    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("w",newline="",encoding="utf-8") as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)

    maxR=np.array([x["max_R"] for x in rows])
    fails=np.array([x["fail_windows"]>0 for x in rows])
    center_row=next(x for x in rows if x["n"]==args.center)

    print(f"genuine square-shell ensemble: n={lo:,}..{hi:,}, samples={len(rows)}")
    print(f"K={args.K}, r={args.r}, window={args.window} odd positions")
    print()
    print(f"center n={args.center:,}: max R={center_row['max_R']:.12f}, "
          f"min cert={center_row['min_certificate']:.12f}")
    print()
    print(f"ensemble mean max R={maxR.mean():.12f}")
    print(f"ensemble sd max R={maxR.std(ddof=1):.12f}")
    for q in [1,5,25,50,75,95,99]:
        print(f"ensemble max R q{q:02d}={np.percentile(maxR,q):.12f}")
    print(f"shells with >=1 failing window={fails.sum()}/{len(rows)}")
    rank=(np.count_nonzero(maxR<=center_row["max_R"]))/len(rows)
    print(f"fraction of sampled square shells with max R <= center={rank:.6f}")
    print(f"CSV: {args.out}")


if __name__=="__main__":
    main()
