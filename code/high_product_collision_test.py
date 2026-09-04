#!/usr/bin/env python3
"""
high_product_collision_test.py

Diagnose the key arithmetic constraint absent from phase-scrambled controls:

For a genuine square shell, if j distinct multiplication rows p_1,...,p_j
hit the same odd position m, then
    d = p_1...p_j divides m,
so necessarily
    d <= m < (n+1)^2.

Independent phase scrambles can create formal row coincidences with
d >> (n+1)^2. This script measures exactly how much of the dangerous
high-order intersection mass in each model comes from such impossible
products.

The comparison uses each realization's WORST K,r local window.

Product bands:
  d <= n
  n < d <= n^2
  n^2 < d < (n+1)^2       (thin genuine shell band)
  (n+1)^2 <= d <= n^3
  n^3 < d <= n^4
  d > n^4

For model-to-model comparison we report ACTUAL weighted intersection mass
sum_j c_j sum_{d in band} A_d. The main term is common to all models, so
differences in R between models equal differences in these actual masses.

Example:
  python high_product_collision_test.py --trials 100 --workers 2

Outputs:
  high_product_trials.csv
  high_product_bands.csv
  high_product_summary.csv
"""

from __future__ import annotations

import argparse, csv, itertools, math
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from math import comb
from pathlib import Path
import numpy as np

G = {}


def primes_upto(n):
    s=np.ones(n+1,dtype=np.bool_); s[:2]=False
    for p in range(2,int(n**0.5)+1):
        if s[p]: s[p*p:n+1:p]=False
    a=np.nonzero(s)[0]
    return a[a>=3]


def lcm(a,b): return a//math.gcd(a,b)*b


def coeffs_P(K,r):
    D=1
    for j in range(K-1): D*=r+j
    vals=[]
    for w in range(K+1):
        z=w-1
        for j in range(K-1): z*=w-r-j
        vals.append(Fraction(1,1)+Fraction(z,D))
    out=[]; cur=vals
    for _ in range(K+1):
        out.append(cur[0])
        cur=[cur[i+1]-cur[i] for i in range(len(cur)-1)]
    return out


def elementary(primes,K):
    e=np.zeros(K+1); e[0]=1
    for p0 in primes:
        z=1/float(p0)
        for j in range(K,0,-1): e[j]+=z*e[j-1]
    return e


def genuine_residues(n,primes):
    delta=1 if n%2==0 else 2
    lo=n*n+delta
    out=np.empty(len(primes),dtype=np.int64)
    for i,p0 in enumerate(primes):
        p=int(p0)
        out[i]=(-(lo%p)*((p+1)//2))%p
    return out


def random_residues(mode,n,primes,rng):
    if mode=="uniform":
        return np.array([rng.integers(0,int(p)) for p in primes],dtype=np.int64)
    if mode=="independent_square":
        delta=1 if n%2==0 else 2
        out=np.empty(len(primes),dtype=np.int64)
        for i,p0 in enumerate(primes):
            p=int(p0); u=int(rng.integers(0,p))
            out[i]=(-((u*u+delta)%p)*((p+1)//2))%p
        return out
    raise ValueError(mode)


def setup(n,K,r,L):
    primes=primes_upto(n)
    coeff=coeffs_P(K,r)
    cf=np.array([float(c) for c in coeff])
    scale=1
    for c in coeff[1:]: scale=lcm(scale,c.denominator)
    point=np.zeros(65,dtype=np.int64)
    for w in range(65):
        Pw=sum(coeff[j]*comb(w,j) for j in range(1,K+1) if w>=j)
        point[w]=int((Fraction(1,1)-Pw)*scale)
    e=elementary(primes,K)
    M=float(sum(cf[j]*e[j] for j in range(1,K+1)))
    mm=L*(1-M)
    return primes,coeff,cf,scale,point,M,mm


def depth(n,primes,res):
    d=np.zeros(n,dtype=np.int16)
    for p0,r0 in zip(primes,res):
        d[int(r0)::int(p0)]+=1
    return d


def worst_window(d,L,point,scale,mm):
    s=point[d]
    pref=np.concatenate(([0],np.cumsum(s,dtype=np.int64)))
    cs=pref[L:]-pref[:-L]
    cert=cs.astype(float)/scale
    R=mm-cert
    i=int(np.argmax(R))
    return i,float(R[i]),float(cert[i]),int(np.count_nonzero(cs<=0))


def hit_lists(L,primes,res,start):
    h=[[] for _ in range(L)]
    end=start+L
    for p0,r0 in zip(primes,res):
        p=int(p0); rr=int(r0)
        if rr<start:
            first=rr+((start-rr+p-1)//p)*p
        else:
            first=rr
        if first>=end: continue
        for t in range(first,end,p):
            h[t-start].append(p)
    return h


def product_band(n,d):
    n2=n*n
    hi=(n+1)*(n+1)
    if d<=n: return "d<=n"
    if d<=n2: return "n<d<=n^2"
    if d<hi: return "n^2<d<(n+1)^2"
    if d<=n**3: return "(n+1)^2<=d<=n^3"
    if d<=n**4: return "n^3<d<=n^4"
    return "d>n^4"


BANDS=[
    "d<=n",
    "n<d<=n^2",
    "n^2<d<(n+1)^2",
    "(n+1)^2<=d<=n^3",
    "n^3<d<=n^4",
    "d>n^4",
]


def actual_product_bands(n,K,cf,hits):
    rows={b:{"counts":np.zeros(K+1,dtype=np.int64),
             "weighted":np.zeros(K+1,float)} for b in BANDS}
    for plist in hits:
        for j in range(1,min(K,len(plist))+1):
            for tup in itertools.combinations(plist,j):
                d=math.prod(tup)
                b=product_band(n,d)
                rows[b]["counts"][j]+=1
                rows[b]["weighted"][j]+=cf[j]
    return rows


def init_worker(n,K,r,L):
    primes,coeff,cf,scale,point,M,mm=setup(n,K,r,L)
    G.update(n=n,K=K,r=r,L=L,primes=primes,coeff=coeff,cf=cf,
             scale=scale,point=point,M=M,mm=mm)


def one(payload):
    model,trial,seed=payload
    if model=="genuine":
        res=genuine_residues(G["n"],G["primes"])
    else:
        rng=np.random.default_rng(seed)
        res=random_residues(model,G["n"],G["primes"],rng)

    d=depth(G["n"],G["primes"],res)
    start,maxR,mincert,nfail=worst_window(
        d,G["L"],G["point"],G["scale"],G["mm"])
    hits=hit_lists(G["L"],G["primes"],res,start)
    bands=actual_product_bands(G["n"],G["K"],G["cf"],hits)

    detail=[]
    impossible_mass=0.0
    impossible_count=0
    total_actual=0.0

    for b in BANDS:
        cnt=bands[b]["counts"]; ww=bands[b]["weighted"]
        mass=float(ww[1:].sum())
        total_actual+=mass
        if b in ["(n+1)^2<=d<=n^3","n^3<d<=n^4","d>n^4"]:
            impossible_mass+=mass
            impossible_count+=int(cnt[1:].sum())

        row={"model":model,"trial":trial,"seed":int(seed),
             "worst_start":start,"max_R":maxR,"band":b,
             "actual_weighted_mass":mass,
             "actual_intersections":int(cnt[1:].sum())}
        for j in range(1,G["K"]+1):
            row[f"count_j{j}"]=int(cnt[j])
            row[f"weighted_j{j}"]=float(ww[j])
        detail.append(row)

    summary={
        "model":model,"trial":trial,"seed":int(seed),
        "worst_start":start,"max_R":maxR,"min_certificate":mincert,
        "fail_windows":nfail,"total_actual_weighted_mass":total_actual,
        "impossible_product_weighted_mass":impossible_mass,
        "impossible_product_intersections":impossible_count,
    }
    return summary,detail


def worker(x): return one(x)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--n",type=int,default=1_000_000)
    ap.add_argument("--K",type=int,default=5)
    ap.add_argument("--r",type=int,default=3)
    ap.add_argument("--window",type=int,default=391)
    ap.add_argument("--trials",type=int,default=100)
    ap.add_argument("--workers",type=int,default=2)
    ap.add_argument("--seed",type=int,default=20260831)
    ap.add_argument("--trials-out",type=Path,default=Path("high_product_trials.csv"))
    ap.add_argument("--bands-out",type=Path,default=Path("high_product_bands.csv"))
    ap.add_argument("--summary-out",type=Path,default=Path("high_product_summary.csv"))
    args=ap.parse_args()

    primes,coeff,cf,scale,point,M,mm=setup(args.n,args.K,args.r,args.window)
    ss=np.random.SeedSequence(args.seed)
    kids=ss.spawn(2*args.trials)
    payload=[("genuine",0,0)]
    z=0
    for model in ["uniform","independent_square"]:
        for i in range(1,args.trials+1):
            seed=int(kids[z].generate_state(1,dtype=np.uint64)[0]); z+=1
            payload.append((model,i,seed))

    if args.workers==1:
        init_worker(args.n,args.K,args.r,args.window)
        results=[worker(x) for x in payload]
    else:
        with ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker,
            initargs=(args.n,args.K,args.r,args.window)) as ex:
            results=list(ex.map(worker,payload,chunksize=1))

    trials=[x[0] for x in results]
    details=[r for x in results for r in x[1]]

    args.trials_out.parent.mkdir(parents=True,exist_ok=True)
    with args.trials_out.open("w",newline="",encoding="utf-8") as f:
        wr=csv.DictWriter(f,fieldnames=list(trials[0].keys()))
        wr.writeheader(); wr.writerows(trials)
    with args.bands_out.open("w",newline="",encoding="utf-8") as f:
        wr=csv.DictWriter(f,fieldnames=list(details[0].keys()))
        wr.writeheader(); wr.writerows(details)

    summaries=[]
    for model in ["genuine","uniform","independent_square"]:
        zz=[x for x in trials if x["model"]==model]
        for b in BANDS:
            bb=[x for x in details if x["model"]==model and x["band"]==b]
            mass=np.array([x["actual_weighted_mass"] for x in bb],float)
            cnt=np.array([x["actual_intersections"] for x in bb],float)
            summaries.append({
                "model":model,"band":b,"count_realizations":len(bb),
                "mean_actual_weighted_mass":float(mass.mean()),
                "median_actual_weighted_mass":float(np.median(mass)),
                "mean_actual_intersections":float(cnt.mean()),
            })

    with args.summary_out.open("w",newline="",encoding="utf-8") as f:
        wr=csv.DictWriter(f,fieldnames=list(summaries[0].keys()))
        wr.writeheader(); wr.writerows(summaries)

    print(f"n={args.n:,}, K={args.K}, r={args.r}, L={args.window}")
    print(f"M={M:.12f}, main margin={mm:.12f}")
    print()

    for model in ["genuine","uniform","independent_square"]:
        zz=[x for x in trials if x["model"]==model]
        a=np.array([x["max_R"] for x in zz])
        im=np.array([x["impossible_product_weighted_mass"] for x in zz])
        ic=np.array([x["impossible_product_intersections"] for x in zz])
        print(model.upper())
        print(f"  mean max R = {a.mean():.12f}")
        print(f"  mean weighted mass from d >= (n+1)^2 = {im.mean():.12f}")
        print(f"  mean count of such formal intersections = {ic.mean():.3f}")
        print()

    print("GENUINE PRODUCT BANDS")
    for x in [r for r in details if r["model"]=="genuine"]:
        print(f"  {x['band']:>22}: weighted={x['actual_weighted_mass']:+.9f}, "
              f"count={x['actual_intersections']}")
    print()
    print(f"trials:  {args.trials_out}")
    print(f"bands:   {args.bands_out}")
    print(f"summary: {args.summary_out}")


if __name__=="__main__":
    main()
