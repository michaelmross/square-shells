#!/usr/bin/env python3
"""
postprocess_exact_k5.py

Read exact prime-stream histograms produced by exact_k5_stream and compute
the K=5,r=3 geometric main term on nested FFT grids.

No PNT tail is used here.  The only remaining approximation is the
logarithmic histogram/product-cutoff discretization, which is assessed by
coarsening the finest grid and extrapolating G -> infinity.

Example
-------
python postprocess_exact_k5.py exact_k5_run \
    --grids 262144,524288,1048576 \
    --out exact_k5_results.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import math
import numpy as np

COEFF = np.array(
    [0.0, 1.0, -14.0/15.0, 4.0/5.0, -3.0/5.0, 1.0/3.0],
    dtype=np.float64
)


def unrestricted_elementary(power: np.ndarray) -> np.ndarray:
    e = np.zeros(6, dtype=np.float64)
    e[0] = 1.0
    for k in range(1, 6):
        e[k] = sum(
            ((-1) ** (i - 1)) * e[k - i] * power[i]
            for i in range(1, k + 1)
        ) / k
    return e


def geometric_e3_e5(hist: list[np.ndarray]) -> tuple[float, float, float]:
    grid = len(hist[0])

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

    a111 = cutoff_mass({1: 3})
    a12 = cutoff_mass({1: 1, 2: 1})
    a3 = float(hist[2].sum())
    E3 = (a111 - 3.0*a12 + 2.0*a3) / 6.0

    a1111 = cutoff_mass({1: 4})
    a112 = cutoff_mass({1: 2, 2: 1})
    a22 = cutoff_mass({2: 2})
    a13 = cutoff_mass({1: 1, 3: 1})
    a4 = float(hist[3].sum())
    E4 = (
        a1111 - 6.0*a112 + 3.0*a22 + 8.0*a13 - 6.0*a4
    ) / 24.0

    a11111 = cutoff_mass({1: 5})
    a1112 = cutoff_mass({1: 3, 2: 1})
    a122 = cutoff_mass({1: 1, 2: 2})
    a113 = cutoff_mass({1: 2, 3: 1})
    a23 = cutoff_mass({2: 1, 3: 1})
    a14 = cutoff_mass({1: 1, 4: 1})
    a5 = float(hist[4].sum())

    E5 = (
        a11111
        - 10.0*a1112
        + 15.0*a122
        + 20.0*a113
        - 20.0*a23
        - 30.0*a14
        + 24.0*a5
    ) / 120.0

    return E3, E4, E5


def coarsen(hist_fine: np.ndarray, fine_grid: int, grid: int) -> np.ndarray:
    if fine_grid % grid != 0:
        raise ValueError(f"Fine grid {fine_grid} is not divisible by {grid}")
    factor = fine_grid // grid
    if factor == 1:
        return hist_fine.copy()
    return hist_fine.reshape(5, grid, factor).sum(axis=2)


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
        A = np.column_stack([np.ones(3), h, h*h])
        c = np.linalg.solve(A, y)
        d = {
            "fit_a": float(c[1]),
            "fit_b": float(c[2]),
            "richardson_12": float("nan"),
            "richardson_23": float("nan"),
        }
        if grids[1] == 2*grids[0] and grids[2] == 2*grids[1]:
            d["richardson_12"] = 2*values[1] - values[0]
            d["richardson_23"] = 2*values[2] - values[1]
        return float(c[0]), d

    raise ValueError("Use exactly two or three grids")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument(
        "--grids",
        default="262144,524288,1048576",
        help="two or three nested grids"
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("exact_k5_results.csv")
    )
    args = ap.parse_args()

    grids = sorted(int(x) for x in args.grids.split(",") if x.strip())
    if len(grids) not in (2, 3):
        raise SystemExit("--grids must contain two or three values")

    manifest_path = args.run_dir / "manifest.csv"
    with manifest_path.open(newline="", encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))

    out_rows = []
    crossing_points = []

    for rec in manifest:
        n = int(rec["n"])
        fine_grid = int(rec["grid"])
        hist_path = args.run_dir / rec["hist_file"]

        raw = np.fromfile(hist_path, dtype=np.float64)
        expected = 5 * fine_grid
        if raw.size != expected:
            raise RuntimeError(
                f"{hist_path}: expected {expected} doubles, found {raw.size}"
            )
        fine = raw.reshape(5, fine_grid)

        power = np.zeros(6, dtype=np.float64)
        for r in range(1, 6):
            power[r] = float(rec[f"p{r}"])

        e = unrestricted_elementary(power)

        Mvals = []
        per_grid = {}
        for G in grids:
            h = coarsen(fine, fine_grid, G)
            E3, E4, E5 = geometric_e3_e5([h[i] for i in range(5)])
            E = np.array([1.0, e[1], e[2], E3, E4, E5], dtype=np.float64)
            M = float(np.dot(COEFF[1:], E[1:]))
            Mvals.append(M)
            per_grid[G] = (M, E3, E4, E5)
            print(
                f"n={n:,}  G={G:,}  Mgeom={M:.12f}  "
                f"1-M={1-M:+.12e}"
            )

        Minf, diag = extrapolate(grids, Mvals)
        crossing_points.append((n, Minf))

        row = {
            "n": n,
            "odd_prime_count": int(rec["odd_prime_count"]),
            "fine_grid": fine_grid,
            "M_extrapolated": Minf,
            "margin_extrapolated": 1.0 - Minf,
            "fit_a_over_G": diag["fit_a"],
            "fit_b_over_G2": diag["fit_b"],
            "richardson_12": diag["richardson_12"],
            "richardson_23": diag["richardson_23"],
        }
        for G in grids:
            row[f"M_G{G}"] = per_grid[G][0]
        out_rows.append(row)

        print(
            f"  -> extrapolated M={Minf:.12f}, "
            f"margin={1-Minf:+.12e}\n"
        )

        del fine

    # Find a neighboring pair that straddles M=1.
    crossing_points.sort()
    root_est = None
    for (n1, m1), (n2, m2) in zip(crossing_points, crossing_points[1:]):
        f1 = m1 - 1.0
        f2 = m2 - 1.0
        if f1 == 0:
            root_est = float(n1)
            break
        if f1 * f2 <= 0:
            root_est = n1 - f1 * (n2 - n1) / (f2 - f1)
            print(
                "Crossing bracketed by exact-prime evaluations:\n"
                f"  n={n1:,}: M={m1:.12f}\n"
                f"  n={n2:,}: M={m2:.12f}\n"
                f"Linear interpolation: n* ~= {root_est:,.0f}"
            )
            break

    if root_est is None:
        print(
            "No evaluated target pair straddles M=1. "
            "Choose a wider or shifted --targets list and rerun the streamer."
        )

    if out_rows:
        keys = list(out_rows[0].keys())
        with args.out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(out_rows)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
