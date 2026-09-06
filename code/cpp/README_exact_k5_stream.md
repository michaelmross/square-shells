# Exact-prime K=5 geometric crossing check

This package removes the PNT-tail approximation from the previous hybrid
calculation.

It has two stages:

1. `exact_k5_stream` (C++ + primesieve) streams **every odd prime** up to the
   largest requested target and accumulates the five logarithmic prime
   measures at one fine grid.
2. `postprocess_exact_k5.py` coarsens that fine grid exactly to the two lower
   grids, computes the FFT/Newton geometric moments, and extrapolates
   `G -> infinity`.

Thus the prime input is exact.  The only remaining numerical approximation is
the logarithmic histogram/product-cutoff discretization.

## Suggested first run

The current hybrid calculation puts the crossing near 112.86 billion, so the
default target triple is

- 112,850,000,000
- 112,860,000,000
- 112,870,000,000

That should bracket the crossing unless the PNT-tail model was off by more
than about 10 million.

The finest grid is 1,048,576.  With three targets the histogram arrays use
about 120 MiB; FFT postprocessing uses substantially more temporary memory.

## Build on Windows

You need a C++ compiler, CMake, and Git.  The supplied `CMakeLists.txt`
automatically fetches primesieve v12.15 if it is not already installed.

From PowerShell in this folder:

```powershell
cmake -S . -B build
cmake --build build --config Release
```

The executable will normally be:

```text
build\Release\exact_k5_stream.exe
```

If you use a single-configuration generator (e.g. Ninja), it may instead be:

```text
build\exact_k5_stream.exe
```

## Run the exact prime stream

```powershell
.\build\Release\exact_k5_stream.exe `
  --targets 112850000000,112860000000,112870000000 `
  --grid 1048576 `
  --out exact_k5_run
```

The program prints progress as it moves through the prime stream.  It does
**not** store the billions of primes.  It stores only the reciprocal-power
histograms.

Output:

- `exact_k5_run\manifest.csv`
- one raw histogram file per target, about 40 MiB each.

## Postprocess

Requires NumPy:

```powershell
python postprocess_exact_k5.py exact_k5_run `
  --grids 262144,524288,1048576 `
  --out exact_k5_results.csv
```

The postprocessor reports:

- `M^geom_{5,3}` on all three nested grids;
- the quadratic fit
  `M(G) = M_inf + a/G + b/G^2`;
- the two adjacent first-order Richardson estimates when the grids double;
- a linearly interpolated crossing if two exact-prime target values straddle
  `M=1`.

## Important interpretation

This is an **exact prime-stream** computation, not an exact evaluation of the
product cutoff itself.  The product cutoff is still represented on a finite
logarithmic grid.  The three-grid extrapolation is intended to diagnose and
reduce that remaining discretization error.

If the first three targets bracket the crossing, there is no reason to run a
large scan.  If they do not, rerun with a shifted target triple.
