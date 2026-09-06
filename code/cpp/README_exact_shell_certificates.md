# Exact square-shell certificate samples

This program implements the "exact shell/certificate samples" calculation.

For a chosen \(n\), it treats the \(n\) odd integers in

\[
n^2<m<(n+1)^2
\]

and marks every irreducible multiplication row indexed by an odd prime
\(p\le n\).  The depth \(w(m)\) is therefore computed exactly at every
odd shell position.

From the depth histogram it obtains

\[
S_j=\sum_m \binom{w(m)}j,\qquad 1\le j\le9,
\]

and reports all three certificates in one pass:

\[
L_{5,3}
=
S_1-\frac{14}{15}S_2+\frac45S_3-\frac35S_4+\frac13S_5,
\]

\[
L_{7,2}=S_1-S_2+S_3-S_4+S_5-S_6+S_7,
\]

\[
L_{9,2}
=
S_1-S_2+S_3-S_4+S_5-S_6+S_7-S_8+S_9.
\]

The zero-depth count is the exact number of prime holes in the shell.

## Memory

The depth array uses one byte per odd shell position:

- \(n=10^7\): about 10 MB;
- \(n=10^8\): about 100 MB;
- \(n=10^9\): about 1 GB.

The program also stores odd primes through \(n\) as 32-bit integers.
At \(10^9\) this is about another 200 MB.

## Compile on WSL

Assuming the same primesieve installation used for the earlier exact-prime
program:

```bash
g++ -O3 -DNDEBUG -std=c++17 -fopenmp \
  exact_shell_certificates.cpp \
  -o exact_shell_certificates \
  -lprimesieve
```

If OpenMP is unavailable, omit `-fopenmp`; the program will run
single-threaded.

## Recommended validation sequence

First:

```bash
./exact_shell_certificates \
  --n 1e7 \
  --threads 8 \
  --out shell_1e7.csv
```

Then:

```bash
./exact_shell_certificates \
  --n 1e8 \
  --threads 8 \
  --out shell_1e8.csv
```

Finally, if those are clean:

```bash
./exact_shell_certificates \
  --n 1e9 \
  --threads 8 \
  --out shell_1e9.csv
```

Choose the thread count to suit the machine.  More threads are not always
faster because the marking stage is memory-heavy.

Each run also writes a depth histogram such as

```text
shell_n1000000000_depth_hist.csv
```

unless `--hist-out` is supplied explicitly.

## Main output

The one-row CSV contains:

- shell endpoints;
- number of odd prime rows;
- maximum depth;
- exact covered positions and exact holes;
- \(S_1,\ldots,S_9\);
- \(L_{5,3}/n\), \(L_{7,2}/n\), \(L_{9,2}/n\);
- certificate margins \(1-L/n\);
- timings.

These exact shell ratios are the quantities to compare with the geometric
main terms.  The fluctuation is

\[
\Delta_{K,r}(n)=\frac{L_{K,r}(n)}n-M^{\rm geom}_{K,r}(n).
\]

The exact-shell computation itself does not use the FFT/PNT approximation.
