
#include <primesieve.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using u128 = unsigned __int128;
using i128 = __int128_t;

static constexpr int JMAX = 9;

static std::string to_string_u128(u128 x)
{
    if (x == 0) return "0";
    std::string s;
    while (x > 0) {
        unsigned digit = static_cast<unsigned>(x % 10);
        s.push_back(static_cast<char>('0' + digit));
        x /= 10;
    }
    std::reverse(s.begin(), s.end());
    return s;
}

static std::string to_string_i128(i128 x)
{
    if (x < 0) {
        u128 y = static_cast<u128>(-(x + 1));
        y += 1;
        return "-" + to_string_u128(y);
    }
    return to_string_u128(static_cast<u128>(x));
}

static long double to_long_double(u128 x)
{
    // Exact enough for reporting ratios; integer values are also written separately.
    const u128 base = static_cast<u128>(1000000000000000000ULL);
    long double ans = 0.0L;
    long double scale = 1.0L;
    while (x > 0) {
        u128 q = x / base;
        u128 r = x % base;
        ans += static_cast<long double>(static_cast<uint64_t>(r)) * scale;
        scale *= 1.0e18L;
        x = q;
    }
    return ans;
}

static long double to_long_double(i128 x)
{
    if (x < 0) {
        u128 y = static_cast<u128>(-(x + 1));
        y += 1;
        return -to_long_double(y);
    }
    return to_long_double(static_cast<u128>(x));
}

static u128 choose_small(unsigned w, unsigned j)
{
    if (j > w) return 0;
    if (j == 0 || j == w) return 1;
    j = std::min(j, w - j);
    u128 c = 1;
    for (unsigned k = 1; k <= j; ++k) {
        c = c * (w - j + k) / k;
    }
    return c;
}

static uint64_t parse_u64(const std::string& s)
{
    long double x = std::stold(s);
    if (!(x >= 1.0L) ||
        x > static_cast<long double>(std::numeric_limits<uint64_t>::max()))
        throw std::runtime_error("Invalid positive integer: " + s);
    return static_cast<uint64_t>(std::llround(x));
}

static std::vector<uint32_t> load_odd_primes(uint64_t n)
{
    if (n > std::numeric_limits<uint32_t>::max())
        throw std::runtime_error(
            "This implementation stores primes as uint32_t; require n <= 2^32-1."
        );

    std::vector<uint32_t> primes;

    if (n >= 100) {
        long double nn = static_cast<long double>(n);
        size_t reserve_n = static_cast<size_t>(
            1.15L * nn / std::log(nn)
        );
        primes.reserve(reserve_n);
    }

    primesieve_iterator it;
    primesieve_init(&it);
    primesieve_jump_to(&it, 3, n);

    while (true) {
        uint64_t p = primesieve_next_prime(&it);
        if (p == PRIMESIEVE_ERROR) {
            primesieve_free_iterator(&it);
            throw std::runtime_error("primesieve iterator error");
        }
        if (p > n)
            break;
        if (p >= 3)
            primes.push_back(static_cast<uint32_t>(p));
    }

    primesieve_free_iterator(&it);
    return primes;
}

static inline uint64_t first_hit_index(uint64_t first_odd, uint32_t p)
{
    // Solve first_odd + 2*i == 0 (mod p).
    // For odd p, 2^{-1} mod p = (p+1)/2.
    uint64_t pp = p;
    uint64_t residue = first_odd % pp;
    uint64_t neg = (residue == 0) ? 0 : (pp - residue);
    uint64_t inv2 = (pp + 1) / 2;
    return (neg * inv2) % pp;
}

static void usage(const char* argv0)
{
    std::cerr
        << "Usage:\n  " << argv0
        << " --n N [--out shell_N.csv] [--hist-out depth_N.csv]"
        << " [--threads T]\n\n"
        << "Examples:\n"
        << "  " << argv0 << " --n 1e7 --threads 8 --out shell_1e7.csv\n"
        << "  " << argv0 << " --n 1e9 --threads 12 --out shell_1e9.csv\n\n"
        << "Compile on WSL/Ubuntu:\n"
        << "  g++ -O3 -DNDEBUG -std=c++17 -fopenmp exact_shell_certificates.cpp "
        << "-o exact_shell_certificates -lprimesieve\n";
}

int main(int argc, char** argv)
{
    try {
        uint64_t n = 0;
        std::string out_path;
        std::string hist_path;
        int requested_threads = 0;

        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--n" && i + 1 < argc) {
                n = parse_u64(argv[++i]);
            } else if (a == "--out" && i + 1 < argc) {
                out_path = argv[++i];
            } else if (a == "--hist-out" && i + 1 < argc) {
                hist_path = argv[++i];
            } else if (a == "--threads" && i + 1 < argc) {
                requested_threads = std::stoi(argv[++i]);
            } else if (a == "--help" || a == "-h") {
                usage(argv[0]);
                return 0;
            } else {
                throw std::runtime_error("Unknown or incomplete argument: " + a);
            }
        }

        if (n < 2) {
            usage(argv[0]);
            throw std::runtime_error("--n must be at least 2");
        }
        if (n > 1000000000ULL) {
            std::cerr
                << "Warning: n > 1e9. The depth array alone uses more than 1 GiB.\n";
        }

        if (out_path.empty())
            out_path = "shell_n" + std::to_string(n) + ".csv";
        if (hist_path.empty())
            hist_path = "shell_n" + std::to_string(n) + "_depth_hist.csv";

        int threads = 1;
#ifdef _OPENMP
        threads = (requested_threads > 0) ? requested_threads : omp_get_max_threads();
#else
        if (requested_threads > 1)
            std::cerr << "Note: binary was compiled without OpenMP; using one thread.\n";
#endif
        if (threads < 1) threads = 1;

        // n <= 1e9 in the intended use, so these products fit comfortably in uint64_t.
        uint64_t shell_lo = n * n;
        uint64_t np1 = n + 1;
        uint64_t shell_hi = np1 * np1;

        // Exactly n odd integers lie strictly between n^2 and (n+1)^2.
        uint64_t first_odd = (shell_lo & 1ULL) ? shell_lo + 2 : shell_lo + 1;

        std::cout << "Exact square-shell certificate computation\n";
        std::cout << "n              = " << n << "\n";
        std::cout << "shell          = (" << shell_lo << ", " << shell_hi << ")\n";
        std::cout << "odd positions  = " << n << "\n";
        std::cout << "first odd      = " << first_odd << "\n";
        std::cout << "threads        = " << threads << "\n";
        std::cout << "depth RAM      = "
                  << std::fixed << std::setprecision(2)
                  << (static_cast<long double>(n) / (1024.0L * 1024.0L * 1024.0L))
                  << " GiB\n\n";

        auto t0 = std::chrono::steady_clock::now();

        std::cout << "Loading odd primes <= n with primesieve ... " << std::flush;
        auto primes = load_odd_primes(n);
        auto t_primes = std::chrono::steady_clock::now();
        std::cout << "done (" << primes.size() << " primes, "
                  << std::setprecision(1)
                  << std::chrono::duration<double>(t_primes - t0).count()
                  << " s)\n";

        std::cout << "prime-vector RAM = "
                  << std::setprecision(2)
                  << (static_cast<long double>(primes.size() * sizeof(uint32_t))
                      / (1024.0L * 1024.0L * 1024.0L))
                  << " GiB\n";

        std::cout << "Allocating depth array ... " << std::flush;
        std::vector<uint8_t> depth;
        try {
            depth.assign(static_cast<size_t>(n), 0);
        } catch (const std::bad_alloc&) {
            throw std::runtime_error(
                "Could not allocate the depth array. "
                "At n=1e9 it requires about 0.93 GiB, plus the prime vector."
            );
        }
        auto t_alloc = std::chrono::steady_clock::now();
        std::cout << "done\n";

        std::cout << "Marking irreducible multiplication rows ... " << std::flush;

#ifdef _OPENMP
#pragma omp parallel num_threads(threads)
        {
            int tid = omp_get_thread_num();
            int nt = omp_get_num_threads();

            uint64_t a = (static_cast<u128>(n) * tid) / nt;
            uint64_t b = (static_cast<u128>(n) * (tid + 1)) / nt;

            for (uint32_t p : primes) {
                uint64_t i0 = first_hit_index(first_odd, p);
                uint64_t i = i0;

                if (i < a) {
                    uint64_t d = a - i;
                    i += ((d + p - 1) / p) * static_cast<uint64_t>(p);
                }

                for (; i < b; i += p) {
                    ++depth[static_cast<size_t>(i)];
                }
            }
        }
#else
        for (uint32_t p : primes) {
            uint64_t i0 = first_hit_index(first_odd, p);
            for (uint64_t i = i0; i < n; i += p) {
                ++depth[static_cast<size_t>(i)];
            }
        }
#endif

        auto t_mark = std::chrono::steady_clock::now();
        std::cout << "done ("
                  << std::setprecision(1)
                  << std::chrono::duration<double>(t_mark - t_alloc).count()
                  << " s)\n";

        std::array<uint64_t, 256> hist{};
        std::cout << "Reducing depth histogram ... " << std::flush;

#ifdef _OPENMP
#pragma omp parallel num_threads(threads)
        {
            std::array<uint64_t, 256> local{};
#pragma omp for schedule(static)
            for (uint64_t i = 0; i < n; ++i) {
                ++local[depth[static_cast<size_t>(i)]];
            }
#pragma omp critical
            {
                for (size_t w = 0; w < hist.size(); ++w)
                    hist[w] += local[w];
            }
        }
#else
        for (uint8_t w : depth)
            ++hist[w];
#endif

        auto t_hist = std::chrono::steady_clock::now();
        std::cout << "done\n";

        unsigned max_depth = 0;
        for (unsigned w = 255; w > 0; --w) {
            if (hist[w] != 0) {
                max_depth = w;
                break;
            }
        }

        std::array<u128, JMAX + 1> S{};
        for (unsigned w = 1; w <= max_depth; ++w) {
            if (hist[w] == 0) continue;
            for (unsigned j = 1; j <= JMAX && j <= w; ++j) {
                S[j] += static_cast<u128>(hist[w]) * choose_small(w, j);
            }
        }

        // K=5, r=3 certificate:
        // L53 = (15 S1 - 14 S2 + 12 S3 - 9 S4 + 5 S5) / 15.
        i128 num53 =
              15 * static_cast<i128>(S[1])
            - 14 * static_cast<i128>(S[2])
            + 12 * static_cast<i128>(S[3])
            -  9 * static_cast<i128>(S[4])
            +  5 * static_cast<i128>(S[5]);

        // Ordinary odd-order Bonferroni for K=7 and K=9.
        i128 L72 = 0;
        for (int j = 1; j <= 7; ++j)
            L72 += (j & 1 ? 1 : -1) * static_cast<i128>(S[j]);

        i128 L92 = 0;
        for (int j = 1; j <= 9; ++j)
            L92 += (j & 1 ? 1 : -1) * static_cast<i128>(S[j]);

        long double L53 = to_long_double(num53) / 15.0L;
        long double L7  = to_long_double(L72);
        long double L9  = to_long_double(L92);

        long double ratio53 = L53 / static_cast<long double>(n);
        long double ratio72 = L7  / static_cast<long double>(n);
        long double ratio92 = L9  / static_cast<long double>(n);

        uint64_t exact_holes = hist[0];
        uint64_t exact_covered = n - exact_holes;

        std::ofstream out(out_path);
        if (!out)
            throw std::runtime_error("Could not open output CSV: " + out_path);

        out << "n,shell_lo,shell_hi,first_odd,odd_positions,odd_prime_rows,"
               "max_depth,exact_covered,exact_holes";
        for (int j = 1; j <= JMAX; ++j)
            out << ",S" << j;
        out << ",L53_numerator_over15,L53_ratio,L53_margin,"
               "L72,L72_ratio,L72_margin,"
               "L92,L92_ratio,L92_margin,"
               "prime_seconds,mark_seconds,hist_seconds,total_seconds\n";

        out << n << ","
            << shell_lo << ","
            << shell_hi << ","
            << first_odd << ","
            << n << ","
            << primes.size() << ","
            << max_depth << ","
            << exact_covered << ","
            << exact_holes;

        for (int j = 1; j <= JMAX; ++j)
            out << "," << to_string_u128(S[j]);

        auto t_end = std::chrono::steady_clock::now();
        double prime_seconds = std::chrono::duration<double>(t_primes - t0).count();
        double mark_seconds  = std::chrono::duration<double>(t_mark - t_alloc).count();
        double hist_seconds  = std::chrono::duration<double>(t_hist - t_mark).count();
        double total_seconds = std::chrono::duration<double>(t_end - t0).count();

        out << "," << to_string_i128(num53) << "/15"
            << "," << std::setprecision(16) << static_cast<double>(ratio53)
            << "," << static_cast<double>(1.0L - ratio53)
            << "," << to_string_i128(L72)
            << "," << static_cast<double>(ratio72)
            << "," << static_cast<double>(1.0L - ratio72)
            << "," << to_string_i128(L92)
            << "," << static_cast<double>(ratio92)
            << "," << static_cast<double>(1.0L - ratio92)
            << "," << prime_seconds
            << "," << mark_seconds
            << "," << hist_seconds
            << "," << total_seconds
            << "\n";

        std::ofstream hout(hist_path);
        if (!hout)
            throw std::runtime_error("Could not open histogram CSV: " + hist_path);
        hout << "depth,count\n";
        for (unsigned w = 0; w <= max_depth; ++w) {
            if (hist[w] != 0)
                hout << w << "," << hist[w] << "\n";
        }

        std::cout << "\nResult\n";
        std::cout << "------\n";
        std::cout << "max depth       = " << max_depth << "\n";
        std::cout << "exact holes     = " << exact_holes << "\n";
        std::cout << std::setprecision(12);
        std::cout << "L_5,3 / n       = " << static_cast<double>(ratio53)
                  << "   margin=" << static_cast<double>(1.0L - ratio53) << "\n";
        std::cout << "L_7,2 / n       = " << static_cast<double>(ratio72)
                  << "   margin=" << static_cast<double>(1.0L - ratio72) << "\n";
        std::cout << "L_9,2 / n       = " << static_cast<double>(ratio92)
                  << "   margin=" << static_cast<double>(1.0L - ratio92) << "\n";
        std::cout << "total seconds   = " << std::setprecision(1) << total_seconds << "\n";
        std::cout << "wrote            " << out_path << "\n";
        std::cout << "wrote            " << hist_path << "\n";

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
