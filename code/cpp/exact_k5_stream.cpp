
#include <primesieve.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static constexpr int RMAX = 5;

struct Tracker {
    uint64_t grid = 0;
    int r = 0;
    long double step = 0.0L;
    uint64_t bin = 0;
    long double upper = 0.0L;

    Tracker() = default;

    Tracker(uint64_t G, int rr, long double logX)
        : grid(G), r(rr), step(logX / (static_cast<long double>(rr) * G)), bin(0)
    {
        upper = std::expl(step);
    }

    inline uint64_t locate(uint64_t p)
    {
        long double pp = static_cast<long double>(p);
        while (bin < grid && pp >= upper) {
            ++bin;
            if (bin < grid)
                upper = std::expl((static_cast<long double>(bin) + 1.0L) * step);
        }
        return bin;
    }
};

struct TargetData {
    uint64_t n = 0;
    uint64_t grid = 0;
    long double logX = 0.0L;
    std::array<std::vector<double>, RMAX> hist;
    std::array<Tracker, RMAX> tracker;
    std::array<double, RMAX + 1> power_sum{};
    uint64_t odd_prime_count = 0;
    bool snapshotted = false;

    TargetData(uint64_t nn, uint64_t G)
        : n(nn), grid(G)
    {
        logX = 2.0L * std::logl(static_cast<long double>(n) + 1.0L);
        for (int r = 1; r <= RMAX; ++r) {
            hist[r - 1].assign(grid, 0.0);
            tracker[r - 1] = Tracker(grid, r, logX);
        }
    }

    size_t bytes() const
    {
        return static_cast<size_t>(RMAX) * static_cast<size_t>(grid) * sizeof(double);
    }
};

static std::vector<uint64_t> parse_targets(const std::string& s)
{
    std::vector<uint64_t> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty())
            continue;
        long double x = std::stold(item);
        if (x < 3.0L || x > static_cast<long double>(std::numeric_limits<uint64_t>::max()))
            throw std::runtime_error("Target out of range: " + item);
        out.push_back(static_cast<uint64_t>(std::llround(x)));
    }
    if (out.empty())
        throw std::runtime_error("No targets supplied");
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

static std::string format_u64(uint64_t x)
{
    std::ostringstream os;
    os << x;
    return os.str();
}

static void write_hist_file(const fs::path& path, const TargetData& t)
{
    std::ofstream out(path, std::ios::binary);
    if (!out)
        throw std::runtime_error("Could not open " + path.string());

    // Raw little-endian doubles, r=1 block first, then r=2,...,r=5.
    // Python postprocessor reads shape (5, grid).
    for (int r = 0; r < RMAX; ++r) {
        out.write(reinterpret_cast<const char*>(t.hist[r].data()),
                  static_cast<std::streamsize>(t.hist[r].size() * sizeof(double)));
        if (!out)
            throw std::runtime_error("Write failed for " + path.string());
    }
}

static void usage(const char* argv0)
{
    std::cerr
        << "Usage:\n  " << argv0
        << " --targets n1,n2[,n3...] --grid 1048576 --out exact_k5_run\n\n"
        << "Example:\n  " << argv0
        << " --targets 112850000000,112860000000,112870000000"
        << " --grid 1048576 --out exact_k5_run\n";
}

int main(int argc, char** argv)
{
    try {
        std::string targets_arg = "112850000000,112860000000,112870000000";
        uint64_t grid = 1048576;
        fs::path outdir = "exact_k5_run";
        uint64_t progress_step = 5000000000ULL;

        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--targets" && i + 1 < argc) {
                targets_arg = argv[++i];
            } else if (a == "--grid" && i + 1 < argc) {
                grid = static_cast<uint64_t>(std::stoull(argv[++i]));
            } else if (a == "--out" && i + 1 < argc) {
                outdir = argv[++i];
            } else if (a == "--progress-step" && i + 1 < argc) {
                progress_step = static_cast<uint64_t>(std::stoull(argv[++i]));
            } else if (a == "--help" || a == "-h") {
                usage(argv[0]);
                return 0;
            } else {
                throw std::runtime_error("Unknown or incomplete argument: " + a);
            }
        }

        if (grid < 1024)
            throw std::runtime_error("Grid is implausibly small");
        if ((grid & (grid - 1)) != 0)
            std::cerr << "Note: grid is not a power of two; postprocessing will still work,\n"
                         "but nested FFT grids are simplest with a power-of-two finest grid.\n";

        auto targets = parse_targets(targets_arg);
        uint64_t max_n = targets.back();

        fs::create_directories(outdir);

        std::vector<TargetData> data;
        data.reserve(targets.size());

        size_t total_hist_bytes = 0;
        for (uint64_t n : targets) {
            data.emplace_back(n, grid);
            total_hist_bytes += data.back().bytes();
        }

        std::cout << "Exact prime-stream K=5,r=3 histogram build\n";
        std::cout << "Targets:";
        for (auto n : targets)
            std::cout << " " << n;
        std::cout << "\nFinest grid: " << grid << "\n";
        std::cout << "Histogram RAM: "
                  << std::fixed << std::setprecision(1)
                  << (static_cast<double>(total_hist_bytes) / (1024.0 * 1024.0))
                  << " MiB\n";
        std::cout << "Prime stream through: " << max_n << "\n\n";

        // Cumulative odd-prime reciprocal power sums.
        std::array<double, RMAX + 1> cumulative{};
        uint64_t odd_prime_count = 0;

        size_t next_snapshot = 0;
        uint64_t next_progress = progress_step;

        auto snapshot_before = [&](uint64_t p) {
            while (next_snapshot < data.size() && data[next_snapshot].n < p) {
                auto& t = data[next_snapshot];
                t.power_sum = cumulative;
                t.odd_prime_count = odd_prime_count;
                t.snapshotted = true;
                std::cout << "Snapshot n=" << t.n
                          << "  odd primes=" << t.odd_prime_count << "\n";
                ++next_snapshot;
            }
        };

        auto start_time = std::chrono::steady_clock::now();

        primesieve_iterator it;
        primesieve_init(&it);
        primesieve_jump_to(&it, 2, max_n);

        while (true) {
            uint64_t p = primesieve_next_prime(&it);
            if (p == PRIMESIEVE_ERROR) {
                primesieve_free_iterator(&it);
                throw std::runtime_error("primesieve iterator error");
            }
            if (p > max_n)
                break;

            snapshot_before(p);

            if (p == 2)
                continue; // Framework is odd-prime only.

            ++odd_prime_count;

            double inv = 1.0 / static_cast<double>(p);
            std::array<double, RMAX + 1> pw{};
            pw[1] = inv;
            for (int r = 2; r <= RMAX; ++r)
                pw[r] = pw[r - 1] * inv;

            for (int r = 1; r <= RMAX; ++r)
                cumulative[r] += pw[r];

            // Update every target that still contains p.
            for (auto& t : data) {
                if (p > t.n)
                    continue;

                for (int r = 1; r <= RMAX; ++r) {
                    uint64_t b = t.tracker[r - 1].locate(p);
                    if (b < grid)
                        t.hist[r - 1][b] += pw[r];
                }
            }

            if (progress_step > 0 && p >= next_progress) {
                auto now = std::chrono::steady_clock::now();
                double sec = std::chrono::duration<double>(now - start_time).count();
                std::cout << "p >= " << next_progress
                          << "  elapsed=" << std::setprecision(1) << sec << " s"
                          << "  odd primes=" << odd_prime_count << "\n";
                while (next_progress <= p &&
                       next_progress <= std::numeric_limits<uint64_t>::max() - progress_step)
                    next_progress += progress_step;
            }
        }

        primesieve_free_iterator(&it);

        // Snapshot targets equal to or above the final processed prime.
        while (next_snapshot < data.size()) {
            auto& t = data[next_snapshot];
            t.power_sum = cumulative;
            t.odd_prime_count = odd_prime_count;
            t.snapshotted = true;
            std::cout << "Snapshot n=" << t.n
                      << "  odd primes=" << t.odd_prime_count << "\n";
            ++next_snapshot;
        }

        // IMPORTANT: cumulative above is valid only for max target. For earlier
        // targets snapshot_before() captured the proper values. Histograms are
        // target-specific and were stopped at p > n automatically.

        fs::path manifest_path = outdir / "manifest.csv";
        std::ofstream manifest(manifest_path);
        if (!manifest)
            throw std::runtime_error("Could not create manifest.csv");

        manifest << "n,grid,odd_prime_count,p1,p2,p3,p4,p5,hist_file\n";
        manifest << std::setprecision(17);

        for (auto& t : data) {
            fs::path hist_name = "hist_n" + format_u64(t.n) + "_G" + format_u64(grid) + ".bin";
            fs::path hist_path = outdir / hist_name;

            std::cout << "Writing " << hist_path.string() << " ...\n";
            write_hist_file(hist_path, t);

            manifest << t.n << ","
                     << grid << ","
                     << t.odd_prime_count;
            for (int r = 1; r <= RMAX; ++r)
                manifest << "," << t.power_sum[r];
            manifest << "," << hist_name.string() << "\n";
        }

        auto end_time = std::chrono::steady_clock::now();
        double total_sec = std::chrono::duration<double>(end_time - start_time).count();

        std::cout << "\nDone.\n";
        std::cout << "Manifest: " << manifest_path.string() << "\n";
        std::cout << "Total elapsed: " << std::setprecision(1) << total_sec << " s\n";
        std::cout << "Now run the Python postprocessor on the output directory.\n";

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
