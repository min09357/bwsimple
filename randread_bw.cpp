// randread_bw.cpp
// Measures random-access read bandwidth (GB/s) over a 1GB-hugepage region.
// Uses HPCC RandomAccess POLY LFSR addressing with UNROLL independent streams
// per thread to maximise MLP (multiple outstanding cacheline loads).
//
// Usage: ./randread_bw [ncores] [iters_per_thread] [hugepages_1gb]
//        e.g.: ./randread_bw 16 100000000 2   (default: ncores=16, iters=100000000, hugepages=2)
// Note: hugepages_1gb must be a power of 2 (LFSR mask addressing requirement).

#include <barrier>
#include <thread>
#include <chrono>
#include <atomic>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <immintrin.h>
#include <sys/mman.h>
#include "bw_width.h"

#ifndef MAP_HUGE_SHIFT
#define MAP_HUGE_SHIFT 26
#endif
#ifndef MAP_HUGE_1GB
#define MAP_HUGE_1GB (30 << MAP_HUGE_SHIFT)
#endif

// HPCC RandomAccess LFSR constants  (from RandomAccess.h)
static constexpr uint64_t POLY   = 0x0000000000000007ULL;
static constexpr int64_t  PERIOD = 1317624576693539401LL;

static constexpr uint64_t GB = 1ULL << 30;

static constexpr int UNROLL = 16;                 // independent RNG streams per thread

// Skip-ahead: returns the n-th value of the HPCC POLY LFSR sequence.
// Copied from hpcc/RandomAccess/utility.c; adapted to use uint64_t/int64_t.
static uint64_t hpcc_starts(int64_t n) {
    uint64_t m2[64], temp, ran;

    while (n < 0)      n += PERIOD;
    while (n > PERIOD) n -= PERIOD;
    if (n == 0) return 0x1ULL;

    temp = 0x1ULL;
    for (int i = 0; i < 64; i++) {
        m2[i] = temp;
        temp = (temp << 1) ^ (static_cast<int64_t>(temp) < 0 ? POLY : 0ULL);
        temp = (temp << 1) ^ (static_cast<int64_t>(temp) < 0 ? POLY : 0ULL);
    }

    int i;
    for (i = 62; i >= 0; i--)
        if ((n >> i) & 1) break;

    ran = 0x2ULL;
    while (i > 0) {
        temp = 0;
        for (int j = 0; j < 64; j++)
            if ((ran >> j) & 1) temp ^= m2[j];
        ran = temp;
        i--;
        if ((n >> i) & 1)
            ran = (ran << 1) ^ (static_cast<int64_t>(ran) < 0 ? POLY : 0ULL);
    }
    return ran;
}

// Thread 0 stores the measurement start time immediately after the barrier.
static std::atomic<std::chrono::steady_clock::time_point> g_t_start;

struct ThreadArg {
    int      tid;
    uint8_t *base;
    int64_t  L;          // iterations per stream  (= iters_per_thread / UNROLL)
    uint64_t mask;       // cacheline index mask (= ncl - 1)
    uint64_t sink_out;   // accumulated read data; returned to main to prevent DCE
};

static void thread_func(ThreadArg &a, std::barrier<> &bar) {
    // CPU affinity is set externally by numactl -C / -m before process launch.

    // Initialise UNROLL independent LFSR streams via skip-ahead.
    // Global stream g = tid*UNROLL + u; each advances L steps → start at g*L.
    uint64_t ran[UNROLL];
    for (int u = 0; u < UNROLL; u++)
        ran[u] = hpcc_starts(static_cast<int64_t>(a.tid * UNROLL + u) * a.L);

    // Wait until all threads are ready, then start simultaneously.
    bar.arrive_and_wait();

    if (a.tid == 0)
        g_t_start.store(std::chrono::steady_clock::now(),
                        std::memory_order_release);

    // --- Measurement loop ---
    const uint8_t  *base = a.base;
    const int64_t   L    = a.L;
    const uint64_t  mask = a.mask;

#if BW_SIMD_WIDTH == 512
    __m512i sink = _mm512_setzero_si512();
#elif BW_SIMD_WIDTH == 256
    __m256i sink = _mm256_setzero_si256();
#else
    uint64_t sink = 0;
#endif

    for (int64_t n = 0; n < L; n++) {
        for (int u = 0; u < UNROLL; u++) {
            // Galois LFSR 1 step.  Logic: ran = (ran<<1) ^ (ran<0 ? POLY : 0)
            // branchless: bit63 → mask (0 or all-ones) → select POLY  (no branch/cmov)
            uint64_t mv = static_cast<uint64_t>(0) - (ran[u] >> 63);
            ran[u] = (ran[u] << 1) ^ (mv & POLY);

            size_t idx = ran[u] & mask;    // cacheline index in [0, ncl)
#if BW_SIMD_WIDTH == 512
            __m512i v = _mm512_load_si512(
                reinterpret_cast<const __m512i *>(base + idx * 64));
            sink = _mm512_xor_si512(sink, v);  // accumulate to prevent DCE
#elif BW_SIMD_WIDTH == 256
            __m256i v = _mm256_load_si256(
                reinterpret_cast<const __m256i *>(base + idx * 64));
            sink = _mm256_xor_si256(sink, v);  // accumulate to prevent DCE
#else
            uint64_t v;
            std::memcpy(&v, base + idx * 64, sizeof(v));
            sink ^= v;                         // accumulate to prevent DCE
#endif
        }
    }

    // Reduce sink to 64-bit for the caller
#if BW_SIMD_WIDTH == 512
    alignas(64) uint64_t buf[8];
    _mm512_store_si512(reinterpret_cast<__m512i *>(buf), sink);
    uint64_t s = 0;
    for (int i = 0; i < 8; i++) s ^= buf[i];
    a.sink_out = s;
#elif BW_SIMD_WIDTH == 256
    alignas(32) uint64_t buf[4];
    _mm256_store_si256(reinterpret_cast<__m256i *>(buf), sink);
    uint64_t s = 0;
    for (int i = 0; i < 4; i++) s ^= buf[i];
    a.sink_out = s;
#else
    a.sink_out = sink;
#endif
}

int main(int argc, char *argv[]) {
    // Probe mode: print the compiled SIMD width and exit (used by sweep_bw.py).
    if (argc > 1 && std::strcmp(argv[1], "--simd") == 0) {
        std::printf("%s\n", BW_SIMD_WIDTH_STR);
        return 0;
    }

    int     ncores           = (argc > 1) ? std::atoi(argv[1]) : 16;
    int64_t iters_per_thread = (argc > 2) ? std::atoll(argv[2]) : 100'000'000LL;
    int     hugepages_1gb    = (argc > 3) ? std::atoi(argv[3]) : 2;

    if (ncores < 1 || ncores > 32) {
        std::fprintf(stderr, "error: ncores must be in [1, 32]\n");
        return 1;
    }
    if (ncores > 16)
        std::fprintf(stderr,
            "warning: ncores=%d > 16 physical cores — SMT siblings will be used\n",
            ncores);
    if (hugepages_1gb < 1) {
        std::fprintf(stderr, "error: hugepages_1gb must be >= 1\n");
        return 1;
    }
    // LFSR mask addressing requires NCL (= hugepages_1gb * GB / 64) to be a power of 2,
    // which holds iff hugepages_1gb is a power of 2.
    if ((hugepages_1gb & (hugepages_1gb - 1)) != 0) {
        std::fprintf(stderr,
            "error: hugepages_1gb=%d must be a power of 2 (1,2,4,8,...) for LFSR mask addressing\n",
            hugepages_1gb);
        return 1;
    }

    uint64_t total = static_cast<uint64_t>(hugepages_1gb) * GB;
    uint64_t ncl   = total / 64;   // number of cachelines
    uint64_t mask  = ncl - 1;

    iters_per_thread = (iters_per_thread / UNROLL) * UNROLL;  // round to UNROLL multiple
    int64_t L = iters_per_thread / UNROLL;

    std::printf("ncores=%d  iters/thread=%lld  streams=%d  region=%d GB  simd=%s\n",
        ncores, (long long)iters_per_thread,
        ncores * UNROLL, hugepages_1gb, BW_SIMD_WIDTH_STR);

    // Allocate hugepages_1gb × 1GB hugepages
    void *base_ptr = mmap(nullptr, total,
                          PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_1GB,
                          -1, 0);
    if (base_ptr == MAP_FAILED) {
        std::perror("mmap 1GB hugepages failed");
        std::fprintf(stderr,
            "hint: check 'grep HugePages /proc/meminfo'  (need HugePages_Free >= %d)\n",
            hugepages_1gb);
        return 1;
    }

    // Warm up: sequential write to commit physical pages before read-only measurement
    std::printf("Warming up %d GB ... ", hugepages_1gb);
    std::fflush(stdout);
    std::memset(base_ptr, 0x5A, total);
    std::printf("done\n");

    // Per-thread arguments
    std::vector<ThreadArg> args(ncores);
    for (int i = 0; i < ncores; i++)
        args[i] = { i, static_cast<uint8_t *>(base_ptr), L, mask, 0 };

    std::barrier<> bar(ncores);

    // Launch threads
    std::vector<std::jthread> threads;
    threads.reserve(ncores);
    for (int i = 0; i < ncores; i++)
        threads.emplace_back([&args, &bar, i] { thread_func(args[i], bar); });

    threads.clear();  // jthread destructor joins each thread

    auto t_end   = std::chrono::steady_clock::now();
    auto t_start = g_t_start.load(std::memory_order_acquire);
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    // XOR all sinks so the compiler cannot elide the loads
    uint64_t total_sink = 0;
    for (int i = 0; i < ncores; i++) total_sink ^= args[i].sink_out;

    double total_bytes = static_cast<double>(ncores) * iters_per_thread * 64.0;
    double total_iters = static_cast<double>(ncores) * iters_per_thread;
    double bw_GBs  = total_bytes / elapsed / 1e9;
    double bw_GiBs = total_bytes / elapsed / static_cast<double>(1ULL << 30);
    double gups    = total_iters / elapsed / 1e9;

    std::printf("\n=== Results ===\n");
    std::printf("Elapsed        : %.3f s\n", elapsed);
    std::printf("Total accesses : %.3e  (%.3e bytes)\n", total_iters, total_bytes);
    std::printf("Bandwidth      : %.3f GB/s  (%.3f GiB/s)\n", bw_GBs, bw_GiBs);
    std::printf("GUPS           : %.4f\n", gups);
    std::printf("Checksum       : %016llx\n", (unsigned long long)total_sink);

    munmap(base_ptr, total);
    return 0;
}
