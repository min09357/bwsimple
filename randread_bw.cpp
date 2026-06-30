// randread_bw.cpp
// Measures random-access read bandwidth (GB/s) over a 2GB 1GB-hugepage region.
// Uses HPCC RandomAccess POLY LFSR addressing with UNROLL independent streams
// per thread to maximise MLP (multiple outstanding cacheline loads).
//
// Usage: ./randread_bw [ncores] [iters_per_thread]
//        e.g.: ./randread_bw 16 100000000   (default: ncores=16, iters=100000000)

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
#include <pthread.h>

#ifndef MAP_HUGE_SHIFT
#define MAP_HUGE_SHIFT 26
#endif
#ifndef MAP_HUGE_1GB
#define MAP_HUGE_1GB (30 << MAP_HUGE_SHIFT)
#endif

// HPCC RandomAccess LFSR constants  (from RandomAccess.h)
static constexpr uint64_t POLY   = 0x0000000000000007ULL;
static constexpr int64_t  PERIOD = 1317624576693539401LL;

static constexpr uint64_t GB    = 1ULL << 30;
static constexpr uint64_t TOTAL = 2ULL * GB;     // 2GB = 2 x 1GB hugepages
static constexpr uint64_t NCL   = TOTAL / 64;    // 2^25 cachelines
static constexpr uint64_t MASK  = NCL - 1;

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
    uint64_t sink_out;   // accumulated read data; returned to main to prevent DCE
};

static void thread_func(ThreadArg &a, std::barrier<> &bar) {
    // Pin to CPU = tid (CPU 0..15 = distinct physical cores on 7950X)
    cpu_set_t cs;
    CPU_ZERO(&cs);
    CPU_SET(a.tid, &cs);
    pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);

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
    __m512i sink = _mm512_setzero_si512();
    const uint8_t *base = a.base;
    const int64_t  L    = a.L;

    for (int64_t n = 0; n < L; n++) {
        for (int u = 0; u < UNROLL; u++) {
            // Galois LFSR 1 step.  Logic: ran = (ran<<1) ^ (ran<0 ? POLY : 0)
            // branchless: bit63 → mask (0 or all-ones) → select POLY  (no branch/cmov)
            uint64_t mv = static_cast<uint64_t>(0) - (ran[u] >> 63);
            ran[u] = (ran[u] << 1) ^ (mv & POLY);

            size_t idx = ran[u] & MASK;    // cacheline index in [0, NCL)
            __m512i v = _mm512_load_si512(
                reinterpret_cast<const __m512i *>(base + idx * 64));
            sink = _mm512_xor_si512(sink, v);  // accumulate to prevent DCE
        }
    }

    // Reduce 512-bit sink to 64-bit for the caller
    alignas(64) uint64_t buf[8];
    _mm512_store_si512(reinterpret_cast<__m512i *>(buf), sink);
    uint64_t s = 0;
    for (int i = 0; i < 8; i++) s ^= buf[i];
    a.sink_out = s;
}

int main(int argc, char *argv[]) {
    int     ncores           = (argc > 1) ? std::atoi(argv[1]) : 16;
    int64_t iters_per_thread = (argc > 2) ? std::atoll(argv[2]) : 100'000'000LL;

    if (ncores < 1 || ncores > 32) {
        std::fprintf(stderr, "error: ncores must be in [1, 32]\n");
        return 1;
    }
    if (ncores > 16)
        std::fprintf(stderr,
            "warning: ncores=%d > 16 physical cores — SMT siblings will be used\n",
            ncores);

    iters_per_thread = (iters_per_thread / UNROLL) * UNROLL;  // round to UNROLL multiple
    int64_t L = iters_per_thread / UNROLL;

    std::printf("ncores=%d  iters/thread=%lld  streams=%d  region=%llu GB\n",
        ncores, (long long)iters_per_thread,
        ncores * UNROLL, (unsigned long long)(TOTAL / GB));

    // Allocate 2GB using two contiguous 1GB hugepages
    void *base_ptr = mmap(nullptr, TOTAL,
                          PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_1GB,
                          -1, 0);
    if (base_ptr == MAP_FAILED) {
        std::perror("mmap 1GB hugepages failed");
        std::fprintf(stderr,
            "hint: check 'grep HugePages /proc/meminfo'  (need HugePages_Free >= 2)\n");
        return 1;
    }

    // Warm up: sequential write to commit physical pages before read-only measurement
    std::printf("Warming up 2 GB ... ");
    std::fflush(stdout);
    std::memset(base_ptr, 0x5A, TOTAL);
    std::printf("done\n");

    // Per-thread arguments
    std::vector<ThreadArg> args(ncores);
    for (int i = 0; i < ncores; i++)
        args[i] = { i, static_cast<uint8_t *>(base_ptr), L, 0 };

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

    munmap(base_ptr, TOTAL);
    return 0;
}
