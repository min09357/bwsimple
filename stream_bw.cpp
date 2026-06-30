// stream_bw.cpp
// Measures sequential read bandwidth (GB/s) over a 2GB 1GB-hugepage region.
// Each thread scans its own contiguous sub-region sequentially (cache-line stride),
// wrapping around until iters_per_thread accesses are done.
// Companion to randread_bw.cpp — same harness, sequential vs random access pattern.
//
// Usage: ./stream_bw [ncores] [iters_per_thread]
//        e.g.: ./stream_bw 16 100000000   (default: ncores=16, iters=100000000)

#include <algorithm>
#include <barrier>
#include <thread>
#include <chrono>
#include <atomic>
#include <vector>
#include <cstdint>
#include <cstdlib>
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

static constexpr uint64_t GB    = 1ULL << 30;
static constexpr uint64_t TOTAL = 2ULL * GB;   // 2GB = 2 x 1GB hugepages
static constexpr uint64_t NCL   = TOTAL / 64;  // number of 64B cache lines

// Thread 0 stores the measurement start time immediately after the barrier.
static std::atomic<std::chrono::steady_clock::time_point> g_t_start;

struct ThreadArg {
    int      tid;
    int      ncores;
    uint8_t *base;
    int64_t  iters_per_thread;
    uint64_t sink_out;
};

static void thread_func(ThreadArg &a, std::barrier<> &bar) {
    // Pin to CPU = tid (CPU 0..15 = distinct physical cores on 7950X)
    cpu_set_t cs;
    CPU_ZERO(&cs);
    CPU_SET(a.tid, &cs);
    pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);

    // Divide the region into contiguous per-thread sub-ranges [cl_begin, cl_end).
    uint64_t cl_per_thread = NCL / static_cast<uint64_t>(a.ncores);
    uint64_t cl_begin      = static_cast<uint64_t>(a.tid) * cl_per_thread;
    uint64_t cl_end        = (a.tid == a.ncores - 1) ? NCL : cl_begin + cl_per_thread;

    // Wait until all threads are ready, then start simultaneously.
    bar.arrive_and_wait();

    if (a.tid == 0)
        g_t_start.store(std::chrono::steady_clock::now(),
                        std::memory_order_release);

    // --- Measurement loop ---
    // Outer loop handles wrap-around; inner loop is a branchless sequential scan
    // so the compiler can freely vectorize and unroll it.
    __m512i sink     = _mm512_setzero_si512();
    int64_t remaining = a.iters_per_thread;

    while (remaining > 0) {
        uint64_t run = std::min<uint64_t>(cl_end - cl_begin,
                                          static_cast<uint64_t>(remaining));
        const uint8_t *p = a.base + cl_begin * 64;
        for (uint64_t i = 0; i < run; i++) {
            __m512i v = _mm512_load_si512(reinterpret_cast<const __m512i *>(p));
            sink = _mm512_xor_si512(sink, v);  // accumulate to prevent DCE
            p += 64;
        }
        remaining -= static_cast<int64_t>(run);
    }

    // Reduce 512-bit sink to 64-bit for the caller.
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

    std::printf("ncores=%d  iters/thread=%lld  region=%llu GB  pattern=sequential\n",
        ncores, (long long)iters_per_thread,
        (unsigned long long)(TOTAL / GB));

    // Allocate 2GB using two contiguous 1GB hugepages.
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

    // Warm up: position-dependent fill commits physical pages and gives a
    // meaningful checksum (uniform memset would always yield checksum=0).
    std::printf("Warming up 2 GB ... ");
    std::fflush(stdout);
    uint64_t *p = static_cast<uint64_t *>(base_ptr);
    for (uint64_t k = 0; k < TOTAL / 8; k++)
        p[k] = k * 0x9E3779B97F4A7C15ULL;
    std::printf("done\n");

    // Per-thread arguments.
    std::vector<ThreadArg> args(ncores);
    for (int i = 0; i < ncores; i++)
        args[i] = { i, ncores, static_cast<uint8_t *>(base_ptr), iters_per_thread, 0 };

    std::barrier<> bar(ncores);

    // Launch threads.
    std::vector<std::jthread> threads;
    threads.reserve(ncores);
    for (int i = 0; i < ncores; i++)
        threads.emplace_back([&args, &bar, i] { thread_func(args[i], bar); });

    threads.clear();  // jthread destructor joins each thread

    auto t_end   = std::chrono::steady_clock::now();
    auto t_start = g_t_start.load(std::memory_order_acquire);
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    // XOR all sinks so the compiler cannot elide the loads.
    uint64_t total_sink = 0;
    for (int i = 0; i < ncores; i++) total_sink ^= args[i].sink_out;

    double total_bytes = static_cast<double>(ncores) * static_cast<double>(iters_per_thread) * 64.0;
    double total_iters = static_cast<double>(ncores) * static_cast<double>(iters_per_thread);
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
