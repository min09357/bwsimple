# config.py — benchmark configuration
# Edit this file to tune hardware/sweep settings before running sweep_bw.py.

# ─── DIMM configuration ───────────────────────────────────────────────────────
# Check with: sudo dmidecode -t memory | grep -E "Speed|Configured"

DIMM_TRANSFER_RATE_MT_S = 4800   # e.g. DDR5-4800 → 4800, DDR4-3200 → 3200
DIMM_CHANNELS           = 1      # number of populated memory channels

# ─── NUMA configuration ──────────────────────────────────────────────────────
# Which NUMA node to use for both CPU affinity (-C) and memory allocation (-m).
# Check topology with: numactl --hardware

NUMA_NODE = 1

# ─── Hugepage configuration ───────────────────────────────────────────────────
# Number of 1GB hugepages to allocate as the benchmark region.
# - For randread_bw: must be a power of 2 (1, 2, 4, 8, ...) due to LFSR masking.
# - For stream_bw: any positive integer is fine.
# Check available pages with: grep HugePages /proc/meminfo
# Allocate with: sudo sh -c 'echo N > /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages'

HUGEPAGES_1GB = 1

# ─── Sweep configuration ─────────────────────────────────────────────────────

CORE_START = 1     # first core count to test
CORE_MAX   = 32    # last  core count to test  (None → use all CPUs in NUMA_NODE)
CORE_STEP  = 1     # increment between steps

ITERS_PER_THREAD = 100_000_000   # passed as 2nd arg to the binary

# ─── Misc ────────────────────────────────────────────────────────────────────

# Set to True if the binary needs sudo to allocate 1GB hugepages
USE_SUDO = False
