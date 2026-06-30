#!/usr/bin/env python3
"""
randread_bw / stream_bw core sweep
Runs ./randread_bw or ./stream_bw from 1 core up to CORE_MAX with CORE_STEP increments,
then reports bandwidth and the percentage of theoretical DRAM peak.

Usage:
  python3 sweep_bw.py          # random access (default)
  python3 sweep_bw.py rand     # random access
  python3 sweep_bw.py stream   # sequential access
"""

import re
import subprocess
import sys
import os

# ─── DIMM configuration ───────────────────────────────────────────────────────
# Check with: sudo dmidecode -t memory | grep -E "Speed|Configured"

DIMM_TRANSFER_RATE_MT_S = 5200   # e.g. DDR5-4800 → 4800, DDR4-3200 → 3200
DIMM_CHANNELS           = 1      # number of populated memory channels

# ─── Sweep configuration ─────────────────────────────────────────────────────

CORE_START = 1     # first core count to test
CORE_MAX   = 16    # last  core count to test  (None → detect logical CPUs)
CORE_STEP  = 1     # increment between steps

ITERS_PER_THREAD = 100_000_000   # passed as 2nd arg to the binary

# Set to True if the binary needs sudo to allocate 1GB hugepages
USE_SUDO = False

# Paths to binaries (resolved relative to this script)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BINARY_RAND   = os.path.join(_SCRIPT_DIR, "randread_bw")
BINARY_STREAM = os.path.join(_SCRIPT_DIR, "stream_bw")

# ─────────────────────────────────────────────────────────────────────────────

PHYSICAL_CORES = 16   # SMT warning threshold


def resolve_mode(argv: list[str]) -> tuple[str, str]:
    """Return (mode, binary_path) based on command-line argument."""
    arg = argv[1].lower() if len(argv) > 1 else "rand"
    if arg == "stream":
        return "stream", BINARY_STREAM
    if arg in ("rand", "random"):
        return "rand", BINARY_RAND
    sys.exit(f"Unknown mode '{argv[1]}'. Use: rand (default) or stream")


def theoretical_peak_gb_s() -> float:
    """Peak BW = rate (MT/s) × 8 B/transfer × channels / 1000"""
    return DIMM_TRANSFER_RATE_MT_S * 8 * DIMM_CHANNELS / 1000.0


def parse_bandwidth(output: str) -> tuple[float, float]:
    """Return (elapsed_s, bw_gb_s) from randread_bw stdout."""
    m_elapsed = re.search(r"Elapsed\s*:\s*([\d.]+)\s*s", output)
    m_bw      = re.search(r"Bandwidth\s*:\s*([\d.]+)\s*GB/s", output)
    if not m_elapsed or not m_bw:
        raise ValueError(f"Could not parse output:\n{output}")
    return float(m_elapsed.group(1)), float(m_bw.group(1))


def run_benchmark(ncores: int, binary: str) -> tuple[float, float]:
    """Run the benchmark binary and return (elapsed_s, bw_gb_s)."""
    cmd = []
    if USE_SUDO:
        cmd.append("sudo")
    cmd += [binary, str(ncores), str(ITERS_PER_THREAD)]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"\n[ERROR] core={ncores} failed (rc={result.returncode})", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return None, None
    return parse_bandwidth(result.stdout)


def bar(value: float, max_value: float, width: int = 30) -> str:
    filled = int(round(value / max_value * width)) if max_value > 0 else 0
    return "█" * filled + "░" * (width - filled)


def main() -> None:
    mode, binary = resolve_mode(sys.argv)

    if not os.path.isfile(binary):
        sys.exit(f"Binary not found: {binary}\nRun 'make' first.")

    max_cores = CORE_MAX if CORE_MAX is not None else os.cpu_count()
    peak      = theoretical_peak_gb_s()
    bus_width = 8  # bytes per transfer (DDR standard)

    core_list = list(range(CORE_START, max_cores + 1, CORE_STEP))
    if core_list[-1] != max_cores:
        core_list.append(max_cores)

    mode_label = "random access" if mode == "rand" else "sequential access"
    print("=" * 65)
    print(f" {os.path.basename(binary)} — core sweep  [{mode_label}]")
    print("=" * 65)
    print(f"  DIMM rate   : {DIMM_TRANSFER_RATE_MT_S} MT/s × {bus_width} B × "
          f"{DIMM_CHANNELS} ch  →  peak {peak:.1f} GB/s")
    print(f"  Core range  : {CORE_START} .. {max_cores}  (step {CORE_STEP})")
    print(f"  Iters/thread: {ITERS_PER_THREAD:,}")
    print(f"  Binary      : {binary}")
    print("=" * 65)
    print()

    header = f"{'Cores':>6}  {'Elapsed':>8}  {'BW (GB/s)':>10}  {'% peak':>7}  Chart"
    print(header)
    print("-" * 65)

    results: list[tuple[int, float, float]] = []

    for ncores in core_list:
        smt_tag = " [SMT]" if ncores > PHYSICAL_CORES else ""
        print(f"  Running {ncores:>2} core(s){smt_tag} ... ", end="", flush=True)

        elapsed, bw = run_benchmark(ncores, binary)
        if bw is None:
            print("FAILED")
            continue

        pct = bw / peak * 100.0
        results.append((ncores, elapsed, bw))
        chart = bar(bw, peak)
        print(f"\r{ncores:>6}  {elapsed:>7.2f}s  {bw:>10.3f}  {pct:>6.1f}%  {chart}")

    if not results:
        sys.exit("No successful measurements.")

    print()
    print("=" * 65)
    max_bw    = max(bw for _, _, bw in results)
    max_cores_result, _, _ = max(results, key=lambda x: x[2])
    print(f"  Peak measured : {max_bw:.3f} GB/s at {max_cores_result} core(s)")
    print(f"  % of DRAM peak: {max_bw / peak * 100:.1f}%  "
          f"(theoretical {peak:.1f} GB/s)")
    print("=" * 65)


if __name__ == "__main__":
    main()
