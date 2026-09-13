"""Shared resource ceilings for every stage in this project.

Deliberately hardcoded rather than derived from os.cpu_count(): this WSL box
reports 80 CPUs to nproc but actually has 64, so auto-detection overshoots.
The cap is a budget shared across ALL concurrently running jobs, not per job --
three 72-worker pools running at once is what froze the machine on 2026-08-18.
"""
import os

N_CPU = 64
MAX_WORKERS = 57          # 90% of 64; never saturate
DOWNLOAD_THREADS = 57     # network bound, but shares the same budget


def atomic_save(path, write_fn):
    """Write via a temp file, fsync, then rename.

    A plain write leaves a truncated file that still looks present if the box
    dies mid-flush -- exactly how the 5m panel's taker_buy_volume ended up at
    96 MiB of an expected 560 MiB, with its meta.json already on disk.
    """
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        write_fn(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# Measured peak RSS for one operator call on a full panel (2026-08-19). ts_corr
# is the worst case: it holds x, y, their product and four rolling sums at once.
#   1h  12.2M cells   ~1.1 GB per worker
#   5m 146.9M cells  ~12.0 GB per worker
# Memory, not CPU, is what binds at short timeframes -- 57 workers on the 5m
# panel would ask for roughly 680 GB against 62 GB of RAM.
RAM_BUDGET_GB = 55          # leave headroom below the 62 GB the box reports
PEAK_GB_PER_WORKER = {      # per operator call, by timeframe
    "1d": 0.05, "4h": 0.3, "1h": 1.1, "30m": 2.2, "15m": 4.3, "5m": 12.0, "1m": 60.0,
}


# A sweep task is not the same shape as a bare operator call, and measuring it
# rather than reusing the table above matters: at 1h, 24 sweep workers sat at
# 0.58 GB each (2026-08-22), half the 1.1 GB the operator table assumes. The
# sweep reads its panels through mmap and slices to the train split before it
# computes anything, so the resident set is far smaller than a full-panel
# operator call. Scaled by train bars from that measurement; the budget is
# lower than RAM_BUDGET_GB because the mmap'd panels want page cache too.
SWEEP_RAM_BUDGET_GB = 45
SWEEP_GB_PER_WORKER = {
    "1d": 0.03, "4h": 0.15, "1h": 0.58, "30m": 1.15, "15m": 2.3, "5m": 7.0, "1m": 28.0,
}


def workers_for(tf, task="operator"):
    """Worker count that respects both the CPU cap and the RAM ceiling."""
    if task == "sweep":
        per = SWEEP_GB_PER_WORKER.get(tf)
        budget = SWEEP_RAM_BUDGET_GB
    else:
        per = PEAK_GB_PER_WORKER.get(tf)
        budget = RAM_BUDGET_GB
    if per is None:
        return MAX_WORKERS
    return max(1, min(MAX_WORKERS, int(budget / per)))
