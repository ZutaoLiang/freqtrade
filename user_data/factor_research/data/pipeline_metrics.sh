#!/usr/bin/env bash
# Download metrics for the tradeable universe. Runs at a low thread count while
# the round-3 sweep still holds the CPU, then switches to the full budget the
# moment that finishes -- the download is network-bound and idling it for two
# hours to avoid a contention that does not exist would be pure waste.
set -u
cd "$(dirname "$0")/.."
log() { echo "STAGE $* $(date -u +%H:%M:%S)"; }

log "manifest"
METRICS_THREADS=24 python3 -u data/fetch_metrics.py manifest || exit 1

log "download at 40 threads while round 3 holds the CPU"
# 40, not the full 57: the only CPU this stage spends is the CRC check on an
# 11 KB archive, which is negligible, but the sweep beside it is already at 57
# workers and the point of the shared budget is that nothing oversubscribes.
METRICS_THREADS=40 timeout 4h python3 -u data/fetch_metrics.py download

while pgrep -f "cost_sweep\.py" > /dev/null; do sleep 60; done
log "round 3 finished, resuming download at full thread budget"
METRICS_THREADS=57 python3 -u data/fetch_metrics.py download || exit 1

log "convert"
python3 -u data/fetch_metrics.py convert || exit 1
log "metrics complete"
