#!/usr/bin/env bash
# Download stage only, with a thread bump once the sweep releases the CPU.
# Kept separate from the manifest stage so neither is edited while running --
# bash reads a script incrementally by byte offset, and editing one mid-run can
# make it execute from the wrong place.
set -u
cd "$(dirname "$0")/.."
log() { echo "STAGE $* $(date -u +%H:%M:%S)"; }

while [ ! -f meta/manifest_metrics.json ]; do sleep 20; done
log "manifest ready"

# Pooled sessions made the thread count matter again: 40 gives 660 files/min
# and 57 gives 904, so run near the cap even beside the sweep -- this stage is
# waiting on the network, and its only CPU is a CRC check on an 11 KB file.
log "download at 48 threads alongside the sweep"
METRICS_THREADS=48 timeout 5h python3 -u data/fetch_metrics.py download

while pgrep -f "cost_sweep\.py" > /dev/null; do sleep 60; done
log "sweep finished, resuming at the full thread budget"
METRICS_THREADS=57 python3 -u data/fetch_metrics.py download || exit 1

log "convert"
python3 -u data/fetch_metrics.py convert || exit 1
log "metrics complete"
