#!/usr/bin/env bash
# Re-run round 3 with funding cash flow included in the P&L. The earlier run is
# kept as *_nofunding.csv: it is the record of what the omission implied, and
# the two together are the measurement of how much it mattered.
set -u
cd "$(dirname "$0")/.."
log() { echo "STAGE $* $(date -u +%H:%M:%S)"; }
for tf in 1h 4h; do
  log "cost sweep round3 $tf (funding included)"
  python3 -u research/cost_sweep.py "$tf" --round round3 --top 30 || exit 1
done
log "round3 refund complete"
