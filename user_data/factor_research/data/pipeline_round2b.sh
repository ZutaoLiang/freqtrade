#!/usr/bin/env bash
# Re-run the 1h round-2 falsification under the fixed decorrelate-then-truncate
# order, then finish round 2 at 4h and 15m.
set -u
cd "$(dirname "$0")/.."
log() { echo "STAGE $* $(date -u +%H:%M:%S)"; }

for gate in ts cs; do
  log "falsify $gate 1h (fixed ordering)"
  python3 -u research/falsify.py 1h --round round2 --gate "$gate" --top 20 || exit 1
done

for tf in 4h 15m; do
  log "sweep round2 $tf"
  python3 -u research/sweep.py "$tf" --round round2 || exit 1
  for gate in ts cs; do
    log "falsify $gate $tf"
    python3 -u research/falsify.py "$tf" --round round2 --gate "$gate" --top 20 || exit 1
  done
done
log "round2 complete"
