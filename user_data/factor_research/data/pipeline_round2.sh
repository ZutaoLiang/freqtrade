#!/usr/bin/env bash
# Round 2: 520 specs at 1h, then confirm whatever survives at 4h and 15m.
set -u
cd "$(dirname "$0")/.."
log() { echo "STAGE $* $(date -u +%H:%M:%S)"; }
for tf in 1h 4h 15m; do
  log "sweep round2 $tf"
  python3 -u research/sweep.py "$tf" --round round2 || exit 1
  log "falsify ts $tf"
  python3 -u research/falsify.py "$tf" --round round2 --gate ts --top 25 || exit 1
  log "falsify cs $tf"
  python3 -u research/falsify.py "$tf" --round round2 --gate cs --top 25 || exit 1
done
log "round2 pipeline complete"
