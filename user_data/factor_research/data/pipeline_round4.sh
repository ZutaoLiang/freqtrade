#!/usr/bin/env bash
# Round 4 under the rule fixed in reports/PREREGISTRATION_round4.md.
set -u
cd "$(dirname "$0")/.."
log() { echo "STAGE $* $(date -u +%H:%M:%S)"; }
log "base cache 4h"
python3 -u research/base.py 4h > logs/base_4h.log 2>&1 || exit 1
for tf in 1h 4h; do
  out="reports/cost_round4_${tf}.csv"
  if [ -f "$out" ]; then log "round4 $tf already on disk, skipping"; continue; fi
  log "cost sweep round4 $tf"
  python3 -u research/cost_sweep.py "$tf" --round round4 --top 30 || exit 1
done
log "round4 cost complete"
