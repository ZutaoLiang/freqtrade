#!/usr/bin/env bash
# Round 3 selected on cost, not on rank IC. The IC gate is skipped entirely --
# it demonstrably selects factors that do not survive trading costs.
set -u
cd "$(dirname "$0")/.."
log() { echo "STAGE $* $(date -u +%H:%M:%S)"; }
for tf in 1h 4h; do
  out="reports/cost_round3_${tf}.csv"
  if [ -f "$out" ]; then log "round3 $tf already on disk, skipping"; continue; fi
  log "cost sweep round3 $tf"
  python3 -u research/cost_sweep.py "$tf" --round round3 --top 40 || exit 1
done
log "round3 cost complete"
