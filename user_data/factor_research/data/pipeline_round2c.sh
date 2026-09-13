#!/usr/bin/env bash
# Finish round 2. Resumable: a stage whose report already exists is skipped,
# because this pipeline has now been interrupted twice by session teardown.
set -u
cd "$(dirname "$0")/.."
log() { echo "STAGE $* $(date -u +%H:%M:%S)"; }

horizon_for() { case "$1" in 1h) echo 24;; 4h) echo 6;; 15m) echo 96;; esac; }

for tf in 1h 4h 15m; do
  h=$(horizon_for "$tf")
  if [ ! -f "reports/sweep_round2_${tf}_train.csv" ]; then
    log "sweep round2 $tf"
    python3 -u research/sweep.py "$tf" --round round2 || exit 1
  else
    log "sweep round2 $tf already on disk, skipping"
  fi
  for gate in ts cs; do
    out="reports/falsify_round2_${gate}_${tf}_train_h${h}_r0.5.csv"
    if [ -f "$out" ]; then
      log "falsify $gate $tf already on disk, skipping"
      continue
    fi
    log "falsify $gate $tf"
    python3 -u research/falsify.py "$tf" --round round2 --gate "$gate" --top 20 || exit 1
  done
done
log "round2 complete"
