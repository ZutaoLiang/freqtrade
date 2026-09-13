#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
log() { echo "STAGE $* $(date -u +%H:%M:%S)"; }
log "merge_aux resample"; python3 -u data/merge_aux.py resample || exit 1
log "merge_aux panel";    python3 -u data/merge_aux.py panel 1h 4h 15m 30m 5m 1d || exit 1
for tf in 1h 4h 15m; do
  log "base cache $tf"; python3 -u research/base.py "$tf" > "logs/base_$tf.log" 2>&1 || exit 1
done
log "refresh complete"
