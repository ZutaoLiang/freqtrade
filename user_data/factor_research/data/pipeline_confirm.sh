#!/usr/bin/env bash
# Refresh the funding fields after the backfill, then confirm the round-1
# survivors at 4h and 15m. Sequential; one pool-spawning stage at a time.
set -u
cd "$(dirname "$0")/.."
log() { echo "STAGE $* $(date -u +%H:%M:%S)"; }

while pgrep -f "fetch_funding_ga[p]" > /dev/null; do sleep 15; done
log "funding backfill finished"

log "merge_aux resample"
python3 -u data/merge_aux.py resample || exit 1
log "merge_aux panel"
python3 -u data/merge_aux.py panel 1h 4h 15m 30m 5m 1d || exit 1

for tf in 1h 4h 15m; do
  log "base cache $tf"
  python3 -u research/base.py "$tf" > "logs/base_$tf.log" 2>&1 || exit 1
done

# 1h reruns too: the sweep now emits a Benjamini-Hochberg q-value that the
# existing report predates, and the funding fields have changed underneath it.
for tf in 1h 4h 15m; do
  log "sweep round1 $tf"
  python3 -u research/sweep.py "$tf" --round round1 || exit 1
  log "falsify ts $tf"
  python3 -u research/falsify.py "$tf" --round round1 --gate ts --top 20 || exit 1
  log "falsify cs $tf"
  python3 -u research/falsify.py "$tf" --round round1 --gate cs --top 20 || exit 1
done

log "confirmation pipeline complete"
