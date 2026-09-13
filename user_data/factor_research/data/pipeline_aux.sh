#!/usr/bin/env bash
# Aux data -> panels -> base series -> round 1 sweep -> falsification.
# Strictly sequential: only one pool-spawning stage may run at a time.
set -u
cd "$(dirname "$0")/.."
log() { echo "STAGE $* $(date -u +%H:%M:%S)"; }

while pgrep -f "fetch_funding_ga[p]" > /dev/null; do sleep 15; done
log "funding gap complete"

log "merge_aux resample"
python3 -u data/merge_aux.py resample || exit 1

log "merge_aux panel"
python3 -u data/merge_aux.py panel 1h 4h 1d 30m 15m 5m || exit 1

for tf in 1h 4h; do
  log "base cache $tf"
  python3 -u research/base.py "$tf" || exit 1
done

log "sweep round1 1h"
python3 -u research/sweep.py 1h --round round1 --workers 24 || exit 1

log "falsify round1 1h"
python3 -u research/falsify.py 1h --round round1 --top 20 || exit 1

log "pipeline complete"
