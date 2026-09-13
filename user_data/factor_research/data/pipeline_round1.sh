#!/usr/bin/env bash
# Round 1 with the corrected statistics, then the funding backfill.
# Sequential on purpose: only one pool-spawning stage at a time.
set -u
cd "$(dirname "$0")/.."
log() { echo "STAGE $* $(date -u +%H:%M:%S)"; }

log "sweep round1 1h"
python3 -u research/sweep.py 1h --round round1 --workers 24 --draws 200 || exit 1

log "falsify ts gate"
python3 -u research/falsify.py 1h --round round1 --gate ts --top 20 || exit 1

log "falsify cs gate"
python3 -u research/falsify.py 1h --round round1 --gate cs --top 20 || exit 1

log "funding backfill"
python3 -u data/fetch_funding_gap.py || exit 1

log "round1 pipeline complete"
