#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
log() { echo "STAGE $* $(date -u +%H:%M:%S)"; }
log "walk-forward + CPCV"
python3 -u research/walkforward.py --names reports/shortlist_round2_ts.txt \
    --tfs 1h,4h,15m --folds 8 --blocks 6 --k 2 || exit 1
log "cost and valid-split scoring"
python3 -u research/report_shortlist.py --names reports/shortlist_round2_ts.txt \
    --tfs 1h,4h --splits train,valid || exit 1
log "all complete"
