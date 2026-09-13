#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
log() { echo "STAGE $* $(date -u +%H:%M:%S)"; }
log "funding backfill"
python3 -u data/fetch_funding_gap.py || exit 1
exec bash data/pipeline_confirm.sh
