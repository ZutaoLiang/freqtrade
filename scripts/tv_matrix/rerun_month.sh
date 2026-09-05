#!/usr/bin/env bash
# Re-freeze one month's pool against the current archive and rerun its 24 units.
# Usage: rerun_month.sh <label> <bt-start> <bt-end> [workers]
set -euo pipefail
LABEL="$1"; START="$2"; END="$3"; WORKERS="${4:-10}"
UNI="user_data/research/tradingview/universe_${LABEL}.json"

python scripts/tv_matrix/select_universe.py \
    --bt-start "$START" --bt-end "$END" --out "$UNI" --workers 16
python scripts/tv_matrix/build_datadir.py \
    --universe "$UNI" --out user_data/data/tv_matrix/futures
python scripts/tv_matrix/run_year.py \
    --year "${LABEL%%-*}" --months "$((10#${LABEL##*-}))" \
    --data-end "$END" --workers "$WORKERS" --skip-select \
    --tag "trendshift_${LABEL}"
