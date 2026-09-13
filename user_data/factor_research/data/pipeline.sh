#!/bin/bash
# Phase 0 end to end: manifest -> download -> convert -> resample -> audit.
# Every stage is resumable, so re-running after an interruption is safe.
set -u
cd "$(dirname "$0")" || exit 1
LOG=../logs
mkdir -p "$LOG"

if [ ! -f ../meta/manifest.json ]; then
  echo "STAGE manifest start"
  python3 -u build_manifest.py > "$LOG/manifest.log" 2>&1 || { echo "STAGE manifest FAILED"; exit 1; }
  echo "STAGE manifest done: $(tail -2 "$LOG/manifest.log" | tr '\n' ' ')"
fi

echo "STAGE download start"
python3 -u download.py > "$LOG/download.log" 2>&1 || { echo "STAGE download FAILED"; exit 1; }
echo "STAGE download done: $(grep '^done:' "$LOG/download.log")"

echo "STAGE convert start"
python3 -u convert.py > "$LOG/convert.log" 2>&1 || { echo "STAGE convert FAILED"; exit 1; }
echo "STAGE convert done: $(grep '^done:' "$LOG/convert.log")"

echo "STAGE resample start"
python3 -u resample.py > "$LOG/resample.log" 2>&1 || { echo "STAGE resample FAILED"; exit 1; }
echo "STAGE resample done: $(grep '^done:' "$LOG/resample.log")"

for tf in 1h 15m 1d; do
  AUDIT_TF=$tf python3 -u audit.py > "$LOG/audit_$tf.log" 2>&1 \
    && echo "STAGE audit $tf done" || echo "STAGE audit $tf FAILED"
done
echo "STAGE ALL COMPLETE"
