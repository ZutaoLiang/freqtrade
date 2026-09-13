#!/usr/bin/env bash
# Finish round 2 at 15m under the fixed correlation sampling, then run the
# validation chain. Resumable: existing reports are skipped.
set -u
cd "$(dirname "$0")/.."
log() { echo "STAGE $* $(date -u +%H:%M:%S)"; }

for gate in ts cs; do
  out="reports/falsify_round2_${gate}_15m_train_h96_r0.5.csv"
  if [ -f "$out" ]; then log "falsify $gate 15m already on disk, skipping"; continue; fi
  log "falsify $gate 15m"
  python3 -u research/falsify.py 15m --round round2 --gate "$gate" --top 20 || exit 1
done

log "build union shortlist"
python3 - <<'PY' || exit 1
import glob, pandas as pd
names = set()
for f in glob.glob("reports/falsify_round2_ts_*_train_h*_r0.5.csv"):
    d = pd.read_csv(f)
    names |= set(d[d["survives"]]["name"])
names = sorted(names)
open("reports/shortlist_round2_ts.txt", "w").write("\n".join(names) + "\n")
print(f"{len(names)} distinct survivors across timeframes")
PY

for tf in 1h 4h 15m; do
  log "falsify --only $tf"
  python3 -u research/falsify.py "$tf" --round round2 --gate ts \
      --only reports/shortlist_round2_ts.txt || exit 1
done

log "walk-forward + CPCV"
python3 -u research/walkforward.py --names reports/shortlist_round2_ts.txt \
    --tfs 1h,4h --folds 8 --blocks 6 --k 2 || exit 1

log "cost and valid-split scoring"
python3 -u research/report_shortlist.py --names reports/shortlist_round2_ts.txt \
    --tfs 1h,4h --splits train,valid || exit 1

log "all complete"
