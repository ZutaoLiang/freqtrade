#!/usr/bin/env bash
# After round 2 finishes: build one shortlist, then confirm it three ways --
# same candidates at every timeframe, forward-in-time out-of-sample, and net of
# trading cost. The holdout is not touched by any stage here.
set -u
cd "$(dirname "$0")/.."
log() { echo "STAGE $* $(date -u +%H:%M:%S)"; }

while pgrep -f "pipeline_round2[c]" > /dev/null; do sleep 30; done
log "round 2 finished"

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

# Same candidates asked of every timeframe, so a blank means "failed", never
# "was crowded out of this timeframe's top-N".
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

log "validation pipeline complete"
