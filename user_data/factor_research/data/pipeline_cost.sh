#!/usr/bin/env bash
# Cost-first selection: sweep every round-2 spec on break-even and P&L t-stat,
# then put whatever clears both splits through the same forward-in-time and
# CPCV tests the IC survivors went through.
set -u
cd "$(dirname "$0")/.."
log() { echo "STAGE $* $(date -u +%H:%M:%S)"; }

for tf in 1h 4h; do
  log "cost sweep round2 $tf"
  python3 -u research/cost_sweep.py "$tf" --round round2 --top 30 || exit 1
done

log "build cost-selected shortlist"
python3 - <<'PY' || exit 1
import pandas as pd, glob
names = set()
for f in glob.glob("reports/cost_round2_*.csv"):
    d = pd.read_csv(f)
    tr = d[d.split == "train"].set_index("name")
    va = d[d.split == "valid"].set_index("name")
    j = tr[["breakeven_bps", "tail_consistent", "pnl_t_nw"]].join(
        va[["breakeven_bps", "tail_consistent", "pnl_t_nw"]],
        lsuffix="_tr", rsuffix="_va")
    ok = j[(j.breakeven_bps_tr > 0) & (j.breakeven_bps_va > 0) &
           j.tail_consistent_tr & j.tail_consistent_va &
           (j.pnl_t_nw_tr > 2.0) & (j.pnl_t_nw_va > 2.0)]
    names |= set(ok.index)
names = sorted(names)
open("reports/shortlist_cost.txt", "w").write("\n".join(names) + "\n")
print(f"{len(names)} specs clear cost and significance in both splits")
PY

log "walk-forward + CPCV on the cost-selected list"
python3 -u research/walkforward.py --names reports/shortlist_cost.txt \
    --tfs 1h,4h --folds 8 --blocks 6 --k 2 \
    --out reports/walkforward_cost.csv || exit 1

log "cost pipeline complete"
