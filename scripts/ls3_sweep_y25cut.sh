#!/bin/bash
# Parameter sensitivity y25 rerun: 2025-01-01 .. 2025-10-01 (was 11-01).
cd /root/freqtrade
OUT=/root/freqtrade/user_data/basket_exp/sweep3
export OUT
mkdir -p "$OUT"
run_one() {
  name=${1%.json}
  cfg=/root/freqtrade/user_data/basket_exp/sweep_cfg/$name.json
  freqtrade backtesting --strategy LongShortV3 --config "$cfg" \
    --timerange 20250101-20251001 --timeframe 5m --cache none \
    > "$OUT/$name.log" 2>&1
  echo "done $name exit=$?" >> "$OUT/progress.log"
}
export -f run_one
ls /root/freqtrade/user_data/basket_exp/sweep_cfg/sw_*_y25.json | xargs -n 1 basename | xargs -P 4 -I {} bash -c 'run_one {}'
echo ALLDONE >> "$OUT/progress.log"
