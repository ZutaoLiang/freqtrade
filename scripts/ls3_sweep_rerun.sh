#!/bin/bash
# Rerun sweep configs that silently failed; full output logged per config.
cd /root/freqtrade
OUT=/root/freqtrade/user_data/basket_exp/sweep2
mkdir -p "$OUT"
run_one() {
  name=$1
  cfg=/root/freqtrade/user_data/basket_exp/sweep_cfg/$name.json
  case "$name" in
    *y25*) tr=20250101-20251101 ;;
    *)     tr=20260101-20260817 ;;
  esac
  freqtrade backtesting --strategy LongShortV3 --config "$cfg" \
    --timerange "$tr" --timeframe 5m --cache none \
    > "$OUT/$name.log" 2>&1
  echo "done $name exit=$?" >> "$OUT/progress.log"
}
export -f run_one
cat /tmp/missing.txt | xargs -P 4 -I {} bash -c 'run_one {}'
echo ALLDONE >> "$OUT/progress.log"
