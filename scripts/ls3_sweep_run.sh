#!/bin/bash
# Parallel parameter sweep for LongShortV3 R3b robustness test.
cd /root/freqtrade
LOG=/root/freqtrade/user_data/basket_exp/sweep_par.log
run_one() {
  cfg=$1
  name=$(basename "$cfg" .json)
  case "$name" in
    *y25*) tr=20250101-20251101 ;;
    *)     tr=20260101-20260817 ;;
  esac
  out=$(freqtrade backtesting --strategy LongShortV3 --config "$cfg" --timerange "$tr" --timeframe 5m --cache none 2>&1 | grep -E "│ (Total profit|Sharpe|Sortino|Absolute drawdown|Total/Daily Avg Trades|Market change) ")
  { echo "=== $name ==="; echo "$out"; } >> "$LOG"
}
export -f run_one
ls /root/freqtrade/user_data/basket_exp/sweep_cfg/sw_*.json | xargs -P 8 -I {} bash -c 'run_one {}'
echo ALLDONE >> "$LOG"
