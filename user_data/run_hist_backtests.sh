#!/bin/bash
# Serial 2023/2024/2025 expanded-universe backtests of XsMomEnsembleV1.
# Detached via setsid; memory-gated per AGENTS.md; logs to user_data/hist_bt_<year>.log
set -u
cd /root/workspace/freqtrade
export MALLOC_ARENA_MAX=2 MALLOC_TRIM_THRESHOLD_=131072

for Y in 2023 2024 2025; do
  # wait for at least 512MiB available before each run
  for i in $(seq 1 120); do
    avail=$(free -m | awk 'NR==2{print $7}')
    [ "$avail" -ge 512 ] && break
    sleep 30
  done
  END=$((Y + 1))0101
  nice -n 10 python3 -m freqtrade backtesting \
    --config "user_data/config_xs_mom_ensemble_v1_hist_${Y}.json" \
    --strategy-path user_data/strategies/neutral \
    --datadir user_data/data/binance-hist \
    --timerange "${Y}0101-${END}" \
    --fee 0.0007 --cache none --export trades \
    --backtest-directory "user_data/analysis/_xsmom_v1_${Y}" \
    > "user_data/hist_bt_${Y}.log" 2>&1
  echo "=== ${Y} exit=$? ===" >> user_data/hist_bt_status.log
done
echo "ALL DONE" >> user_data/hist_bt_status.log
