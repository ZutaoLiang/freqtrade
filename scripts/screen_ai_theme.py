"""Screen previous strategies on Theme 2: AI Tokens (10 tokens).
Runs TRAIN (20250101-20251001) and VALID-C (20251201-20260301).
Calculates trades, winrate, total profit %, profit factor, day-clustered t-stat, max drawdown, and per-token performance.
"""

import glob
import json
import os
import subprocess
import sys
import zipfile
import numpy as np
import pandas as pd

# Define the candidates to screen
CANDIDATES = [
    {
        "name": "FundingExhaustionShort5m (R24)",
        "strategy": "FundingExhaustionShort5m",
        "config": "user_data/minute_research/ai_theme/config_ai_5m.json",
        "tf": "5m",
    },
    {
        "name": "AllWeatherV2_Exhaust (R24 inside V2)",
        "strategy": "AllWeatherV2_Exhaust",
        "config": "user_data/minute_research/ai_theme/config_ai_5m.json",
        "tf": "5m",
    },
    {
        "name": "AllWeatherV2_DualSqLong",
        "strategy": "AllWeatherV2_DualSqLong",
        "config": "user_data/minute_research/ai_theme/config_ai_5m.json",
        "tf": "5m",
    },
    {
        "name": "AllWeatherV2_DualSqShort",
        "strategy": "AllWeatherV2_DualSqShort",
        "config": "user_data/minute_research/ai_theme/config_ai_5m.json",
        "tf": "5m",
    },
    {
        "name": "AllWeatherV2_Flushout",
        "strategy": "AllWeatherV2_Flushout",
        "config": "user_data/minute_research/ai_theme/config_ai_5m.json",
        "tf": "5m",
    },
    {
        "name": "AllWeatherRegimeAdaptiveV2 (Combined)",
        "strategy": "AllWeatherRegimeAdaptiveV2",
        "config": "user_data/minute_research/ai_theme/config_ai_5m.json",
        "tf": "5m",
    },
    {
        "name": "DualSqueezeBtcTrend1h",
        "strategy": "DualSqueezeBtcTrend1h",
        "config": "user_data/minute_research/ai_theme/config_ai_1h.json",
        "tf": "1h",
    },
    {
        "name": "TrendShiftADXPureSignal",
        "strategy": "TrendShiftADXPureSignal",
        "config": "user_data/minute_research/ai_theme/config_ai_1h.json",
        "tf": "1h",
    },
    {
        "name": "MacroRelativeStrengthSqueeze1h",
        "strategy": "MacroRelativeStrengthSqueeze1h",
        "config": "user_data/minute_research/ai_theme/config_ai_1h.json",
        "tf": "1h",
    },
]

PERIODS = [
    ("TRAIN", "20250101-20251001"),
    ("VALID-C", "20251201-20260301"),
]


def run_backtest(strat_name, config_path, timerange):
  cmd = [
      "freqtrade",
      "backtesting",
      "-c",
      config_path,
      "-c",
      "config-funding-exhaustion-local.json",
      "--strategy",
      strat_name,
      "--datadir",
      "user_data/data/r3b",
      "--timerange",
      timerange,
      "--export",
      "trades",
      "--cache",
      "day",
  ]
  print(f"Running: {strat_name} on {timerange} ...")
  res = subprocess.run(
      cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
  )
  if res.returncode != 0:
    print(f"Error running {strat_name} on {timerange}:")
    print(res.stderr[-500:])
    return None

  # Read latest backtest result
  with open("user_data/backtest_results/.last_result.json") as f:
    last_zip_name = json.load(f)["latest_backtest"]
  zip_path = os.path.join("user_data/backtest_results", last_zip_name)

  with zipfile.ZipFile(zip_path) as zf:
    names = [
        n
        for n in zf.namelist()
        if n.endswith(".json") and not n.endswith(".meta.json")
    ]
    data = json.loads(zf.read(names[0]))

  strat_data = data["strategy"][strat_name]
  trades = strat_data.get("trades", [])

  # Compute stats
  total_trades = strat_data.get("total_trades", 0)
  profit_total_pct = round(strat_data.get("profit_total", 0.0) * 100, 2)
  profit_factor = strat_data.get("profit_factor")
  pf = (
      round(profit_factor, 2)
      if profit_factor is not None and profit_factor < 1000
      else ("Inf" if profit_factor else 0.0)
  )
  max_dd = round(strat_data.get("max_drawdown_account", 0.0) * 100, 2)
  winrate = round(strat_data.get("winrate", 0.0) * 100, 1)

  # Day clustered t-stat
  t_stat = 0.0
  if trades:
    df_trades = pd.DataFrame(trades)
    df_trades["open_date"] = pd.to_datetime(df_trades["open_date"], utc=True)
    df_trades["day"] = df_trades["open_date"].dt.floor("D")
    daily_ret = df_trades.groupby("day")["profit_ratio"].sum()
    if len(daily_ret) > 1 and daily_ret.std() > 0:
      t_stat = round(
          daily_ret.mean() / daily_ret.std() * np.sqrt(len(daily_ret)), 2
      )

  # Pair breakdown
  pair_results = {}
  for p in strat_data.get("results_per_pair", []):
    pair_name = p.get("key", "").split("/")[0]
    pair_profit = round(p.get("profit_total", 0.0) * 100, 2)
    pair_results[pair_name] = (p.get("trades", 0), pair_profit)

  return {
      "strategy": strat_name,
      "trades": total_trades,
      "winrate": winrate,
      "profit_pct": profit_total_pct,
      "pf": pf,
      "t": t_stat,
      "max_dd": max_dd,
      "pair_results": pair_results,
  }


def main():
  all_results = []
  for cand in CANDIDATES:
    c_res = {"name": cand["name"], "strategy": cand["strategy"]}
    for period_name, timerange in PERIODS:
      res = run_backtest(cand["strategy"], cand["config"], timerange)
      if res:
        c_res[f"{period_name}_trades"] = res["trades"]
        c_res[f"{period_name}_winrate"] = res["winrate"]
        c_res[f"{period_name}_profit"] = res["profit_pct"]
        c_res[f"{period_name}_pf"] = res["pf"]
        c_res[f"{period_name}_t"] = res["t"]
        c_res[f"{period_name}_dd"] = res["max_dd"]
        c_res[f"{period_name}_pairs"] = res["pair_results"]
      else:
        c_res[f"{period_name}_trades"] = 0
        c_res[f"{period_name}_winrate"] = 0.0
        c_res[f"{period_name}_profit"] = 0.0
        c_res[f"{period_name}_pf"] = 0.0
        c_res[f"{period_name}_t"] = 0.0
        c_res[f"{period_name}_dd"] = 0.0
        c_res[f"{period_name}_pairs"] = {}
    all_results.append(c_res)

  # Output markdown summary
  print("\n" + "=" * 80)
  print("AI THEME (10 TOKENS) SCREENING REPORT")
  print("=" * 80 + "\n")

  header = "| 策略名称 | TRAIN 笔数 | TRAIN 收益 | TRAIN PF | TRAIN t | VALID 笔数 | VALID 收益 | VALID PF | VALID t | VALID DD |"
  separator = "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
  print(header)
  print(separator)

  for r in all_results:
    row = (
        f"| **{r['name']}** | {r.get('TRAIN_trades', 0)} |"
        f" {r.get('TRAIN_profit', 0)}% | {r.get('TRAIN_pf', 0)} |"
        f" {r.get('TRAIN_t', 0)} | {r.get('VALID-C_trades', 0)} |"
        f" {r.get('VALID-C_profit', 0)}% | {r.get('VALID-C_pf', 0)} |"
        f" {r.get('VALID-C_t', 0)} | {r.get('VALID-C_dd', 0)}% |"
    )
    print(row)

  # Save to json for further analysis
  out_path = "user_data/minute_research/ai_theme/screening_results.json"
  with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
  print(f"\nSaved full results to {out_path}")


if __name__ == "__main__":
  main()
