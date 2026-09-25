"""Screen and evaluate strategies on Theme 1: Layer 1 Blue Chips (10 tokens).
Tokens: BTC, ETH, SOL, BNB, SUI, ADA, AVAX, NEAR, APT, TON
Runs TRAIN, VALID-C, and HOLDOUT with max_open_trades = 5.
"""

import json
import os
import subprocess
import zipfile
import numpy as np
import pandas as pd

STRATEGIES = [
    (
        "Dual Squeeze (Baseline 18h Exit)",
        "DualSqueezeBtcTrend1h",
        "user_data/minute_research/l1_theme/config_l1_1h.json",
    ),
    (
        "Dual Squeeze + Vol Filter (> 1.3x)",
        "DualSqAi_V1_VolFilter",
        "user_data/minute_research/l1_theme/config_l1_1h.json",
    ),
    (
        "Dual Squeeze + Vol + 6h Stagnation",
        "DualSqAi_V3_VolAndStagnation",
        "user_data/minute_research/l1_theme/config_l1_1h.json",
    ),
    (
        "Conditional Time Exit (18h/36h) + Vol",
        "DualSqAi_ConditionalTimeExit_Vol",
        "user_data/minute_research/l1_theme/config_l1_1h.json",
    ),
    (
        "Funding Exhaustion Short (R24 5m)",
        "FundingExhaustionShort5m",
        "user_data/minute_research/l1_theme/config_l1_5m.json",
    ),
    (
        "AllWeather V2 Combined (5m)",
        "AllWeatherRegimeAdaptiveV2",
        "user_data/minute_research/l1_theme/config_l1_5m.json",
    ),
    (
        "AllWeather V2 Flushout Dip-buy (5m)",
        "AllWeatherV2_Flushout",
        "user_data/minute_research/l1_theme/config_l1_5m.json",
    ),
]

PERIODS = [
    ("TRAIN", "20250101-20251001"),
    ("VALID-C", "20251201-20260301"),
    ("HOLDOUT", "20260301-20260831"),
]


def run_bt(strat, config, trange):
  cmd = [
      "freqtrade",
      "backtesting",
      "-c",
      config,
      "-c",
      "config-funding-exhaustion-local.json",
      "--strategy",
      strat,
      "--datadir",
      "user_data/data/r3b",
      "--timerange",
      trange,
      "--export",
      "trades",
      "--cache",
      "day",
  ]
  res = subprocess.run(
      cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
  )
  if res.returncode != 0:
    print(f"Error {strat} on {trange}: {res.stderr[-300:]}")
    return None

  with open("user_data/backtest_results/.last_result.json") as f:
    last_zip = json.load(f)["latest_backtest"]

  with zipfile.ZipFile(
      os.path.join("user_data/backtest_results", last_zip)
  ) as zf:
    names = [
        n
        for n in zf.namelist()
        if n.endswith(".json") and not n.endswith(".meta.json")
    ]
    data = json.loads(zf.read(names[0]))

  st = data["strategy"][strat]
  trades = st.get("trades", [])

  t_stat = 0.0
  avg_dur = "0h"
  if trades:
    df = pd.DataFrame(trades)
    df["open_date"] = pd.to_datetime(df["open_date"], utc=True)
    df["day"] = df["open_date"].dt.floor("D")
    daily_ret = df.groupby("day")["profit_ratio"].sum()
    if len(daily_ret) > 1 and daily_ret.std() > 0:
      t_stat = round(
          daily_ret.mean() / daily_ret.std() * np.sqrt(len(daily_ret)), 2
      )
    if "trade_duration" in df.columns:
      avg_dur = f"{round(df['trade_duration'].mean() / 60, 1)}h"

  pf = st.get("profit_factor", 0.0)
  pf_clean = round(pf, 2) if pf and pf < 1000 else ("Inf" if pf else 0.0)

  return {
      "trades": st.get("total_trades", 0),
      "winrate": round(st.get("winrate", 0.0) * 100, 1),
      "profit": round(st.get("profit_total", 0.0) * 100, 2),
      "pf": pf_clean,
      "t": t_stat,
      "dd": round(st.get("max_drawdown_account", 0.0) * 100, 2),
      "avg_dur": avg_dur,
  }


def main():
  print("=" * 80)
  print("SCREENING THEME 1: LAYER 1 BLUE CHIPS (10 TOKENS)")
  print("=" * 80)

  results = []
  for label, strat, cfg in STRATEGIES:
    print(f"\n--- Testing: {label} ({strat}) ---")
    row = {"label": label, "strategy": strat}
    for p_name, trange in PERIODS:
      res = run_bt(strat, cfg, trange)
      if res:
        row[f"{p_name}_trades"] = res["trades"]
        row[f"{p_name}_profit"] = res["profit"]
        row[f"{p_name}_pf"] = res["pf"]
        row[f"{p_name}_t"] = res["t"]
        row[f"{p_name}_dd"] = res["dd"]
        row[f"{p_name}_dur"] = res["avg_dur"]
      else:
        row[f"{p_name}_trades"] = 0
        row[f"{p_name}_profit"] = 0.0
        row[f"{p_name}_pf"] = 0.0
        row[f"{p_name}_t"] = 0.0
        row[f"{p_name}_dd"] = 0.0
        row[f"{p_name}_dur"] = "0h"
    results.append(row)

  print("\n" + "=" * 80)
  print("LAYER 1 BLUE CHIPS (10 TOKENS) SCREENING REPORT")
  print("=" * 80 + "\n")

  header = (
      "| 策略名称 | TRAIN 收益 (PF, t, 笔数) | VALID-C 收益 (PF, t, 笔数) |"
      " HOLDOUT 收益 (PF, t, 笔数) | HOLDOUT 回撤 |"
  )
  sep = (
      "| :--- | :---: | :---: | :---: | :---: |"
  )
  print(header)
  print(sep)

  for r in results:
    train_str = (
        f"{r.get('TRAIN_profit', 0)}% (PF {r.get('TRAIN_pf', 0)}, t"
        f" {r.get('TRAIN_t', 0)}, {r.get('TRAIN_trades', 0)}笔)"
    )
    valid_str = (
        f"{r.get('VALID-C_profit', 0)}% (PF {r.get('VALID-C_pf', 0)}, t"
        f" {r.get('VALID-C_t', 0)}, {r.get('VALID-C_trades', 0)}笔)"
    )
    hold_str = (
        f"{r.get('HOLDOUT_profit', 0)}% (PF {r.get('HOLDOUT_pf', 0)}, t"
        f" {r.get('HOLDOUT_t', 0)}, {r.get('HOLDOUT_trades', 0)}笔)"
    )
    dd_str = f"{r.get('HOLDOUT_dd', 0)}%"
    print(f"| **{r['label']}** | {train_str} | {valid_str} | {hold_str} | {dd_str} |")

  out_path = "user_data/minute_research/l1_theme/l1_screening_results.json"
  with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
  print(f"\nSaved full results to {out_path}")


if __name__ == "__main__":
  main()
