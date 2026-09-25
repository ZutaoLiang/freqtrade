"""Test Dual Squeeze AI Sector Variants on TRAIN, VALID-C, and HOLDOUT.
Evaluates the impact of:
- Volume expansion gating (vol > 1.3x)
- Breakeven & trailing stops
- Stagnation early exits (6h)
- Relative strength filtering
"""

import glob
import json
import os
import subprocess
import zipfile
import numpy as np
import pandas as pd

VARIANTS = [
    ("Baseline (Original)", "DualSqueezeBtcTrend1h"),
    ("V1 (Vol Filter > 1.3x)", "DualSqAi_V1_VolFilter"),
    ("V2 (Vol + Breakeven)", "DualSqAi_V2_VolAndBreakeven"),
    ("V3 (Vol + Stagnation 6h)", "DualSqAi_V3_VolAndStagnation"),
    ("V4 (Full Suite: Vol+BE+Stag)", "DualSqAi_V4_FullSuite"),
    ("V5 (Strict RS vs BTC)", "DualSqAi_V5_StrictRS"),
]

PERIODS = [
    ("TRAIN", "20250101-20251001"),
    ("VALID-C", "20251201-20260301"),
    ("HOLDOUT", "20260301-20260831"),
]


def run_test(strat_name, timerange):
  cmd = [
      "freqtrade",
      "backtesting",
      "-c",
      "user_data/minute_research/ai_theme/config_ai_1h.json",
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
  res = subprocess.run(
      cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
  )
  if res.returncode != 0:
    print(f"Error {strat_name} on {timerange}: {res.stderr[-300:]}")
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

  st = data["strategy"][strat_name]
  trades = st.get("trades", [])
  t_stat = 0.0
  if trades:
    df = pd.DataFrame(trades)
    df["open_date"] = pd.to_datetime(df["open_date"], utc=True)
    df["day"] = df["open_date"].dt.floor("D")
    daily_ret = df.groupby("day")["profit_ratio"].sum()
    if len(daily_ret) > 1 and daily_ret.std() > 0:
      t_stat = round(
          daily_ret.mean() / daily_ret.std() * np.sqrt(len(daily_ret)), 2
      )

  pf = st.get("profit_factor", 0.0)
  pf_clean = round(pf, 2) if pf and pf < 1000 else ("Inf" if pf else 0.0)

  return {
      "trades": st.get("total_trades", 0),
      "winrate": round(st.get("winrate", 0.0) * 100, 1),
      "profit": round(st.get("profit_total", 0.0) * 100, 2),
      "pf": pf_clean,
      "t": t_stat,
      "dd": round(st.get("max_drawdown_account", 0.0) * 100, 2),
      "exits": st.get("exit_reason_summary", []),
  }


def main():
  print("=" * 80)
  print("DUAL SQUEEZE AI SECTOR ITERATION MATRIX")
  print("=" * 80)

  all_results = []
  for label, strat in VARIANTS:
    print(f"\nTesting {label} ({strat}) ...")
    row = {"label": label, "strategy": strat}
    for p_name, trange in PERIODS:
      res = run_test(strat, trange)
      if res:
        row[f"{p_name}_trades"] = res["trades"]
        row[f"{p_name}_profit"] = res["profit"]
        row[f"{p_name}_pf"] = res["pf"]
        row[f"{p_name}_t"] = res["t"]
        row[f"{p_name}_dd"] = res["dd"]
        row[f"{p_name}_exits"] = res["exits"]
      else:
        row[f"{p_name}_trades"] = 0
        row[f"{p_name}_profit"] = 0.0
        row[f"{p_name}_pf"] = 0.0
        row[f"{p_name}_t"] = 0.0
        row[f"{p_name}_dd"] = 0.0
    all_results.append(row)

  # Markdown Table
  print("\n" + "=" * 80)
  print("SUMMARY TABLE: AI SECTOR DUAL SQUEEZE EVOLUTION")
  print("=" * 80 + "\n")

  header = (
      "| 策略变体 | TRAIN 收益 (PF, t) | VALID-C 收益 (PF, t) | HOLDOUT 收益"
      " (PF, t) | HOLDOUT 回撤 | 核心机制评价 |"
  )
  sep = (
      "| :--- | :---: | :---: | :---: | :---: | :---"
      " |"
  )
  print(header)
  print(sep)

  for r in all_results:
    train_str = (
        f"{r.get('TRAIN_profit', 0)}% (PF {r.get('TRAIN_pf', 0)}, t"
        f" {r.get('TRAIN_t', 0)})"
    )
    valid_str = (
        f"{r.get('VALID-C_profit', 0)}% (PF {r.get('VALID-C_pf', 0)}, t"
        f" {r.get('VALID-C_t', 0)})"
    )
    hold_str = (
        f"{r.get('HOLDOUT_profit', 0)}% (PF {r.get('HOLDOUT_pf', 0)}, t"
        f" {r.get('HOLDOUT_t', 0)})"
    )
    dd_str = f"{r.get('HOLDOUT_dd', 0)}%"
    print(f"| **{r['label']}** | {train_str} | {valid_str} | {hold_str} | {dd_str} | |")

  # Save to file
  with open(
      "user_data/minute_research/ai_theme/dualsq_iteration_results.json", "w"
  ) as f:
    json.dump(all_results, f, indent=2)
  print("\nSaved to user_data/minute_research/ai_theme/dualsq_iteration_results.json")


if __name__ == "__main__":
  main()
