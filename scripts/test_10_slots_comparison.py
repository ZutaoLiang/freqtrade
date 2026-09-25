"""Compare 5 slots vs 10 slots across key strategies on AI Theme Sector.
Tests:
1. Baseline (With 18h Exit, TP 14% / SL 7%)
2. No Time Exit (Pure TP 14% / SL 7%)
3. No Time Exit + Vol Filter (TP 14% / SL 7%)
4. Conditional Time Exit (Cut at 18h if loss, let run to 36h if win)
5. Conditional Time Exit + Vol Filter
Across TRAIN, VALID-C, and HOLDOUT with max_open_trades = 10.
"""

import json
import os
import subprocess
import zipfile
import numpy as np
import pandas as pd

STRATEGIES = [
    ("Baseline (18h Exit, TP14)", "DualSqueezeBtcTrend1h"),
    ("No Time Exit (Pure TP14/SL7)", "DualSqAi_NoTimeExit_TP14"),
    ("No Time Exit + Vol (TP14/SL7)", "DualSqAi_NoTimeExit_Vol_TP14"),
    ("Conditional Exit (18h/36h)", "DualSqAi_ConditionalTimeExit"),
    ("Conditional Exit + Vol", "DualSqAi_ConditionalTimeExit_Vol"),
]

PERIODS = [
    ("TRAIN", "20250101-20251001"),
    ("VALID-C", "20251201-20260301"),
    ("HOLDOUT", "20260301-20260831"),
]


def run_backtest_slot(strat, config_file, trange):
  cmd = [
      "freqtrade",
      "backtesting",
      "-c",
      config_file,
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
    print(f"Error {strat} {trange}: {res.stderr[-300:]}")
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
  print("10 SLOTS vs 5 SLOTS MATRIX COMPARISON ON AI THEME")
  print("=" * 80)

  results = []
  for label, strat in STRATEGIES:
    print(f"\n--- Testing: {label} ({strat}) ---")
    row = {"label": label, "strategy": strat}
    for p_name, trange in PERIODS:
      # Run with 10 slots
      res_10 = run_backtest_slot(
          strat,
          "user_data/minute_research/ai_theme/config_ai_1h_10slots.json",
          trange,
      )
      if res_10:
        row[f"{p_name}_trades_10"] = res_10["trades"]
        row[f"{p_name}_profit_10"] = res_10["profit"]
        row[f"{p_name}_pf_10"] = res_10["pf"]
        row[f"{p_name}_t_10"] = res_10["t"]
        row[f"{p_name}_dd_10"] = res_10["dd"]
        row[f"{p_name}_dur_10"] = res_10["avg_dur"]
      else:
        row[f"{p_name}_trades_10"] = 0
        row[f"{p_name}_profit_10"] = 0.0
        row[f"{p_name}_pf_10"] = 0.0
        row[f"{p_name}_t_10"] = 0.0
        row[f"{p_name}_dd_10"] = 0.0
        row[f"{p_name}_dur_10"] = "0h"
    results.append(row)

  print("\n" + "=" * 80)
  print("10 SLOTS EXPERIMENT RESULTS TABLE")
  print("=" * 80 + "\n")

  header = (
      "| 策略变体 (10 仓位) | TRAIN 收益 (PF, t, 笔数) | VALID-C 收益 (PF, t,"
      " 笔数) | HOLDOUT 收益 (PF, t, 笔数) | HOLDOUT 最大回撤 |"
  )
  sep = (
      "| :--- | :---: | :---: | :---: | :---: |"
  )
  print(header)
  print(sep)

  for r in results:
    train_str = (
        f"{r.get('TRAIN_profit_10', 0)}% (PF {r.get('TRAIN_pf_10', 0)}, t"
        f" {r.get('TRAIN_t_10', 0)}, {r.get('TRAIN_trades_10', 0)}笔)"
    )
    valid_str = (
        f"{r.get('VALID-C_profit_10', 0)}% (PF {r.get('VALID-C_pf_10', 0)}, t"
        f" {r.get('VALID-C_t_10', 0)}, {r.get('VALID-C_trades_10', 0)}笔)"
    )
    hold_str = (
        f"{r.get('HOLDOUT_profit_10', 0)}% (PF {r.get('HOLDOUT_pf_10', 0)}, t"
        f" {r.get('HOLDOUT_t_10', 0)}, {r.get('HOLDOUT_trades_10', 0)}笔)"
    )
    dd_str = f"{r.get('HOLDOUT_dd_10', 0)}%"
    print(f"| **{r['label']}** | {train_str} | {valid_str} | {hold_str} | {dd_str} |")

  out_path = "user_data/minute_research/ai_theme/ten_slots_results.json"
  with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
  print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
  main()
