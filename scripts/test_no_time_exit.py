"""Test impact of removing the 18-hour mandatory exit on AI Theme Sector.
Compares:
1. Baseline: With 18h exit, TP 14%, SL 7% (DualSqueezeBtcTrend1h)
2. No Time Exit: TP 14%, SL 7% (DualSqAi_NoTimeExit_TP14)
3. No Time Exit + Vol Filter: TP 14%, SL 7% (DualSqAi_NoTimeExit_Vol_TP14)
4. No Time Exit + Vol Filter: TP 8%, SL 6% (DualSqAi_NoTimeExit_Vol_TP8)
Across TRAIN, VALID-C, and HOLDOUT.
"""

import glob
import json
import os
import subprocess
import zipfile
import numpy as np
import pandas as pd

COMPARISONS = [
    (
        "Baseline (With 18h Exit, TP 14%, SL 7%)",
        "DualSqueezeBtcTrend1h",
        "user_data/minute_research/ai_theme/config_ai_1h.json",
    ),
    (
        "No Time Exit (Pure TP 14% / SL 7%)",
        "DualSqAi_NoTimeExit_TP14",
        "user_data/minute_research/ai_theme/config_ai_1h.json",
    ),
    (
        "No Time Exit + Vol Filter (TP 14% / SL 7%)",
        "DualSqAi_NoTimeExit_Vol_TP14",
        "user_data/minute_research/ai_theme/config_ai_1h.json",
    ),
    (
        "No Time Exit + Vol Filter (TP 8% / SL 6%)",
        "DualSqAi_NoTimeExit_Vol_TP8",
        "user_data/minute_research/ai_theme/config_ai_1h.json",
    ),
]

PERIODS = [
    ("TRAIN", "20250101-20251001"),
    ("VALID-C", "20251201-20260301"),
    ("HOLDOUT", "20260301-20260831"),
]


def run_single(strat, config, trange):
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
    # Average holding duration in hours
    if "trade_duration" in df.columns:
      avg_dur = f"{round(df['trade_duration'].mean() / 60, 1)}h"

  pf = st.get("profit_factor", 0.0)
  pf_clean = round(pf, 2) if pf and pf < 1000 else ("Inf" if pf else 0.0)

  # Exit reasons
  reasons = {}
  for item in st.get("exit_reason_summary", []):
    reasons[item.get("key", "other")] = item.get("trades", 0)

  return {
      "trades": st.get("total_trades", 0),
      "winrate": round(st.get("winrate", 0.0) * 100, 1),
      "profit": round(st.get("profit_total", 0.0) * 100, 2),
      "pf": pf_clean,
      "t": t_stat,
      "dd": round(st.get("max_drawdown_account", 0.0) * 100, 2),
      "avg_dur": avg_dur,
      "reasons": reasons,
  }


def main():
  print("=" * 80)
  print("TESTING: REMOVING 18-HOUR TIME EXIT ON AI THEME")
  print("=" * 80)

  results = []
  for label, strat, config in COMPARISONS:
    print(f"\nRunning {label} ({strat}) ...")
    row = {"label": label, "strategy": strat}
    for p_name, trange in PERIODS:
      res = run_single(strat, config, trange)
      if res:
        row[f"{p_name}_trades"] = res["trades"]
        row[f"{p_name}_winrate"] = res["winrate"]
        row[f"{p_name}_profit"] = res["profit"]
        row[f"{p_name}_pf"] = res["pf"]
        row[f"{p_name}_t"] = res["t"]
        row[f"{p_name}_dd"] = res["dd"]
        row[f"{p_name}_dur"] = res["avg_dur"]
        row[f"{p_name}_reasons"] = res["reasons"]
      else:
        row[f"{p_name}_trades"] = 0
        row[f"{p_name}_winrate"] = 0.0
        row[f"{p_name}_profit"] = 0.0
        row[f"{p_name}_pf"] = 0.0
        row[f"{p_name}_t"] = 0.0
        row[f"{p_name}_dd"] = 0.0
        row[f"{p_name}_dur"] = "0h"
        row[f"{p_name}_reasons"] = {}
    results.append(row)

  # Output markdown table
  print("\n" + "=" * 80)
  print("COMPARISON RESULTS: WITH VS WITHOUT 18H MANDATORY EXIT")
  print("=" * 80 + "\n")

  header = "| 策略版本 | TRAIN 收益 (PF, t, 持仓) | VALID-C 收益 (PF, t, 持仓) | HOLDOUT 收益 (PF, t, 持仓) | HOLDOUT 回撤 | 出场原因构成 (HOLDOUT) |"
  sep = "| :--- | :---: | :---: | :---: | :---: | :--- |"
  print(header)
  print(sep)

  for r in results:
    train_str = f"{r.get('TRAIN_profit', 0)}% (PF {r.get('TRAIN_pf', 0)}, t {r.get('TRAIN_t', 0)}, 均持{r.get('TRAIN_dur', '0h')})"
    valid_str = f"{r.get('VALID-C_profit', 0)}% (PF {r.get('VALID-C_pf', 0)}, t {r.get('VALID-C_t', 0)}, 均持{r.get('VALID-C_dur', '0h')})"
    hold_str = f"{r.get('HOLDOUT_profit', 0)}% (PF {r.get('HOLDOUT_pf', 0)}, t {r.get('HOLDOUT_t', 0)}, 均持{r.get('HOLDOUT_dur', '0h')})"
    dd_str = f"{r.get('HOLDOUT_dd', 0)}%"
    reasons_str = str(r.get("HOLDOUT_reasons", {}))
    print(
        f"| **{r['label']}** | {train_str} | {valid_str} | {hold_str} | {dd_str}"
        f" | {reasons_str} |"
    )

  with open(
      "user_data/minute_research/ai_theme/no_time_exit_results.json", "w"
  ) as f:
    json.dump(results, f, indent=2)
  print(
      "\nSaved full results to"
      " user_data/minute_research/ai_theme/no_time_exit_results.json"
  )


if __name__ == "__main__":
  main()
