"""
Screen 3 new themes: DeFi Blue Chips, Layer 2, GameFi (10 tokens each).
Runs TRAIN, VALID-C, and HOLDOUT with max_open_trades = 5.
"""

import json
import os
import subprocess
import zipfile
import pandas as pd
import numpy as np

THEMES = [
    ("DeFi Blue Chips", "defi"),
    ("Layer 2 Scaling", "l2"),
    ("GameFi & Metaverse", "gamefi"),
]

STRATEGIES = [
    ("Dual Squeeze (Baseline 18h Exit)", "DualSqueezeBtcTrend1h", "1h"),
    ("Dual Squeeze + Vol Filter (> 1.3x)", "DualSqAi_V1_VolFilter", "1h"),
    ("Conditional Time Exit (18h/36h) + Vol", "DualSqAi_ConditionalTimeExit_Vol", "1h"),
    ("Funding Exhaustion Short (R24 5m)", "FundingExhaustionShort5m", "5m"),
    ("AllWeather V2 Combined (5m)", "AllWeatherRegimeAdaptiveV2", "5m"),
]

PERIODS = [
    ("TRAIN", "20250101-20251001"),
    ("VALID-C", "20251201-20260301"),
    ("HOLDOUT", "20260301-20260831"),
]

def run_bt(strat, theme_slug, tf, trange):
    config = f"user_data/minute_research/{theme_slug}_theme/config_{theme_slug}_{tf}.json"
    cmd = [
        "freqtrade", "backtesting",
        "-c", config,
        "-c", "config-funding-exhaustion-local.json",
        "--strategy", strat,
        "--datadir", "user_data/data/r3b",
        "--timerange", trange,
        "--export", "trades",
        "--cache", "day"
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print(f"Error {strat} on {theme_slug} ({trange}): {res.stderr[-200:]}")
        return None
    
    with open("user_data/backtest_results/.last_result.json") as f:
        last_zip = json.load(f)["latest_backtest"]
    
    with zipfile.ZipFile(os.path.join("user_data/backtest_results", last_zip)) as zf:
        names = [n for n in zf.namelist() if n.endswith(".json") and not n.endswith(".meta.json")]
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
            t_stat = round(daily_ret.mean() / daily_ret.std() * np.sqrt(len(daily_ret)), 2)
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
        "avg_dur": avg_dur
    }

def main():
    all_results = {}
    for theme_name, theme_slug in THEMES:
        print("\n" + "=" * 80)
        print(f"SCREENING THEME: {theme_name.upper()} (10 TOKENS)")
        print("=" * 80)
        
        theme_results = []
        for label, strat, tf in STRATEGIES:
            print(f"\n--- [{theme_name}] Testing: {label} ({strat}) ---")
            row = {"theme": theme_name, "label": label, "strategy": strat}
            for p_name, trange in PERIODS:
                res = run_bt(strat, theme_slug, tf, trange)
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
            theme_results.append(row)
            
            train_s = f"{row.get('TRAIN_profit',0)}% (PF {row.get('TRAIN_pf',0)}, {row.get('TRAIN_trades',0)}t)"
            valid_s = f"{row.get('VALID-C_profit',0)}% (PF {row.get('VALID-C_pf',0)}, {row.get('VALID-C_trades',0)}t)"
            hold_s = f"{row.get('HOLDOUT_profit',0)}% (PF {row.get('HOLDOUT_pf',0)}, {row.get('HOLDOUT_trades',0)}t)"
            print(f"--> {label}: TRAIN={train_s} | VALID-C={valid_s} | HOLDOUT={hold_s}")
            
        all_results[theme_slug] = theme_results
        out_single = f"user_data/minute_research/{theme_slug}_theme/{theme_slug}_results.json"
        with open(out_single, "w") as f:
            json.dump(theme_results, f, indent=2)

    out_all = "user_data/minute_research/three_new_themes_results.json"
    with open(out_all, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll 3 themes finished! Saved aggregate to {out_all}")

if __name__ == "__main__":
    main()
