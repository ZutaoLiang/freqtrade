"""Reconcile the native DelistShort1m backtest against the research event study (delay 5)."""
import glob
import os

import numpy as np
import pandas as pd

from freqtrade.data.btanalysis import load_backtest_data, load_backtest_stats

import os as _os; RES = _os.environ.get("DELIST_BT_DIR", "/root/freqtrade/user_data/delist/backtest_results")
RESEARCH = _os.environ.get("DELIST_RESEARCH", "/root/freqtrade/user_data/delist_short_20260906/trades_all_delays.parquet")


def main():
    zips = sorted(glob.glob(f"{RES}/*.zip"), key=os.path.getmtime)
    path = zips[-1]
    stats = load_backtest_stats(path)["strategy"]["DelistShort1m"]
    tr = load_backtest_data(path)
    tr = tr.sort_values("open_date")
    print(f"file: {path}")
    print(f"trades {len(tr)}, profit_total {stats['profit_total']*100:+.2f}% ({stats['profit_total_abs']:+.2f} USDT), "
          f"PF {stats.get('profit_factor', float('nan')):.3f}, max dd {stats['max_drawdown_account']*100:.2f}% / {stats['max_drawdown_abs']:.2f} USDT, "
          f"funding total {tr.funding_fees.sum():+.2f} USDT, wins {stats['wins']} losses {stats['losses']}")
    print("exit reasons:", tr.exit_reason.value_counts().to_dict())
    print("rejected signals:", stats.get("rejected_signals"), " timedout entries:", stats.get("timedout_entry_orders"))
    print("\nmonthly:")
    for m in stats.get("periodic_breakdown", {}).get("month", []):
        print(f"  {m['date']}: {m['profit_abs']:+8.2f} USDT  wins {m['wins']} losses {m['loses'] if 'loses' in m else m.get('losses')}")
    r = pd.read_parquet(RESEARCH)
    r = r[r.delay == 5].copy()
    r["pair"] = r.symbol.str[:-4] + "/USDT:USDT"
    r["entry_t"] = pd.to_datetime(r.entry_t, utc=True)
    j = tr.merge(r[["pair", "entry_t", "exit_t", "exit_reason", "net10", "funding", "hold_h"]].rename(columns={"exit_reason": "res_reason", "exit_t": "res_exit"}), on="pair", how="outer", indicator=True)
    j["engine_ret"] = j.profit_ratio
    j["dt_entry_min"] = (pd.to_datetime(j.open_date, utc=True) - j.entry_t).dt.total_seconds() / 60
    j["dt_exit_min"] = (pd.to_datetime(j.close_date, utc=True) - pd.to_datetime(j.res_exit, utc=True)).dt.total_seconds() / 60
    pd.set_option("display.width", 250)
    cols = ["pair", "open_date", "dt_entry_min", "dt_exit_min", "exit_reason", "res_reason", "engine_ret", "net10", "funding_fees", "funding", "_merge"]
    out = j[cols].copy()
    out["engine_ret"] = (out.engine_ret * 100).round(2); out["net10"] = (out.net10 * 100).round(2); out["funding"] = (out.funding * 100).round(2)
    print("\nper-trade engine vs research (returns %, funding: engine USDT vs research % of notional):")
    print(out.sort_values("open_date").to_string(index=False))
    both = j[j._merge == "both"]
    d = (both.engine_ret - both.net10) * 1e4
    print(f"\nmatched {len(both)}: mean engine {both.engine_ret.mean()*100:+.2f}% vs research {both.net10.mean()*100:+.2f}%; "
          f"per-trade diff bps mean {d.mean():+.1f}, max |diff| {d.abs().max():.1f}; entry-time offsets (min): {both.dt_entry_min.describe()[['min','max']].to_dict()}; "
          f"exit-time |offset| > 1 min: {(both.dt_exit_min.abs() > 1).sum()}")


if __name__ == "__main__":
    main()
