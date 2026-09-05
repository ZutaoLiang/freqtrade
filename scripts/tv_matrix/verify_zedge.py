"""Verification pass for the W67tJltC (Z-Edge) port, per SKILL.md section 7.

Three independent checks, all on real local candles:

1. Formula cross-check - a deliberately naive, Pine-literal reference
   implementation (plain python loops, no pandas tricks) is compared against
   the vectorised ``zedge_core`` used by the strategies.
2. Truncated-prefix causality - recomputing on a prefix of the data must give
   the same last-row values as the full series. A future reference anywhere in
   the chain breaks this.
3. Real callback wrapper - the strategy object is built through freqtrade's own
   resolver and its populate_* callbacks are run on the same candles, so the
   columns the backtest will read are the ones verified above.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path("user_data/strategies/tradingview").resolve()))

from freqtrade.configuration import Configuration  # noqa: E402
from freqtrade.data.history import load_pair_history  # noqa: E402
from freqtrade.enums import CandleType  # noqa: E402
from freqtrade.resolvers import StrategyResolver  # noqa: E402

import zedge_core as core  # noqa: E402


# --- 1. Pine-literal reference ------------------------------------------------

def ref_sma(x, n, i):
    if i + 1 < n:
        return float("nan")
    window = x[i - n + 1 : i + 1]
    return sum(window) / n if not any(np.isnan(window)) else float("nan")


def ref_stdev(x, n, i):
    m = ref_sma(x, n, i)
    if np.isnan(m):
        return float("nan")
    window = x[i - n + 1 : i + 1]
    return (sum((v - m) ** 2 for v in window) / n) ** 0.5


def ref_zscore(x, n):
    out = []
    for i in range(len(x)):
        m, sd = ref_sma(x, n, i), ref_stdev(x, n, i)
        if np.isnan(m):
            out.append(float("nan"))
        elif sd > 0:
            out.append((x[i] - m) / sd)
        else:
            out.append(0.0)
    return np.array(out)


def ref_percentrank(x, n):
    out = np.full(len(x), np.nan)
    for i in range(n, len(x)):
        cur = x[i]
        window = x[i - n : i]
        if np.isnan(cur) or np.isnan(window).any():
            continue
        out[i] = sum(1 for v in window if v <= cur) * 100.0 / n
    return out


def ref_adaptive_ema(composite, alpha):
    out = np.full(len(composite), np.nan)
    prev = float("nan")
    for i in range(len(composite)):
        cur = composite[i] if np.isnan(prev) else prev + alpha[i] * (composite[i] - prev)
        out[i] = cur
        prev = cur
    return out


def max_diff(a: np.ndarray, b: np.ndarray) -> float:
    both = ~np.isnan(a) & ~np.isnan(b)
    if (np.isnan(a) != np.isnan(b)).any():
        return float("inf")  # na patterns must match too
    return float(np.abs(a[both] - b[both]).max()) if both.any() else 0.0


def check_formulas(df: pd.DataFrame) -> list[str]:
    out = []
    sub = df.tail(1500).reset_index(drop=True)
    rsi = core.ta.RSI(sub, timeperiod=14).to_numpy(dtype=float)
    fast = core.zscore(pd.Series(rsi), 100).to_numpy(dtype=float)
    slow = ref_zscore(rsi, 100)
    out.append(f"zscore(rsi,100) max|diff| = {max_diff(fast, slow):.3e}")

    atr = core.ta.ATR(sub, timeperiod=14).to_numpy(dtype=float)
    fast = core.percentrank(pd.Series(atr), 100).to_numpy(dtype=float)
    slow = ref_percentrank(atr, 100)
    out.append(f"percentrank(atr,100) max|diff| = {max_diff(fast, slow):.3e}")

    comp = core.add_signal_columns(sub.copy())
    alpha = comp["alpha"].to_numpy(dtype=float)
    slow = ref_adaptive_ema(comp["composite"].to_numpy(dtype=float), alpha)
    out.append(f"adaptive ema max|diff| = {max_diff(comp['smooth'].to_numpy(dtype=float), slow):.3e}")
    return out


# --- 2. Prefix causality ------------------------------------------------------

def check_causality(df: pd.DataFrame, cuts: int = 6) -> list[str]:
    full = core.add_order_columns(core.add_signal_columns(df.copy()))
    cols = ["composite", "smooth", "atr_pct_rank", "long_entry", "short_entry",
            "long_exit", "short_exit"]
    rng = np.random.default_rng(7)
    idx = sorted(rng.integers(600, len(df), size=cuts).tolist())
    worst = 0.0
    bad = 0
    for cut in idx:
        prefix = core.add_order_columns(core.add_signal_columns(df.iloc[:cut].copy()))
        for c in cols:
            a = float(prefix[c].iloc[-1]) if prefix[c].dtype != bool else float(prefix[c].iloc[-1])
            b = float(full[c].iloc[cut - 1]) if full[c].dtype != bool else float(full[c].iloc[cut - 1])
            if np.isnan(a) and np.isnan(b):
                continue
            d = abs(a - b)
            worst = max(worst, d)
            if d > 1e-9:
                bad += 1
    return [f"prefix causality: {cuts} cuts x {len(cols)} columns, "
            f"mismatches = {bad}, max|diff| = {worst:.3e}"]


# --- 3. Real strategy callbacks ----------------------------------------------

def check_strategy(df: pd.DataFrame, config_path: str, strategy: str) -> list[str]:
    config = Configuration.from_files([config_path])
    config["strategy"] = strategy
    config["timeframe"] = "1h"
    strat = StrategyResolver.load_strategy(config)
    out = strat.advise_indicators(df.copy(), {"pair": "BTC/USDT:USDT"})
    out = strat.advise_entry(out, {"pair": "BTC/USDT:USDT"})
    out = strat.advise_exit(out, {"pair": "BTC/USDT:USDT"})
    warm = strat.startup_candle_count
    early = out.iloc[:warm]
    lines = [
        f"{strategy}: entry_mode={strat.entry_mode} startup={warm} "
        f"stoploss={strat.stoploss} roi={strat.minimal_roi}",
        f"{strategy}: enter_long={int(out['enter_long'].sum())} "
        f"enter_short={int(out['enter_short'].sum())} "
        f"exit_long={int(out['exit_long'].sum())} exit_short={int(out['exit_short'].sum())}",
        f"{strategy}: signals inside startup window = "
        f"{int(early[['enter_long', 'enter_short']].sum().sum())}",
    ]
    both = ((out["enter_long"] == 1) & (out["enter_short"] == 1)).sum()
    lines.append(f"{strategy}: bars with both long and short entry = {int(both)}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="BTC/USDT:USDT")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--datadir", default="user_data/data/tv_matrix")
    ap.add_argument("--config", default="user_data/research/tradingview/config_base.json")
    args = ap.parse_args()

    df = load_pair_history(
        pair=args.pair,
        timeframe=args.timeframe,
        datadir=Path(args.datadir),
        data_format="feather",
        candle_type=CandleType.FUTURES,
    )
    print(f"loaded {len(df)} candles {df['date'].iloc[0]} -> {df['date'].iloc[-1]}")

    for line in check_formulas(df):
        print(line)
    for line in check_causality(df):
        print(line)
    for strategy in ("ZEdgeSignal", "ZEdgeReversion"):
        for line in check_strategy(df, args.config, strategy):
            print(line)


if __name__ == "__main__":
    main()
