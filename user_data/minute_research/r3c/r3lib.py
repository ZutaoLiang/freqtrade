"""Shared helpers for r3: load Binance Vision 1m parquet with flow columns, split labels, costs."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/root/freqtrade/.claude/skills/binance-minute-strategy-research/scripts")
import harness as H  # noqa: E402

ROOT = "/root/freqtrade/user_data/data/binance_public"
TRAIN = ("2025-01-01", "2025-10-01")
VALID = ("2025-12-01", "2026-03-01")  # SKILL scheme C: 2025-10..11 excluded
HOLD = ("2026-03-01", "2026-09-01")
U60 = open("/root/freqtrade/user_data/minute_research/r3c/U60.txt").read().split()


def cost_bps(base: str) -> float:
    return 7.5 if base in ("BTC", "ETH") else 10.0


def load(base: str, end: str | None = TRAIN[1]) -> dict:
    """1m bars up to `end` (exclusive). end=None loads everything -- only when a segment is being read."""
    d = pd.read_parquet(f"{ROOT}/klines_1m/{base}USDT.parquet")
    if end is not None:
        d = d[d.date < end]
    d = d.reset_index(drop=True)
    out = {k: d[k].to_numpy(np.float64) for k in ("open", "high", "low", "close", "volume", "quote_volume",
                                                  "count", "taker_buy_quote_volume")}
    out["date"] = d["date"]
    return out


def roll_med(x: np.ndarray, w: int) -> np.ndarray:
    """Trailing median over the previous w bars, excluding the current bar (no lookahead)."""
    return pd.Series(x).rolling(w, min_periods=w // 2).median().shift(1).to_numpy()


def segment(t: pd.Series, seg: tuple[str, str]) -> np.ndarray:
    return ((t >= pd.Timestamp(seg[0], tz="UTC")) & (t < pd.Timestamp(seg[1], tz="UTC"))).to_numpy()


def summarize(trades: dict[str, pd.DataFrame]) -> dict:
    allr = np.concatenate([t.ret.to_numpy() for t in trades.values() if len(t)]) if trades else np.array([])
    s = H.stats(allr)
    s.update(H.day_clustered(trades))
    if s.get("n"):
        days = pd.concat([pd.Series(pd.DatetimeIndex(t.t).floor("D")) for t in trades.values() if len(t)])
        s["tr_days"] = int(days.nunique())
    return s
