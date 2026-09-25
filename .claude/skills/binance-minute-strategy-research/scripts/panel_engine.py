"""High-Performance Numba Vectorized Panel Research Engine.

Implementation reference for high-throughput strategy pre-screening on Binance Vision panels.
Guarantees:
  - Zero lookahead bias (all indicators computed causally; higher timeframes shifted by 1 bar).
  - Strict taker friction: 10 bps/side (20 bps round trip; 7.5 bps for BTC/ETH).
  - Next-bar open execution (signal on candle close, fill at next candle open).
  - Realistic one-position-per-coin constraint.
  - Day-clustered t-statistics, Profit Factor, win rate, and concentration audits.
  - Strict Option C split (TRAIN: 2025-01..10, EXCLUDED: 2025-10..11, VALID: 2025-12..2026-03, HOLDOUT: unread).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd

PANELS_DIR = os.environ.get("PANELS_DIR", "user_data/data/binance_public/panels")
UNIVERSE_CSV = os.environ.get("UNIVERSE_CSV", "user_data/minute_research/r3/universe_rank.csv")

# Standard Option C Time slices
TRAIN_START = os.environ.get("TRAIN_START", "2025-01-01 00:00:00+00:00")
TRAIN_END = os.environ.get("TRAIN_END", "2025-10-01 00:00:00+00:00")
VALID_START = os.environ.get("VALID_START", "2025-12-01 00:00:00+00:00")
VALID_END = os.environ.get("VALID_END", "2026-03-01 00:00:00+00:00")


class PanelData:
    """Memory-mapped multi-symbol OHLCV and derivative panels."""

    def __init__(self, tf: str = "1h", panels_dir: str | None = None):
        self.tf = tf
        self.dir = Path(panels_dir or PANELS_DIR) / tf
        meta_file = self.dir / "meta.json"
        if not meta_file.exists():
            raise FileNotFoundError(f"Panel metadata not found at {meta_file}")
            
        with open(meta_file) as f:
            meta = json.load(f)
            
        self.symbols = meta["symbols"]
        self.dates = pd.date_range(meta["start"], meta["end"], freq="15min" if tf == "15m" else tf)
        self.n_bars = len(self.dates)
        self.n_symbols = len(self.symbols)

        # Core price arrays with memory mapping (zero RAM overhead)
        self.open = np.load(self.dir / "open.npy", mmap_mode="r")
        self.high = np.load(self.dir / "high.npy", mmap_mode="r")
        self.low = np.load(self.dir / "low.npy", mmap_mode="r")
        self.close = np.load(self.dir / "close.npy", mmap_mode="r")
        self.volume = np.load(self.dir / "volume.npy", mmap_mode="r")

        # Optional derivative fields
        self.funding_rate = np.load(self.dir / "funding_rate.npy", mmap_mode="r") if (self.dir / "funding_rate.npy").exists() else None
        self.oi = np.load(self.dir / "sum_open_interest_value.npy", mmap_mode="r") if (self.dir / "sum_open_interest_value.npy").exists() else None
        self.toptrader_ls = np.load(self.dir / "count_toptrader_long_short_ratio.npy", mmap_mode="r") if (self.dir / "count_toptrader_long_short_ratio.npy").exists() else None
        self.quote_volume = np.load(self.dir / "quote_volume.npy", mmap_mode="r") if (self.dir / "quote_volume.npy").exists() else None
        self.taker_buy_volume = np.load(self.dir / "taker_buy_volume.npy", mmap_mode="r") if (self.dir / "taker_buy_volume.npy").exists() else None
        self.count = np.load(self.dir / "count.npy", mmap_mode="r") if (self.dir / "count.npy").exists() else None

        # Universe filtering (e.g. U162 or U60)
        if os.path.exists(UNIVERSE_CSV):
            u_df = pd.read_csv(UNIVERSE_CSV)
            u_set = set(b + "USDT" for b in u_df[u_df.med_qv >= 1e7].base)
            self.u_indices = np.array([i for i, s in enumerate(self.symbols) if s in u_set], dtype=np.int32)
        else:
            self.u_indices = np.arange(self.n_symbols, dtype=np.int32)

        # Segment masks
        self.train_mask = (self.dates >= TRAIN_START) & (self.dates < TRAIN_END)
        self.valid_mask = (self.dates >= VALID_START) & (self.dates < VALID_END)
        self.train_idx_range = (np.where(self.train_mask)[0][0], np.where(self.train_mask)[0][-1] + 1)
        self.valid_idx_range = (np.where(self.valid_mask)[0][0], np.where(self.valid_mask)[0][-1] + 1)

    def map_htf_to_ltf(self, htf_arr: np.ndarray, htf_tf: str = "4h") -> np.ndarray:
        """Map a higher-timeframe array causally to lower timeframe.
        Each bar t receives the value from the last COMPLETED HTF bar (shifted by 1).
        """
        out = np.full((self.n_bars, htf_arr.shape[1]), np.nan, dtype=htf_arr.dtype)
        ratio = 4 if (htf_tf == "4h" and self.tf == "1h") or (htf_tf == "1h" and self.tf == "15m") else (
            24 if (htf_tf == "1d" and self.tf == "1h") else (
                96 if (htf_tf == "1d" and self.tf == "15m") else (
                    16 if (htf_tf == "4h" and self.tf == "15m") else 1
                )
            )
        )
        for t in range(self.n_bars):
            htf_idx = (t // ratio) - 1
            if 0 <= htf_idx < htf_arr.shape[0]:
                out[t] = htf_arr[htf_idx]
        return out


@nb.njit(fastmath=True)
def simulate_fast(
    o: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    sig_long: np.ndarray,
    sig_short: np.ndarray,
    hold_bars: int,
    fee_bp: float = 10.0,
    start_bar: int = 0,
    end_bar: int = -1,
    sl_pct: float = 0.0,
    tp_pct: float = 0.0,
):
    """Fast Numba single-coin simulation strictly matching freqtrade execution semantics."""
    n = len(o)
    if end_bar < 0:
        end_bar = n
        
    ent_bars = np.empty(n, dtype=np.int32)
    ext_bars = np.empty(n, dtype=np.int32)
    sides = np.empty(n, dtype=np.int8)
    rets = np.empty(n, dtype=np.float64)
    
    k = 0
    busy = -1
    fee = 2.0 * fee_bp / 10000.0  # Round-trip taker fee
    
    for i in range(start_bar, end_bar - 1):
        if i < busy:
            continue
            
        sl = sig_long[i]
        ss = sig_short[i]
        if not (sl ^ ss):
            continue
            
        side = 1 if sl else -1
        e = i + 1  # Next-bar open fill
        if e >= end_bar:
            break
            
        px = o[e]
        if px <= 0.0 or np.isnan(px):
            continue
            
        max_x = min(e + hold_bars - 1, end_bar - 1)
        x_bar = max_x
        exit_px = c[max_x]
        
        # Intra-bar SL / TP checking
        if sl_pct > 0.0 or tp_pct > 0.0:
            for b in range(e, max_x + 1):
                if side == 1:
                    if sl_pct > 0.0 and l[b] <= px * (1.0 - sl_pct):
                        exit_px = min(o[b], px * (1.0 - sl_pct))
                        x_bar = b
                        break
                    if tp_pct > 0.0 and h[b] >= px * (1.0 + tp_pct):
                        exit_px = max(o[b], px * (1.0 + tp_pct))
                        x_bar = b
                        break
                else:
                    if sl_pct > 0.0 and h[b] >= px * (1.0 + sl_pct):
                        exit_px = max(o[b], px * (1.0 + sl_pct))
                        x_bar = b
                        break
                    if tp_pct > 0.0 and l[b] <= px * (1.0 - tp_pct):
                        exit_px = min(o[b], px * (1.0 - tp_pct))
                        x_bar = b
                        break
                        
        if exit_px <= 0.0 or np.isnan(exit_px):
            continue
            
        ret = side * (exit_px / px - 1.0) - fee
        ent_bars[k] = e
        ext_bars[k] = x_bar
        sides[k] = side
        rets[k] = ret
        k += 1
        busy = x_bar
        
    return ent_bars[:k], ext_bars[:k], sides[:k], rets[:k]


def evaluate_strategy(
    panel: PanelData,
    sig_long: np.ndarray,
    sig_short: np.ndarray,
    hold_bars: int,
    seg: str = "TRAIN",
    coin_indices: np.ndarray | None = None,
    sl_pct: float = 0.0,
    tp_pct: float = 0.0,
) -> dict:
    """Evaluates strategy performance across the specified universe and segment."""
    if coin_indices is None:
        coin_indices = panel.u_indices
        
    start_bar, end_bar = panel.train_idx_range if seg == "TRAIN" else panel.valid_idx_range
    days_count = 273 if seg == "TRAIN" else 90
    
    all_rets = []
    all_dates = []
    all_coins = []
    
    btc_idx = panel.symbols.index("BTCUSDT") if "BTCUSDT" in panel.symbols else -1
    
    for col in coin_indices:
        cost = 7.5 if (col == btc_idx or panel.symbols[col] == "ETHUSDT") else 10.0
        o = np.array(panel.open[:, col])
        h = np.array(panel.high[:, col])
        l = np.array(panel.low[:, col])
        c = np.array(panel.close[:, col])
        sl = sig_long[:, col]
        ss = sig_short[:, col]
        
        ent, ext, side, rets = simulate_fast(
            o, h, l, c, sl, ss, hold_bars, cost, start_bar, end_bar, sl_pct, tp_pct
        )
        if len(rets) > 0:
            all_rets.append(rets)
            all_dates.append(panel.dates[ent])
            all_coins.append(np.full(len(rets), panel.symbols[col]))
            
    if not all_rets:
        return {
            "n": 0, "mean_bp": 0.0, "med_bp": 0.0, "win": 0.0, "pf": 0.0,
            "days": 0, "day_mean_bp": 0.0, "day_t": 0.0, "days_pos": 0.0,
            "per_day": 0.0, "max_coin_share": 0.0, "max_day_share": 0.0,
            "sharpe": 0.0
        }
        
    r = np.concatenate(all_rets)
    d = np.concatenate(all_dates)
    syms = np.concatenate(all_coins)
    
    df = pd.DataFrame({"ret": r, "t": d, "sym": syms})
    df["day"] = pd.to_datetime(df.t).dt.floor("D")
    
    n = len(df)
    mean_bp = df.ret.mean() * 10000.0
    med_bp = df.ret.median() * 10000.0
    win = (df.ret > 0).mean()
    
    pos = df.ret[df.ret > 0].sum()
    neg = -df.ret[df.ret < 0].sum()
    pf = (pos / neg) if neg > 0 else 999.0
    
    # Day-clustered statistics
    day_series = df.groupby("day")["ret"].sum() * 10000.0
    n_days = len(day_series)
    day_mean = day_series.mean()
    day_std = day_series.std(ddof=1) if n_days > 1 else 1e-9
    day_t = (day_mean / (day_std / np.sqrt(n_days))) if day_std > 0 else 0.0
    
    # Annualized daily Sharpe
    sharpe = day_t * np.sqrt(365.0 / days_count) if days_count > 0 else 0.0
    
    # Concentration metrics
    tot_pnl = df.ret.sum()
    coin_pnl = df.groupby("sym")["ret"].sum()
    max_coin_share = (coin_pnl.max() / tot_pnl) if tot_pnl > 0 else 0.0
    day_pnl = df.groupby("day")["ret"].sum()
    max_day_share = (day_pnl.max() / tot_pnl) if tot_pnl > 0 else 0.0
    
    return {
        "n": n,
        "mean_bp": round(mean_bp, 2),
        "med_bp": round(med_bp, 2),
        "win": round(win, 3),
        "pf": round(pf, 3),
        "days": n_days,
        "day_mean_bp": round(day_mean, 2),
        "day_t": round(day_t, 2),
        "days_pos": round((day_series > 0).mean(), 3),
        "per_day": round(n / days_count, 2),
        "max_coin_share": round(max_coin_share, 3),
        "max_day_share": round(max_day_share, 3),
        "sharpe": round(sharpe, 2)
    }
