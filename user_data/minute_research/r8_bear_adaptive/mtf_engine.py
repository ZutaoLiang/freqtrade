"""Vectorized MTF Research Engine with Asymmetric Multi-Timeframe Execution (r8_bear_adaptive).

Features:
- Asymmetric parameters for Long vs Short (holding time, stop-loss, take-profit).
- Memory-mapped zero-copy multi-timeframe panel loader (1d, 4h, 1h).
- Strictly causal mapping from HTF to LTF (Zero Lookahead).
- Full derivative fields: OI, Funding Rate, Taker Volume.
- Scheme-C triple-split evaluation:
  * TRAIN: 2025-01-01 to 2025-10-01 (273 days, bull market)
  * VALID-C: 2025-12-01 to 2026-03-01 (90 days, choppy/distribution market)
  * HOLDOUT: 2026-03-01 to 2026-08-31 (184 days, brutal bear market)
- Realistic taker fees (20 bps round trip, 15 bps for BTC/ETH).
- Separate Long/Short and combined breakdown metrics.
"""
import os
import json
import numba as nb
import numpy as np
import pandas as pd

PANELS_DIR = "/root/freqtrade/user_data/data/binance_public/panels"
UNIVERSE_CSV = "/root/freqtrade/user_data/minute_research/r3/universe_rank.csv"

TRAIN_START = "2025-01-01 00:00:00+00:00"
TRAIN_END = "2025-10-01 00:00:00+00:00"
VALID_START = "2025-12-01 00:00:00+00:00"
VALID_END = "2026-03-01 00:00:00+00:00"
HOLDOUT_START = "2026-03-01 00:00:00+00:00"
HOLDOUT_END = "2026-08-31 00:00:00+00:00"


class MTFPanels:
    def __init__(self, base_tf: str = "1h"):
        self.base_tf = base_tf
        self.dir = f"{PANELS_DIR}/{base_tf}"
        meta = json.load(open(f"{self.dir}/meta.json"))
        self.symbols = meta["symbols"]
        self.dates = pd.date_range(meta["start"], meta["end"], freq="15min" if base_tf == "15m" else base_tf)
        self.n_bars = len(self.dates)
        self.n_symbols = len(self.symbols)

        # Base OHLCV arrays
        self.open = np.load(f"{self.dir}/open.npy", mmap_mode="r")
        self.high = np.load(f"{self.dir}/high.npy", mmap_mode="r")
        self.low = np.load(f"{self.dir}/low.npy", mmap_mode="r")
        self.close = np.load(f"{self.dir}/close.npy", mmap_mode="r")
        self.volume = np.load(f"{self.dir}/volume.npy", mmap_mode="r")

        # Taker buy volume
        self.taker_buy = None
        if os.path.exists(f"{self.dir}/taker_buy_volume.npy"):
            self.taker_buy = np.load(f"{self.dir}/taker_buy_volume.npy", mmap_mode="r")

        # Funding rate & OI
        self.funding_rate = None
        if os.path.exists(f"{self.dir}/funding_rate.npy"):
            self.funding_rate = np.load(f"{self.dir}/funding_rate.npy", mmap_mode="r")

        self.oi = None
        if os.path.exists(f"{self.dir}/sum_open_interest_value.npy"):
            self.oi = np.load(f"{self.dir}/sum_open_interest_value.npy", mmap_mode="r")

        # Universe U162
        u_df = pd.read_csv(UNIVERSE_CSV)
        u162_set = set(b + "USDT" for b in u_df[u_df.med_qv >= 1e7].base)
        self.u162_indices = np.array([i for i, s in enumerate(self.symbols) if s in u162_set], dtype=np.int32)

        # Slice indices
        self.train_mask = (self.dates >= TRAIN_START) & (self.dates < TRAIN_END)
        self.valid_mask = (self.dates >= VALID_START) & (self.dates < VALID_END)
        self.holdout_mask = (self.dates >= HOLDOUT_START) & (self.dates < HOLDOUT_END)

        self.train_range = (np.where(self.train_mask)[0][0], np.where(self.train_mask)[0][-1] + 1)
        self.valid_range = (np.where(self.valid_mask)[0][0], np.where(self.valid_mask)[0][-1] + 1)
        self.holdout_range = (np.where(self.holdout_mask)[0][0], np.where(self.holdout_mask)[0][-1] + 1)

    def map_htf(self, htf_arr: np.ndarray, htf_tf: str = "4h") -> np.ndarray:
        """Strict causal mapping: LTF bar t gets HTF bar from (t // ratio) - 1."""
        if htf_arr.dtype == bool or np.issubdtype(htf_arr.dtype, np.bool_):
            out = np.zeros((self.n_bars, htf_arr.shape[1]), dtype=bool)
        else:
            out = np.full((self.n_bars, htf_arr.shape[1]), np.nan, dtype=htf_arr.dtype)

        if htf_tf == "4h" and self.base_tf == "1h":
            ratio = 4
        elif htf_tf == "1d" and self.base_tf == "1h":
            ratio = 24
        elif htf_tf == "1h" and self.base_tf == "15m":
            ratio = 4
        elif htf_tf == "4h" and self.base_tf == "15m":
            ratio = 16
        elif htf_tf == "1d" and self.base_tf == "15m":
            ratio = 96
        else:
            ratio = 1

        for t in range(self.n_bars):
            htf_idx = (t // ratio) - 1
            if 0 <= htf_idx < htf_arr.shape[0]:
                out[t] = htf_arr[htf_idx]
        return out


def calc_safe_ret(c: np.ndarray) -> np.ndarray:
    """Strictly causal returns: ret[k] is (c[k] - c[k-1])/c[k-1]. ret[0] is NaN."""
    ret = np.full_like(c, np.nan, dtype=np.float64)
    ret[1:] = (c[1:] - c[:-1]) / np.maximum(c[:-1], 1e-12)
    return ret


@nb.njit(cache=True)
def nb_simulate_asymmetric(
    o: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    sig_long: np.ndarray,
    sig_short: np.ndarray,
    hold_bars_l: int,
    hold_bars_s: int,
    cost_bps: float,
    start_bar: int,
    end_bar: int,
    sl_pct_l: float = 0.0,
    sl_pct_s: float = 0.0,
    tp_pct_l: float = 0.0,
    tp_pct_s: float = 0.0,
):
    n_bars = end_bar - start_bar
    ent_bars = np.empty(n_bars, dtype=np.int32)
    ext_bars = np.empty(n_bars, dtype=np.int32)
    sides = np.empty(n_bars, dtype=np.int8)
    rets = np.empty(n_bars, dtype=np.float64)

    k = 0
    busy = -1
    fee = 2.0 * cost_bps / 1e4

    for i in range(start_bar, end_bar - 1):
        if i < busy:
            continue
        is_long = sig_long[i]
        is_short = sig_short[i]
        if not (is_long or is_short):
            continue

        ent_idx = i + 1
        entry_price = o[ent_idx]
        if np.isnan(entry_price) or entry_price <= 0:
            continue

        if is_long:
            side = 1
            hold_bars = hold_bars_l
            sl_pct = sl_pct_l
            tp_pct = tp_pct_l
        else:
            side = -1
            hold_bars = hold_bars_s
            sl_pct = sl_pct_s
            tp_pct = tp_pct_s

        max_ext = min(ent_idx + hold_bars, end_bar - 1)
        ext_idx = max_ext
        exit_price = c[max_ext]

        for step_bar in range(ent_idx, max_ext + 1):
            cur_l = l[step_bar]
            cur_h = h[step_bar]
            if side == 1:
                if sl_pct > 0.0 and cur_l <= entry_price * (1.0 - sl_pct):
                    exit_price = entry_price * (1.0 - sl_pct)
                    ext_idx = step_bar
                    break
                elif tp_pct > 0.0 and cur_h >= entry_price * (1.0 + tp_pct):
                    exit_price = entry_price * (1.0 + tp_pct)
                    ext_idx = step_bar
                    break
            else:
                if sl_pct > 0.0 and cur_h >= entry_price * (1.0 + sl_pct):
                    exit_price = entry_price * (1.0 + sl_pct)
                    ext_idx = step_bar
                    break
                elif tp_pct > 0.0 and cur_l <= entry_price * (1.0 - tp_pct):
                    exit_price = entry_price * (1.0 - tp_pct)
                    ext_idx = step_bar
                    break

        ret = side * (exit_price - entry_price) / entry_price - fee
        ent_bars[k] = ent_idx
        ext_bars[k] = ext_idx
        sides[k] = side
        rets[k] = ret
        k += 1
        busy = ext_idx

    return ent_bars[:k], ext_bars[:k], sides[:k], rets[:k]


def eval_signals_asymmetric(
    panels: MTFPanels,
    sig_long: np.ndarray,
    sig_short: np.ndarray,
    hold_bars_l: int = 18,
    hold_bars_s: int = 24,
    sl_pct_l: float = 0.06,
    sl_pct_s: float = 0.07,
    tp_pct_l: float = 0.10,
    tp_pct_s: float = 0.16,
    seg: str = "TRAIN",
):
    if seg == "TRAIN":
        start_idx, end_idx = panels.train_range
    elif seg == "VALID":
        start_idx, end_idx = panels.valid_range
    else:
        start_idx, end_idx = panels.holdout_range

    if sig_long.shape != panels.close.shape:
        sig_long = np.broadcast_to(sig_long, panels.close.shape)
    if sig_short.shape != panels.close.shape:
        sig_short = np.broadcast_to(sig_short, panels.close.shape)

    all_rets = []
    all_dates = []
    all_sides = []

    o = np.array(panels.open)
    h = np.array(panels.high)
    l = np.array(panels.low)
    c = np.array(panels.close)

    for col in panels.u162_indices:
        cost = 7.5 if panels.symbols[col] in ["BTCUSDT", "ETHUSDT"] else 10.0
        ent, ext, side, rets = nb_simulate_asymmetric(
            o[:, col], h[:, col], l[:, col], c[:, col],
            sig_long[:, col], sig_short[:, col],
            hold_bars_l, hold_bars_s, cost, start_idx, end_idx,
            sl_pct_l, sl_pct_s, tp_pct_l, tp_pct_s
        )
        if len(rets) > 0:
            all_rets.extend(rets)
            all_dates.extend(panels.dates[ent])
            all_sides.extend(side)

    if len(all_rets) == 0:
        return {
            "n_trades": 0, "win_rate": 0.0, "mean_bp": 0.0,
            "pf": 0.0, "t_stat": 0.0, "sharpe": 0.0, "n_days": 0,
            "n_long": 0, "win_long": 0.0, "pf_long": 0.0, "mean_long_bp": 0.0,
            "n_short": 0, "win_short": 0.0, "pf_short": 0.0, "mean_short_bp": 0.0,
        }

    df = pd.DataFrame({
        "date": pd.to_datetime(all_dates).date,
        "ret": all_rets,
        "side": all_sides
    })
    n_trades = len(df)
    win_rate = (df["ret"] > 0).mean() * 100.0
    mean_bp = df["ret"].mean() * 10000.0

    wins = df.loc[df["ret"] > 0, "ret"].sum()
    losses = -df.loc[df["ret"] < 0, "ret"].sum()
    pf = wins / losses if losses > 0 else 999.0

    # Daily aggregation for robust cluster t-stat
    daily = df.groupby("date")["ret"].sum()
    n_days = len(daily)
    d_mean = daily.mean()
    d_std = daily.std(ddof=1) if n_days > 1 else 1e-6
    t_stat = (d_mean / (d_std + 1e-12)) * np.sqrt(n_days) if n_days > 1 else 0.0
    sharpe = (d_mean / (d_std + 1e-12)) * np.sqrt(365.0) if n_days > 1 else 0.0

    # Separate Long and Short metrics
    df_l = df[df["side"] == 1]
    n_long = len(df_l)
    win_l = (df_l["ret"] > 0).mean() * 100.0 if n_long > 0 else 0.0
    mean_l = df_l["ret"].mean() * 10000.0 if n_long > 0 else 0.0
    wins_l = df_l.loc[df_l["ret"] > 0, "ret"].sum() if n_long > 0 else 0.0
    loss_l = -df_l.loc[df_l["ret"] < 0, "ret"].sum() if n_long > 0 else 0.0
    pf_l = wins_l / loss_l if loss_l > 0 else (999.0 if wins_l > 0 else 0.0)

    df_s = df[df["side"] == -1]
    n_short = len(df_s)
    win_s = (df_s["ret"] > 0).mean() * 100.0 if n_short > 0 else 0.0
    mean_s = df_s["ret"].mean() * 10000.0 if n_short > 0 else 0.0
    wins_s = df_s.loc[df_s["ret"] > 0, "ret"].sum() if n_short > 0 else 0.0
    loss_s = -df_s.loc[df_s["ret"] < 0, "ret"].sum() if n_short > 0 else 0.0
    pf_s = wins_s / loss_s if loss_s > 0 else (999.0 if wins_s > 0 else 0.0)

    return {
        "n_trades": n_trades,
        "win_rate": win_rate,
        "mean_bp": mean_bp,
        "pf": pf,
        "t_stat": t_stat,
        "sharpe": sharpe,
        "n_days": n_days,
        "n_long": n_long, "win_long": win_l, "pf_long": pf_l, "mean_long_bp": mean_l,
        "n_short": n_short, "win_short": win_s, "pf_short": pf_s, "mean_short_bp": mean_s,
    }
