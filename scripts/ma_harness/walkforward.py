"""Walk-forward test of the "moving-average screening" harness described in
https://zhuanlan.zhihu.com/p/2032120164750205868 .

The article's strategy is a three-layer rolling selection framework:

  layer 1  universe   top N USDT perpetuals by dollar volume
  layer 2  two-stage race
              stage a  for every coin, backtest a grid of (fast, slow) MA
                       crossover parameters over recent history and keep the
                       parameter set with the highest composite score
              stage b  rank the coins by that best score and keep the top K
                       as the whitelist, each with its own parameters
  layer 3  live        trade the whitelist with stop-and-reverse crossovers
                       plus a dynamic trailing take profit

The composite score is quoted verbatim in the article:

    score = min(winRate * 100, 100)              * 0.30
          + min(profitFactor * 20, 60)           * 0.30
          + max(0, 1 - maxDrawdown / maxMDD)*100 * 0.20
          + volPct * volPctBonus

Two constants the article does not give are set here and flagged as
assumptions: ``maxMDD`` (the drawdown that scores zero) and the parameter grid.
The article also never states a candle interval, so the timeframe is a flag.

The test question is the one the article itself leaves open: does the two-layer
race select coins and parameters that keep working out of sample? Every
rebalance therefore selects on a closed lookback window and is then traded on
the untouched days that follow, against three controls -- a random draw from
the same universe, the bottom of the same ranking, and buy-and-hold BTC.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

ROOT = "/root/freqtrade/user_data/factor_research"
sys.path.insert(0, ROOT)
from research.panel import Panel  # noqa: E402

BARS_PER_DAY = {"5m": 288, "15m": 96, "30m": 48, "1h": 24, "4h": 6, "1d": 1}

# Assumption: a plausible grid for a crossover screen. The article shows the
# loop over `maParamsList` but never prints its contents.
FAST = [5, 8, 10, 12, 15, 20, 25, 30, 40, 50]
SLOW = [20, 30, 40, 50, 60, 80, 100, 120, 150, 200]
GRID = [(f, s) for f in FAST for s in SLOW if s >= 2 * f]

# Layer 3 defaults. The tiers are the article's; the activation threshold and
# the market-state cutoffs are not stated there and are assumptions.
TIERS = [(7.0, 3.0), (4.0, 2.0), (0.0, 1.5)]
_OVERLAY = {"trail": False, "long_only": False, "trail_start": 2.0,
            "tiers": TIERS, "allow_short": None, "scale": None}

MAX_MDD = 30.0        # assumption: drawdown in percent that scores zero
VOL_BONUS = 10.0      # article default


def sma_matrix(close, windows):
    """Simple moving averages for several windows at once, shape (T, W)."""
    c = np.concatenate([[0.0], np.cumsum(close)])
    out = np.full((close.size, len(windows)), np.nan)
    for j, w in enumerate(windows):
        out[w - 1:, j] = (c[w:] - c[:-w]) / w
    return out


def cross_stats(close, fast_ma, slow_ma, fee):
    """Stop-and-reverse crossover backtest on one series.

    Returns win rate, profit factor, max drawdown (percent, on the closed-trade
    equity curve) and the trade count -- the four numbers the article's score
    consumes.
    """
    diff = fast_ma - slow_ma
    sig = np.sign(diff)
    ok = np.isfinite(diff)
    sig[~ok] = 0.0
    # a cross is a sign change; the position is taken on the next bar's close
    change = np.flatnonzero((sig[1:] != sig[:-1]) & (sig[1:] != 0) & ok[:-1])
    if change.size < 2:
        return None
    entry = change + 1              # bar whose close we transact at
    entry = entry[entry < close.size]
    if entry.size < 2:
        return None
    side = sig[entry]               # the new side the cross puts us on
    px = close[entry]
    ret = side[:-1] * (px[1:] / px[:-1] - 1.0) - 2.0 * fee
    if ret.size == 0:
        return None
    wins = ret[ret > 0]
    losses = ret[ret < 0]
    gross_loss = -losses.sum()
    pf = wins.sum() / gross_loss if gross_loss > 0 else (10.0 if wins.sum() > 0 else 0.0)
    eq = np.cumprod(1.0 + ret)
    dd = 100.0 * (1.0 - eq / np.maximum.accumulate(eq))
    return float(wins.size / ret.size), float(pf), float(dd.max()), int(ret.size)


def vol_percentile(close, win, hist):
    """Where the trailing realised volatility sits inside its own history."""
    r = np.diff(np.log(close))
    if r.size < win + 10:
        return 0.5
    s = pd.Series(r).rolling(win).std().to_numpy()
    cur = s[-1]
    past = s[-hist:] if s.size > hist else s
    past = past[np.isfinite(past)]
    if not np.isfinite(cur) or past.size < 10:
        return 0.5
    return float((past <= cur).mean())


def score_coin(close, fee, vol_win, vol_hist):
    """Stage a: race every parameter set on one coin, return the winner."""
    windows = sorted(set(FAST) | set(SLOW))
    mas = sma_matrix(close, windows)
    col = {w: i for i, w in enumerate(windows)}
    volpct = vol_percentile(close, vol_win, vol_hist)
    best = None
    for f, s in GRID:
        st = cross_stats(close, mas[:, col[f]], mas[:, col[s]], fee)
        if st is None:
            continue
        wr, pf, mdd, n = st
        if n < 5:                    # too few signals to rank on
            continue
        sc = (min(wr * 100.0, 100.0) * 0.30
              + min(pf * 20.0, 60.0) * 0.30
              + max(0.0, 1.0 - mdd / MAX_MDD) * 100.0 * 0.20
              + volpct * VOL_BONUS)
        if best is None or sc > best[0]:
            best = (sc, f, s, wr, pf, mdd, n)
    return best


def trail_positions(sig, close, trail_start, tiers, allow_short):
    """Layer 3 overlay: dynamic trailing take profit on close prices.

    ``sig`` is the raw crossover side per bar, unshifted. The state returned is
    the position *decided* at each bar's close; the caller shifts it by one bar
    before applying returns, so no decision uses the bar it earns.

    The article gives the tier table but not the activation threshold, so
    ``trail_start`` is an assumption. Once a position's paper profit has passed
    it, giving back the tier amount from the peak closes the sleeve, which then
    stays flat until the next crossover.
    """
    out = np.zeros_like(sig)
    held = 0.0
    entry = np.nan
    peak = 0.0
    prev = 0.0
    for i in range(sig.size):
        cur = sig[i]
        if cur != prev and cur != 0.0:        # a fresh crossover opens
            held, entry, peak = cur, close[i], 0.0
        elif held != 0.0:
            pnl = 100.0 * held * (close[i] / entry - 1.0)
            peak = max(peak, pnl)
            if peak >= trail_start:
                give = next(g for lvl, g in tiers if peak >= lvl)
                if peak - pnl >= give:
                    held = 0.0                # flat until the signal flips
        if allow_short is not None and not allow_short[i] and held < 0:
            held = 0.0
        prev = cur
        out[i] = held
    return out


def oos_returns(close, funding, f, s, fee, bar_hours):
    """Bar returns of a stop-and-reverse sleeve, net of fees and funding.

    ``close`` must start ``s`` bars before the out-of-sample window so the
    moving averages are warm on its first bar; the returned array covers the
    whole slice and the caller keeps its tail.
    """
    windows = sorted({f, s})
    mas = sma_matrix(close, windows)
    d = mas[:, windows.index(f)] - mas[:, windows.index(s)]
    sig = np.sign(d)
    sig[~np.isfinite(d)] = 0.0
    if _OVERLAY["long_only"]:
        sig = np.maximum(sig, 0.0)
    if _OVERLAY["trail"]:
        state = trail_positions(sig, close, _OVERLAY["trail_start"],
                                _OVERLAY["tiers"], _OVERLAY.get("allow_short"))
    else:
        state = sig
        if _OVERLAY.get("allow_short") is not None:
            state = np.where(_OVERLAY["allow_short"] | (state > 0), state, 0.0)
    pos = np.concatenate([[0.0], state[:-1]])    # decided at i-1, earned at i
    if _OVERLAY.get("scale") is not None:
        pos = pos * _OVERLAY["scale"]
    px_ret = np.concatenate([[0.0], close[1:] / close[:-1] - 1.0])
    turn = np.abs(np.diff(np.concatenate([[0.0], pos])))
    fund = np.nan_to_num(funding) * (bar_hours / 8.0)
    return pos * px_ret - turn * fee - pos * fund, pos


def rebalance(args):
    """One walk-forward step: select on the lookback, trade the days after."""
    (t, cfg, tf) = args
    p = Panel(tf, mmap=True)
    close = p["close"]
    qv = p["quote_volume"]
    mask = p["universe_mask"]
    fr = p["funding_rate"]
    bpd = BARS_PER_DAY[tf]
    bar_hours = 24.0 / bpd

    is_lo = t - cfg["lookback_days"] * bpd
    vol_lo = t - cfg["volume_days"] * bpd
    oos_hi = min(t + cfg["oos_days"] * bpd, close.shape[0])
    if is_lo < 0 or oos_hi - t < bpd:
        return None

    # layer 1: dollar-volume universe, tradable and fully observed in sample
    dollar = np.nansum(qv[vol_lo:t], axis=0)
    valid = (np.isfinite(close[is_lo:t]).all(axis=0)
             & mask[t - 1] & (close[t - 1] > 0))
    dollar = np.where(valid, dollar, -1.0)
    universe = np.argsort(-dollar)[: cfg["top_n"]]
    universe = universe[dollar[universe] > 0]
    if universe.size < cfg["top_k"] * 2:
        return None

    # layer 2: two-stage race
    rows = []
    for c in universe:
        best = score_coin(np.asarray(close[is_lo:t, c], dtype=np.float64),
                          cfg["sel_fee"], cfg["vol_win"] * bpd,
                          cfg["vol_hist"] * bpd)
        if best is None:
            continue
        sc, f, s, wr, pf, mdd, n = best
        rows.append(dict(col=int(c), symbol=p.symbols[c], score=sc, fast=f, slow=s,
                         is_winrate=wr, is_pf=pf, is_mdd=mdd, is_trades=n,
                         dollar=float(dollar[c])))
    if len(rows) < cfg["top_k"] * 2:
        return None
    rows.sort(key=lambda r: -r["score"])
    for i, r in enumerate(rows):
        r["rank"] = i

    # layer 3: trade every candidate out of sample so the controls come free
    warm = max(SLOW) + 5
    lo = max(0, t - warm)
    rng = np.random.default_rng(t)
    _OVERLAY["trail"] = cfg["trail"]
    _OVERLAY["long_only"] = cfg["long_only"]
    _OVERLAY["trail_start"] = cfg["trail_start"]
    _OVERLAY["allow_short"] = None
    _OVERLAY["scale"] = None
    if cfg["market_state"]:
        btc = p.symbols.index("BTCUSDT")
        b = np.asarray(close[max(0, lo - cfg["vol_hist"] * bpd):oos_hi, btc],
                       dtype=np.float64)
        r = np.diff(np.log(b))
        v = pd.Series(r).rolling(cfg["vol_win"] * bpd).std().to_numpy()
        pct = pd.Series(v).rolling(cfg["vol_hist"] * bpd, min_periods=200)\
                .rank(pct=True).to_numpy()
        pct = np.concatenate([[np.nan], pct])[-(oos_hi - lo):]
        pct = np.nan_to_num(pct, nan=0.5)
        scale = np.ones_like(pct)
        scale[pct > 0.9] = 0.5          # volatile
        scale[(pct > 0.7) & (pct <= 0.9)] = 0.8
        scale[pct < 0.3] = 0.7          # low_vol
        _OVERLAY["scale"] = scale
        _OVERLAY["allow_short"] = pct <= 0.9   # no shorts in extreme regimes
    for r in rows:
        c = r["col"]
        cl = np.asarray(close[lo:oos_hi, c], dtype=np.float64)
        fu = np.asarray(fr[lo:oos_hi, c], dtype=np.float64)
        good = np.isfinite(cl)
        if not good[t - lo:].all():
            cl = pd.Series(cl).ffill().to_numpy()
            good2 = np.isfinite(cl[t - lo:])
            if good2.mean() < 0.9:
                r["oos_total"] = None
                continue
        ret, pos = oos_returns(cl, fu, r["fast"], r["slow"], cfg["fee"], bar_hours)
        seg = ret[t - lo:]
        # control: the same coin traded with a parameter set drawn at random,
        # which isolates what the parameter race itself contributes
        rf, rs = GRID[rng.integers(len(GRID))]
        rret, _ = oos_returns(cl, fu, rf, rs, cfg["fee"], bar_hours)
        r["oos_total_rand"] = float(np.prod(1.0 + np.nan_to_num(rret[t - lo:])) - 1.0)
        # control: one fixed slow crossover for everyone, no race at all
        fret, _ = oos_returns(cl, fu, cfg["fixed_fast"], cfg["fixed_slow"],
                              cfg["fee"], bar_hours)
        r["oos_total_fixed"] = float(np.prod(1.0 + np.nan_to_num(fret[t - lo:])) - 1.0)
        # bar-level detail only for the ranks a portfolio could hold; the
        # period total is enough for the ranking and control statistics
        r["oos_ret"] = (np.nan_to_num(seg).tolist()
                        if r["rank"] < cfg["detail_ranks"] else [])
        r["oos_total"] = float(np.prod(1.0 + np.nan_to_num(seg)) - 1.0)
        r["oos_bars"] = int(seg.size)
    return {"t": int(t), "rows": [r for r in rows if r.get("oos_total") is not None]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="1h")
    ap.add_argument("--lookback-days", type=int, default=90)
    ap.add_argument("--oos-days", type=int, default=7)
    ap.add_argument("--volume-days", type=int, default=30)
    ap.add_argument("--top-n", type=int, default=150)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--fee", type=float, default=0.0005, help="per side, out of sample")
    ap.add_argument("--sel-fee", type=float, default=0.0, help="per side, in the screen")
    ap.add_argument("--vol-win", type=int, default=7, help="days of realised vol")
    ap.add_argument("--vol-hist", type=int, default=180, help="days of vol history")
    ap.add_argument("--trail", action="store_true", help="dynamic trailing take profit")
    ap.add_argument("--trail-start", type=float, default=2.0)
    ap.add_argument("--long-only", action="store_true")
    ap.add_argument("--market-state", action="store_true")
    ap.add_argument("--fixed-fast", type=int, default=50)
    ap.add_argument("--fixed-slow", type=int, default=200)
    ap.add_argument("--detail-ranks", type=int, default=40)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg = dict(lookback_days=a.lookback_days, oos_days=a.oos_days,
               volume_days=a.volume_days, top_n=a.top_n, top_k=a.top_k,
               fee=a.fee, sel_fee=a.sel_fee, vol_win=a.vol_win,
               vol_hist=a.vol_hist, detail_ranks=a.detail_ranks,
               trail=a.trail, trail_start=a.trail_start,
               long_only=a.long_only, market_state=a.market_state,
               fixed_fast=a.fixed_fast, fixed_slow=a.fixed_slow)
    p = Panel(a.tf, mmap=True)
    bpd = BARS_PER_DAY[a.tf]
    start = max(a.lookback_days, a.volume_days) * bpd
    steps = list(range(start, p.shape[0] - bpd, a.oos_days * bpd))
    print(f"{a.tf}: {len(steps)} rebalances, {len(GRID)} parameter sets, "
          f"{p.index[steps[0]].date()} -> {p.index[steps[-1]].date()}", flush=True)

    out = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for res in ex.map(rebalance, [(t, cfg, a.tf) for t in steps]):
            if res is not None:
                out.append(res)
                print(f"  step {p.index[res['t']].date()} "
                      f"{len(res['rows'])} coins", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"config": {**cfg, "tf": a.tf, "grid": GRID,
                              "max_mdd": MAX_MDD, "vol_bonus": VOL_BONUS},
                   "index_start": str(p.index[0]), "steps": out}, fh)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
