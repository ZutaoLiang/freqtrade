#!/usr/bin/env python3
"""Second signal-family battery: jump reversal, BTC lead-lag, listing age,
time-of-day seasonality.

Families already falsified in the first sweep (funding carry, plain
reversal, single-horizon momentum, low-vol, volume growth, tsmom, blends)
are NOT retested — see the high-volume-trend-research skill.

Universes: 2025 rotating pool (72 pairs, binance-2025) and 2026 all-perps
(binance/futures), 1h candles, liquidity floor 30M as before.  Every effect
must agree in sign across both universes and their halves before any
freqtrade implementation.  Fees quoted at 0.07%/side taker.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analyze_smallcap_carry_xs import load_universe, MIN_QVOL

FEE_RT = 0.0014  # round trip


def halves(idx, year_tag):
    if year_tag == "2025":
        cuts = [("25H1", "2025-02-01", "2025-08-01"), ("25H2", "2025-08-01", "2026-02-01")]
    else:
        cuts = [("26H1", "2026-01-15", "2026-05-01"), ("26H2", "2026-05-01", "2026-08-14")]
    return {lbl: (idx >= pd.Timestamp(a, tz="UTC")) & (idx < pd.Timestamp(b, tz="UTC"))
            for lbl, a, b in cuts}


def tstat(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < 20 or s.std() == 0:
        return float("nan")
    return float(s.mean() / s.std() * np.sqrt(len(s)))


def jump_reversal(close, qvol, elig, spl):
    """Fade a single-candle move beyond 4 trailing sigmas."""
    ret1 = close.pct_change()
    sig = ret1.rolling(168, min_periods=84).std()
    z = ret1 / sig
    print("  jump fade (|z|>4, enter next close, fade direction):")
    for h in (4, 12, 24, 48):
        fwd = close.shift(-1 - h) / close.shift(-1) - 1.0  # entered one candle later
        pnl = (-np.sign(ret1) * fwd).where((z.abs() > 4) & elig)
        flat = pnl.stack().dropna()
        cells = []
        for lbl, m in spl.items():
            seg = pnl[m].stack().dropna()
            cells.append(f"{lbl}: n={len(seg):5d} {seg.mean()*100:+6.2f}% t={tstat(seg):5.1f}")
        print(f"    H{h:2d} net_of_fee_mean={(flat.mean()-FEE_RT)*100:+6.2f}% | " + " | ".join(cells))


def btc_leadlag(close, elig, spl, btc):
    """Alt forward return conditioned on BTC trailing return sign."""
    btc = btc.reindex(close.index).ffill()
    fwd24 = close.shift(-24) / close - 1.0
    print("  BTC lead-lag (alt fwd 24h by BTC trailing sign, eligible alts):")
    for look in (4, 24, 72):
        cond = np.sign(btc / btc.shift(look) - 1.0)
        pnl = fwd24.where(elig).mul(cond, axis=0)  # long alts when BTC up, short when down
        cells = []
        for lbl, m in spl.items():
            seg = pnl[m].stack().dropna()
            # daily-ish sampling to reduce overlap bias in t-stat
            cells.append(f"{lbl}: {seg.mean()*100:+6.2f}%")
        # switch frequency of the BTC sign -> turnover estimate
        flips = cond.diff().abs().gt(0).mean()
        print(f"    look{look:3d}h flips/h={flips:.3f} | " + " | ".join(cells))


def listing_age(close, elig, spl):
    """Forward 24h return bucketed by days since first candle."""
    first = close.apply(lambda s: s.first_valid_index())
    age_days = pd.DataFrame(
        {p: (close.index - first[p]).days if first[p] is not None else np.nan
         for p in close.columns}, index=close.index)
    fwd24 = close.shift(-24) / close - 1.0
    hourly0 = close.index.hour == 0  # one observation per day per pair
    buckets = [(1, 3), (4, 7), (8, 14), (15, 30), (31, 10_000)]
    print("  listing age (fwd 24h mean at 00:00, eligible only):")
    for lo, hi in buckets:
        mask = (age_days >= lo) & (age_days <= hi) & elig
        pnl = fwd24.where(mask)
        pnl = pnl[hourly0]
        cells = []
        for lbl, m in spl.items():
            seg = pnl[m[hourly0]].stack().dropna()
            cells.append(f"{lbl}: n={len(seg):5d} {seg.mean()*100:+6.2f}% t={tstat(seg):5.1f}")
        print(f"    d{lo:3d}-{hi:5d} | " + " | ".join(cells))


def seasonality(close, elig, spl):
    """Mean 1h return by UTC hour and by weekday (eligible pairs pooled)."""
    ret1 = close.pct_change().where(elig)
    pooled = ret1.mean(axis=1)
    print("  seasonality (pooled mean 1h ret, bp):")
    for lbl, m in spl.items():
        seg = pooled[m]
        byh = seg.groupby(seg.index.hour).mean() * 1e4
        top = byh.sort_values()
        print(f"    {lbl} hours: worst {top.index[0]:2d}h {top.iloc[0]:+5.1f} | "
              f"best {top.index[-1]:2d}h {top.iloc[-1]:+5.1f}")
        byd = seg.groupby(seg.index.dayofweek).mean() * 24 * 1e4
        print(f"      weekdays(bp/day): " + " ".join(f"{d}:{v:+5.0f}" for d, v in byd.items()))


def main() -> None:
    uni = {
        "2025": ("user_data/data/binance-2025/futures", "2025-02-01", "2026-01-31"),
        "2026": ("user_data/data/binance/futures", "2026-01-15", "2026-08-13"),
    }
    btc = None
    btc_px = pd.read_feather("user_data/data/binance/futures/BTC_USDT_USDT-30m-futures.feather")
    btc = btc_px.set_index("date")["close"].sort_index()
    btc = btc[~btc.index.duplicated(keep="last")].resample("1h").last()

    for tag, (pdir, a, b) in uni.items():
        close, qvol, _ = load_universe(pdir, a, b)
        # load_universe returns 8h grid; rebuild 1h here instead
        print(f"\n### {tag} ({pdir}) — rebuilding 1h panel")
        import glob, os
        closes, qvols = {}, {}
        for f in sorted(glob.glob(f"{pdir}/*-1h-futures.feather")):
            p = os.path.basename(f).split("-1h-")[0]
            try:
                px = pd.read_feather(f, columns=["date", "close", "volume"]).set_index("date").sort_index()
            except Exception:
                continue
            if px.empty:
                continue
            px = px[~px.index.duplicated(keep="last")]
            closes[p] = px["close"]
            qvols[p] = px["close"] * px["volume"]
        close = pd.DataFrame(closes).sort_index()
        close = close[(close.index >= pd.Timestamp(a, tz="UTC")) & (close.index <= pd.Timestamp(b, tz="UTC"))]
        qvol = pd.DataFrame(qvols).sort_index().reindex(close.index)
        qtrail = qvol.rolling(24, min_periods=12).sum().rolling(168, min_periods=84).mean()
        elig = qtrail > MIN_QVOL
        spl = halves(close.index, tag)
        print(f"pairs={close.shape[1]} eligible median={elig.sum(axis=1).median():.0f}")
        jump_reversal(close, qvol, elig, spl)
        btc_leadlag(close, elig, spl, btc)
        listing_age(close, elig, spl)
        seasonality(close, elig, spl)


if __name__ == "__main__":
    main()
