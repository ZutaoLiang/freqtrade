"""Overshoot reversion with a maker limit entry: does the entry discount create the edge?

Entry: on an idiosyncratic |z| spike with a volume spike and BTC quiet, rest a limit
`k * ATR` deeper into the overshoot direction. Fill requires the price to trade through
the limit by `buffer` bps within `fill_window` bars (the SKILL.md 穿价版 rule).
Exit: fixed horizon after the fill bar. Reports fills, gross bps and hit rate.
"""
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

DATA = "/root/freqtrade/user_data/data/binance_public/freqtrade/futures"
SCRATCH = "/root/freqtrade/user_data/niche_work"
WINDOWS = {"train": ("2025-11-01", "2026-05-31"), "hold": ("2026-06-01", "2026-08-16")}
SIGMA_WIN = 240
HORIZONS = (5, 10, 20, 30)
GRID = [(m, k, buf) for m in (4.0, 6.0) for k in (0.5, 1.0) for buf in (0.0, 8.0)]
FILL_WINDOW = 3


def load(sym):
    df = pd.read_feather(f"{DATA}/{sym}-1m-futures.feather").set_index("date")
    return df[~df.index.duplicated()]


def _init():
    b = load("BTC_USDT_USDT")
    r = np.log(b["close"]).diff()
    _run.btc = (r / r.rolling(SIGMA_WIN).std()).rename("btc_z")


def _run(sym):
    try:
        df = load(sym).join(_run.btc, how="left")
    except Exception:
        return []
    c, h, lo = df["close"], df["high"], df["low"]
    r = np.log(c).diff()
    sigma = r.rolling(SIGMA_WIN).std()
    move_z = r / sigma
    tr = pd.concat([h - lo, (h - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    v = df["volume"]
    vol_z = (v - v.rolling(SIGMA_WIN).mean()) / v.rolling(SIGMA_WIN).std()

    # best price reached over the next FILL_WINDOW bars (bar t+1 .. t+FILL_WINDOW)
    fut_low = lo.shift(-1).rolling(FILL_WINDOW).min().shift(-(FILL_WINDOW - 1))
    fut_high = h.shift(-1).rolling(FILL_WINDOW).max().shift(-(FILL_WINDOW - 1))
    exit_px = {n: c.shift(-(FILL_WINDOW + n)) for n in HORIZONS}

    ok = sigma.notna() & atr.notna() & vol_z.notna() & df["btc_z"].notna()
    rows = []
    for label, (a, b) in WINDOWS.items():
        win = ok & (df.index >= a) & (df.index <= b)
        for m, k, buf in GRID:
            base = win & (vol_z > 2.0) & (df["btc_z"].abs() < 2.0)
            dn = base & (move_z < -m)          # overshot down -> rest a bid below
            up = base & (move_z > m)           # overshot up   -> rest an ask above
            n_ev = int(dn.sum() + up.sum())
            if n_ev < 20:
                continue
            bid = c * (1 - buf / 1e4) - k * atr
            ask = c * (1 + buf / 1e4) + k * atr
            f_l = dn & (fut_low <= bid)
            f_s = up & (fut_high >= ask)
            n_fill = int(f_l.sum() + f_s.sum())
            if n_fill < 20:
                continue
            row = dict(sym=sym, window=label, m=m, k=k, buf=buf,
                       n_ev=n_ev, n_fill=n_fill, fill_rate=n_fill / n_ev)
            for hz in HORIZONS:
                pnl = pd.concat([
                    np.log(exit_px[hz][f_l] / bid[f_l]),
                    -np.log(exit_px[hz][f_s] / ask[f_s]),
                ]).dropna()
                if len(pnl) < 20:
                    continue
                row[f"bps_{hz}"] = 1e4 * pnl.mean()
                row[f"hit_{hz}"] = (pnl > 0).mean()
            rows.append(row)
    return rows


def main():
    u = pd.read_csv(f"{SCRATCH}/universe.csv")
    syms = u[(u.med_qv > 3e6) & (u.med_qv < 6e7) & (u.days >= 250)].sym.tolist()
    print(f"pairs: {len(syms)}", flush=True)
    out = []
    with ProcessPoolExecutor(max_workers=int(sys.argv[1]), initializer=_init) as ex:
        for rows in ex.map(_run, syms, chunksize=2):
            out.extend(rows)
    d = pd.DataFrame(out)
    d.to_csv(f"{SCRATCH}/overshoot_maker.csv", index=False)
    for (w, m, k, buf), g in d.groupby(["window", "m", "k", "buf"]):
        tot = g.n_fill.sum()
        parts = [f"{w:5s} m={m} k={k} buf={buf:4.1f} pairs={len(g):3d} "
                 f"fills={tot:6d} fill_rate={g.n_fill.sum()/g.n_ev.sum():.3f}"]
        for hz in HORIZONS:
            col = f"bps_{hz}"
            if col in g and g[col].notna().any():
                gg = g[g[col].notna()]
                parts.append(f"{hz}m:{(gg[col]*gg.n_fill).sum()/gg.n_fill.sum():+6.1f}bps "
                             f"hit{(gg[f'hit_{hz}']*gg.n_fill).sum()/gg.n_fill.sum():.3f}")
        print("  ".join(parts))


if __name__ == "__main__":
    main()
