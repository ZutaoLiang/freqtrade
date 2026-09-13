"""Build 1d_long / 4h_long panels (2022-11..2026-08) from the existing 1h_hist and 1h panels.

No download: 1h_hist covers 2022-11-01..2025-12-31 (598 symbols, OHLCV + funding on settlement
bars), the 1h panel supplies 2026-01-01..2026-08-16 (860 symbols). Symbols are the union.
Funding per output bar = mean of the settled rates inside the bar, carried forward (limit 3 bars).
Universe mask: >= 30 days history and trailing 7-day median daily quote volume >= 2.4M.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from config import atomic_save  # noqa: E402
from panel import CACHE, Panel  # noqa: E402

MIN_DAILY_QV = 2.4e6
MIN_HISTORY_DAYS = 30


def _frames(p, fields, lo, hi):
    a, b = int(np.searchsorted(p.index, lo)), int(np.searchsorted(p.index, hi))
    out = {}
    for f in fields:
        out[f] = pd.DataFrame(np.asarray(p[f][a:b], dtype="float64"), index=p.index[a:b], columns=p.symbols)
    return out


def _funding_1h(p, lo, hi):
    a, b = int(np.searchsorted(p.index, lo)), int(np.searchsorted(p.index, hi))
    fr = np.asarray(p["funding_rate"][a:b], dtype="float64")
    if (fr != 0).mean() < 0.5:     # settlement-only storage -> NaN off settlement bars
        fr = np.where(fr != 0, fr, np.nan)
    else:                          # carried storage -> keep only bars where the value changes (settlements)
        chg = np.ones_like(fr, dtype=bool); chg[1:] = fr[1:] != fr[:-1]
        fr = np.where(chg, fr, np.nan)
    return pd.DataFrame(fr, index=p.index[a:b], columns=p.symbols)


def build(tf_out):
    rule = {"1d": "1D", "4h": "4h"}[tf_out]
    hist, cur = Panel("1h_hist"), Panel("1h")
    cut = pd.Timestamp("2026-01-01", tz="UTC")
    fields = ["open", "high", "low", "close", "volume", "quote_volume"]
    fh = _frames(hist, fields, hist.index[0], cut)
    fc = _frames(cur, fields, cut, cur.index[-1] + pd.Timedelta("1h"))
    frh, frc = _funding_1h(hist, hist.index[0], cut), _funding_1h(cur, cut, cur.index[-1] + pd.Timedelta("1h"))
    syms = sorted(set(hist.symbols) | set(cur.symbols))
    out = {}
    for f in fields:
        x = pd.concat([fh[f], fc[f]]).reindex(columns=syms)
        agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "quote_volume": "sum"}[f]
        out[f] = x.resample(rule).agg(agg)
    # bars where the pair had no data at all: volume NaN -> make OHLC NaN too (resample 'sum' gives 0)
    nodata = pd.concat([fh["close"], fc["close"]]).reindex(columns=syms).resample(rule).count() == 0
    for f in fields:
        out[f] = out[f].mask(nodata)
    fr = pd.concat([frh, frc]).reindex(columns=syms).resample(rule).mean().ffill(limit=3)
    out["funding_rate"] = fr.reindex(out["close"].index)
    idx = out["close"].index
    # universe mask
    bpd = {"1d": 1, "4h": 6}[tf_out]
    qv_d = out["quote_volume"].resample("1D").sum()
    med7 = qv_d.rolling(7, min_periods=4).median().reindex(idx, method="ffill")
    first = out["close"].notna().cumsum() > 0
    hist_ok = first.rolling(MIN_HISTORY_DAYS * bpd, min_periods=MIN_HISTORY_DAYS * bpd).sum() >= MIN_HISTORY_DAYS * bpd
    mask = (med7 >= MIN_DAILY_QV) & hist_ok & out["close"].notna() & (out["volume"] > 0)
    d = os.path.join(CACHE, f"{tf_out}_long")
    os.makedirs(d, exist_ok=True)
    for k, v in out.items():
        atomic_save(os.path.join(d, f"{k}.npy"), lambda f, v=v: np.save(f, v.to_numpy(dtype="float32")))
    atomic_save(os.path.join(d, "universe_mask.npy"), lambda f: np.save(f, mask.to_numpy(dtype=bool)))
    meta = {"tf": f"{tf_out}_long", "symbols": syms, "fields": list(out) + ["universe_mask"],
            "start": str(idx[0]), "end": str(idx[-1]), "n_bars": int(len(idx))}
    atomic_save(os.path.join(d, "meta.json"), lambda f: f.write(json.dumps(meta).encode()))
    print(f"{tf_out}_long: {len(idx)} bars x {len(syms)} symbols {idx[0].date()}..{idx[-1].date()}; "
          f"tradeable per bar median {mask.sum(axis=1).median():.0f}, ever {mask.any().sum()}; funding finite {out['funding_rate'].notna().mean().mean():.2f}")


if __name__ == "__main__":
    for tf in sys.argv[1:] or ["1d", "4h"]:
        build(tf)
