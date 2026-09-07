"""Build a 2022-11..2025-12 panel from the historical backfill.

The research engine reads panels of aligned (time, symbol) float32 matrices.
The existing one covers 2025-01 onward, which is the whole reason the k-fold
could only ever test one regime. This assembles the same layout from
``user_data/data/binance-hist/futures`` so the study can run on three years.

The tradability mask reproduces the rules recorded in the 2025 panel's
``universe_meta.json``: at least 30 days of history, and a trailing 7-day mean
daily quote volume above 2.4M USDT.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("/root/freqtrade/user_data/data/binance-hist/futures")
OUT = Path("/root/freqtrade/user_data/data/binance_public/panels/1h_hist")
MIN_HISTORY_DAYS = 30
LIQ_WINDOW_DAYS = 7
MIN_DAILY_QUOTE_VOLUME = 2_400_000.0


def main():
    files = sorted(SRC.glob("*-1h-futures.feather"))
    print(f"{len(files)} symbols")
    frames = {}
    for i, f in enumerate(files):
        sym = f.name.split("_")[0] + "USDT"
        df = pd.read_feather(f).set_index("date")
        if len(df) < MIN_HISTORY_DAYS * 24:
            continue
        frames[sym] = df
        if (i + 1) % 150 == 0:
            print(f"  read {i+1}/{len(files)}", flush=True)

    syms = sorted(frames)
    start = min(d.index[0] for d in frames.values())
    end = max(d.index[-1] for d in frames.values())
    index = pd.date_range(start, end, freq="1h", tz="UTC")
    print(f"{len(syms)} symbols, {len(index)} bars {index[0]} .. {index[-1]}")

    OUT.mkdir(parents=True, exist_ok=True)
    fields = ["open", "high", "low", "close", "volume", "quote_volume"]
    panels = {}
    for field in fields:
        if field == "quote_volume" and "quote_volume" not in next(iter(frames.values())).columns:
            # the backfill stores base volume only; approximate the quote
            # volume with close x volume, which is what the screen ranks on
            wide = pd.concat({s: frames[s]["volume"] * frames[s]["close"] for s in syms}, axis=1)
        else:
            wide = pd.concat({s: frames[s][field] for s in syms}, axis=1)
        arr = wide.reindex(index)[syms].to_numpy(dtype="float32")
        np.save(OUT / f"{field}.npy", arr)
        panels[field] = arr
        print(f"  wrote {field}", flush=True)

    fr = {}
    for s in syms:
        p = SRC / f"{s.removesuffix('USDT')}_USDT_USDT-1h-funding_rate.feather"
        if p.exists():
            # download_hist_universe_1h writes the rate into `open` and
            # leaves high/low/close at zero
            d = pd.read_feather(p).set_index("date")["open"]
            fr[s] = d[~d.index.duplicated()]
    wide = pd.concat(fr, axis=1).reindex(index)[[s for s in syms if s in fr]]
    full = pd.DataFrame(0.0, index=index, columns=syms)
    full[wide.columns] = wide.fillna(0.0)
    np.save(OUT / "funding_rate.npy", full.to_numpy(dtype="float32"))
    print(f"  wrote funding_rate ({len(fr)} symbols had a file)")

    close = panels["close"]
    qv = panels["quote_volume"]
    observed = np.isfinite(close)
    age = np.cumsum(observed, axis=0)
    liq = pd.DataFrame(qv).rolling(LIQ_WINDOW_DAYS * 24, min_periods=24).mean().to_numpy()
    mask = (age >= MIN_HISTORY_DAYS * 24) & observed & (liq * 24 >= MIN_DAILY_QUOTE_VOLUME)
    np.save(OUT / "universe_mask.npy", mask)
    print(f"  wrote universe_mask, tradable share {100*mask.mean():.1f}%")

    meta = {"tf": "1h_hist", "symbols": syms,
            "fields": fields + ["funding_rate", "universe_mask"],
            "start": str(index[0]), "end": str(index[-1]), "n_bars": len(index)}
    json.dump(meta, open(OUT / "meta.json", "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
