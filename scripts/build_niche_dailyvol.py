"""Trailing 30-day median daily quote volume per pair -- the strategy's liquidity floor.

Batched rather than computed in the strategy: a 30-day window on 5m candles needs 8640
startup candles and Freqtrade refuses more than 5x what the exchange serves. The window is
slow-moving, so a daily refresh is enough. The series is shifted one day, so a value dated D
uses only days strictly before D and the gate never sees the day it is trading.

Live: run daily after `freqtrade download-data --timeframes 1d`, pointing --datadir at the
same directory the bot uses. Backtest: point it at the backtest datadir.
"""
import argparse
import os

import pandas as pd

VOL_DAYS = 30
VOL_MIN_DAYS = 10
DEFAULT_DATADIR = "/root/freqtrade/user_data/data/niche_funding/futures"
DEFAULT_OUT = "/root/freqtrade/user_data/niche_work/daily_qv.parquet"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datadir", default=DEFAULT_DATADIR,
                   help="directory holding <PAIR>-1d-futures.feather (or -1m- as fallback)")
    p.add_argument("--out", default=DEFAULT_OUT)
    a = p.parse_args()

    frames, missing = [], 0
    for f in sorted(os.listdir(a.datadir)):
        if f.endswith("-1d-futures.feather"):
            sym, resample = f.replace("-1d-futures.feather", ""), False
        elif f.endswith("-1m-futures.feather"):
            sym, resample = f.replace("-1m-futures.feather", ""), True
        else:
            continue
        if resample and os.path.exists(f"{a.datadir}/{sym}-1d-futures.feather"):
            continue                                    # prefer the native daily candles
        df = pd.read_feather(f"{a.datadir}/{f}").set_index("date")
        df = df[~df.index.duplicated()]
        quote_volume = df["close"] * df["volume"]
        daily = quote_volume.resample("1D").sum() if resample else quote_volume
        # Shift one day by re-dating rather than .shift(1): a value dated D still uses only
        # days strictly before D, but the newest median now lands on a row dated "today"
        # instead of being dropped -- live, todays candles must find a row or the qv gate
        # blocks every entry (this cost the first dry-run day all of its signals).
        trail = daily.rolling(VOL_DAYS, min_periods=VOL_MIN_DAYS).median().dropna()
        trail.index = trail.index + pd.Timedelta(days=1)
        if trail.empty:
            missing += 1
            continue
        frames.append(pd.DataFrame({"sym": sym, "date": trail.index, "qv": trail.values}))

    d = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    d.to_parquet(a.out)
    print(f"{d.sym.nunique()} pairs, {len(d)} pair-days, through {d.date.max():%Y-%m-%d} "
          f"-> {a.out} ({missing} with too little history)")


if __name__ == "__main__":
    main()
