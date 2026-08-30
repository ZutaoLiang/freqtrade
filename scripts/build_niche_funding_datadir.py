"""Assemble a Freqtrade futures datadir for the funding-skew universe.

Klines are symlinked from the public dump; funding_rate and mark are rebuilt from the raw
parquet feeds at the 1h grid this Freqtrade version asks for (`funding_fee_timeframe` and
`mark_ohlcv_timeframe` are both "1h" in exchange.py). Settlement stamps are kept EXACTLY as
the exchange published them -- 1h, 4h and 8h pairs all coexist in this universe -- because
the strategy reads this same file for its signal, so a resampled rate would be a different
number than the one traded on. Funding fees are applied by inner-joining funding to mark on
the timestamp, so mark is written on the same 1h grid.
"""
import os

import pandas as pd

SRC = "/root/freqtrade/user_data/data/binance_public"
KL = f"{SRC}/freqtrade/futures"
DST = "/root/freqtrade/user_data/data/niche_funding/futures"
WORK = "/root/freqtrade/user_data/niche_work"
TFS = ("1m", "5m", "1h")


def main():
    os.makedirs(DST, exist_ok=True)
    syms = pd.read_csv(f"{WORK}/wide_syms.csv")["sym"].tolist()
    done, skipped = 0, []
    for sym in syms:
        raw = sym.replace("_USDT_USDT", "USDT")
        fpath, mpath = f"{SRC}/funding/{raw}.parquet", f"{SRC}/markprice_1m/{raw}.parquet"
        if not (os.path.exists(fpath) and os.path.exists(mpath)):
            skipped.append(sym)
            continue

        fr = pd.read_parquet(fpath).set_index("date")["funding_rate"]
        fr = fr[~fr.index.duplicated()].sort_index()
        # snap to the 1h grid without merging distinct settlements
        fr = fr.resample("1h").sum().loc[fr.index.min().floor("h"):]
        pd.DataFrame({"date": fr.index, "open": fr.values, "high": 0.0,
                      "low": 0.0, "close": 0.0, "volume": 0.0}
                     ).to_feather(f"{DST}/{sym}-1h-funding_rate.feather")

        mk = pd.read_parquet(mpath).set_index("date")["mark_close"].resample("1h").ohlc()
        mk = mk.reindex(fr.index).ffill().dropna()
        pd.DataFrame({"date": mk.index, "open": mk["open"].values, "high": mk["high"].values,
                      "low": mk["low"].values, "close": mk["close"].values, "volume": 0.0}
                     ).to_feather(f"{DST}/{sym}-1h-mark.feather")

        for tf in TFS:
            src, dst = f"{KL}/{sym}-{tf}-futures.feather", f"{DST}/{sym}-{tf}-futures.feather"
            if os.path.exists(src) and not os.path.exists(dst):
                os.symlink(src, dst)
        done += 1
    print(f"built {done} pairs at the 1h grid, skipped {len(skipped)}: {skipped[:5]}")


if __name__ == "__main__":
    main()
