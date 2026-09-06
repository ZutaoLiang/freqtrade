"""Freqtrade futures datadir for the delisting-short backtest.

For every symbol in user_data/delist/events.json: 1m klines trimmed to the non-zero-volume
range (the public archive pads delisted symbols with zero-volume rows), plus funding_rate and
mark at the 1h grid this freqtrade version reads (see funding-skew-momentum.md §6.4).
"""
import json
import os

import pandas as pd

SRC = "/root/freqtrade/user_data/data/binance_public"
DST = "/root/freqtrade/user_data/data/delist/futures"


def main():
    os.makedirs(DST, exist_ok=True)
    events = json.load(open("/root/freqtrade/user_data/delist/events.json"))
    syms = sorted({e["symbol"] for e in events})
    built, skipped = [], []
    for raw in syms:
        pair = f"{raw[:-4]}_USDT_USDT"
        kp, fp, mp = f"{SRC}/klines_1m/{raw}.parquet", f"{SRC}/funding/{raw}.parquet", f"{SRC}/markprice_1m/{raw}.parquet"
        if not (os.path.exists(kp) and os.path.exists(fp) and os.path.exists(mp)):
            skipped.append(raw)
            continue
        k = pd.read_parquet(kp)
        nz = k[k.volume > 0]
        if nz.empty:
            skipped.append(raw)
            continue
        k = k[(k.date >= nz.date.min()) & (k.date <= nz.date.max())]
        k[["date", "open", "high", "low", "close", "volume"]].reset_index(drop=True).to_feather(f"{DST}/{pair}-1m-futures.feather")
        fr = pd.read_parquet(fp).set_index("date")["funding_rate"]
        fr = fr[(fr.index >= nz.date.min()) & (fr.index <= nz.date.max())]
        fr = fr[~fr.index.duplicated()].sort_index()
        fr = fr.resample("1h").sum()
        pd.DataFrame({"date": fr.index, "open": fr.values, "high": 0.0, "low": 0.0, "close": 0.0, "volume": 0.0}).to_feather(f"{DST}/{pair}-1h-funding_rate.feather")
        mk = pd.read_parquet(mp).set_index("date")["mark_close"].resample("1h").ohlc().reindex(fr.index).ffill().dropna()
        pd.DataFrame({"date": mk.index, "open": mk["open"].values, "high": mk["high"].values, "low": mk["low"].values, "close": mk["close"].values, "volume": 0.0}).to_feather(f"{DST}/{pair}-1h-mark.feather")
        built.append(f"{raw[:-4]}/USDT:USDT")
    json.dump(built, open("/root/freqtrade/user_data/delist/backtest_pairs.json", "w"), indent=1)
    print(f"built {len(built)} pairs, skipped {skipped}")


if __name__ == "__main__":
    main()
