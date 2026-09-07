"""Build a 1h-only freqtrade datadir for the vol-screen Donchian reconciliation.

Same conversion the TradingView matrix uses, trimmed to the single timeframe
this strategy trades: OHLCV is symlinked from the public archive, funding and
mark price are converted from the raw parquet archives to the 1h feathers the
Binance exchange class asks for.
"""
import json
from pathlib import Path

import pandas as pd

SRC_OHLCV = Path("/root/freqtrade/user_data/data/binance_public/freqtrade/futures")
SRC_FUNDING = Path("/root/freqtrade/user_data/data/binance_public/funding")
SRC_MARK = Path("/root/freqtrade/user_data/data/binance_public/markprice_1m")
UNIVERSE = Path("/root/freqtrade/user_data/research/ma_harness/universe_vsd.json")
OUT = Path("/root/freqtrade/user_data/data/vsd/futures")


def main():
    pairs = json.loads(UNIVERSE.read_text())["groups"]["vsd"]
    OUT.mkdir(parents=True, exist_ok=True)
    missing_ohlcv, missing_funding, missing_mark = [], [], []
    for i, pair in enumerate(pairs):
        base = pair.split("/")[0]
        stem = f"{base}_USDT_USDT"
        src = SRC_OHLCV / f"{stem}-1h-futures.feather"
        if not src.exists():
            missing_ohlcv.append(pair)
            continue
        dst = OUT / f"{stem}-1h-futures.feather"
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src.resolve())

        fsrc = SRC_FUNDING / f"{base}USDT.parquet"
        if not fsrc.exists():
            missing_funding.append(pair)
        else:
            f = pd.read_parquet(fsrc).sort_values("date")
            fr = pd.DataFrame({"date": f["date"],
                               "open": f["funding_rate"].astype("float64"),
                               "high": f["funding_rate"].astype("float64"),
                               "low": f["funding_rate"].astype("float64"),
                               "close": f["funding_rate"].astype("float64"),
                               "volume": 0.0})
            fr = fr[fr["date"].dt.minute.eq(0) & fr["date"].dt.second.eq(0)]
            fr.reset_index(drop=True).to_feather(
                OUT / f"{stem}-1h-funding_rate.feather", compression="lz4")

        msrc = SRC_MARK / f"{base}USDT.parquet"
        if msrc.exists():
            m = pd.read_parquet(msrc).sort_values("date").set_index("date")
            agg = m.resample("1h").agg({"mark_open": "first", "mark_high": "max",
                                        "mark_low": "min", "mark_close": "last"}).dropna()
            agg.columns = ["open", "high", "low", "close"]
        else:
            missing_mark.append(pair)
            m = pd.read_feather(src).set_index("date")
            agg = m.resample("1h").agg({"open": "first", "high": "max",
                                        "low": "min", "close": "last"}).dropna()
            agg = agg[["open", "high", "low", "close"]]
        mk = agg.reset_index()
        mk["volume"] = 0.0
        mk.to_feather(OUT / f"{stem}-1h-mark.feather", compression="lz4")
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(pairs)}", flush=True)

    print(f"pairs={len(pairs)} missing_ohlcv={len(missing_ohlcv)} "
          f"missing_funding={len(missing_funding)} missing_mark={len(missing_mark)}")
    if missing_ohlcv:
        print("  no OHLCV:", missing_ohlcv[:10])
    Path("/root/freqtrade/user_data/research/ma_harness/datadir_vsd_report.json"
         ).write_text(json.dumps({"missing_ohlcv": missing_ohlcv,
                                  "missing_funding": missing_funding,
                                  "missing_mark": missing_mark}, indent=1))


if __name__ == "__main__":
    main()
