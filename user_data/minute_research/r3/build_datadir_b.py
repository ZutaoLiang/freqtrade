"""Write a freqtrade 2026.x futures datadir for U60 from binance_public parquet (1m klines, 1h funding, 1h mark)."""
import sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.feather as pf
R = "/root/freqtrade/user_data/data/binance_public"
OUT = "/root/freqtrade/user_data/data/r3b/futures"
END = pd.Timestamp("2026-09-01", tz="UTC")          # funding archive ends 2026-08-31 16:00

def w(df, path):
    df = df[["date", "open", "high", "low", "close", "volume"]].copy()
    df["date"] = df["date"].astype("datetime64[ms, UTC]")
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype("float64")
    pf.write_feather(pa.Table.from_pandas(df.reset_index(drop=True), preserve_index=False), path, compression="lz4")

def one(base):
    p = f"{base}_USDT_USDT"
    k = pd.read_parquet(f"{R}/klines_1m/{base}USDT.parquet", columns=["date", "open", "high", "low", "close", "volume"])
    k = k[k.date < END].drop_duplicates("date").sort_values("date")
    k5 = k.set_index("date").resample("5min").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna().reset_index()
    w(k5, f"{OUT}/{p}-5m-futures.feather")
    f = pd.read_parquet(f"{R}/funding/{base}USDT.parquet")
    f["date"] = f.date.dt.round("h")
    f = f.drop_duplicates("date", keep="last")
    f = pd.DataFrame({"date": f.date, "open": f.funding_rate, "high": 0.0, "low": 0.0, "close": 0.0, "volume": 0.0})
    w(f[f.date < END], f"{OUT}/{p}-1h-funding_rate.feather")
    m = pd.read_parquet(f"{R}/markprice_1m/{base}USDT.parquet").set_index("date")
    mh = m.resample("1h").agg({"mark_open": "first", "mark_high": "max", "mark_low": "min", "mark_close": "last"}).dropna()
    mh.columns = ["open", "high", "low", "close"]; mh["volume"] = 0.0
    kh = k.set_index("date").resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    miss = kh.index.difference(mh.index)
    mh = pd.concat([mh, kh.loc[miss].assign(volume=0.0)]).sort_index()   # last-price proxy for missing mark hours
    mh.index.name = "date"
    mh = mh.reset_index()
    w(mh[mh.date < END], f"{OUT}/{p}-1h-mark.feather")
    return base, len(miss), len(k), f.date.min(), f.date.max(), mh.date.min(), mh.date.max()

if __name__ == "__main__":
    import os; os.makedirs(OUT, exist_ok=True)
    from r11_presettle import U162 as bases
    with ProcessPoolExecutor(12) as ex:
        for r in ex.map(one, bases):
            print(*r, flush=True)
