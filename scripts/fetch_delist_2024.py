"""Fetch 2024 data for the contracts named in 2024 futures-delisting notices and build
(a) a binance_public-style parquet dir for the event-study replay and
(b) a freqtrade datadir for the native backtest.

Sources (no proxy; data.binance.vision answers directly):
  futures/um/monthly/klines/{S}/1m/{S}-1m-YYYY-MM.zip
  futures/um/monthly/markPriceKlines/{S}/1m/{S}-1m-YYYY-MM.zip
  futures/um/monthly/fundingRate/{S}/{S}-fundingRate-YYYY-MM.zip
Months fetched: from (release - 40 days) through the settlement month.
"""
import io
import json
import os
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

BASE = "https://data.binance.vision/data/futures/um/monthly"
ROOT = "/root/freqtrade/user_data/delist2024"
RAW = f"{ROOT}/raw"
PUB = f"{ROOT}/public"          # klines_1m/, markprice_1m/, funding/  (parquet, binance_public layout)
FT = f"{ROOT}/data/futures"     # freqtrade datadir
KCOLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))


def months_between(a: pd.Timestamp, b: pd.Timestamp):
    return [p.strftime("%Y-%m") for p in pd.period_range(a.to_period("M"), b.to_period("M"), freq="M")]


def fetch(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return "skip"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest + ".part")
        os.replace(dest + ".part", dest)
        return "ok"
    except Exception as e:  # noqa
        if os.path.exists(dest + ".part"):
            os.remove(dest + ".part")
        return f"miss:{type(e).__name__}"


def read_zip_csv(path, names):
    with zipfile.ZipFile(path) as z:
        raw = z.read(z.namelist()[0])
    head = raw[:120].split(b"\n", 1)[0].lower()
    has_header = b"open" in head or b"time" in head
    df = pd.read_csv(io.BytesIO(raw), header=0 if has_header else None, names=None if has_header else names)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def to_utc(v):
    v = pd.to_numeric(v, errors="coerce").astype("int64")
    return pd.to_datetime(np.where(v > 1e14, v // 1000, v), unit="ms", utc=True)


def main():
    notices = json.load(open("/root/freqtrade/user_data/delist_short_20260906/announcements_2024_raw.json"))
    events = []
    for n in notices:
        for sym, t in n["parsed"]:
            events.append({"symbol": sym, "pair": f"{sym[:-4]}/USDT:USDT", "release": pd.Timestamp(n["release_utc"], tz="UTC").isoformat(),
                           "settle": pd.Timestamp(t, tz="UTC").isoformat(), "notice": n["title"]})
    # first notice per symbol (postponements are later notices; the entry rule fires on the first)
    first = {}
    for e in sorted(events, key=lambda e: e["release"]):
        first.setdefault(e["symbol"], e)
    events = list(first.values())
    os.makedirs(ROOT, exist_ok=True)
    json.dump(events, open(f"{ROOT}/events.json", "w"), indent=1)
    jobs = []
    for e in events:
        s = e["symbol"]
        for m in months_between(pd.Timestamp(e["release"]) - pd.Timedelta(days=40), pd.Timestamp(e["settle"])):
            jobs += [(f"{BASE}/klines/{s}/1m/{s}-1m-{m}.zip", f"{RAW}/klines/{s}/{s}-1m-{m}.zip"),
                     (f"{BASE}/markPriceKlines/{s}/1m/{s}-1m-{m}.zip", f"{RAW}/mark/{s}/{s}-1m-{m}.zip"),
                     (f"{BASE}/fundingRate/{s}/{s}-fundingRate-{m}.zip", f"{RAW}/funding/{s}/{s}-fundingRate-{m}.zip")]
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda j: fetch(*j), jobs))
    print("downloads:", pd.Series(res).value_counts().to_dict())
    for d in ("klines_1m", "markprice_1m", "funding"):
        os.makedirs(f"{PUB}/{d}", exist_ok=True)
    os.makedirs(FT, exist_ok=True)
    built = []
    for e in events:
        s = e["symbol"]; pair = f"{s[:-4]}_USDT_USDT"
        kd = sorted(os.listdir(f"{RAW}/klines/{s}")) if os.path.isdir(f"{RAW}/klines/{s}") else []
        if not kd:
            print("no klines for", s); continue
        k = pd.concat([read_zip_csv(f"{RAW}/klines/{s}/{f}", KCOLS) for f in kd]); k["date"] = to_utc(k.open_time)
        k = k[["date", "open", "high", "low", "close", "volume", "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume"]].astype({c: "float64" for c in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_volume", "taker_buy_quote_volume"]})
        k = k.drop_duplicates("date").sort_values("date").reset_index(drop=True)
        k.to_parquet(f"{PUB}/klines_1m/{s}.parquet", index=False)
        k[["date", "open", "high", "low", "close", "volume"]].to_feather(f"{FT}/{pair}-1m-futures.feather")
        md = sorted(os.listdir(f"{RAW}/mark/{s}")) if os.path.isdir(f"{RAW}/mark/{s}") else []
        m = pd.concat([read_zip_csv(f"{RAW}/mark/{s}/{f}", KCOLS) for f in md]); m["date"] = to_utc(m.open_time)
        m = m[["date", "open", "high", "low", "close"]].astype({c: "float64" for c in ["open", "high", "low", "close"]}).drop_duplicates("date").sort_values("date")
        m.columns = ["date", "mark_open", "mark_high", "mark_low", "mark_close"]
        m.to_parquet(f"{PUB}/markprice_1m/{s}.parquet", index=False)
        fd = sorted(os.listdir(f"{RAW}/funding/{s}")) if os.path.isdir(f"{RAW}/funding/{s}") else []
        f = pd.concat([read_zip_csv(f"{RAW}/funding/{s}/{x}", ["calc_time", "funding_interval_hours", "last_funding_rate"]) for x in fd])
        f = f.rename(columns={"calc_time": "date", "last_funding_rate": "funding_rate"})
        f["date"] = pd.DatetimeIndex(to_utc(f.date)).floor("min")
        f = f[["date", "funding_rate", "funding_interval_hours"]].astype({"funding_rate": "float64"}).drop_duplicates("date").sort_values("date")
        f.to_parquet(f"{PUB}/funding/{s}.parquet", index=False)
        fr = f.set_index("date")["funding_rate"].resample("1h").sum()
        pd.DataFrame({"date": fr.index, "open": fr.values, "high": 0.0, "low": 0.0, "close": 0.0, "volume": 0.0}).to_feather(f"{FT}/{pair}-1h-funding_rate.feather")
        mk = m.set_index("date")["mark_close"].resample("1h").ohlc().reindex(fr.index).ffill().dropna()
        pd.DataFrame({"date": mk.index, "open": mk["open"].values, "high": mk["high"].values, "low": mk["low"].values, "close": mk["close"].values, "volume": 0.0}).to_feather(f"{FT}/{pair}-1h-mark.feather")
        built.append((s, str(k.date.min())[:16], str(k.date.max())[:16], len(k), len(f)))
    for b in built:
        print(b)
    json.dump([f"{b[0][:-4]}/USDT:USDT" for b in built], open(f"{ROOT}/backtest_pairs.json", "w"), indent=1)


if __name__ == "__main__":
    main()
