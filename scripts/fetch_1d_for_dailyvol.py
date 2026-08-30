"""Fetch 1d candles only, for build_niche_dailyvol.py.

`freqtrade download-data` in futures mode also pulls 1h mark and funding history for every
pair, which triples the request count and gets the IP banned (403) partway through a
600-pair universe. The liquidity table needs nothing but daily candles, so fetch those alone,
in freqtrade's feather layout, and let the bot fetch its own funding data live.

Run daily, then:  python3 scripts/build_niche_dailyvol.py --datadir <out> --out <parquet>
"""
import argparse
import os
import time

import ccxt
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/root/freqtrade/user_data/data/binance/futures")
    p.add_argument("--days", type=int, default=45)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ex = ccxt.binanceusdm({"enableRateLimit": True})
    markets = ex.load_markets()
    syms = sorted(m["symbol"] for m in markets.values()
                  if m.get("swap") and m.get("quote") == "USDT" and m.get("active"))
    since = ex.milliseconds() - a.days * 86400_000
    ok = fail = 0
    for i, s in enumerate(syms, 1):
        try:
            rows = ex.fetch_ohlcv(s, "1d", since=since, limit=a.days + 5)
        except Exception as e:  # noqa: BLE001
            fail += 1
            print(f"{s}: {type(e).__name__}: {str(e)[:80]}", flush=True)
            time.sleep(2)
            continue
        if not rows:
            continue
        df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["date"], unit="ms", utc=True)
        df = df.iloc[:-1]  # drop the still-open daily candle
        fname = s.replace("/", "_").replace(":", "_") + "-1d-futures.feather"
        df.reset_index(drop=True).to_feather(os.path.join(a.out, fname))
        ok += 1
        if i % 100 == 0:
            print(f"{i}/{len(syms)} done", flush=True)
    print(f"{ok} pairs written, {fail} failed -> {a.out}")


if __name__ == "__main__":
    main()
