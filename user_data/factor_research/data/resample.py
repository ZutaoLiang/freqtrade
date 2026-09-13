"""Resample the 1m parquet base layer into every timeframe we research on.

Two artefacts per timeframe: a rich parquet (all Binance columns, for factor
work) and a freqtrade-compatible feather (OHLCV only, for backtesting). Driving
every timeframe off one 1m base guarantees identical pair coverage and span
across them, which the previously downloaded data did not have.
"""
import os
import sys
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import MAX_WORKERS  # noqa: E402

SRC = "/root/freqtrade/user_data/data/binance_public/klines_1m"
RICH = "/root/freqtrade/user_data/data/binance_public/resampled"
FT = "/root/freqtrade/user_data/data/binance_public/freqtrade/futures"
TIMEFRAMES = {"5m": "5min", "15m": "15min", "30m": "30min",
              "1h": "1h", "4h": "4h", "1d": "1D"}
QUOTES = ("USDT", "USDC", "USD1")
AGG = {"open": "first", "high": "max", "low": "min", "close": "last",
       "volume": "sum", "quote_volume": "sum", "count": "sum",
       "taker_buy_volume": "sum", "taker_buy_quote_volume": "sum"}
OHLCV = ["date", "open", "high", "low", "close", "volume"]


def ft_pair(sym):
    """BTCUSDT -> BTC_USDT_USDT, the freqtrade futures filename convention."""
    name = sym[: -len("SETTLED")] if sym.endswith("SETTLED") else sym
    for q in QUOTES:
        if name.endswith(q) and len(name) > len(q):
            return f"{name[: -len(q)]}_{q}_{q}"
    return None


def process(sym):
    pair = ft_pair(sym)
    if pair is None:
        return sym, None, "unrecognised quote asset"
    try:
        df = pd.read_parquet(os.path.join(SRC, f"{sym}.parquet"))
        if df.empty:
            return sym, {}, None
        df = df.set_index("date")
        os.makedirs(FT, exist_ok=True)
        counts = {}
        for tf, rule in TIMEFRAMES.items():
            out = df.resample(rule, label="left", closed="left", origin="epoch").agg(AGG)
            out = out.dropna(subset=["open"]).reset_index()
            os.makedirs(os.path.join(RICH, tf), exist_ok=True)
            out.to_parquet(os.path.join(RICH, tf, f"{sym}.parquet"),
                           compression="zstd", index=False)
            out[OHLCV].to_feather(os.path.join(FT, f"{pair}-{tf}-futures.feather"),
                                  compression="lz4")
            counts[tf] = len(out)
        base = df.reset_index()[OHLCV]
        base.to_feather(os.path.join(FT, f"{pair}-1m-futures.feather"), compression="lz4")
        counts["1m"] = len(base)
        return sym, counts, None
    except Exception as exc:
        return sym, None, repr(exc)


if __name__ == "__main__":
    syms = sorted(f[:-8] for f in os.listdir(SRC) if f.endswith(".parquet"))
    print(f"resampling {len(syms)} symbols ...", flush=True)
    ok = bad = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for i, (sym, counts, err) in enumerate(ex.map(process, syms), 1):
            if err:
                bad += 1
                sys.stderr.write(f"FAIL {sym}: {err}\n")
            else:
                ok += 1
            if i % 100 == 0:
                print(f"  {i}/{len(syms)}  ok={ok} bad={bad}", flush=True)
    print(f"done: ok={ok} bad={bad}")
