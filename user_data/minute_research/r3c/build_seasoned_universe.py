"""Daily causal 'seasoned liquid' universe for R24 (rolling version of U162's definition).

A USDT perp is eligible on day D if, over the 270 calendar days before D (D itself excluded), it has >= 250 days of data
and a median daily quote volume >= 10M USDT. Quote volume: binance_public 1m klines (2025+), binance-hist 1h close*volume
before 2025. Equity/commodity-linked contracts excluded (as for U162). Output: parquet (date, pair, eligible) for
2025-01-01..2026-08-31, written into the backtest datadir so the strategy can read it.
"""
import glob, os, sys
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

PUB = "/root/freqtrade/user_data/data/binance_public"
HIST = "/root/freqtrade/user_data/data/binance-hist/futures"
EQUITY = {"NVDA", "MSTR", "TSLA", "AAPL", "COIN", "HOOD", "AMZN", "GOOGL", "META", "MSFT", "CRCL", "INTC", "PLTR", "QQQ", "SPY",
          "AMD", "NFLX", "BABA", "TSM", "XAU", "XAG", "PAXG", "XAUT", "CL", "BZ", "NG", "AAOI"}
DAYS = pd.date_range("2025-01-01", "2026-08-31", freq="D", tz="UTC")


def daily_qv(sym):
    base = sym[:-4]
    parts = []
    h = f"{HIST}/{base}_USDT_USDT-1h-futures.feather"
    if os.path.exists(h):
        x = pd.read_feather(h, columns=["date", "close", "volume"]); x = x[x.date < "2025-01-01"]
        parts.append((x.close * x.volume).groupby(x.date.dt.floor("D")).sum())
    k = pd.read_parquet(f"{PUB}/klines_1m/{sym}.parquet", columns=["date", "quote_volume"]).set_index("date")
    parts.append(k.quote_volume.resample("1D").sum())
    v = pd.concat(parts).sort_index()
    return v[~v.index.duplicated(keep="last")]


def one(sym):
    v = daily_qv(sym)
    full = v.reindex(pd.date_range(min(v.index.min(), DAYS[0] - pd.Timedelta(days=300)), DAYS[-1], freq="D", tz="UTC"))
    days = full.notna().rolling(270, min_periods=1).sum().shift(1)
    med = full.rolling(270, min_periods=1).median().shift(1)
    e = ((days >= 250) & (med >= 1e7)).reindex(DAYS).fillna(False)
    base = sym[:-4]
    return pd.DataFrame({"date": DAYS, "pair": f"{base}/USDT:USDT", "eligible": e.to_numpy()})


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/root/freqtrade/user_data/data/r3s/seasoned_universe.parquet"
    syms = sorted(os.path.basename(p)[:-8] for p in glob.glob(f"{PUB}/klines_1m/*USDT.parquet") if os.path.basename(p)[:-12] not in EQUITY)
    with ProcessPoolExecutor(24) as ex:
        df = pd.concat(ex.map(one, syms), ignore_index=True)
    df = df[df.pair.isin(df[df.eligible].pair.unique())]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_parquet(out, index=False)
    per_day = df.groupby("date").eligible.sum()
    print("pairs ever eligible:", df.pair.nunique(), "| eligible per day: min", per_day.min(), "median", per_day.median(), "max", per_day.max())
