"""R24 production diagnostics (not a parameter search). Rebuilds R24 from raw 1m klines + raw funding for every USDT perp.

Rule unchanged: at settlement T, if the last three settled rates (T, T-1, T-2) are all >= 0.03%, short at the 1m open T+delay,
exit at the 1m open 480 min after entry, 10 bp/side, funding settled in (entry, exit] credited to the short.
Outputs per trade: universe flags (static U162, dynamic = trailing-30d median daily quote volume >= 10M at T), funding interval,
return for delays {1,5,15,30,60} min, maximum adverse excursion (1m highs), quote volume in the first 60 min after entry.
"""
import glob, os
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd

R = "/root/freqtrade/user_data/data/binance_public"
U = pd.read_csv("/root/freqtrade/user_data/minute_research/r3/universe_rank.csv")
U162 = set(U[U.med_qv >= 1e7].base + "USDT")
DELAYS = (1, 5, 15, 30, 60)
EQUITY = {"NVDA", "MSTR", "TSLA", "AAPL", "COIN", "HOOD", "AMZN", "GOOGL", "META", "MSFT", "CRCL", "INTC", "PLTR", "QQQ", "SPY",
          "AMD", "NFLX", "BABA", "TSM", "XAU", "XAG", "PAXG", "XAUT", "CL", "BZ", "NG", "AAOI"}


def one(sym):
    fp, kp = f"{R}/funding/{sym}.parquet", f"{R}/klines_1m/{sym}.parquet"
    if not (os.path.exists(fp) and os.path.exists(kp)) or sym[:-4] in EQUITY:
        return []
    f = pd.read_parquet(fp); f["date"] = f.date.dt.round("h")
    f = f.drop_duplicates("date", keep="last").set_index("date").sort_index()
    fr = f.funding_rate
    sig = fr[(fr >= 3e-4) & (fr.shift(1) >= 3e-4) & (fr.shift(2) >= 3e-4)]
    sig = sig[(sig.index >= "2025-01-01") & (sig.index < "2026-09-01")]
    if sig.empty:
        return []
    k = pd.read_parquet(kp, columns=["date", "open", "high", "quote_volume"]).set_index("date")
    dq = k.quote_volume.resample("1D").sum()
    liq30 = dq.rolling(30, min_periods=10).median().shift(1)             # known before day T
    out = []
    busy_until = pd.Timestamp("1970-01-01", tz="UTC")
    for T in sig.index:
        row = dict(sym=sym, T=T, interval=float(f.funding_interval_hours.get(T, np.nan)), in_u162=sym in U162,
                   liq30=float(liq30.get(T.floor("D"), np.nan)))
        ok = True
        for dly in DELAYS:
            e = T + pd.Timedelta(minutes=dly); x = e + pd.Timedelta(minutes=480)
            if e not in k.index or x not in k.index:
                ok = False; break
            fund = fr[(fr.index > e) & (fr.index <= x)].sum()
            row[f"r{dly}"] = -(k.open[x] / k.open[e] - 1) + fund - 20e-4
            if dly == 5:
                win = k.loc[e:x - pd.Timedelta(minutes=1)]
                row["mae"] = win.high.max() / k.open[e] - 1                   # worst move against the short
                row["qv60"] = k.quote_volume.loc[e:e + pd.Timedelta(minutes=59)].sum()
        if not ok:
            continue
        row["overlap"] = T < busy_until                                      # R24 holds one position per coin
        if not row["overlap"]:
            busy_until = T + pd.Timedelta(minutes=485)
        out.append(row)
    return out


if __name__ == "__main__":
    syms = sorted(os.path.basename(p)[:-8] for p in glob.glob(f"{R}/funding/*USDT.parquet"))
    with ProcessPoolExecutor(24) as ex:
        rows = [r for rs in ex.map(one, syms) for r in rs]
    df = pd.DataFrame(rows); df = df[~df.overlap]
    df.to_parquet("r24_prod_diag.parquet"); print("trades", len(df), "coins", df.sym.nunique())
