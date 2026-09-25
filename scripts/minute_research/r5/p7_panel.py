"""Independent re-check of the 7 strategies in PROFITABLE_STRATEGIES.md.
Panel: U160 (TRAIN 2025-01..09 median daily quote volume rank, crypto only), 1h Vision klines incl. taker_buy_volume,
settled funding (binance-hist + binance 2026 + r4new feathers), metrics (OI / top-trader L/S) for the liquid 30 only.
Engine: signal on bar close -> fill next open, fixed hold (bars), one position per coin, 10bp/side (BTC/ETH 7.5),
funding optionally charged (short receives +fr, long pays), arithmetic returns. Scheme-C segments."""
import glob, io, json, os, sys, zipfile
import numpy as np, pandas as pd
from numba import njit

RAW = "user_data/data/r5-vision1h/data/futures/um"
OUT = os.environ.get("P7_OUT", "user_data/minute_research/r5/p7_panel.npz")
U = json.load(open(os.environ.get("P7_UNIV", "user_data/minute_research/r5/u160.json")))
IDX = pd.date_range("2024-11-01", "2026-08-14", freq="h", tz="UTC", inclusive="left")
FDIRS = ["user_data/data/binance-hist/futures", "user_data/data/binance/futures", "user_data/data/r4new/futures"]


def read_sym(sym):
    files = sorted(glob.glob(f"{RAW}/monthly/klines/{sym}/1h/*.zip")) + sorted(glob.glob(f"{RAW}/daily/klines/{sym}/1h/*.zip"))
    parts = []
    for z in files:
        try:
            with zipfile.ZipFile(z) as zf:
                df = pd.read_csv(zf.open(zf.namelist()[0]), header=None, usecols=[0, 1, 2, 3, 4, 5, 9])
        except Exception:
            continue
        if not str(df.iloc[0, 0]).isdigit():
            df = df.iloc[1:]
        parts.append(df)
    if not parts:
        return None
    d = pd.concat(parts).astype(float)
    d.columns = ["ot", "open", "high", "low", "close", "volume", "tbv"]
    d["ot"] = np.where(d.ot > 1e14, d.ot // 1000, d.ot)
    d["date"] = pd.to_datetime(d.ot.astype("int64"), unit="ms", utc=True)
    return d.drop_duplicates("date", keep="last").set_index("date").reindex(IDX)


def funding(b):
    fs = []
    um = f"{RAW}"                                   # raw Vision settlements first (monthly zips + REST csv)
    vf = [pd.read_csv(z) for z in sorted(glob.glob(f"{um}/monthly/fundingRate/{b}USDT/*.zip"))]
    vf += [pd.read_csv(x) for x in sorted(glob.glob(f"{um}/rest/fundingRate/{b}USDT/*.csv"))]
    if vf:
        v = pd.concat(vf)
        t = pd.to_numeric(v.calc_time); t = np.where(t > 1e14, t // 1000, t)
        fs.append(pd.DataFrame({"date": pd.to_datetime(t, unit="ms", utc=True).round("h"), "open": v.last_funding_rate.astype(float)}))
    for D in FDIRS:
        p = f"{D}/{b}_USDT_USDT-1h-funding_rate.feather"
        if os.path.exists(p):
            try:
                fs.append(pd.read_feather(p, columns=["date", "open"]))
            except Exception:
                pass
    if not fs:
        return pd.Series(np.nan, index=IDX)
    f = pd.concat(fs); f["date"] = pd.to_datetime(f.date, utc=True).dt.floor("h")
    return f.drop_duplicates("date", keep="first").set_index("date").open.reindex(IDX)   # NaN = no settlement that hour


def build():
    if os.path.exists(OUT):
        return dict(np.load(OUT, allow_pickle=True))
    T, N = len(IDX), len(U)
    arr = {k: np.full((T, N), np.nan, np.float32) for k in ("open", "high", "low", "close", "volume", "tbv", "fr")}
    for j, b in enumerate(U):
        d = read_sym(b + "USDT")
        if d is not None:
            for k in ("open", "high", "low", "close", "volume", "tbv"):
                arr[k][:, j] = d[k].to_numpy(np.float32)
        arr["fr"][:, j] = funding(b).to_numpy(np.float32)
        if j % 20 == 0:
            print("panel", j, b, flush=True)
    np.savez(OUT, **arr, syms=np.array(U), dates=IDX.astype("int64").to_numpy())
    return dict(np.load(OUT, allow_pickle=True))


@njit(cache=True)
def sim(o, sig, side, hold, cost, fr, use_fund):
    """o, sig, fr: (T, N). One position per coin. Entry at o[i+1], exit at o[i+1+hold]. Returns trade arrays."""
    T, N = o.shape
    cap = T * N // max(hold, 1) + N
    ents = np.empty(cap, np.int64); coins = np.empty(cap, np.int64); sides = np.empty(cap, np.int64)
    rets = np.empty(cap, np.float64); funds = np.empty(cap, np.float64)
    k = 0
    for j in range(N):
        busy = -1
        for i in range(T - 1):
            if not sig[i, j] or i < busy:
                continue
            e = i + 1; x = e + hold
            if x >= T or not (o[e, j] > 0) or not (o[x, j] > 0):
                continue
            d = side[i, j]
            r = d * (o[x, j] / o[e, j] - 1.0) - 2.0 * cost[j]
            f = 0.0
            for q in range(e + 1, x + 1):        # settlements strictly after entry, up to and incl. the exit hour's stamp
                if fr[q, j] == fr[q, j]:
                    f += -d * fr[q, j]
            if use_fund:
                r += f
            ents[k] = e; coins[k] = j; sides[k] = d; rets[k] = r; funds[k] = f
            k += 1
            busy = x
    return ents[:k], coins[:k], sides[:k], rets[:k], funds[:k]


def resample_4h(a, agg):
    """(T,N) hourly -> (T4,N) 4h aggregated on UTC 4h buckets; returns array and bucket index per hour."""
    T = a.shape[0]; b = np.arange(T) // 4
    df = pd.DataFrame(a)
    g = df.groupby(b)
    out = getattr(g, agg)().to_numpy()
    return out, b


def to_ltf_prev(a4, b):
    """map 4h values to 1h using the PREVIOUS completed 4h bucket (zero lookahead)."""
    prev = b - 1
    res = np.full((len(b), a4.shape[1]), np.nan)
    ok = prev >= 0
    res[ok] = a4[prev[ok]]
    return res
