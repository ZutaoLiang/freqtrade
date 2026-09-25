"""freqtrade datadir for U160 from the independently built Vision 1h panel (p7_panel.npz): 1h / 4h / 1d futures,
plus 1h funding + mark merged from binance-hist, binance (2026), r4new, r2."""
import os
import numpy as np, pandas as pd
OUT = "user_data/data/r5u160/futures"; os.makedirs(OUT, exist_ok=True)
D = np.load("user_data/minute_research/r5/p7_panel.npz", allow_pickle=True)
syms = list(D["syms"]); idx = pd.to_datetime(D["dates"], unit="us", utc=True).astype("datetime64[ms, UTC]")
AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
SRC = ["user_data/data/binance-hist/futures", "user_data/data/r2/futures", "user_data/data/binance/futures", "user_data/data/r4new/futures"]
for j, b in enumerate(syms):
    k = pd.DataFrame({c: D[c][:, j].astype("float64") for c in ("open", "high", "low", "close", "volume")}, index=idx).dropna()
    if k.empty:
        continue
    k.index.name = "date"
    k.reset_index().to_feather(f"{OUT}/{b}_USDT_USDT-1h-futures.feather")
    for tf in ("4h", "1D"):
        r = k.resample(tf).agg(AGG).dropna(); r.index = r.index.astype("datetime64[ms, UTC]"); r.index.name = "date"
        r.reset_index().to_feather(f"{OUT}/{b}_USDT_USDT-{tf.lower()}-futures.feather")
    for kind in ("funding_rate", "mark"):
        fs = []
        for d in SRC:
            p = f"{d}/{b}_USDT_USDT-1h-{kind}.feather"
            if os.path.exists(p):
                try: fs.append(pd.read_feather(p))
                except Exception: pass
        if fs:
            f = pd.concat(fs); f["date"] = f.date.dt.floor("h").astype("datetime64[ms, UTC]")
            f.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True).to_feather(f"{OUT}/{b}_USDT_USDT-1h-{kind}.feather")
print("files", len(os.listdir(OUT)))
