"""TRAIN-window liquidity ranking (2025-01-01..2025-09-30), USDT perps only, no equity/commodity-linked."""
import glob, os, pandas as pd
EXCL = {"NVDA","MSTR","TSLA","AAPL","COIN","HOOD","AMZN","GOOGL","META","MSFT","CL","PAXG","XAUT","XAU","XAG","CRCL","AAOI","BZ","NG","QQQ","SPY","INTC","PLTR"}
rows = []
for f in glob.glob("user_data/data/binance_public/resampled/1d/*.parquet"):
    s = os.path.basename(f)[:-8]
    if not s.endswith("USDT"): continue
    b = s[:-4]
    if b in EXCL: continue
    d = pd.read_parquet(f, columns=["date", "quote_volume"])
    tr = d[(d.date >= "2025-01-01") & (d.date < "2025-10-01")]
    if len(tr) < 250: continue                      # must exist through most of TRAIN
    rows.append((b, tr.quote_volume.median(), d.date.min(), d.date.max(), len(d)))
u = pd.DataFrame(rows, columns=["base", "med_qv", "first", "last", "days"]).sort_values("med_qv", ascending=False)
u.to_csv("user_data/minute_research/r3c/universe_rank.csv", index=False)
print(u.head(70).to_string())
