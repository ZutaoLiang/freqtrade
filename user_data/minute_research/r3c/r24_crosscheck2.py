"""Candidate explanations for 1452 TRAIN trades: (a) no one-position rule, (b) hourly re-check without position rule,
(c) chain of the last 3 *hours* of an hourly-forward-filled funding series (treats every hour as a settlement)."""
import warnings; warnings.filterwarnings("ignore")
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd

B = "/root/freqtrade/user_data/data/binance_public"
U = pd.read_csv("universe_rank.csv"); U162 = U[U.med_qv >= 1e7].base.tolist()

def one(base):
    k = pd.read_parquet(f"{B}/klines_1m/{base}USDT.parquet", columns=["date", "open"])
    k = k[(k.date >= "2025-01-01") & (k.date < "2025-10-02")]
    o = k.set_index("date").open.resample("1h").first().dropna()
    f = pd.read_parquet(f"{B}/funding/{base}USDT.parquet"); f["date"] = f.date.dt.round("h")
    f = f.drop_duplicates("date", keep="last").set_index("date").funding_rate.sort_index()
    chain = (f >= 3e-4) & (f.shift(1) >= 3e-4) & (f.shift(2) >= 3e-4)
    ff = f.reindex(o.index, method="ffill")
    out = {}
    def trades(sig):
        idx = np.where(sig.to_numpy())[0]; idx = idx[idx + 9 < len(o)]
        return -(o.to_numpy()[idx + 9] / o.to_numpy()[idx + 1] - 1) - 20e-4      # entry next open, exit 8h later
    out["a_settle_nobusy"] = trades(chain.reindex(o.index).fillna(False).astype(bool))
    out["b_hourly_nobusy"] = trades(chain.reindex(o.index, method="ffill").fillna(False).astype(bool))
    out["c_hourly_ffill_chain"] = trades(((ff >= 3e-4) & (ff.shift(1) >= 3e-4) & (ff.shift(2) >= 3e-4)))
    return out

if __name__ == "__main__":
    with ProcessPoolExecutor(16) as ex:
        res = list(ex.map(one, U162))
    for key in res[0]:
        r = np.concatenate([x[key] for x in res]); r = r[~np.isnan(r)]
        print(f"{key:22s} n={len(r):5d} mean={r.mean()*1e4:6.1f}bp pf={r[r>0].sum()/-r[r<0].sum():.2f}")
