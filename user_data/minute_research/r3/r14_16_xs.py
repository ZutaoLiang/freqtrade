"""R14-R16 XS books (pre-registered in LOG.md). TRAIN only."""
import itertools, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
import r3lib as L, depthlib as D, spotlib as S, xs_engine as X
from r08_liq_magnet import heat, BIN, LEVS

ALTS = [b for b in L.U60 if b != "BTC"]


def feats(base):
    d = L.load(base)
    idx = pd.DatetimeIndex(d["date"])
    out = pd.DataFrame(index=idx)
    out["o"] = d["open"]
    dep = D.build(f"{base}USDT")
    out["imb5"] = D.features(dep, idx)["imb5"] if dep is not None else np.nan
    h = out.resample("1h").agg({"o": "first", "imb5": "mean"})
    # spot taker imbalance per hour
    sp = S.load_spot(base)
    if sp is not None:
        sh = sp.set_index("date").resample("1h").agg({"quote_volume": "sum", "taker_buy_quote_volume": "sum"})
        h["s_flow"] = (2 * sh.taker_buy_quote_volume - sh.quote_volume).reindex(h.index)
        h["s_qv"] = sh.quote_volume.reindex(h.index)
    # heatmap imbalance at hour snapshots (W 6%), same construction as R08
    m = pd.read_parquet(f"{L.ROOT}/metrics/{base}USDT.parquet", columns=["date", "sum_open_interest_value"])
    m = m[m.date < L.TRAIN[1]].drop_duplicates("date").set_index("date").sort_index()
    k = pd.DataFrame({"h": d["high"], "l": d["low"], "c": d["close"]}, index=idx)
    k5 = k.resample("5min", label="right", closed="left").agg({"h": "max", "l": "min", "c": "last"})
    x = k5.join(m, how="inner").dropna()
    lp = np.log(x.c.to_numpy())
    b0 = int(np.floor(lp.min() / BIN)) - 200; nb = int(np.ceil(lp.max() / BIN)) - b0 + 200
    hour = np.asarray(x.index.minute == 0)
    imb = heat(lp, x.h.to_numpy(), x.l.to_numpy(), x.sum_open_interest_value.to_numpy(), b0, nb, LEVS, 7 * 288.0, hour,
               np.array([0.06]))[0]
    hm = pd.Series(imb, index=x.index)[hour]
    hm.index = hm.index - pd.Timedelta(hours=1)          # snapshot at t = close of hour starting t-1h
    h["heat"] = hm.reindex(h.index)
    return base, h


if __name__ == "__main__":
    H = {}
    with ProcessPoolExecutor(12) as ex:
        for b, h in ex.map(feats, ALTS):
            H[b] = h
    O = pd.DataFrame({b: h.o for b, h in H.items()})
    col = lambda c: pd.DataFrame({b: h[c] for b, h in H.items() if c in h})
    rows = []
    imb = col("imb5")
    for reb, N in itertools.product([72, 168], [5, 10]):
        rows.append({"round": "R14 depth", "LB": 168, "REB": reb, "N": N, **X.book(-imb.rolling(168, min_periods=84).mean(), O, N, reb)})
    heat_ = col("heat")
    for LB, N in itertools.product([24, 72], [5, 10]):
        rows.append({"round": "R15 heat", "LB": LB, "REB": 24, "N": N, **X.book(heat_.rolling(LB, min_periods=LB // 2).mean(), O, N, 24)})
    fl, qv = col("s_flow"), col("s_qv")
    for LB, N in itertools.product([24, 72], [5, 10]):
        rows.append({"round": "R16 spotflow", "LB": LB, "REB": 24, "N": N, **X.book(fl.rolling(LB).sum() / qv.rolling(LB).sum(), O[fl.columns], N, 24)})
    df = pd.DataFrame(rows); print(df.to_string()); df.to_csv("r14_16_train.csv", index=False)
