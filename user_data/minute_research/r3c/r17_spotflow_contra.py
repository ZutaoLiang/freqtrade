"""R17 contrarian spot taker flow XS (pre-registered). TRAIN only unless SEG set."""
import itertools, os
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
import r3lib as L, spotlib as S, xs_engine as X

ALTS = [b for b in L.U60 if b != "BTC"]
SEG = {"TRAIN": L.TRAIN, "VALID": L.VALID, "HOLD": L.HOLD}[os.environ.get("SEG", "TRAIN")]


def feats(base):
    d = L.load(base, end=SEG[1])
    p = pd.DataFrame({"o": d["open"], "qv": d["quote_volume"], "tb": d["taker_buy_quote_volume"]}, index=pd.DatetimeIndex(d["date"]))
    h = p.resample("1h").agg({"o": "first", "qv": "sum", "tb": "sum"})
    sp = S.load_spot(base)
    if sp is None:
        return base, None
    sh = sp.set_index("date").resample("1h").agg({"quote_volume": "sum", "taker_buy_quote_volume": "sum"})
    h["s_flow"] = (2 * sh.taker_buy_quote_volume - sh.quote_volume).reindex(h.index)
    h["s_qv"] = sh.quote_volume.reindex(h.index)
    h["p_flow"] = 2 * h.tb - h.qv
    return base, h


if __name__ == "__main__":
    H = {}
    with ProcessPoolExecutor(12) as ex:
        for b, h in ex.map(feats, ALTS):
            if h is not None:
                H[b] = h
    t0 = pd.Timestamp(SEG[0], tz="UTC")
    col = lambda c: pd.DataFrame({b: h[c] for b, h in H.items()})
    O = col("o")
    rows = []
    for kind, fl, qv in (("spot", col("s_flow"), col("s_qv")), ("perp_control", col("p_flow"), col("qv"))):
        for LB, N in itertools.product([24, 72], [5, 10]):
            sig = -(fl.rolling(LB).sum() / qv.rolling(LB).sum())
            rows.append({"kind": kind, "LB": LB, "N": N, **X.book(sig[sig.index >= t0], O[O.index >= t0], N, 24)})
    df = pd.DataFrame(rows); print("coins", len(H)); print(df.to_string())
    df.to_csv(f"r17_{os.environ.get('SEG','TRAIN')}.csv", index=False)
