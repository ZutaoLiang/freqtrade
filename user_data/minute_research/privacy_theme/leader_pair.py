"""BTC-regime leader/follower long-short book in the privacy theme (pre-registered in this docstring, 2026-09-25).

Theme P = ZEC, XMR, DASH, ZEN, SCRT, ROSE. Each day d (decided on day d-1's close, traded at day d's open):
  regime s = +1 if BTC daily close > EMA(n) on d-1 else -1;  n in {20, 50}
  leader: "ZEC" (fixed; hindsight reference only) | "mom7" / "mom28" = theme member with the highest trailing 7 / 28-day
          return, re-chosen every 7 days (all 7 weekday offsets evaluated, median reported)
  weights: leader +0.5*s; each other member -0.5*s/(N-1)  (dollar neutral)
Daily return open-to-open; cost 10 bp per unit of weight change; real daily funding (sum of settlements) charged on weights.
Segments DEV 2025-01-01..10-01, VALID-C 2025-12-01..2026-03-01, HOLDOUT 2026-03-01..09-01 (read once, --holdout, survivors only).
Gates on daily book returns: DEV and VALID-C mean > 0 and PF >= 1.2, DEV t >= 2.
"""
import itertools, sys
import numpy as np, pandas as pd
import privacy_theme as PT

SEG = PT.SEG


def load():
    D = {b: PT.bars(b, "1D") for b in PT.P + ["BTC"]}; idx = D["ZEC"].index
    O = pd.DataFrame({b: D[b].open for b in D}).reindex(idx); C = pd.DataFrame({b: D[b].close for b in D}).reindex(idx)
    FR = pd.DataFrame({b: PT.funding(b, idx) for b in PT.P})
    return idx, O, C, FR


def trend_filters(idx):
    """Pre-registered chop filters on each member's daily bars (value on day d uses data through d's close)."""
    out = {}
    for b in PT.P:
        k = pd.read_parquet(f"{PT.R}/klines_1m/{b}USDT.parquet", columns=["date", "high", "low", "close"]).set_index("date")
        k = k[k.index < "2026-09-01"].resample("1D").agg({"high": "max", "low": "min", "close": "last"}).reindex(idx)
        h, l, c = k.high, k.low, k.close; pc = c.shift(1)
        tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
        er = (c - c.shift(20)).abs() / c.diff().abs().rolling(20).sum()
        up, dn = h.diff(), -l.diff()
        pdm = up.where((up > dn) & (up > 0), 0.0); mdm = dn.where((dn > up) & (dn > 0), 0.0)
        w = lambda x: x.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        atr = w(tr); pdi, mdi = 100 * w(pdm) / atr, 100 * w(mdm) / atr
        adx = w(100 * (pdi - mdi).abs() / (pdi + mdi))
        chop = 100 * np.log10(tr.rolling(14).sum() / (h.rolling(14).max() - l.rolling(14).min())) / np.log10(14)
        out[b] = pd.DataFrame({"ER": er >= 0.3, "ADX": adx >= 25, "CHOP": chop <= 50})
    return out


def book(idx, O, C, FR, n, leader, off, regime="btc", filt=None, TF=None):
    btc = C.BTC; s = np.where(btc > btc.ewm(span=n, adjust=False).mean(), 1.0, -1.0)
    own = (C[PT.P] > C[PT.P].ewm(span=n, adjust=False).mean()).astype(float) * 2 - 1     # each member's own trend
    r = (O[PT.P].shift(-1) / O[PT.P] - 1)                      # return from open d to open d+1
    W = pd.DataFrame(0.0, index=idx, columns=PT.P); cur = None
    for d in range(1, len(idx)):
        if leader == "ZEC":
            cur = "ZEC"
        elif (d - off) % 7 == 0 or cur is None:
            L = 7 if leader == "mom7" else 28
            m = (C[PT.P].iloc[d - 1] / C[PT.P].iloc[d - 1 - L] - 1) if d - 1 - L >= 0 else None
            if m is not None and m.notna().sum() >= 4:
                cur = m.idxmax()
        if cur is None:
            continue
        sd = s[d - 1] if regime == "btc" else own[cur].iloc[d - 1]
        if filt is not None and not bool(TF[cur][filt].iloc[d - 1]):
            W.iloc[d] = 0.0; continue                                           # choppy leader -> flat
        others = [b for b in PT.P if b != cur]
        W.iloc[d] = 0.0; W.loc[idx[d], cur] = 0.5 * sd
        for b in others:
            W.loc[idx[d], b] = -0.5 * sd / len(others)
    turn = W.diff().abs().sum(axis=1).fillna(W.abs().sum(axis=1))
    pnl = (W * r).sum(axis=1) - (W * FR).sum(axis=1) - turn * PT.COST
    return pnl


def stats(p, a, b):
    x = p[(p.index >= a) & (p.index < b)].dropna()
    g, l = x[x > 0].sum(), -x[x < 0].sum(); eq = (1 + x).cumprod()
    return dict(days=len(x), mean_bp=round(x.mean() * 1e4, 1), pf=round(g / l, 2) if l else np.inf,
                t=round(x.mean() / x.std() * np.sqrt(len(x)), 2), total_pct=round((eq.iloc[-1] - 1) * 100, 1),
                maxdd_pct=round((eq / eq.cummax() - 1).min() * 100, 1))


if __name__ == "__main__":
    segs = ["DEV", "VALID-C"] + (["HOLDOUT"] if "--holdout" in sys.argv else [])
    idx, O, C, FR = load(); rows = []
    regime = "own" if "--own" in sys.argv else "btc"
    filters = [None]
    if "--filters" in sys.argv:
        filters = ["ER", "ADX", "CHOP"]; TF = trend_filters(idx)
    else:
        TF = None
    for n, leader, filt in itertools.product((20, 50), ("ZEC", "mom7", "mom28"), filters):
        offs = [0] if leader == "ZEC" else range(7)
        res = {s: [stats(book(idx, O, C, FR, n, leader, o, regime, filt, TF), *SEG[s]) for o in offs] for s in segs}
        row = dict(ema=n, leader=leader + (" (hindsight)" if leader == "ZEC" else ""), filter=filt or "-")
        for s in segs:
            v = pd.DataFrame(res[s]).median()
            row.update({f"{s}:{k}": round(float(v[k]), 2) for k in ("mean_bp", "pf", "t", "total_pct", "maxdd_pct")})
        row["PASS"] = row["DEV:mean_bp"] > 0 and row["DEV:pf"] >= 1.2 and row["DEV:t"] >= 2 and row["VALID-C:mean_bp"] > 0 and row["VALID-C:pf"] >= 1.2
        rows.append(row)
    pd.set_option("display.width", 250); print(pd.DataFrame(rows).to_string(index=False))
