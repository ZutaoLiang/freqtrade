"""Beta-neutral cross-sectional book with turnover cost; all rebalance offsets evaluated."""
import numpy as np, pandas as pd


def book(sig: pd.DataFrame, O: pd.DataFrame, N: int, reb: int, cost=10e-4):
    """sig, O: hourly frames (index = hour start, columns = coins). Decide at close of hour a, trade next open."""
    sig = sig.reindex(O.index)
    res = []
    for off in range(reb):
        ts = np.arange(off, len(O) - 1, reb)
        prev = pd.Series(0.0, index=O.columns)
        rets, days = [], []
        for a, b in zip(ts[:-1], ts[1:]):
            s = sig.iloc[a].dropna()
            if len(s) < 2 * N:
                continue
            w = pd.Series(0.0, index=O.columns)
            w[s.nlargest(N).index] = 1.0 / N
            w[s.nsmallest(N).index] = -1.0 / N
            if b + 1 >= len(O):
                break
            r = (O.iloc[b + 1] / O.iloc[a + 1] - 1).fillna(0)
            turn = (w - prev).abs().sum()
            rets.append((w * r).sum() - turn * cost)
            days.append(O.index[a])
            prev = w * (1 + r) / (1 + (w * r).sum())
        rr = pd.Series(rets, index=pd.DatetimeIndex(days))
        daily = rr / (reb / 24)                                    # per-day equivalent
        res.append((daily.mean() * 1e4, rr.mean() / rr.std() * np.sqrt(len(rr)), len(rr)))
    r = np.array(res)
    return {"bp_day": round(r[:, 0].mean(), 2), "t_med": round(float(np.median(r[:, 1])), 2),
            "t_min": round(r[:, 1].min(), 2), "t_max": round(r[:, 1].max(), 2),
            "pos_frac": round(float((r[:, 0] > 0).mean()), 2), "periods": int(r[:, 2].mean())}
