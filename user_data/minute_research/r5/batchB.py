"""r5 Batch B: mark/last dislocation (R9-R15). TRAIN only. 15m + 5m panels."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/root/freqtrade/user_data/minute_research/r5")
import eng5 as E


def zscore(x: np.ndarray, w: int) -> np.ndarray:
    s = pd.DataFrame(x)
    mu = s.rolling(w, min_periods=w // 2).mean().shift(1)
    sd = s.rolling(w, min_periods=w // 2).std().shift(1)
    return ((s - mu) / sd.replace(0, np.nan)).to_numpy()


rows = []
def rec(rid, name, cell, res):
    rows.append({"round": rid, "idea": name, **cell, **res})


# ---------- 15m panel ----------
p = E.Panel("15m")
u = E.universe_cols(p)
n, m = p.n, p.m
close = np.asarray(p.close, np.float64)
mark = np.asarray(p.mark_close, np.float64)
qv = np.asarray(p.quote_volume, np.float64)
prem = np.where((mark > 0) & (close > 0), close / mark - 1.0, np.nan)
z = zscore(prem, 96)  # 24h of 15m bars
vz = zscore(np.log1p(qv), 96)

# R9: dislocation z extreme -> reversion; R10: continuation
for thr in (2.0, 2.5, 3.0):
    for hold in (4, 8, 16):
        L9 = z <= -thr; S9 = z >= thr
        L10 = z >= thr; S10 = z <= -thr
        rec("R9", "dislocation reversion", {"thr": thr, "hold": hold},
            E.evaluate(p, L9, S9, hold, E.TRAIN, u))
        rec("R10", "dislocation continuation", {"thr": thr, "hold": hold},
            E.evaluate(p, L10, S10, hold, E.TRAIN, u))

# R11: dislocation + taker flow agreement (taker buy share > 0.65 while prem z>thr)
tb = np.asarray(p.taker_buy_volume, np.float64)
share = np.where(qv > 0, tb / np.maximum(qv, 1e-9), np.nan)
for thr in (2.5,):
    for hold in (4, 8, 16):
        # premium high AND aggressive buying -> fade (reversion, both agree) vs follow
        L = (z <= -thr) & (share < 0.35); S = (z >= thr) & (share > 0.65)
        rec("R11", "dislocation + taker agree fade", {"thr": thr, "hold": hold},
            E.evaluate(p, L, S, hold, E.TRAIN, u))
        L = (z >= thr) & (share > 0.65); S = (z <= -thr) & (share < 0.35)
        rec("R11f", "dislocation + taker agree follow", {"thr": thr, "hold": hold},
            E.evaluate(p, L, S, hold, E.TRAIN, u))

# R14: dislocation + volume spike
for thr in (2.5,):
    for vthr in (2.0,):
        for hold in (8, 16):
            L = (z <= -thr) & (vz >= vthr); S = (z >= thr) & (vz >= vthr)
            rec("R14", "dislocation + vol spike fade", {"thr": thr, "vthr": vthr, "hold": hold},
                E.evaluate(p, L, S, hold, E.TRAIN, u))

# R12: cross-sectional daily rebalance: long lowest prem, short highest prem
i0, i1 = p.idx(E.TRAIN)
L = np.zeros((n, m), bool); S = np.zeros((n, m), bool)
for i in range(i0, i1):
    if i % 96 != 0: continue  # 00:00 UTC daily
    vals = z[i - 1]
    valid = [c for c in u if np.isfinite(vals[c])]
    if len(valid) < 10: continue
    order = sorted(valid, key=lambda c: vals[c])
    for c in order[:5]: L[i, c] = True
    for c in order[-5:]: S[i, c] = True
for hold in (96, 192):
    rec("R12", "XS prem daily LS", {"hold_h": hold // 4},
        E.evaluate(p, L, S, hold, E.TRAIN, u))

# R15: mid caps (rank 31-90)
u_mid = E.universe_cols(p, min_rank=31, max_rank=90)
for thr in (2.5,):
    for hold in (8, 16):
        L = z <= -thr; S = z >= thr
        rec("R15", "mid-cap dislocation fade", {"thr": thr, "hold": hold},
            E.evaluate(p, L, S, hold, E.TRAIN, u_mid))

# ---------- R13: 5m panel dislocation fade ----------
p5 = E.Panel("5m")
close5 = np.asarray(p5.close, np.float64)
mark5 = np.asarray(p5.mark_close, np.float64)
prem5 = np.where((mark5 > 0) & (close5 > 0), close5 / mark5 - 1.0, np.nan)
z5 = zscore(prem5, 288)
for thr in (2.5, 3.0):
    for hold in (12, 24, 48):
        L = z5 <= -thr; S = z5 >= thr
        rec("R13", "5m dislocation fade", {"thr": thr, "hold_m": hold * 5},
            E.evaluate(p5, L, S, hold, E.TRAIN, u))

out = pd.DataFrame(rows)
out.to_csv("/root/freqtrade/user_data/minute_research/r5/batchB.csv", index=False)
print(out.sort_values("day_t", ascending=False).to_string())
