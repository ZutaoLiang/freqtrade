"""r5 Batch E: listing / structure events (R41-R45). TRAIN only. 1h panel."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/root/freqtrade/user_data/minute_research/r5")
import eng5 as E

p = E.Panel("1h")
u = E.universe_cols(p)
n, m = p.n, p.m
dates = p.dates
cl = np.asarray(p.close, np.float64)
qv = np.asarray(p.quote_volume, np.float64)
rows = []
def rec(rid, name, cell, res):
    rows.append({"round": rid, "idea": name, **cell, **res})

finite = np.isfinite(cl)
first_i = np.full(m, -1); last_i = np.full(m, -1)
for c in range(m):
    w = np.where(finite[:, c])[0]
    if len(w): first_i[c], last_i[c] = w[0], w[-1]

DATA_START = np.where(dates >= pd.Timestamp("2025-01-15", tz="UTC"))[0][0]
TRAIN_END_I = p.idx(E.TRAIN)[1]

new_listings = [c for c in range(m) if 0 < first_i[c] and first_i[c] >= DATA_START]
delisted = [c for c in range(m) if last_i[c] < n - 24 * 30 and first_i[c] >= 0]

# R41/R42: first 48h after listing: enter at age a hours, hold h
for a in (2, 6, 12):
    for hold in (12, 24, 48):
        L = np.zeros((n, m), bool); S = np.zeros((n, m), bool)
        for c in new_listings:
            i = first_i[c] + a
            if i >= TRAIN_END_I: continue
            L[i, c] = True; S[i, c] = True
        rec("R41", "new listing momentum long", {"age_h": a, "hold": hold},
            E.evaluate(p, L, np.zeros((n, m), bool), hold, E.TRAIN))
        rec("R42", "new listing fade short", {"age_h": a, "hold": hold},
            E.evaluate(p, np.zeros((n, m), bool), S, hold, E.TRAIN))

# R43: age 30-90d momentum: 7d return sign -> hold 48h (weekly check)
ret7 = pd.DataFrame(cl).pct_change(168).shift(1).to_numpy()
age_ok = np.zeros((n, m), bool)
for c in new_listings:
    a0, a1 = first_i[c] + 720, first_i[c] + 2160
    age_ok[max(a0, 0):min(a1, n), c] = True
for hold in (24, 48):
    L = (ret7 > 0) & age_ok; S = (ret7 < 0) & age_ok
    rec("R43", "age 30-90d momentum", {"hold": hold}, E.evaluate(p, L, S, hold, E.TRAIN))

# R44: dormant coin volume awakening: baseline qv rank > 400, then 24h qv z > 4
qv_med = pd.DataFrame(qv).rolling(30 * 24, min_periods=240).median().shift(1).to_numpy()
qv_now = pd.DataFrame(qv).rolling(24).sum().shift(1).to_numpy()
awake = np.zeros((n, m), bool)
for c in u:
    base = qv_med[:, c]
    for i in range(240, n):
        if np.isfinite(base[i]) and base[i] > 0:
            pass
    break
# vectorized: ratio = 24h qv sum / 30d median daily qv
qv_day_med = pd.DataFrame(qv).rolling(720, min_periods=240).median().shift(1).to_numpy()
qv_24 = pd.DataFrame(qv).rolling(24).sum().shift(1).to_numpy()
with np.errstate(invalid="ignore", divide="ignore"):
    ratio = qv_24 / qv_day_med
for thr in (10.0, 20.0):
    for hold in (24, 48):
        L = np.zeros((n, m), bool); S = np.zeros((n, m), bool)
        for i in range(721, n):
            if i % 4 != 0: continue
            for c in u:
                if np.isfinite(ratio[i, c]) and ratio[i, c] >= thr:
                    if cl[i, c] > cl[i - 24, c]: L[i, c] = True
                    else: S[i, c] = True
        rec("R44", "dormant awakening follow", {"thr": thr, "hold": hold}, E.evaluate(p, L, S, hold, E.TRAIN))

# R45: delisting last 14 days short
for hold in (24, 48):
    L = np.zeros((n, m), bool); S = np.zeros((n, m), bool)
    for c in delisted:
        for i in range(max(first_i[c], last_i[c] - 336), min(last_i[c] - hold, TRAIN_END_I)):
            if i % 24 == 0:
                S[i, c] = True
    rec("R45", "pre-delist short", {"hold": hold}, E.evaluate(p, np.zeros((n, m), bool), S, hold, E.TRAIN))

out = pd.DataFrame(rows)
out.to_csv("/root/freqtrade/user_data/minute_research/r5/batchE.csv", index=False)
print(out.sort_values("day_t", ascending=False).to_string())
