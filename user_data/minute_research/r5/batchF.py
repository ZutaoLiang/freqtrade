"""r5 Batch F: R47 R7-variants, R48 interval-change (fixed), R49 hourly-skew XS fade, R50 Tuesday anchor check."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/root/freqtrade/user_data/minute_research/r5")
import eng5 as E

p = E.Panel("1h")
u = E.universe_cols(p)
u_mid = E.universe_cols(p, min_rank=31, max_rank=90)
n, m = p.n, p.m
cl = np.asarray(p.close, np.float64)
mark = np.asarray(p.mark_close, np.float64)
fr = np.asarray(p.funding_rate, np.float64)
fih = np.asarray(p.funding_interval_hours, np.float64)
rows = []
def rec(rid, name, cell, res):
    rows.append({"round": rid, "idea": name, **cell, **res})

fr_t = np.where(np.abs(fr) > 1e-12, fr, 0.0)
prem = np.where((mark > 0) & (cl > 0), cl / mark - 1.0, np.nan)

# ---- R47: funding z + premium agree variants ----
for w in (72, 168):
    cum = pd.DataFrame(fr_t).rolling(w, min_periods=w // 6).sum().to_numpy()
    for hold in (48, 72, 96):
        for uu, utag in ((u, "U60"), (u_mid, "U31-90")):
            L = np.zeros((n, m), bool); S = np.zeros((n, m), bool)
            for i in range(w + 8, n):
                if i % 8 != 1: continue
                for c in uu:
                    f = cum[i - 1, c]; pr = prem[i - 1, c]
                    if not (np.isfinite(f) and np.isfinite(pr)): continue
                    if f >= 0.0012 and pr > 0: S[i, c] = True
                    elif f <= -0.0012 and pr < 0: L[i, c] = True
            rec("R47", "funding z + premium agree var", {"w": w, "hold": hold, "uni": utag},
                E.evaluate(p, L, S, hold, E.TRAIN, uu))

# ---- R48: funding interval change event (fade current funding direction) ----
L = np.zeros((n, m), bool); S = np.zeros((n, m), bool)
nev = 0
for c in u:
    v = fih[:, c]
    ch = np.where((v[1:] != v[:-1]) & (v[1:] > 0) & (v[:-1] > 0))[0] + 1
    for i in ch:
        if i < 25: continue
        f = fr_t[i, c]
        if f > 0: S[i, c] = True; nev += 1
        elif f < 0: L[i, c] = True; nev += 1
rec("R48", "interval-change fade funding", {"events": nev}, E.evaluate(p, L, S, 24, E.TRAIN, u))

# ---- R49: hourly return skew XS (7d window, daily rebalance, short high skew) ----
r1 = pd.DataFrame(cl).pct_change().shift(1)
skew = r1.rolling(168, min_periods=84).skew()
for hold in (24, 48):
    L = np.zeros((n, m), bool); S = np.zeros((n, m), bool)
    for i in range(169, n):
        if i % 24 != 0: continue
        vals = skew.iloc[i - 1].to_numpy()
        valid = [c for c in u if np.isfinite(vals[c])]
        if len(valid) < 10: continue
        order = sorted(valid, key=lambda c: vals[c])
        for c in order[:5]: L[i, c] = True
        for c in order[-5:]: S[i, c] = True
    rec("R49", "hourly skew XS daily LS", {"hold": hold}, E.evaluate(p, L, S, hold, E.TRAIN, u))

# ---- R50: Tuesday long anchor robustness (trap 14) ----
uu = u
for anchor in ("hour0_Tue", "avg24h_Tue", "hour0_Wed", "hour0_Thu", "hour0_Mon"):
    for hold in (24,):
        L = np.zeros((n, m), bool)
        i0, i1 = p.idx(E.TRAIN)
        if anchor == "hour0_Tue":
            for i in range(i0, i1):
                if p.dates[i].hour == 0 and p.dates[i].dayofweek == 2: L[i, uu] = True
        elif anchor == "avg24h_Tue":
            for i in range(i0, i1):
                if p.dates[i].dayofweek == 2: L[i, uu] = True
        elif anchor == "hour0_Wed":
            for i in range(i0, i1):
                if p.dates[i].hour == 0 and p.dates[i].dayofweek == 3: L[i, uu] = True
        elif anchor == "hour0_Thu":
            for i in range(i0, i1):
                if p.dates[i].hour == 0 and p.dates[i].dayofweek == 4: L[i, uu] = True
        elif anchor == "hour0_Mon":
            for i in range(i0, i1):
                if p.dates[i].hour == 0 and p.dates[i].dayofweek == 1: L[i, uu] = True
        rec("R50", "Tuesday anchor check", {"anchor": anchor, "hold": hold},
            E.evaluate(p, L, np.zeros((n, m), bool), hold, E.TRAIN, uu))

out = pd.DataFrame(rows)
out.to_csv("/root/freqtrade/user_data/minute_research/r5/batchF.csv", index=False)
print(out.sort_values("day_t", ascending=False).to_string())
