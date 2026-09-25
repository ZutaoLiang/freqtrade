"""r5 fresh-stage gates on 1h_hist (2022-11..2025-12): R27 Tuesday, R42 listing fade, R21 FOMC.
Only TRAIN-era ideas; 2025-01..09 of hist overlaps TRAIN (exclude), use 2022-11..2024-12 as independent stage.
No mark/OI fields here; the three ideas don't need them. Universe: top-60 by median daily qv in the stage itself.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/root/freqtrade/user_data/minute_research/r5")
import eng5 as E

p = E.Panel("1h_hist")
n, m = p.n, p.m
dates = p.dates
cl = np.asarray(p.close, np.float64)
qv = np.asarray(p.quote_volume, np.float64)
rows = []
def rec(rid, name, cell, res):
    rows.append({"round": rid, "idea": name, **cell, **res})

# stage windows: independent eras only
S1 = ("2023-01-01", "2024-06-01")
S2 = ("2024-06-01", "2025-01-01")
S22 = ("2022-11-01", "2023-01-01")

def universe_for(seg):
    i0, i1 = p.idx(seg)
    dqv = pd.DataFrame(qv[i0:i1]).rolling(24).mean()
    rank = dqv.median().sort_values(ascending=False)
    return np.array([int(c) for c in rank.index[:60]], dtype=np.int64)

def universe_for_22():
    i0, i1 = p.idx(S22)
    dqv = pd.DataFrame(qv[i0:i1]).rolling(24).mean()
    rank = dqv.median().sort_values(ascending=False)
    return np.array([p.symbols.index(s) for s in rank.index[:60] if s in p.symbols], dtype=np.int64)

# ---- R27 Tuesday long universe ----
for seg, tag in ((S1, "2023-01..2024-05"), (S2, "2024-06..2024-12")):
    uu = universe_for(seg)
    for hold in (24, 48):
        L = np.zeros((n, m), bool)
        i0, i1 = p.idx(seg)
        for i in range(i0 + 1, i1):
            if dates[i].hour == 0 and dates[i].dayofweek == 2:
                L[i, uu] = True
        rec("R27g", "Tuesday long universe", {"stage": tag, "hold": hold}, E.evaluate(p, L, np.zeros((n, m), bool), hold, seg, uu))

# all weekdays in stage1 for baseline contrast
uu = universe_for(S1)
for wd in range(7):
    L = np.zeros((n, m), bool)
    i0, i1 = p.idx(S1)
    for i in range(i0 + 1, i1):
        if dates[i].hour == 0 and dates[i].dayofweek == wd:
            L[i, uu] = True
    rec("R27g", "weekday baseline S1", {"weekday": wd}, E.evaluate(p, L, np.zeros((n, m), bool), 24, S1, uu))

# ---- R42 listing fade short (age 6h, hold 48h) ----
finite = np.isfinite(cl)
for seg, tag in ((S1, "2023-01..2024-05"), (S2, "2024-06..2024-12")):
    i0, i1 = p.idx(seg)
    first_i = {}
    for c in range(m):
        w = np.where(finite[i0:i1, c])[0]
        if len(w): first_i[c] = i0 + w[0]
    # listing = first bar inside stage with no prior data
    for age, hold in ((6, 48), (2, 48)):
        S = np.zeros((n, m), bool)
        cnt = 0
        for c, fi in first_i.items():
            prior = finite[:i0, c].any() if i0 > 0 else False
            if prior or fi < 24: continue
            j = fi + age
            if j < i1:
                S[j, c] = True; cnt += 1
        rec("R42g", "listing fade short", {"stage": tag, "age": age, "hold": hold, "listings": cnt},
            E.evaluate(p, np.zeros((n, m), bool), S, hold, seg))

# ---- R21 FOMC long BTC/ETH (extend sample: 2023+2024 FOMC dates) ----
FOMC = ["2023-02-01","2023-03-22","2023-05-03","2023-06-14","2023-07-26","2023-09-20","2023-11-01","2023-12-13",
        "2024-01-31","2024-03-20","2024-05-01","2024-06-12","2024-07-31","2024-09-18","2024-11-07","2024-12-18"]
def event_windows(day_strs, hour_utc):
    idxs = []
    for ds in day_strs:
        ts = pd.Timestamp(ds, tz="UTC")
        hit = np.where((dates >= ts) & (dates < ts + pd.Timedelta(days=1)))[0]
        for i in hit:
            if dates[i].hour == hour_utc:
                idxs.append(int(i)); break
    return idxs
fomc_i = event_windows(FOMC, 18) + event_windows(FOMC, 19)
for hold in (4, 8):
    Lb = np.zeros((n, m), bool)
    btc = p.symbols.index("BTCUSDT"); eth = p.symbols.index("ETHUSDT")
    for i in fomc_i:
        Lb[i, btc] = True; Lb[i, eth] = True
    rec("R21g", "FOMC long BTC/ETH", {"hold": hold, "events": len(fomc_i)},
        E.evaluate(p, Lb, np.zeros((n, m), bool), hold, ("2023-01-01", "2025-01-01"), np.array([btc, eth])))

out = pd.DataFrame(rows)
out.to_csv("/root/freqtrade/user_data/minute_research/r5/freshgate.csv", index=False)
print(out.to_string())
