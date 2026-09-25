"""r5 Batch C: calendar / macro events (R17-R27). TRAIN only. 1h panel, BTC/ETH focus + universe."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/root/freqtrade/user_data/minute_research/r5")
import eng5 as E

p = E.Panel("1h")
u = E.universe_cols(p)
n, m = p.n, p.m
dates = p.dates
btc = p.symbols.index("BTCUSDT"); eth = p.symbols.index("ETHUSDT")
L = np.zeros((n, m), bool); S = np.zeros((n, m), bool)
rows = []
def rec(rid, name, cell, res):
    rows.append({"round": rid, "idea": name, **cell, **res})

def hours_of(y, mo, d, h):
    return pd.Timestamp(year=y, month=mo, day=d, hour=h, tz="UTC")

# Deribit weekly expiry: every Friday 08:00 UTC; monthly: last Friday of month.
# Event window: enter at expiry-2h, hold H.
fri_idx = [i for i in range(n) if dates[i].dayofweek == 4 and dates[i].hour == 6]
all_fri = set(fri_idx)
last_fri = set()
seen_month = set()
for i in sorted(fri_idx, reverse=True):
    key = (dates[i].year, dates[i].month)
    if key not in seen_month:
        seen_month.add(key); last_fri.add(i)

for tag, idxs in (("R17_weekly", all_fri), ("R18_monthly", last_fri)):
    for hold in (2, 4, 8):
        Lb = np.zeros((n, m), bool)
        for i in idxs:
            for j in range(max(i, 0), min(i + 2, n)):
                Lb[j, btc] = True; Lb[j, eth] = True
        rec(tag, "expiry-window long BTC/ETH", {"hold": hold}, E.evaluate(p, Lb, S, hold, E.TRAIN, universe=np.array([btc, eth])))
        Lb = np.zeros((n, m), bool)
        for i in idxs:
            for j in range(max(i, 0), min(i + 2, n)):
                S[j, btc] = True; S[j, eth] = True
        rec(tag + "s", "expiry-window short BTC/ETH", {"hold": hold}, E.evaluate(p, L, S, hold, E.TRAIN, universe=np.array([btc, eth])))
        S = np.zeros((n, m), bool)

# CME quarterly expiry: last Friday of Mar/Jun/Sep/Dec, 08:00 CT = 13/14 UTC -> use 13:00 UTC bar entry
q_idx = [i for i in sorted(last_fri) if dates[i].month in (3, 6, 9, 12)]
Lb = np.zeros((n, m), bool); Sb = np.zeros((n, m), bool)
for i in q_idx:
    j = i + 7  # 06:00 + 7h = 13:00
    if j < n:
        for k in range(j, min(j + 2, n)):
            Lb[k, btc] = True; Sb[k, eth] = False
for hold in (2, 4):
    rec("R19", "CME quarterly expiry long BTC", {"hold": hold},
        E.evaluate(p, Lb, np.zeros((n, m), bool), hold, E.TRAIN, universe=np.array([btc])))

# US macro events 2025-2026 (8:30 ET = 12:30 UTC EST / 13:30 UTC EDT). Enter at release hour, hold H.
CPI = ["2025-01-15","2025-02-12","2025-03-12","2025-04-10","2025-05-13","2025-06-11","2025-07-15",
       "2025-08-12","2025-09-11","2025-10-24","2025-11-13","2025-12-10",
       "2026-01-13","2026-02-11","2026-03-11","2026-04-10","2026-05-12","2026-06-10","2026-07-14","2026-08-12"]
FOMC = ["2025-01-29","2025-03-19","2025-05-07","2025-06-18","2025-07-30","2025-09-17","2025-10-29","2025-12-10",
        "2026-01-28","2026-03-18","2026-04-29","2026-06-17","2026-07-29"]  # announcement 14:00 ET = 18/19 UTC
nfp_idx = [i for i in range(n) if dates[i].dayofweek == 4 and dates[i].day <= 7 and dates[i].hour == 12]

def event_windows(day_strs, hour_utc):
    idxs = []
    for ds in day_strs:
        ts = pd.Timestamp(ds, tz="UTC")
        hit = np.where((dates >= ts) & (dates < ts + pd.Timedelta(days=1)))[0]
        for i in hit:
            if dates[i].hour == hour_utc:
                idxs.append(int(i)); break
    return idxs

cpi_i = event_windows(CPI, 13) + event_windows(CPI, 12)
fomc_i = event_windows(FOMC, 18) + event_windows(FOMC, 19)

for tag, idxs in (("R20_cpi", cpi_i), ("R21_fomc", fomc_i), ("R22_nfp", nfp_idx)):
    for hold in (1, 2, 4):
        for side, Lb in (("long", True), ("short", False)):
            Ls = np.zeros((n, m), bool); Ss = np.zeros((n, m), bool)
            for i in idxs:
                for j in range(i, min(i + 1, n)):
                    (Ls if Lb else Ss)[j, btc] = True
                    (Ls if Lb else Ss)[j, eth] = True
            rec(tag, f"event {side} BTC/ETH", {"hold": hold},
                E.evaluate(p, Ls, Ss, hold, E.TRAIN, universe=np.array([btc, eth])))

# R23: post-event 15m momentum continuation grid — handled on 15m panel below
# R24: month-boundary first 3 days long universe
Lb = np.zeros((n, m), bool)
for i in range(n):
    if dates[i].day <= 3 and dates[i].hour == 0:
        Lb[i, u] = True
for hold in (24, 48):
    rec("R24", "month-start long universe", {"hold": hold}, E.evaluate(p, Lb, np.zeros((n, m), bool), hold, E.TRAIN, u))

# R25: US equity close 20:00 UTC reversal on BTC/ETH
Lb = np.zeros((n, m), bool); Sb = np.zeros((n, m), bool)
op = np.asarray(p.open, np.float64); cl = np.asarray(p.close, np.float64)
for i in range(1, n):
    if dates[i].hour == 20:
        for c in (btc, eth):
            if cl[i - 1, c] > op[i - 1, c]: Sb[i, c] = True
            else: Lb[i, c] = True
for hold in (2, 4, 8):
    rec("R25", "US close reversal BTC/ETH", {"hold": hold},
        E.evaluate(p, Lb, Sb, hold, E.TRAIN, universe=np.array([btc, eth])))

# R26: Asia open 00:00 UTC momentum
Lb = np.zeros((n, m), bool); Sb = np.zeros((n, m), bool)
for i in range(1, n):
    if dates[i].hour == 0:
        for c in u:
            if cl[i - 1, c] > op[i - 1, c]: Lb[i, c] = True
            else: Sb[i, c] = True
for hold in (4, 8, 12):
    rec("R26", "asia-open momentum", {"hold": hold}, E.evaluate(p, Lb, Sb, hold, E.TRAIN, u))

# R27: weekday seasonality grid (long universe each weekday)
for wd in range(7):
    Lb = np.zeros((n, m), bool)
    for i in range(n):
        if dates[i].hour == 0 and dates[i].dayofweek == wd:
            Lb[i, u] = True
    rec("R27", "weekday long universe", {"weekday": wd}, E.evaluate(p, Lb, np.zeros((n, m), bool), 24, E.TRAIN, u))

out = pd.DataFrame(rows)
out.to_csv("/root/freqtrade/user_data/minute_research/r5/batchC.csv", index=False)
print(out.sort_values("day_t", ascending=False).to_string())
