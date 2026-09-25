"""r5 Batch A: funding-family structural fixes (R1-R7). TRAIN only."""
import sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, "/root/freqtrade/user_data/minute_research/r5")
import eng5 as E

p = E.Panel("1h")
u = E.universe_cols(p)
n, m = p.n, p.m
fr = np.asarray(p.funding_rate, dtype=np.float64)
mark = np.asarray(p.mark_close, dtype=np.float64)
close = np.asarray(p.close, dtype=np.float64)
oi = np.asarray(p.sum_open_interest, dtype=np.float64) if p.sum_open_interest is not None else None

# settlements: rows where funding != 0 within trading columns; trailing 24h sum of settlement rates
fr_t = np.where(np.abs(fr) > 1e-12, fr, 0.0)
cum24 = pd.DataFrame(fr_t).rolling(24, min_periods=6).sum().to_numpy()
settle8 = (fr_t > 0)  # a settlement hour with positive rate

rows = []
def rec(rid, name, cell, res):
    rows.append({"round": rid, "idea": name, **cell, **res})

LONG = np.zeros((n, m), bool); SHORT = np.zeros((n, m), bool)

# --- R1: daily top-3 trailing-24h funding short (equal-weight basket, E-fix) ---
for theta in (0.0006, 0.0009, 0.0012):
    for hold in (24, 48, 72):
        L = np.zeros((n, m), bool); S = np.zeros((n, m), bool)
        for i in range(n):
            day0 = i - (i % 24)
            if i != day0:  # rebalance at 00:00 UTC only
                continue
            if i < 24: continue
            vals = cum24[i - 1]
            cols = [c for c in u if np.isfinite(vals[c]) and vals[c] >= theta]
            cols = sorted(cols, key=lambda c: -vals[c])[:3]
            for c in cols: S[i, c] = True
        r = E.evaluate(p, L, S, hold, E.TRAIN, u)
        rec("R1", "top3 cum24F short", {"theta": theta, "hold": hold}, r)

# --- R2: negative-funding exhaustion long (mirror) ---
for theta in (-0.0006, -0.0009, -0.0012):
    for hold in (24, 48, 72):
        L = np.zeros((n, m), bool); S = np.zeros((n, m), bool)
        for i in range(n):
            day0 = i - (i % 24)
            if i != day0 or i < 24: continue
            vals = cum24[i - 1]
            cols = [c for c in u if np.isfinite(vals[c]) and vals[c] <= theta]
            cols = sorted(cols, key=lambda c: vals[c])[:3]
            for c in cols: L[i, c] = True
        r = E.evaluate(p, L, S, hold, E.TRAIN, u)
        rec("R2", "bottom3 cum24F long", {"theta": theta, "hold": hold}, r)

# --- R3: exhaustion short + OI confirmation (OI rising over 24h) ---
for theta in (0.0009, 0.0012):
    for hold in (24, 48):
        L = np.zeros((n, m), bool); S = np.zeros((n, m), bool)
        for i in range(n):
            day0 = i - (i % 24)
            if i != day0 or i < 48: continue
            vals = cum24[i - 1]
            for c in u:
                if not (np.isfinite(vals[c]) and vals[c] >= theta): continue
                if oi is None: continue
                o0, o1 = oi[i - 24, c], oi[i - 1, c]
                if np.isfinite(o0) and np.isfinite(o1) and o0 > 0 and o1 / o0 > 1.02:
                    S[i, c] = True
        r = E.evaluate(p, L, S, hold, E.TRAIN, u)
        rec("R3", "exh short + OI up", {"theta": theta, "hold": hold}, r)

# --- R4: exhaustion short entered T+4h after settlement, exit before next settlement (hold 20h) ---
# signal at settlement bar itself (funding known), enter i+5 -> ~T+5h, hold 20h -> exit ~T+25h
for theta in (0.0009, 0.0012, 0.0015):
    L = np.zeros((n, m), bool); S = np.zeros((n, m), bool)
    for i in range(n):
        if i < 24: continue
        vals = cum24[i - 1]
        for c in u:
            if np.isfinite(vals[c]) and vals[c] >= theta:
                S[i, c] = True
    r = E.evaluate(p, L, S, 20, E.TRAIN, u)
    rec("R4", "exh short @settle+5h hold20", {"theta": theta}, r)

# --- R5: funding interval change event (event-type criteria) ---
fih = np.asarray(p.funding_interval_hours, dtype=np.float64) if p.funding_interval_hours is not None else None
if fih is not None:
    for direction in ("short", "long"):
        L = np.zeros((n, m), bool); S = np.zeros((n, m), bool)
        for c in u:
            v = fih[:, c]
            ch = np.where((v[1:] != v[:-1]) & np.isfinite(v[1:]) & np.isfinite(v[:-1]) & (v[1:] > 0))[0] + 1
            for i in ch:
                if v[i] < v[i - 1]:  # interval shortened -> funding pressure intensifies
                    (S if direction == "short" else L)[i:i + 48, c] = False
                # fade direction of funding: pay side crowded -> short
        # simpler: interval change -> trade in direction opposite to current funding sign
        L = np.zeros((n, m), bool); S = np.zeros((n, m), bool)
        for c in u:
            v = fih[:, c]
            ch = np.where((v[1:] != v[:-1]) & np.isfinite(v[1:]) & np.isfinite(v[:-1]) & (v[1:] > 0))[0] + 1
            for i in ch:
                if fr_t[i, c] > 0: S[i, c] = True
                elif fr_t[i, c] < 0: L[i, c] = True
        r = E.evaluate(p, L, S, 24, E.TRAIN, u)
        rec("R5", "interval-change fade funding", {"event": "ch"}, r)

# --- R6: settlement candle range -> next settlement direction ---
import numpy.ma as ma
for q in (0.9, 0.95):
    L = np.zeros((n, m), bool); S = np.zeros((n, m), bool)
    rng = (np.asarray(p.high, np.float64) - np.asarray(p.low, np.float64)) / np.asarray(p.close, np.float64)
    for i in range(25, n):
        # settlement hours: use global 8h marks 00/08/16; check funding settlement effect
        if i % 8 != 0: continue
        prev = rng[i - 8:i]
        thr = np.nanquantile(prev, q, axis=0)
        up = (np.asarray(p.close, np.float64)[i - 1] > np.asarray(p.open, np.float64)[i - 1])
        pass
    break  # replaced below with vectorized simple version

rng = (np.asarray(p.high, np.float64) - np.asarray(p.low, np.float64)) / np.asarray(p.close, np.float64)
op = np.asarray(p.open, np.float64); cl = np.asarray(p.close, np.float64)
for hold in (8, 16):
    L = np.zeros((n, m), bool); S = np.zeros((n, m), bool)
    for c in u:
        r_ = rng[:, c]
        for i in range(9, n):
            if i % 8 != 0: continue
            prev = r_[i - 8:i]
            if not np.isfinite(prev).all(): continue
            top = np.quantile(prev, 0.9)
            if r_[i - 1] >= top:  # wide settlement candle
                if cl[i - 1, c] > op[i - 1, c]: S[i, c] = True
                else: L[i, c] = True
    r = E.evaluate(p, L, S, hold, E.TRAIN, u)
    rec("R6", "wide settlement candle fade", {"hold": hold}, r)

# --- R7: funding z extreme + mark>last premium same direction ---
z = pd.DataFrame(fr_t).rolling(24 * 7, min_periods=24).sum().to_numpy()  # 7d cum funding
prem = close / mark - 1.0
for theta in (0.0009, 0.0012):
    for hold in (24, 48):
        L = np.zeros((n, m), bool); S = np.zeros((n, m), bool)
        for i in range(24 * 8, n):
            if i % 8 != 1: continue
            for c in u:
                if np.isfinite(z[i - 1, c]) and z[i - 1, c] >= theta and np.isfinite(prem[i - 1, c]) and prem[i - 1, c] > 0:
                    S[i, c] = True
                elif np.isfinite(z[i - 1, c]) and z[i - 1, c] <= -theta and np.isfinite(prem[i - 1, c]) and prem[i - 1, c] < 0:
                    L[i, c] = True
        r = E.evaluate(p, L, S, hold, E.TRAIN, u)
        rec("R7", "funding z + premium agree", {"theta": theta, "hold": hold}, r)

out = pd.DataFrame(rows).sort_values(["round", "day_t"], ascending=[True, False])
out.to_csv("/root/freqtrade/user_data/minute_research/r5/batchA.csv", index=False)
print(out.to_string())
