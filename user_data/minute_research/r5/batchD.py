"""r5 Batch D: flow / positioning untested cells (R29-R39). TRAIN only. 1h panel."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/root/freqtrade/user_data/minute_research/r5")
import eng5 as E

p = E.Panel("1h")
u = E.universe_cols(p)
n, m = p.n, p.m
rows = []
def rec(rid, name, cell, res):
    rows.append({"round": rid, "idea": name, **cell, **res})

op = np.asarray(p.open, np.float64); cl = np.asarray(p.close, np.float64)
qv = np.asarray(p.quote_volume, np.float64)
oi = np.asarray(p.sum_open_interest, np.float64) if p.sum_open_interest is not None else None
top_pos = np.asarray(p.sum_toptrader_long_short_ratio, np.float64) if p.sum_toptrader_long_short_ratio is not None else None
top_cnt = np.asarray(p.count_toptrader_long_short_ratio, np.float64) if p.count_toptrader_long_short_ratio is not None else None
cnt_ls = np.asarray(p.count_long_short_ratio, np.float64) if p.count_long_short_ratio is not None else None
tb = np.asarray(p.taker_buy_volume, np.float64)
fr = np.asarray(p.funding_rate, np.float64)

def roll_z(x, w):
    s = pd.DataFrame(x)
    mu = s.rolling(w, min_periods=w // 3).mean().shift(1)
    sd = s.rolling(w, min_periods=w // 3).std().shift(1)
    return ((s - mu) / sd.replace(0, np.nan)).to_numpy()

# R29: toptrader position ratio extreme fade (daily)
if top_pos is not None:
    tpz = roll_z(np.log(top_pos.clip(0.01)), 168)
    for thr in (1.5, 2.0):
        for hold in (24, 48):
            L = tpz <= -thr; S = tpz >= thr
            rec("R29", "toptrader pos ratio fade", {"thr": thr, "hold": hold}, E.evaluate(p, L, S, hold, E.TRAIN, u))
# R30: position-ratio vs count-ratio divergence (smart minus retail)
if top_pos is not None and cnt_ls is not None:
    div = np.log(top_pos.clip(0.01)) - np.log(cnt_ls.clip(0.01))
    divz = roll_z(div, 168)
    for thr in (1.5, 2.0):
        for hold in (24, 48):
            L = divz <= -thr; S = divz >= thr
            rec("R30", "smart-retail divergence fade", {"thr": thr, "hold": hold}, E.evaluate(p, L, S, hold, E.TRAIN, u))

# R31: volume dry-up + OI rising -> next-day breakout (trade direction of prior 24h)
if oi is not None:
    qvz = roll_z(np.log1p(qv), 168)
    for hold in (24, 48):
        L = np.zeros((n, m), bool); S = np.zeros((n, m), bool)
        for i in range(169, n):
            if i % 24 != 0: continue
            for c in u:
                if not (np.isfinite(qvz[i - 1, c]) and qvz[i - 1, c] <= -1.5): continue
                o0, o1 = oi[i - 24, c], oi[i - 1, c]
                if not (np.isfinite(o0) and np.isfinite(o1) and o0 > 0 and o1 / o0 > 1.01): continue
                if cl[i - 1, c] > op[i - 1, c]: L[i, c] = True
                else: S[i, c] = True
        rec("R31", "vol dry-up + OI up breakout", {"hold": hold}, E.evaluate(p, L, S, hold, E.TRAIN, u))

# R32: taker share term structure: 1h share vs 24h share divergence
share = np.where(qv > 0, tb / np.maximum(qv, 1e-9), np.nan)
sh_short = pd.DataFrame(share).rolling(6).mean().to_numpy()
sh_long = pd.DataFrame(share).rolling(72, min_periods=36).mean().shift(1).to_numpy()
div = sh_short - sh_long
for thr in (0.10, 0.15):
    for hold in (6, 12, 24):
        L = div <= -thr; S = div >= thr   # fade the burst
        rec("R32", "taker share term fade", {"thr": thr, "hold": hold}, E.evaluate(p, L, S, hold, E.TRAIN, u))
        L = div >= thr; S = div <= -thr   # follow the burst
        rec("R32f", "taker share term follow", {"thr": thr, "hold": hold}, E.evaluate(p, L, S, hold, E.TRAIN, u))

# R33: taker share 1h extreme reversal
for thr in (0.75, 0.8):
    for hold in (6, 12, 24):
        L = share <= 1 - thr; S = share >= thr
        rec("R33", "taker share extreme fade", {"thr": thr, "hold": hold}, E.evaluate(p, L, S, hold, E.TRAIN, u))

# R34: OI velocity extreme + price direction
if oi is not None:
    oiv = pd.DataFrame(oi).pct_change(24).shift(1).to_numpy()
    ret24 = pd.DataFrame(cl / np.roll(op, 24) - 1.0).shift(1).to_numpy()
    for thr in (0.05, 0.08):
        for hold in (24, 48):
            L = (oiv >= thr) & (ret24 > 0); S = (oiv >= thr) & (ret24 < 0)  # OI+price up: long continuation; short fade alt
            rec("R34", "OI velocity + price follow", {"thr": thr, "hold": hold}, E.evaluate(p, L, S, hold, E.TRAIN, u))
            L = (oiv >= thr) & (ret24 > 0); S = (oiv >= thr) & (ret24 < 0)
            S2 = (oiv >= thr) & (ret24 > 0); L2 = np.zeros_like(S2)
            rec("R34f", "OI velocity + price fade", {"thr": thr, "hold": hold}, E.evaluate(p, L2, S2, hold, E.TRAIN, u))

# R35: realized vol ratio 6h/72h extreme -> expansion follow
hi = np.asarray(p.high, np.float64); lo = np.asarray(p.low, np.float64)
tr = np.maximum(hi - lo, np.maximum(abs(hi - np.roll(cl, 1)), abs(lo - np.roll(cl, 1)))) / cl
rv6 = pd.DataFrame(tr).rolling(6).mean().to_numpy()
rv72 = pd.DataFrame(tr).rolling(72, min_periods=36).mean().shift(1).to_numpy()
ratio = rv6 / rv72
for thr in (1.8, 2.2):
    for hold in (12, 24):
        L = ratio >= thr  # expansion: follow direction of last 6h
        Lb = L & (pd.DataFrame(cl).pct_change(6).shift(1).to_numpy() > 0)
        Sb = L & (pd.DataFrame(cl).pct_change(6).shift(1).to_numpy() < 0)
        rec("R35", "RV expansion follow", {"thr": thr, "hold": hold}, E.evaluate(p, Lb, Sb, hold, E.TRAIN, u))

# R36: avg trade size z (qv/count) extreme -> direction
ats = np.where((np.asarray(p.count, np.float64) > 0) & (qv > 0), qv / np.asarray(p.count, np.float64), np.nan)
atz = roll_z(np.log(ats.clip(1)), 168)
for thr in (2.0, 2.5):
    for hold in (12, 24):
        L = (atz >= thr) & (pd.DataFrame(cl).pct_change().shift(1).to_numpy() > 0)
        S = (atz >= thr) & (pd.DataFrame(cl).pct_change().shift(1).to_numpy() < 0)
        rec("R36", "avg trade size z follow", {"thr": thr, "hold": hold}, E.evaluate(p, L, S, hold, E.TRAIN, u))

# R37: market-wide OI change as alt timing (not tradeable alone; gate: OI up -> long alts)
if oi is not None:
    mkt_oi = np.nanmean(oi[:, u], axis=1)
    mo = pd.Series(mkt_oi).pct_change(24).shift(1).to_numpy()
    for thr in (0.01, 0.02):
        for hold in (24, 48):
            Lb = np.zeros((n, m), bool)
            for i in range(25, n):
                if mo[i] >= thr: Lb[i, u] = True
            rec("R37", "market OI up -> long alts", {"thr": thr, "hold": hold}, E.evaluate(p, Lb, np.zeros((n, m), bool), hold, E.TRAIN, u))

# R38: OI 30d high + price 30d low quadrant (short squeeze anticipation)
if oi is not None:
    for hold in (24, 48):
        L = np.zeros((n, m), bool); S = np.zeros((n, m), bool)
        for i in range(24 * 30, n):
            if i % 8 != 0: continue
            for c in u:
                o90 = oi[i - 720:i, c]; p90 = cl[i - 720:i, c]
                if not (np.isfinite(o90).all() and np.isfinite(p90).all()): continue
                if o1 := oi[i - 1, c]:
                    pass
                if oi[i - 1, c] >= np.nanquantile(o90, 0.95) and cl[i - 1, c] <= np.nanquantile(p90, 0.05):
                    L[i, c] = True
        rec("R38", "OI high + price low long", {"hold": hold}, E.evaluate(p, L, np.zeros((n, m), bool), hold, E.TRAIN, u))

# R39: funding extreme x OI percentile quadrant (crowded pay-side fade)
fr_t = np.where(np.abs(fr) > 1e-12, fr, 0.0)
cum24 = pd.DataFrame(fr_t).rolling(24, min_periods=6).sum().to_numpy()
if oi is not None:
    for hold in (24, 48):
        L = np.zeros((n, m), bool); S = np.zeros((n, m), bool)
        for i in range(25, n):
            if i % 8 != 1: continue
            for c in u:
                f = cum24[i - 1, c]
                if not np.isfinite(f): continue
                opct = oi[i - 1, c] / np.nanquantile(oi[i - 720:i, c], 0.5) if np.isfinite(oi[i - 720:i, c]).all() else np.nan
                if not np.isfinite(opct): continue
                if f >= 0.0009 and opct > 1.3: S[i, c] = True
                elif f <= -0.0009 and opct > 1.3: L[i, c] = True
        rec("R39", "funding extreme + high OI fade", {"hold": hold}, E.evaluate(p, L, S, hold, E.TRAIN, u))

out = pd.DataFrame(rows)
out.to_csv("/root/freqtrade/user_data/minute_research/r5/batchD.csv", index=False)
print(out.sort_values("day_t", ascending=False).to_string())
