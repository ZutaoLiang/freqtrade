"""r5 R51-R56: R47 refinements + ablations + beta decomposition. TRAIN only."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/root/freqtrade/user_data/minute_research/r5")
import eng5 as E

p = E.Panel("1h")
u_mid = E.universe_cols(p, min_rank=31, max_rank=90)
u_tail = E.universe_cols(p, min_rank=61, max_rank=150)
n, m = p.n, p.m
cl = np.asarray(p.close, np.float64)
mark = np.asarray(p.mark_close, np.float64)
fr = np.asarray(p.funding_rate, np.float64)
op = np.asarray(p.open, np.float64)
rows = []
def rec(rid, name, cell, res):
    rows.append({"round": rid, "idea": name, **cell, **res})

fr_t = np.where(np.abs(fr) > 1e-12, fr, 0.0)
prem = np.where((mark > 0) & (cl > 0), cl / mark - 1.0, np.nan)
cum72 = pd.DataFrame(fr_t).rolling(72, min_periods=12).sum().to_numpy()
cum168 = pd.DataFrame(fr_t).rolling(168, min_periods=28).sum().to_numpy()

def build(cum, theta, theta_lo, phase):
    L = np.zeros((n, m), bool); S = np.zeros((n, m), bool)
    for i in range(180, n):
        if i % 8 != phase: continue
        for c in u_mid:
            f = cum[i - 1, c]; pr = prem[i - 1, c]
            if not (np.isfinite(f) and np.isfinite(pr)): continue
            if f >= theta and pr > 0: S[i, c] = True
            elif f <= theta_lo and pr < 0: L[i, c] = True
    return L, S

# R51: theta grid + phase stagger
for theta in (0.0006, 0.0012, 0.0018):
    for phase in (1, 5):
        L, S = build(cum72, theta, -theta, phase)
        rec("R51", "R47 theta/phase grid", {"theta": theta, "phase": phase},
            E.evaluate(p, L, S, 96, E.TRAIN, u_mid))

# R52: tail universe 61-150
L, S = build(cum72, 0.0012, -0.0012, 1)
rec("R52", "R47 tail 61-150", {"hold": 96}, E.evaluate(p, L, S, 96, E.TRAIN, u_tail))

# R53: ablation funding only (no prem condition)
L = np.zeros((n, m), bool); S = np.zeros((n, m), bool)
for i in range(180, n):
    if i % 8 != 1: continue
    for c in u_mid:
        f = cum72[i - 1, c]
        if not np.isfinite(f): continue
        if f >= 0.0012: S[i, c] = True
        elif f <= -0.0012: L[i, c] = True
rec("R53", "ablation funding only", {"hold": 96}, E.evaluate(p, L, S, 96, E.TRAIN, u_mid))

# R54: ablation prem only (no funding condition)
z = pd.DataFrame(prem).rolling(168, min_periods=84).mean().shift(1).to_numpy()
sd = pd.DataFrame(prem).rolling(168, min_periods=84).std().shift(1).to_numpy()
pz = (pd.DataFrame(prem) - pd.DataFrame(prem).rolling(168, min_periods=84).mean().shift(1)) / \
     pd.DataFrame(prem).rolling(168, min_periods=84).std().shift(1)
pz = pz.to_numpy()
L = np.zeros((n, m), bool); S = np.zeros((n, m), bool)
for i in range(180, n):
    if i % 8 != 1: continue
    for c in u_mid:
        v = pz[i - 1, c]
        if not np.isfinite(v): continue
        if v > 2.0: S[i, c] = True
        elif v < -2.0: L[i, c] = True
rec("R54", "ablation prem only", {"hold": 96}, E.evaluate(p, L, S, 96, E.TRAIN, u_mid))

# R55: beta decomposition of R47 trades: residual = trade ret - beta*market ret over same window
mkt = pd.DataFrame(cl[:, u_mid]).mean(axis=1).pct_change().to_numpy()  # equal-weight index 1h ret
L, S = build(cum72, 0.0012, -0.0012, 1)
sigs = L | S
side_arr = np.where(L, 1, np.where(S, -1, 0))
hold = 96
resid, raw, mkts = [], [], []
for i in range(180, n - hold):
    if not sigs[i].any(): continue
    mr = np.nanprod(1 + mkt[i + 1:i + 1 + hold]) - 1
    for c in u_mid:
        if not sigs[i, c]: continue
        d = side_arr[i, c]
        ret = d * (cl[min(i + hold, n - 1), c] / op[i + 1, c] - 1.0) - 2 * 10 / 1e4
        resid.append(ret - mr)  # beta 1 hedge
        raw.append(ret); mkts.append(mr)
resid = np.array(resid); raw = np.array(raw); mkts = np.array(mkts)
g, b = resid[resid > 0].sum(), -resid[resid < 0].sum()
rec("R55", "R47 beta-1 hedge residual", {"hold": 96}, {"n": len(resid),
    "resid_mean_bp": round(float(resid.mean()) * 1e4, 2), "raw_mean_bp": round(float(raw.mean()) * 1e4, 2),
    "mkt_mean_bp": round(float(mkts.mean()) * 1e4, 2), "resid_pf": round(float(g / b), 3) if b > 0 else 99.0})

# R56: barriers sl/tp on R47
for sl, tp in ((0.05, 0.15), (0.08, 0.0), (0.0, 0.15)):
    L, S = build(cum72, 0.0012, -0.0012, 1)
    rec("R56", "R47 barriers", {"sl": sl, "tp": tp}, E.evaluate(p, L, S, 96, E.TRAIN, u_mid, sl=sl, tp=tp))

out = pd.DataFrame(rows)
out.to_csv("/root/freqtrade/user_data/minute_research/r5/batchG.csv", index=False)
print(out.to_string())
