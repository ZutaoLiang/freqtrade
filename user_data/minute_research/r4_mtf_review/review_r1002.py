"""Review of R1002: reproduce with the author's engine, then fix the 4h-return alignment (np.diff shifts by one bar)."""
import sys
import numpy as np
sys.path.insert(0, "/root/freqtrade/user_data/minute_research/r4_mtf")
from mtf_engine import MTFPanels, eval_signals
import indicators_mtf as ind

p1h, p4h, p1d = MTFPanels("1h"), MTFPanels("4h"), MTFPanels("1d")
b = p1h.symbols.index("BTCUSDT")
c1h, h1h, l1h, v1h = (np.array(x) for x in (p1h.close, p1h.high, p1h.low, p1h.volume))
tr = np.where(v1h > 0, np.array(p1h.taker_buy) / (v1h + 1e-12), 0.5)
c4h, c1d = np.array(p4h.close), np.array(p1d.close)
bc1d, bc4h = c1d[:, b:b + 1], c4h[:, b:b + 1]
bull = p1h.map_htf(bc1d > ind.nb_sma(bc1d, 50), "1d") & p1h.map_htf(bc4h > ind.nb_ema(bc4h, 50), "4h")
bear = (~p1h.map_htf(bc1d > ind.nb_sma(bc1d, 50), "1d")) & (~p1h.map_htf(bc4h > ind.nb_ema(bc4h, 50), "4h"))
_, bbu, bbl = ind.nb_bollinger(c1h, 20, 2.0)
_, ku, kl = ind.nb_keltner(h1h, l1h, c1h, 20, 1.5)
sq = (bbu < ku) & (bbl > kl)
lr = np.log(np.maximum(c4h, 1e-8))

def run(label, ret4h):
    rs = ret4h - ret4h[:, b:b + 1]
    lead, lag = p1h.map_htf(rs > 0.02, "4h"), p1h.map_htf(rs < -0.02, "4h")
    sl = (c1h > ku) & np.roll(sq, 1, axis=0) & bull & (tr > 0.60) & lead
    ss = (c1h < kl) & np.roll(sq, 1, axis=0) & bear & (tr < 0.40) & lag
    for seg in ("TRAIN", "VALID", "HOLDOUT"):
        r = eval_signals(p1h, sl, ss, hold_bars=12, sl_pct=0.07, tp_pct=0.14, seg=seg)
        print(f"{label:28s} {seg:8s} n={r['n_trades']:4d} mean={r['mean_bp']:7.1f}bp pf={r['pf']:5.2f} t={r['t_stat']:5.2f}")

run("author (np.diff, leaks)", np.diff(lr, axis=0))
fixed = np.full_like(lr, np.nan); fixed[1:] = np.diff(lr, axis=0)   # ret[j] = return of 4h bar j
run("fixed alignment", fixed)
