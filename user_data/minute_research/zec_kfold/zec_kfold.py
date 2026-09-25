"""ZEC relative-threshold screen with time-series folds and CSCV (see PREREGISTRATION.md).

  python zec_kfold.py                      # screen on the six folds (TRAIN + VALID-C only)
  python zec_kfold.py --holdout F11:0.9 …  # read HOLDOUT once for the listed family:q combos
"""
from __future__ import annotations

import itertools
import sys

import numba as nb
import numpy as np
import pandas as pd

R = "/root/freqtrade/user_data/data/binance_public"
SYM = "ZECUSDT"
COST = 10.0
QS = (0.80, 0.90, 0.95, 0.98)
FOLDS = [("F1", "2025-01-01", "2025-03-01"), ("F2", "2025-03-01", "2025-05-01"), ("F3", "2025-05-01", "2025-07-01"),
         ("F4", "2025-07-01", "2025-09-01"), ("F5a", "2025-09-01", "2025-10-01"), ("F5b", "2025-12-01", "2026-01-01"),
         ("F6", "2026-01-01", "2026-03-01")]
FOLD_OF = {"F1": "F1", "F2": "F2", "F3": "F3", "F4": "F4", "F5a": "F5", "F5b": "F5", "F6": "F6"}
HOLD = ("2026-03-01", "2026-09-01")


# ----------------------------------------------------------------------------- data
def hourly(sym: str) -> pd.DataFrame:
    k = pd.read_parquet(f"{R}/klines_1m/{sym}.parquet").set_index("date")
    h = k.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
                              "quote_volume": "sum", "count": "sum", "taker_buy_quote_volume": "sum"}).dropna()
    return h[h.index < "2026-09-01"]


def build() -> pd.DataFrame:
    h = hourly(SYM)
    btc = hourly("BTCUSDT").close.reindex(h.index).ffill()
    h["btc"] = btc
    f = pd.read_parquet(f"{R}/funding/{SYM}.parquet"); f["date"] = f.date.dt.round("h")
    f = f.drop_duplicates("date", keep="last").set_index("date").funding_rate
    h["fr_settle"] = f.reindex(h.index).fillna(0.0)                       # paid at this bar's open
    h["fr_last"] = f.reindex(h.index).ffill().shift(0)                    # known at bar close (settled at its open)
    h["fr_mean3"] = f.rolling(3).mean().reindex(h.index).ffill()
    m = pd.read_parquet(f"{R}/markprice_1m/{SYM}.parquet").set_index("date").mark_close.resample("1h").last()
    h["mark"] = m.reindex(h.index)
    o = pd.read_parquet(f"{R}/metrics/{SYM}.parquet").set_index("date")
    o.index = o.index.floor("5min")
    oh = o.resample("1h", label="right", closed="right").last()          # value known at the hour's close
    oh.index = oh.index - pd.Timedelta(hours=1)                           # align to the bar that closes then
    for c in ("sum_open_interest_value", "sum_toptrader_long_short_ratio", "count_long_short_ratio"):
        h[c] = oh[c].reindex(h.index)
    return h


def rsi(c, n=14):
    d = c.diff(); g = d.clip(lower=0).ewm(com=n - 1, adjust=False, min_periods=n).mean()
    l = (-d.clip(upper=0)).ewm(com=n - 1, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def atr(h, n=14):
    pc = h.close.shift(1)
    tr = pd.concat([h.high - h.low, (h.high - pc).abs(), (h.low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False, min_periods=n).mean()


def z(x, n=168):
    return (x - x.rolling(n, min_periods=48).mean()) / x.rolling(n, min_periods=48).std()


def families(h: pd.DataFrame) -> dict[str, tuple[pd.Series, int, bool]]:
    c, o = h.close, h.open
    r1, r4, r12, r24 = c.pct_change(1), c.pct_change(4), c.pct_change(12), c.pct_change(24)
    bar = np.sign(c - o)
    a = atr(h)
    RS = rsi(c)
    sma20, sd20 = c.rolling(20).mean(), c.rolling(20).std(ddof=0)
    pctb = (c - (sma20 - 2 * sd20)) / (4 * sd20)
    kmid = c.ewm(span=20, adjust=False).mean()
    sq = (sma20 + 2 * sd20 < kmid + 1.5 * a) & (sma20 - 2 * sd20 > kmid - 1.5 * a)
    hh, ll = h.high.rolling(24).max().shift(1), h.low.rolling(24).min().shift(1)
    don = np.where(c > hh, (c - hh) / a, np.where(c < ll, (c - ll) / a, 0.0))
    vr = h.volume / h.volume.rolling(24, min_periods=12).mean()
    tk = h.taker_buy_quote_volume / h.quote_volume - 0.5
    delta = (2 * h.taker_buy_quote_volume - h.quote_volume).rolling(12).sum()
    ema200 = c.ewm(span=200, adjust=False, min_periods=100).mean()
    btc24 = h.btc.pct_change(24)
    oi = h.sum_open_interest_value
    S = {
        "F01_mom24": (r24, 12, False), "F02_rev24": (-r24, 12, False), "F03_mom4": (r4, 6, False),
        "F04_rev1": (-r1, 4, False), "F05_rsi_rev": (-(RS - 50), 12, False), "F06_bb_rev": (-(pctb - 0.5), 12, False),
        "F07_donchian": (pd.Series(don, index=h.index), 12, False),
        "F08_squeeze_break": (((c - kmid) / a).where(sq.shift(1).fillna(False).astype(bool), 0.0), 18, False),
        "F09_vol_follow": (bar * vr, 12, False), "F10_vol_fade": (-bar * vr, 12, False),
        "F11_taker_follow": (tk, 12, False), "F12_taker_fade": (-tk, 12, False),
        "F13_cvd_div": (z(delta) - z(r12), 12, False),
        "F14_flushout": ((-r12).where((c > o) & (vr > 1.5) & (r12 < 0), 0.0), 12, True),
        "F15_fund_fade": (-h.fr_last, 8, False), "F16_fund_follow": (h.fr_last, 8, False),
        "F17_fund3_exh": (-h.fr_mean3, 8, False),
        "F18_oi_follow": (np.sign(r4) * oi.pct_change(4), 12, False),
        "F19_oi_flush": ((-oi.pct_change(12)).where(r12 < 0, 0.0), 12, True),
        "F20_btc_lead": (h.btc.pct_change(1), 4, False),
        "F21_rs_follow": (r24 - btc24, 12, False), "F22_rs_fade": (-(r24 - btc24), 12, False),
        "F23_atr_break": (bar * a / a.rolling(48, min_periods=24).mean(), 12, False),
        "F24_trend_pullback": ((50 - RS).where((c > ema200) & (RS < 50), 0.0) - (RS - 50).where((c < ema200) & (RS > 50), 0.0), 12, False),
        "F25_basis_fade": (-(c / h.mark - 1), 8, False),
        "F26_count_follow": (bar * h["count"] / h["count"].rolling(24, min_periods=12).mean(), 12, False),
        "F27_toptrader_follow": (h.sum_toptrader_long_short_ratio.diff(12), 12, False),
        "F28_crowd_fade": (-(h.count_long_short_ratio - h.count_long_short_ratio.rolling(720, min_periods=168).median()), 12, False),
    }
    return S


def signal(s: pd.Series, q: float, long_only: bool) -> np.ndarray:
    nz = s.where(s != 0)
    up = nz.rolling(2160, min_periods=240).quantile(q).shift(1)
    lo = nz.rolling(2160, min_periods=240).quantile(1 - q).shift(1)
    sig = np.where(s >= np.maximum(up, 0), 1, 0)
    if not long_only:
        sig = np.where(s <= np.minimum(lo, 0), -1, sig)
    sig = np.where(np.isnan(up) | s.isna(), 0, sig)
    return sig.astype(np.int8)


# ----------------------------------------------------------------------------- numba simulation
@nb.njit(parallel=True, cache=True)
def simulate(o, fr, sigs, holds, cost_bps):
    """sigs (K, T) in {-1,0,1}; entry next open, exit at open hold bars later, one position per combo.
    Returns per-combo arrays padded to T: entry idx, side, return (price + funding - cost)."""
    K, T = sigs.shape
    ent = np.full((K, T), -1, np.int64); sd = np.zeros((K, T), np.int8); ret = np.zeros((K, T))
    cnt = np.zeros(K, np.int64)
    for k in nb.prange(K):
        busy = -1; n = 0; hold = holds[k]
        for i in range(T - 1):
            s = sigs[k, i]
            if s == 0 or i < busy:
                continue
            e = i + 1; x = e + hold
            if x >= T:
                break
            f = 0.0
            for j in range(e + 1, x + 1):          # settlements at bar opens strictly after entry, up to exit
                f += fr[j]
            ret[k, n] = s * (o[x] / o[e] - 1.0) - s * f - 2.0 * cost_bps / 1e4
            ent[k, n] = e; sd[k, n] = s; n += 1
            busy = x
        cnt[k] = n
    return ent, sd, ret, cnt


# ----------------------------------------------------------------------------- evaluation
def fold_label(t: pd.DatetimeIndex) -> np.ndarray:
    lab = np.full(len(t), "", dtype=object)
    for name, a, b in FOLDS:
        lab[(t >= a) & (t < b)] = FOLD_OF[name]
    return lab


def stats(r: np.ndarray, days: pd.Series) -> dict:
    if len(r) < 3:
        return dict(n=len(r))
    g, l = r[r > 0].sum(), -r[r < 0].sum()
    dm = pd.Series(r).groupby(days.to_numpy()).sum()
    return dict(n=len(r), mean_bp=r.mean() * 1e4, pf=g / l if l > 0 else np.inf,
                t=dm.mean() / dm.std() * np.sqrt(len(dm)) if dm.std() > 0 else np.nan)


def run(combos, h, S, cost=COST):
    o = h.open.to_numpy(np.float64); fr = h.fr_settle.to_numpy(np.float64)
    sigs = np.stack([signal(S[f][0], q, S[f][2]) for f, q in combos])
    holds = np.array([S[f][1] for f, q in combos], np.int64)
    return simulate(o, fr, sigs, holds, cost)


def drift(h: pd.DataFrame, hold: int) -> pd.Series:
    """unconditional forward return (next-open entry, exit hold bars later), per bar."""
    o = h.open
    return (o.shift(-(hold + 1)) / o.shift(-1) - 1)


def screen():
    h = build(); S = families(h)
    combos = list(itertools.product(S.keys(), QS))
    idx = h.index
    fl = fold_label(idx)
    res = {}
    for cost in (COST, 15.0):
        ent, sd, ret, cnt = run(combos, h, S, cost)
        res[cost] = (ent, sd, ret, cnt)
    ent, sd, ret, cnt = res[COST]; _, _, ret15, _ = res[15.0]
    active_days = sum((pd.Timestamp(b) - pd.Timestamp(a)).days for _, a, b in FOLDS)
    rows, daily = [], {}
    for k, (f, q) in enumerate(combos):
        e = ent[k, :cnt[k]]; s = sd[k, :cnt[k]]; r = ret[k, :cnt[k]]; r15 = ret15[k, :cnt[k]]
        lab = fl[e]; keep = lab != ""
        e, s, r, r15, lab = e[keep], s[keep], r[keep], r15[keep], lab[keep]
        t = idx[e]; day = pd.Series(t.floor("D"))
        st = stats(r, day)
        dr = drift(h, S[f][1]).to_numpy()
        adj = r - s * np.array([np.nanmean(dr[fl == L]) for L in lab]) if len(r) else r
        fold_pos = sum(r[lab == L].sum() > 0 for L in ("F1", "F2", "F3", "F4", "F5", "F6"))
        rows.append(dict(family=f, q=q, hold=S[f][1], **st, per_day=len(r) / active_days, folds_pos=fold_pos,
                         mean15_bp=r15.mean() * 1e4 if len(r15) else np.nan, adj_bp=adj.mean() * 1e4 if len(adj) else np.nan,
                         long_n=int((s > 0).sum()), short_n=int((s < 0).sum())))
        daily[(f, q)] = (pd.Series(r).groupby(day.to_numpy()).sum(), pd.Series(lab).groupby(day.to_numpy()).first())
    df = pd.DataFrame(rows)
    df["pass_rel"] = (df.mean_bp > 0) & (df.pf >= 1.2) & (df.t >= 2.0) & (df.folds_pos >= 4) & (df.per_day >= 0.4) & (df.mean15_bp > 0) & (df.adj_bp > 0)
    df["pass_std"] = df.pass_rel & (df.per_day >= 1.0)
    df.to_csv("screen_folds.csv", index=False)
    return df, daily


def cscv_pbo(daily: dict) -> float:
    folds = ["F1", "F2", "F3", "F4", "F5", "F6"]
    keys = list(daily)
    per = {}
    for kk in keys:
        d, lab = daily[kk]
        per[kk] = {L: d[lab == L] for L in folds}
    logits = []
    for IS in itertools.combinations(folds, 3):
        OOS = [x for x in folds if x not in IS]
        sh = lambda kk, part: (lambda v: v.mean() / v.std() * np.sqrt(365) if len(v) > 5 and v.std() > 0 else -np.inf)(pd.concat([per[kk][L] for L in part]))
        is_s = np.array([sh(kk, IS) for kk in keys]); oos_s = np.array([sh(kk, OOS) for kk in keys])
        best = int(np.argmax(is_s))
        valid = np.isfinite(oos_s)
        rank = (oos_s[valid] < oos_s[best]).mean() if np.isfinite(oos_s[best]) else 0.0
        rank = min(max(rank, 1e-3), 1 - 1e-3)
        logits.append(np.log(rank / (1 - rank)))
    return float((np.array(logits) <= 0).mean())


if __name__ == "__main__":
    if "--holdout" in sys.argv:
        pick = [(a.split(":")[0], float(a.split(":")[1])) for a in sys.argv[sys.argv.index("--holdout") + 1:]]
        h = build(); S = families(h)
        for cost in (COST, 15.0):
            ent, sd, ret, cnt = run(pick, h, S, cost)
            for k, (f, q) in enumerate(pick):
                e = ent[k, :cnt[k]]; t = h.index[e]; m = (t >= HOLD[0]) & (t < HOLD[1])
                r = ret[k, :cnt[k]][m]; s = sd[k, :cnt[k]][m]; d = pd.Series(t[m].floor("D"))
                dr = drift(h, S[f][1]).to_numpy(); hm = (h.index >= HOLD[0]) & (h.index < HOLD[1])
                adj = r - s * np.nanmean(dr[hm])
                print(f"HOLDOUT cost {cost:4.1f} {f} q{q}: {stats(r, d)} per_day {len(r)/184:.2f} long {int((s>0).sum())} short {int((s<0).sum())} drift-adj {adj.mean()*1e4:.1f} bp")
        sys.exit()
    df, daily = screen()
    pbo = cscv_pbo(daily)
    cols = ["family", "q", "hold", "n", "per_day", "mean_bp", "pf", "t", "folds_pos", "mean15_bp", "adj_bp", "long_n", "short_n"]
    print(f"combos {len(df)} | PBO (CSCV, 20 splits) = {pbo:.2f} | pass relaxed {int(df.pass_rel.sum())} | pass standard {int(df.pass_std.sum())}")
    print(df[df.pass_rel].sort_values("t", ascending=False)[cols].round(2).to_string(index=False))
    print("\nTop 15 by pooled t (any gate):")
    print(df.sort_values("t", ascending=False)[cols].head(15).round(2).to_string(index=False))
