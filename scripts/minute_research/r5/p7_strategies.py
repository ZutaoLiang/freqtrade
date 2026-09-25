"""The 7 strategies of PROFITABLE_STRATEGIES.md, implemented independently from the document's rules.
Usage: python p7_strategies.py [--holdout NAME,...]   (HOLDOUT is read only for the names given)"""
import sys
sys.path.insert(0, "scripts/minute_research/r5")
import numpy as np, pandas as pd
import p7_panel as P
import recheck_schemeC as R

D = P.build()
o, h, l, c, v, tbv, fr = (D[k].astype(np.float64) for k in ("open", "high", "low", "close", "volume", "tbv", "fr"))
syms = list(D["syms"]); dates = pd.to_datetime(D["dates"], unit="us", utc=True)
T, N = c.shape
cost = np.array([7.5e-4 if s in ("BTC", "ETH") else 10e-4 for s in syms])
import json as _json, os as _os
SUBSET = set(_json.load(open(_os.environ["P7_SUBSET"]))) if _os.environ.get("P7_SUBSET") else None
TAG = _os.environ.get("P7_TAG", "")
ONLY = set(_os.environ["P7_ONLY"].split(",")) if _os.environ.get("P7_ONLY") else None


def ema(a, n):
    return pd.DataFrame(a).ewm(span=n, adjust=False, min_periods=n).mean().to_numpy()


def sma(a, n):
    return pd.DataFrame(a).rolling(n, min_periods=n).mean().to_numpy()


def rsi(a, n=14):
    df = pd.DataFrame(a).diff()
    up, dn = df.clip(lower=0), -df.clip(upper=0)
    ru = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean(); rd = dn.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return (100 - 100 / (1 + ru / rd)).to_numpy()


def atr(hh, ll, cc, n):
    pc = np.vstack([np.full((1, cc.shape[1]), np.nan), cc[:-1]])
    tr = np.nanmax(np.stack([hh - ll, np.abs(hh - pc), np.abs(ll - pc)]), axis=0)
    return pd.DataFrame(tr).ewm(span=n, adjust=False, min_periods=n).mean().to_numpy()


def squeeze(hh, ll, cc):
    mid = sma(cc, 20); sd = pd.DataFrame(cc).rolling(20, min_periods=20).std(ddof=0).to_numpy()
    bu, bl = mid + 2 * sd, mid - 2 * sd
    em = ema(cc, 20); a = atr(hh, ll, cc, 20)
    ku, kl = em + 1.5 * a, em - 1.5 * a
    return (bu < ku) & (bl > kl), ku, kl


# 4h aggregates from the hourly panel (UTC buckets), mapped back with the previous COMPLETED bucket
h4, b4 = P.resample_4h(h, "max"); l4, _ = P.resample_4h(l, "min"); c4, _ = P.resample_4h(c, "last")
ema50_4h = P.to_ltf_prev(ema(c4, 50), b4); c4_prev = P.to_ltf_prev(c4, b4)
trend_up_4h = c4_prev > ema50_4h
ratio = np.where(v > 0, tbv / v, np.nan)
rsi14 = rsi(c)
last_fr = pd.DataFrame(fr).ffill().to_numpy()             # latest settled rate known at each hour close


def settle_chain():
    """bool (T,N): at settlement hour t (stamp t), this and the previous two settled rates >= 0.03%."""
    out = np.zeros((T, N), bool)
    for j in range(N):
        idx = np.where(~np.isnan(fr[:, j]))[0]
        if len(idx) < 3:
            continue
        f = fr[idx, j]
        ok = (f[2:] >= 3e-4) & (f[1:-1] >= 3e-4) & (f[:-2] >= 3e-4)
        out[idx[2:][ok], j] = True
    return out


def S_R291():
    sq4, _, _ = squeeze(h4, l4, c4); sq4_l = P.to_ltf_prev(sq4.astype(float), b4) == 1.0
    sq1, ku, kl = squeeze(h, l, c)
    dual_prev = np.vstack([np.zeros((1, N), bool), (sq4_l & sq1)[:-1]])
    lo = dual_prev & (c > ku) & (ratio > 0.65); sh = dual_prev & (c < kl) & (ratio < 0.35)
    return lo | sh, np.where(lo, 1, -1), 24


def S_R24():
    ch = settle_chain()        # stamp hour t = settlement; signal on that bar's close -> entry at t+1h open (conservative vs doc's T+5m)
    return ch, np.full((T, N), -1), 7


def S_R102():
    body = np.abs(c - o); lower = np.minimum(o, c) - l
    s = (v > 3 * sma(v, 24)) & (lower > 2 * body) & (rsi14 < 30)
    return s, np.ones((T, N), int), 18


def S_R362():
    r3 = (ratio > 0.66) & (np.roll(ratio, 1, 0) > 0.66) & (np.roll(ratio, 2, 0) > 0.66)
    r3[:2] = False
    return r3 & trend_up_4h, np.ones((T, N), int), 24


def S_R105():
    return (rsi14 < 30) & (ratio > 0.60) & trend_up_4h, np.ones((T, N), int), 12


def S_R145():
    return (last_fr < -5e-4) & (rsi14 < 25) & (c > o), np.ones((T, N), int), 18


def S_R234():
    """Liquid-30 subset only (metrics available); OI = sum_open_interest_value, LS = count_toptrader_long_short_ratio (1h last)."""
    s = np.zeros((T, N), bool)
    for j, b in enumerate(syms):
        p = f"user_data/minute_research/r3/metrics/{b}USDT_5m.parquet"
        try:
            m = pd.read_parquet(p, columns=["date", "sum_open_interest_value", "count_toptrader_long_short_ratio"]).set_index("date")
        except Exception:
            continue
        g = m.groupby(m.index.floor("h")).last().reindex(dates)
        oi, ls = g.sum_open_interest_value, g.count_toptrader_long_short_ratio
        cond = (oi / oi.rolling(12).mean() - 1 < -0.15) & (ls - ls.rolling(12).mean() > 0.15)
        s[:, j] = cond.to_numpy() & trend_up_4h[:, j]
    return s, np.ones((T, N), int), 24


STRATS = {"R291 dual squeeze + taker": S_R291, "R24 funding exhaustion short": S_R24, "R102 volume climax pin": S_R102,
          "R362 taker persistence": S_R362, "R105 oversold taker reversal": S_R105, "R145 negative funding rebound": S_R145,
          "R234 OI flush + top-trader (liq30)": S_R234}
DOC = {"R291 dual squeeze + taker": "TRAIN n58 +237bp PF4.69 t2.50 | VALID n72 +103bp PF1.95 t1.44",
       "R24 funding exhaustion short": "TRAIN n338 +107bp PF1.73 t2.86 | VALID n99-132 +177~203bp PF2.16-2.42 t2.30-2.58",
       "R102 volume climax pin": "TRAIN n825 +192bp PF2.07 t1.05 | VALID n422 +527bp PF10.5 t1.32",
       "R362 taker persistence": "TRAIN n65 +146bp PF4.34 t2.08 | VALID n54 +110bp PF1.98 t1.34",
       "R105 oversold taker reversal": "VALID n90 +107bp PF2.68 t2.28",
       "R145 negative funding rebound": "TRAIN n62 +209bp PF1.97 t1.36 | VALID n39 +175bp PF2.22 t1.39",
       "R234 OI flush + top-trader (liq30)": "TRAIN n120 +1620bp PF9.92 t1.12 | VALID n32 +230bp PF7.09 t2.58"}

if __name__ == "__main__":
    read_ho = set(sys.argv[sys.argv.index("--holdout") + 1].split(",")) if "--holdout" in sys.argv else set()
    rows = []
    for name, fn in STRATS.items():
        if ONLY and name.split()[0] not in ONLY:
            continue
        sig, side, hold = fn()
        sig = np.nan_to_num(sig).astype(np.bool_); side = np.asarray(side, np.int64)
        if SUBSET is not None:
            sig[:, [k for k, s_ in enumerate(syms) if s_ not in SUBSET]] = False
        for fund in (True, False):
            e, j, sd, r, f = P.sim(o, sig, side, hold, cost, fr, fund)
            t = pd.DataFrame({"t": dates[e], "ret": r, "side": sd, "pair": [syms[k] for k in j], "cost": cost[j], "fund": f})
            t["seg"] = R.segC(t.t)
            segs = ("TRAIN", "VALID") + (("HOLDOUT",) if name.split()[0] in read_ho else ())
            out = {s: R.stats(t[t.seg == s]) for s in segs}
            tag = "with funding" if fund else "no funding"
            print(f"\n### {name} [{tag}]  doc: {DOC[name]}")
            for s in segs:
                st = out[s]
                print(f"  {s:7s} n{st.get('n')} days{st.get('days')} pd{st.get('per_day')} {st.get('net_bp')}bp PF{st.get('pf')} t{st.get('t')} mon+{st.get('mon+')} "
                      f"top_pair{st.get('top_pair')} top_day{st.get('top_day')} L{st.get('L_bp')} S{st.get('S_bp')}" + (f" fund/trade {t[t.seg==s].fund.mean()*1e4:.1f}bp" if len(t[t.seg==s]) else ""), flush=True)
            if fund:
                rows.append({"strategy": name, **{f"{s}_{k}": out[s].get(k) for s in segs for k in ("n", "net_bp", "pf", "t", "per_day", "mon+", "top_pair", "top_day")},
                             "verdict_VALID": R.verdict(out["TRAIN"], out["VALID"])})
                t.to_parquet(f"user_data/minute_research/r5/p7{TAG}_{name.split()[0]}_trades.parquet")
    pd.DataFrame(rows).to_csv(f"user_data/minute_research/r5/p7{TAG}_recheck.csv", index=False)
    print("\n" + pd.DataFrame(rows)[["strategy", "TRAIN_n", "TRAIN_net_bp", "TRAIN_t", "VALID_n", "VALID_net_bp", "VALID_pf", "VALID_t", "verdict_VALID"]].to_string(index=False))
