"""Signal search on a fixed coin screen.

The screen is settled: rank the top 150 perpetuals by 30d dollar volume, keep
the 30 with the highest 3d realised volatility, refresh every 3 days. Only the
trend rule is under test here.

Unlike the earlier per-window tests this runs one continuous portfolio: a
position survives a rebalance if its coin is still in the basket, is closed when
the coin drops out, and pays a fee only when it actually changes. Every rule is
scored on the same bars, and the sample is split in half so a rule that only
works in one regime is visible as such.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import os

import numpy as np
import pandas as pd
from pathlib import Path

NON_COIN = Path("/root/freqtrade/user_data/research/ma_harness/non_coin_symbols.json")

sys.path.insert(0, "/root/freqtrade/user_data/factor_research")
from research.panel import Panel  # noqa: E402

BPD = 24
PANEL = {}


class _RawPanel:
    """Minimal stand-in for research.panel.Panel over a plain npy directory.

    The historical backfill panel is named 1h_hist, which the Panel class
    cannot parse as a timeframe, and its funding column carries settlement
    values rather than the hourly-repeated rate.
    """

    def __init__(self, name):
        import json as _json
        d = Path("/root/freqtrade/user_data/data/binance_public/panels") / name
        self.dir = d
        self.meta = _json.loads((d / "meta.json").read_text())
        self.symbols = self.meta["symbols"]
        self.index = pd.date_range(self.meta["start"], self.meta["end"],
                                   freq="1h", tz="UTC")
        self._cache = {}

    def __getitem__(self, field):
        if field not in self._cache:
            self._cache[field] = np.load(self.dir / f"{field}.npy", mmap_mode="r")
        return self._cache[field]

    @property
    def shape(self):
        return len(self.index), len(self.symbols)


def load():
    if "close" not in PANEL:
        name = os.environ.get("VSD_PANEL", "1h")
        if name == "1h":
            p = Panel("1h", mmap=True)
            PANEL["fund_div"] = 8.0      # hourly-repeated 8h rate
        else:
            p = _RawPanel(name)
            PANEL["fund_div"] = 1.0      # settlement-time values only
        PANEL["p"] = p
        PANEL["close"] = np.asarray(p["close"], dtype=np.float32)
        PANEL["high"] = np.asarray(p["high"], dtype=np.float32)
        PANEL["low"] = np.asarray(p["low"], dtype=np.float32)
        PANEL["fund"] = np.nan_to_num(np.asarray(p["funding_rate"], dtype=np.float32))
        PANEL["qv"] = np.asarray(p["quote_volume"], dtype=np.float32)
        PANEL["mask"] = np.asarray(p["universe_mask"])
    return PANEL


def selection_mask(sel_vol_days=3, step_days=3, top_n=150, top_vol=30,
                   start_days=90, basket="movers"):
    """Which coins the basket holds at each bar."""
    d = load()
    close, qv, mask = d["close"], d["qv"], d["mask"]
    T, N = close.shape
    # The archive carries USDC-quoted twins of the same coin and a few
    # non-crypto index products with CJK names; both would be selected
    # alongside the USDT contract and double the exposure to one coin.
    # Binance also lists tokenised equity and index perpetuals (CRCL, MU,
    # SNDK, EWY ...) under a USDT quote; exchangeInfo marks them
    # underlyingType != COIN and freqtrade cannot even backtest them.
    non_coin = set(json.load(open(NON_COIN))) if NON_COIN.exists() else set()
    eligible = np.array([s.isascii() and s.endswith("USDT")
                         and s not in non_coin for s in d["p"].symbols])
    out = np.zeros((T, N), dtype=bool)
    step = step_days * BPD
    for t in range(start_days * BPD, T, step):
        dollar = np.nansum(qv[t - 30 * BPD:t], axis=0)
        ok = np.isfinite(close[t - 90 * BPD:t]).all(axis=0) & mask[t - 1]
        dollar = np.where(ok & eligible, dollar, -1.0)
        liq = np.argsort(-dollar)[:top_n]
        liq = liq[dollar[liq] > 0]
        if liq.size < top_vol:
            continue
        c = np.asarray(close[t - sel_vol_days * BPD:t, liq], dtype=np.float64)
        v = np.nanstd(np.diff(np.log(c), axis=0), axis=0)
        if basket == "movers":
            pick = liq[np.argsort(-v)[:top_vol]]
        elif basket == "calm":                     # the other end of the screen
            pick = liq[np.argsort(v)[:top_vol]]
        elif basket == "mid":
            mid = liq.size // 2
            pick = liq[np.argsort(-v)[mid - top_vol // 2:mid + top_vol // 2]]
        elif basket == "movers_fund":
            # the funding control suggested the gate acts as a coin filter more
            # than a timing one, so try it as part of the screen instead
            wide = liq[np.argsort(-v)[:top_vol * 2]]
            af = np.abs(np.nanmean(np.asarray(
                d["fund"][t - 30 * BPD:t, wide], dtype=np.float64), axis=0))
            pick = wide[np.argsort(-af)[:top_vol]]
        elif basket == "movers_fund_rank":
            wide = liq[np.argsort(-v)[:top_vol * 2]]
            af = np.abs(np.nanmean(np.asarray(
                d["fund"][t - 30 * BPD:t, wide], dtype=np.float64), axis=0))
            vr = np.argsort(np.argsort(-v[np.argsort(-v)[:top_vol * 2]]))
            fr = np.argsort(np.argsort(-af))
            pick = wide[np.argsort(vr + fr)[:top_vol]]
        elif basket == "all":
            pick = liq
        else:
            raise ValueError(basket)
        out[t:t + step, pick] = True
    return out


def roll_max(a, w):
    return pd.DataFrame(a).rolling(w).max().shift(1).to_numpy()


def roll_min(a, w):
    return pd.DataFrame(a).rolling(w).min().shift(1).to_numpy()


def sma(a, w):
    return pd.DataFrame(a).rolling(w).mean().to_numpy()


def atr(high, low, close, w):
    prev = np.vstack([np.full((1, close.shape[1]), np.nan), close[:-1]])
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))
    return pd.DataFrame(tr).ewm(alpha=1.0 / w, adjust=False).mean().to_numpy()


def donchian_state(close, entry, exit_, high, low, atr_mult=None, atr_w=24,
                   max_hold=None):
    """Breakout entries with a channel exit, optionally an ATR trail as well."""
    hh, ll = roll_max(high, entry), roll_min(low, entry)
    xl, xh = roll_min(low, exit_), roll_max(high, exit_)
    a = atr(high, low, close, atr_w) if atr_mult else None
    T, N = close.shape
    pos = np.zeros((T, N), dtype=np.float32)
    cur = np.zeros(N, dtype=np.float32)
    stop = np.full(N, np.nan)
    held = np.zeros(N, dtype=np.int32)
    for i in range(T):
        c = close[i]
        long_, short_ = cur > 0, cur < 0
        if atr_mult:
            lvl_l, lvl_s = c - atr_mult * a[i], c + atr_mult * a[i]
            stop = np.where(long_, np.fmax(stop, lvl_l),
                            np.where(short_, np.fmin(stop, lvl_s), stop))
            cur = np.where(long_ & (c < stop), 0.0, cur)
            cur = np.where(short_ & (c > stop), 0.0, cur)
        cur = np.where((cur > 0) & (c < xl[i]), 0.0, cur)
        cur = np.where((cur < 0) & (c > xh[i]), 0.0, cur)
        if max_hold is not None:
            held = np.where(cur != 0, held + 1, 0)
            cur = np.where(held >= max_hold, 0.0, cur)
        fresh_l = (cur == 0) & (c > hh[i])
        fresh_s = (cur == 0) & (c < ll[i])
        cur = np.where(fresh_l, 1.0, np.where(fresh_s, -1.0, cur))
        if atr_mult:
            stop = np.where(fresh_l, c - atr_mult * a[i],
                            np.where(fresh_s, c + atr_mult * a[i], stop))
        held = np.where(fresh_l | fresh_s, 0, held)
        pos[i] = cur
    return pos


def cross_state(close, fast, slow):
    d = sma(close, fast) - sma(close, slow)
    s = np.sign(d).astype(np.float32)
    s[~np.isfinite(d)] = 0.0
    return s


def build_state(cfg):
    if cfg["kind"] == "ensemble":
        # averaging several breakout speeds gives a partial position rather
        # than one all-or-nothing bet, which also smooths turnover
        parts = [build_state(c) for c in cfg["members"]]
        st = np.mean(parts, axis=0).astype(np.float32)
        if cfg.get("vol_target"):
            st = _vol_scale(st, cfg)
        return st
    d = load()
    close, high, low = d["close"], d["high"], d["low"]
    kind = cfg["kind"]
    if kind == "donchian":
        st = donchian_state(close, cfg["entry"], cfg["exit"], high, low,
                            cfg.get("atr_mult"), cfg.get("atr_w", 24),
                            cfg.get("max_hold"))
    elif kind == "cross":
        st = cross_state(close, cfg["fast"], cfg["slow"])
    else:
        raise ValueError(kind)
    if cfg.get("long_only"):
        st = np.maximum(st, 0.0)
    if cfg.get("short_only"):
        st = np.minimum(st, 0.0)
    if cfg.get("trend_filter"):                       # coin's own slow trend
        base = sma(close, cfg["trend_filter"])
        up = close > base
        st = np.where((st > 0) & ~up, 0.0, st)
        st = np.where((st < 0) & up, 0.0, st)
    if cfg.get("btc_filter"):                         # market regime gate
        p = d["p"]
        b = close[:, p.symbols.index("BTCUSDT")]
        bs = pd.Series(b).rolling(cfg["btc_filter"]).mean().to_numpy()
        up = (b > bs)[:, None]
        st = np.where((st > 0) & ~up, 0.0, st)
        st = np.where((st < 0) & up, 0.0, st)
    if cfg.get("fund_abs_pctile") is not None:
        # attribution found the middle third of |funding| losing in both halves
        # while both extremes paid: crowded positioning, either direction
        # shifted by one bar: the funding used to gate a bar must be known
        # before that bar opens, not settled during it
        f = pd.DataFrame(d["fund"]).rolling(3 * BPD).mean().shift(1).to_numpy()
        af = np.abs(f)
        thr = np.nanquantile(np.where(np.isfinite(af), af, np.nan),
                             cfg["fund_abs_pctile"], axis=1, keepdims=True)
        st = np.where(af >= thr, st, 0.0)
    if cfg.get("fund_side") == "contrarian":
        # long only when funding is negative, short only when it is positive
        f = pd.DataFrame(d["fund"]).rolling(3 * BPD).mean().shift(1).to_numpy()
        st = np.where((st > 0) & (f > 0), 0.0, st)
        st = np.where((st < 0) & (f < 0), 0.0, st)
    if cfg.get("vol_target"):                         # size by inverse vol
        st = _vol_scale(st, cfg)
    return np.nan_to_num(st).astype(np.float32)


def _vol_scale(st, cfg):
    close = load()["close"]
    r = np.diff(np.log(np.maximum(close, 1e-12)), axis=0)
    r = np.vstack([np.full((1, close.shape[1]), np.nan), r])
    v = pd.DataFrame(r).rolling(cfg.get("vol_win", 7 * BPD)).std().to_numpy()
    v = v * np.sqrt(24 * 365)
    scale = np.clip(cfg["vol_target"] / np.where(v > 0, v, np.nan), 0.25, 2.0)
    return (st * np.nan_to_num(scale, nan=1.0)).astype(np.float32)


def run(cfg, sel, fee=0.0005):
    """One continuous equal-weight portfolio over the selected basket."""
    d = load()
    close, fund = d["close"], d["fund"]
    state = build_state(cfg) * sel                    # flat when out of basket
    pos = np.vstack([np.zeros((1, close.shape[1]), np.float32), state[:-1]])
    ret = np.zeros_like(close)
    ret[1:] = close[1:] / close[:-1] - 1.0
    ret[~np.isfinite(ret)] = 0.0
    turn = np.abs(np.diff(np.vstack([np.zeros((1, close.shape[1]), np.float32),
                                     pos]), axis=0))
    pnl = pos * ret - turn * fee - pos * fund / PANEL["fund_div"]
    k = max(1, int(sel[0].size and sel.sum(axis=1).max()))
    port = np.nansum(pnl, axis=1) / k                 # equal weight, cash idle
    PANEL["last_turnover"] = float(turn.sum() / k / (len(port) / (24 * 365)))
    return port


def stats(port, label, split=None):
    eq = np.cumprod(1.0 + port)
    years = len(port) / (24 * 365)
    mdd = float((1.0 - eq / np.maximum.accumulate(eq)).max())
    ann = np.sqrt(24 * 365)
    sh = port.mean() / port.std(ddof=1) * ann if port.std(ddof=1) > 0 else 0.0
    daily = port.reshape(-1, 24).sum(axis=1) if len(port) % 24 == 0 else port
    t = daily.mean() / (daily.std(ddof=1) / np.sqrt(daily.size))
    out = dict(label=label, total=float(eq[-1] - 1.0),
               cagr=float(eq[-1] ** (1 / years) - 1.0), sharpe=float(sh),
               mdd=mdd, t=float(t), exposure=float((port != 0).mean()))
    if split:
        h = len(port) // 2
        for name, seg in (("h1", port[:h]), ("h2", port[h:])):
            e = np.cumprod(1.0 + seg)
            out[name + "_total"] = float(e[-1] - 1.0)
            out[name + "_sharpe"] = float(seg.mean() / seg.std(ddof=1) * ann)
    return out


def job(args):
    cfg, sel, fee = args
    port = run(cfg, sel, fee)
    s = stats(port, cfg["label"], split=True)
    s["turnover"] = PANEL.get("last_turnover", float("nan"))
    return s, port


def configs_round2():
    """Focused grid around what round one liked, plus ensembles of it."""
    out = []
    for entry in (48, 96, 168):
        for ex in (6, 12):
            for atr_m in (None, 2.0, 3.0):
                for vt in (None, 0.6):
                    c = dict(kind="donchian", entry=entry, exit=ex)
                    lab = f"donchian {entry}/{ex}"
                    if atr_m:
                        c["atr_mult"] = atr_m
                        lab += f" +{atr_m:g}ATR"
                    if vt:
                        c["vol_target"] = vt
                        lab += f" +vt{vt:g}"
                    out.append({**c, "label": lab})
    fam = [dict(kind="donchian", entry=e, exit=x, atr_mult=2.0)
           for e in (48, 96, 168) for x in (6, 12)]
    out.append(dict(kind="ensemble", members=fam,
                    label="ensemble 6x donchian +2ATR"))
    out.append(dict(kind="ensemble", members=fam, vol_target=0.6,
                    label="ensemble 6x donchian +2ATR +vt0.6"))
    fam2 = fam + [dict(kind="cross", fast=6, slow=24),
                  dict(kind="cross", fast=12, slow=48)]
    out.append(dict(kind="ensemble", members=fam2,
                    label="ensemble 6x donchian + 2 cross"))
    return out


def configs_round5():
    """The chosen rule, with and without the bar-level funding gate."""
    base = dict(kind="donchian", entry=96, exit=12, atr_mult=2.0,
                vol_target=0.6)
    return [{**base, "label": "rule only"},
            {**base, "fund_abs_pctile": 0.8, "label": "rule + gate 80th"},
            {**base, "fund_abs_pctile": 0.9, "label": "rule + gate 90th"}]


def configs_round4():
    """Filters suggested by the trade attribution, each on both halves."""
    base = dict(kind="donchian", entry=96, exit=12, atr_mult=2.0,
                vol_target=0.6)
    out = [{**base, "label": "base 96/12+2ATR+vt0.6"}]
    for q in (0.2, 0.33, 0.5, 0.6, 0.67, 0.75, 0.8, 0.9):
        out.append({**base, "fund_abs_pctile": q,
                    "label": f"+ |funding| above {int(q*100)}th pct"})
    out.append({**base, "fund_side": "contrarian",
                "label": "+ only against the funding side"})
    out.append({**base, "fund_abs_pctile": 0.33, "fund_side": "contrarian",
                "label": "+ |funding| 33rd pct and against it"})
    for w in (2.5, 3.0):
        out.append({**base, "atr_mult": w, "fund_abs_pctile": 0.33,
                    "label": f"+ |funding| 33rd, {w:g}ATR"})
    nvt = {k: v for k, v in base.items() if k != "vol_target"}
    out.append({**nvt, "fund_abs_pctile": 0.33,
                "label": "no vol target, |funding| 33rd"})
    return out


def configs_round3():
    """Is the peak a plateau? Neighbours of donchian 96/12 + 2ATR."""
    out = []
    for entry in (72, 96, 120, 144):
        for ex in (8, 12, 16, 24):
            for m in (1.5, 2.0, 2.5):
                out.append(dict(kind="donchian", entry=entry, exit=ex,
                                atr_mult=m, vol_target=0.6,
                                label=f"{entry}/{ex} +{m:g}ATR +vt0.6"))
    for w in (12, 48):
        out.append(dict(kind="donchian", entry=96, exit=12, atr_mult=2.0,
                        atr_w=w, vol_target=0.6,
                        label=f"96/12 +2ATR(w{w}) +vt0.6"))
    return out


def configs():
    out = []
    for entry in (12, 24, 48, 96, 168):
        for ex in (6, 12, 24, 48):
            if ex < entry:
                out.append(dict(kind="donchian", entry=entry, exit=ex,
                                label=f"donchian {entry}/{ex}"))
    for f, s in ((6, 24), (12, 48), (24, 96), (12, 96), (48, 192)):
        out.append(dict(kind="cross", fast=f, slow=s, label=f"cross {f}/{s}"))
    base = dict(kind="donchian", entry=48, exit=12)
    for m in (2.0, 3.0, 4.0):
        out.append({**base, "atr_mult": m, "label": f"donchian 48/12 + {m:g}ATR"})
    for h in (3 * BPD, 7 * BPD):
        out.append({**base, "max_hold": h, "label": f"donchian 48/12 + {h//BPD}d stop"})
    out.append({**base, "long_only": True, "label": "donchian 48/12 long only"})
    for w in (200, 500):
        out.append({**base, "trend_filter": w,
                    "label": f"donchian 48/12 + own MA{w}"})
        out.append({**base, "btc_filter": w,
                    "label": f"donchian 48/12 + BTC MA{w}"})
    for tv in (0.6, 1.0):
        out.append({**base, "vol_target": tv,
                    "label": f"donchian 48/12 + vol target {tv:g}"})
    out.append({**base, "atr_mult": 3.0, "trend_filter": 200,
                "label": "donchian 48/12 + 3ATR + own MA200"})
    out.append(dict(kind="donchian", entry=96, exit=24, atr_mult=3.0,
                    label="donchian 96/24 + 3ATR"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fee", type=float, default=0.0005)
    ap.add_argument("--sel-vol-days", type=int, default=3)
    ap.add_argument("--step-days", type=int, default=3)
    ap.add_argument("--top-vol", type=int, default=30)
    ap.add_argument("--basket", default="movers",
                    choices=["movers", "calm", "mid", "all", "movers_fund",
                             "movers_fund_rank"])
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default="/root/freqtrade/user_data/research/ma_harness/trend_lab.json")
    a = ap.parse_args()

    sel = selection_mask(a.sel_vol_days, a.step_days, top_vol=a.top_vol,
                         basket=a.basket)
    d = load()
    print(f"basket: {a.basket}, {a.top_vol} coins, refreshed every {a.step_days}d, "
          f"{sel.any(axis=1).sum()} live bars of {sel.shape[0]}, "
          f"fee {a.fee*100:.3f}%/side")

    cfgs = {1: configs, 2: configs_round2, 3: configs_round3,
            4: configs_round4, 5: configs_round5}[a.round]()
    rows, curves = [], {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for s, port in ex.map(job, [(c, sel, a.fee) for c in cfgs]):
            rows.append(s)
            curves[s["label"]] = port
    rows.sort(key=lambda r: -r["sharpe"])
    print(f"\n{'rule':<36}{'total':>9}{'CAGR':>8}{'Sharpe':>8}{'maxDD':>7}"
          f"{'t/day':>7}{'turn/yr':>8}{'h1 Sh':>7}{'h2 Sh':>7}")
    for r in rows:
        print(f"{r['label']:<36}{100*r['total']:>+8.1f}%{100*r['cagr']:>+7.1f}%"
              f"{r['sharpe']:>8.2f}{100*r['mdd']:>6.1f}%{r['t']:>+7.2f}"
              f"{r.get('turnover', float('nan')):>8.0f}"
              f"{r['h1_sharpe']:>7.2f}{r['h2_sharpe']:>7.2f}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=1)
    np.savez_compressed(a.out.replace(".json", "_curves.npz"), **curves)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
