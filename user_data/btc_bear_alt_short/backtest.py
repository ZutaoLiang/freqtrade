"""BTC-bear regime -> equal-weight short basket of alts (market cap >= 200M, ex BTC/ETH) with per-coin stops.

Timing (all causal):
  * The regime is evaluated on BTC daily closes up to and including day D-1 and acted on at the
    first 4h bar of day D (00:00 UTC open).
  * Eligibility uses the CMC snapshot available at that time and the 4h_long liquidity mask
    of the previous bar.
  * Each day at 00:00 the book is rebalanced to -gross/N per eligible, not-stopped coin.
    Fees are charged on the traded notional.
  * Per-coin stop: once the 4h high reaches entry * (1 + stop), the short is closed at
    max(open, stop price) plus slippage. The coin is then barred until the regime switches
    off and on again (bar="episode") or for `cooldown` days (bar="cooldown").
  * Funding: the short receives the positive rates and pays the negative ones.
  * A coin whose price disappears (delisting) is closed at its last close.
"""
import argparse, itertools, json, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FEE = 0.0005     # taker per side
SLIP = 0.0005    # per side, every fill
STOP_SLIP = 0.0010  # extra on stop fills (market order into a squeeze)


def load():
    z = np.load(os.path.join(HERE, "inputs.npz"))
    d = {k: z[k] for k in z.files}
    d["grid"] = pd.DatetimeIndex(d["grid"], tz="UTC")
    d["btc"] = pd.Series(d["btc_d"], index=pd.DatetimeIndex(d["btc_d_idx"], tz="UTC"))
    return d


STRICT = {"s_ma3", "s_ma3_slope", "s_ma3_slope_ret", "s_all8", "s_7of8", "s_6of8"}


def bear_conditions(c):
    """Eight daily bearish conditions on BTC closes."""
    s20, s50, s200 = c.rolling(20).mean(), c.rolling(50).mean(), c.rolling(200).mean()
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    cond = pd.DataFrame({
        "c<s20": c < s20, "c<s50": c < s50, "c<s200": c < s200,
        "s20<s50": s20 < s50, "s50<s200": s50 < s200,
        "s50_down": s50 < s50.shift(10), "ret30<0": c / c.shift(30) - 1 < 0, "macd<0": macd < 0,
    })
    return cond[s200.notna()].reindex(c.index, fill_value=False).astype(bool)


def strict_conditions(c, name):
    k = bear_conditions(c)
    ma3 = k["c<s50"] & k["c<s200"] & k["s50<s200"]
    if name == "s_ma3":
        return ma3
    if name == "s_ma3_slope":
        return ma3 & k["s50_down"]
    if name == "s_ma3_slope_ret":
        return ma3 & k["s50_down"] & k["ret30<0"]
    n = k.sum(axis=1)
    return {"s_all8": n == 8, "s_7of8": n >= 7, "s_6of8": n >= 6}[name]


def regime_series(btc, name):
    c = btc
    if name == "always":
        on = pd.Series(True, index=c.index)
    elif name == "sma50":
        on = c < c.rolling(50).mean()
    elif name == "sma200":
        on = c < c.rolling(200).mean()
    elif name == "cross":
        on = c.rolling(50).mean() < c.rolling(200).mean()
    elif name == "ret30":
        on = c / c.shift(30) - 1 < -0.05
    elif name == "sma20x50":
        s20, s50 = c.rolling(20).mean(), c.rolling(50).mean()
        on = (c < s20) & (s20 < s50)
    elif name in STRICT:
        on = strict_conditions(c, name)
    elif name.startswith("not_"):
        on = ~regime_series(btc, name[4:])
    else:
        raise ValueError(name)
    # value at day D-1 close is used on day D
    on.index = on.index + pd.Timedelta("1D")
    return on


def run(d, regime="sma50", stop=0.15, bar="episode", cooldown=7, mcap_min=2e8, gross=1.0, use_funding=1.0):
    grid = d["grid"]
    o, h, cl = d["open"], d["high"], d["close"]
    T, N = cl.shape
    on_day = regime_series(d["btc"], regime).reindex(grid.normalize(), method="ffill").fillna(False).to_numpy()
    is_day_start = grid.hour == 0
    mcap, liq, fund = d["mcap"], d["liq"], d["fund"]

    E = 1.0
    q = np.zeros(N)            # coins held (negative = short)
    entry = np.full(N, np.nan)
    barred_until = np.zeros(N, dtype=np.int64)   # bar index; episode-bar uses a large sentinel
    last_px = np.full(N, np.nan)
    fbar = np.zeros(T); cbar = np.zeros(T)
    eq = np.empty(T); expo = np.zeros(T); nheld = np.zeros(T, dtype=int)
    fees_tot = fund_tot = 0.0
    n_stops = n_entries = 0
    trades = []   # (sym, entry_time, exit_time, entry, exit, reason)
    entry_t = np.full(N, -1)
    prev_on = False

    t_now = [0]

    def close_pos(i, px, t, reason, extra_slip=0.0):
        nonlocal E, fees_tot
        cost = abs(q[i]) * px * (FEE + SLIP + extra_slip)
        E -= cost; fees_tot += cost; cbar[t_now[0]] += cost
        trades.append((i, entry_t[i], t, entry[i], px, reason))
        q[i] = 0.0; entry[i] = np.nan; entry_t[i] = -1

    for t in range(T):
        t_now[0] = t
        op = o[t]
        valid_now = np.isfinite(op)
        # delistings / gaps: settle at last known close
        for i in np.nonzero((q != 0) & ~valid_now)[0]:
            close_pos(i, last_px[i], t, "delist")
        # gap from previous close to this open
        m = (q != 0)
        if m.any():
            E += np.sum(q[m] * (op[m] - last_px[m]))
        if is_day_start[t]:
            on = bool(on_day[t])
            if on and not prev_on:
                barred_until[barred_until >= 10**9] = 0   # new episode clears episode bars
            prev_on = on
            if on:
                ok = (mcap[t] >= mcap_min) & (liq[t - 1] if t else False) & valid_now & (barred_until <= t)
            else:
                ok = np.zeros(N, dtype=bool)
            # exits first
            for i in np.nonzero((q != 0) & ~ok)[0]:
                close_pos(i, op[i], t, "regime" if not on else "ineligible")
            n_ok = ok.sum()
            if n_ok:
                tgt = -gross * E / n_ok / op[ok]
                idx = np.nonzero(ok)[0]
                dq = tgt - q[idx]
                cost = np.sum(np.abs(dq) * op[idx]) * (FEE + SLIP)
                E -= cost; fees_tot += cost; cbar[t] += cost
                new = idx[q[idx] == 0]
                entry[new] = op[new]; entry_t[new] = t; n_entries += len(new)
                q[idx] = tgt
        # intrabar stop, else mark to close
        m = q != 0
        if m.any() and stop is not None:
            sp = entry * (1 + stop)
            hit = m & (h[t] >= sp)
            for i in np.nonzero(hit)[0]:
                fill = max(op[i], sp[i])
                E += q[i] * (fill - op[i])
                close_pos(i, fill, t, "stop", STOP_SLIP)
                barred_until[i] = 10**12 if bar == "episode" else t + cooldown * 6
                n_stops += 1
        m = q != 0
        c = np.where(np.isfinite(cl[t]), cl[t], op)
        if m.any():
            E += np.sum(q[m] * (c[m] - op[m]))
            f = -np.sum(q[m] * c[m] * fund[t, m])   # long pays positive funding
            E += f * use_funding; fund_tot += f; fbar[t] = f
        last_px = np.where(np.isfinite(c), c, last_px)
        eq[t] = E
        expo[t] = -np.sum(q[m] * c[m]) / E if E > 0 else 0
        nheld[t] = m.sum()
        if E <= 0:
            eq[t:] = 0; break
    eq_prev = np.concatenate([[1.0], eq[:-1]])
    return dict(fund_ret=pd.Series(fbar / eq_prev, index=grid), fee_ret=pd.Series(cbar / eq_prev, index=grid),
                eq=pd.Series(eq, index=grid), expo=pd.Series(expo, index=grid), nheld=pd.Series(nheld, index=grid),
                fees=fees_tot, funding=fund_tot, stops=n_stops, entries=n_entries, trades=trades)


PERIODS = {"2022-11..2023": ("2022-11-01", "2024-01-01"), "2024": ("2024-01-01", "2025-01-01"),
           "2025": ("2025-01-01", "2026-01-01"), "2026 (..08-16)": ("2026-01-01", "2026-08-17")}


def stats(eq, expo=None):
    eq = eq[eq > 0]
    daily = eq.resample("1D").last().dropna()
    r = daily.pct_change().dropna()
    yrs = (daily.index[-1] - daily.index[0]).days / 365.25
    tot = daily.iloc[-1] / daily.iloc[0] - 1
    cagr = (1 + tot) ** (1 / yrs) - 1 if yrs > 0 and tot > -1 else np.nan
    dd = (daily / daily.cummax() - 1).min()
    sh = r.mean() / r.std() * np.sqrt(365) if r.std() > 0 else np.nan
    out = dict(ret=tot, cagr=cagr, sharpe=sh, maxdd=dd)
    if expo is not None:
        out["in_mkt"] = (expo.resample("1D").mean() > 0.01).mean()
    return out


def period_table(res):
    rows = {}
    for k, (a, b) in PERIODS.items():
        e = res["eq"][a:b]
        rows[k] = stats(e, res["expo"][a:b])
        rows[k]["funding"] = res["fund_ret"][a:b].sum()
        rows[k]["fees"] = -res["fee_ret"][a:b].sum()
    rows["ALL"] = stats(res["eq"], res["expo"])
    rows["ALL"]["funding"] = res["fund_ret"].sum()
    rows["ALL"]["fees"] = -res["fee_ret"].sum()
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", default="always,sma50,sma200,cross,ret30,sma20x50")
    ap.add_argument("--stops", default="none,0.10,0.15,0.20,0.30")
    ap.add_argument("--out", default=os.path.join(HERE, "grid.csv"))
    a = ap.parse_args()
    d = load()
    rows = []
    for reg, st in itertools.product(a.regimes.split(","), a.stops.split(",")):
        stop = None if st == "none" else float(st)
        res = run(d, reg, stop)
        tab = period_table(res)
        for p, r in tab.iterrows():
            rows.append(dict(regime=reg, stop=st, period=p, **r.to_dict(), stops_hit=res["stops"],
                             entries=res["entries"], fees=res["fees"], funding=res["funding"]))
        al = tab.loc["ALL"]
        yr = " ".join(f"{p[:4]}:{tab.loc[p,'ret']:+.1%}" for p in PERIODS)
        print(f"{reg:10s} stop={st:5s} ALL ret {al.ret:+8.1%} sharpe {al.sharpe:+.2f} maxDD {al.maxdd:.1%} "
              f"in_mkt {al.in_mkt:.0%} stops {res['stops']:4d} | {yr}", flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
