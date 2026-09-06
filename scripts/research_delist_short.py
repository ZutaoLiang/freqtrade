"""Event study: short Binance USDT perpetuals after a futures delisting announcement.

Frozen design (written before any result was looked at):
- Events: announcement (release timestamp, minute precision) -> per-symbol settlement time,
  parsed from the Binance CMS delisting catalog (user_data/delist_short_20260906/events_parsed.csv).
- Entry: short at the open of the first 1m candle whose open time >= release + delay,
  delay in {0, 1, 5, 15, 60} minutes. 0 is an unattainable reference, not a tradeable arm.
- Exit: whichever comes first:
    (a) forced exit at settlement - 60 min (open of that candle),
    (b) close-based stop: 1m close >= entry * 1.15 -> exit at next open.
  Also report fixed horizons 5/15/30/60/240/1440 min (no stop) for the path.
- Costs: 0.10% per side (fee + friction assumption), stress 0.20% per side.
- Funding: every settlement in (entry, exit] with rate r and mark price m at that minute
  contributes +r * m / entry to a short's return on entry notional (short receives positive rate).
  Missing mark at a settlement -> event excluded from the funding-inclusive arms, counted.
- Padding: klines archives are padded with zero-volume rows; an event is only valid if the
  entry candle lies inside the symbol's non-zero-volume range and the exit candle exists.
- Controls: BTCUSDT short over the identical window (price only).
- Splits by announcement date: 2025 = train, 2026-01..05 = valid, 2026-06+ = holdout.
- Robustness: cluster bootstrap by announcement notice, drop best 3 events, 20bp/side stress.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/root/freqtrade")
B = ROOT / "user_data/data/binance_public"
OUT = ROOT / "user_data/delist_short_20260906"
DELAYS = [0, 1, 5, 15, 60]
HORIZONS = [5, 15, 30, 60, 240, 1440]
STOP = 0.15
FORCE_BEFORE_MIN = 60
COST = 0.001
COST_STRESS = 0.002
DATA_START = pd.Timestamp("2025-01-01", tz="UTC")


def load_symbol(sym: str):
    k = pd.read_parquet(B / f"klines_1m/{sym}.parquet")
    nz = k[k.volume > 0]
    if nz.empty:
        return None
    k = k[(k.date >= nz.date.min()) & (k.date <= nz.date.max())].set_index("date")
    m = pd.read_parquet(B / f"markprice_1m/{sym}.parquet").set_index("date")["mark_close"]
    f = pd.read_parquet(B / f"funding/{sym}.parquet")
    f = f[(f.date > nz.date.min()) & (f.date <= nz.date.max())].set_index("date")["funding_rate"]
    return k, m, f


def period_of(ts: pd.Timestamp) -> str:
    if ts.year == 2025:
        return "2025_train"
    if ts < pd.Timestamp("2026-06-01", tz="UTC"):
        return "2026H1_valid"
    return "2026Jun+_holdout"


def run_event(ev, data, btc, delay: int):
    k, m, f = data
    release = ev.release
    entry_t = (release + pd.Timedelta(minutes=delay)).ceil("1min")
    force_t = (ev.settle - pd.Timedelta(minutes=FORCE_BEFORE_MIN)).floor("1min")
    if entry_t not in k.index:
        return {"symbol": ev.symbol, "delay": delay, "excluded": "no_entry_candle"}
    if entry_t >= force_t:
        return {"symbol": ev.symbol, "delay": delay, "excluded": "entry_after_force_exit"}
    entry = k.at[entry_t, "open"]
    if not np.isfinite(entry) or entry <= 0:
        return {"symbol": ev.symbol, "delay": delay, "excluded": "bad_entry_price"}
    path = k.loc[entry_t:force_t]
    # close-based stop: first close >= entry*(1+STOP), exit at next candle open
    hit = path.index[path.close >= entry * (1 + STOP)]
    exit_reason = "force"
    exit_t = force_t
    if len(hit):
        nxt = path.index[path.index > hit[0]]
        if len(nxt):
            exit_t, exit_reason = nxt[0], "close_stop"
    if exit_t not in k.index:
        # data ends before force exit: exit at last available candle, flagged
        exit_t, exit_reason = path.index[-1], "data_end"
    exit_px = k.at[exit_t, "open"] if exit_reason != "data_end" else k.at[exit_t, "close"]
    gross = 1.0 - exit_px / entry  # short P&L on entry notional (arithmetic; a 100% rise = -100%)
    # funding
    fs = f[(f.index > entry_t) & (f.index <= exit_t)]
    fund = 0.0
    fund_missing = 0
    for t, r in fs.items():
        if t in m.index and np.isfinite(m.at[t]):
            fund += r * m.at[t] / entry
        else:
            fund_missing += 1
    rec = {
        "symbol": ev.symbol, "notice": ev.title, "period": period_of(ev.release), "delay": delay,
        "release": release, "entry_t": entry_t, "exit_t": exit_t, "exit_reason": exit_reason,
        "hold_h": (exit_t - entry_t).total_seconds() / 3600, "entry": entry, "exit": exit_px,
        "gross": gross, "funding": fund, "n_funding": len(fs), "fund_missing": fund_missing,
        "net10": gross + fund - 2 * COST, "net20": gross + fund - 2 * COST_STRESS,
        "excluded": "",
    }
    # BTC control over same window
    if entry_t in btc.index and exit_t in btc.index:
        rec["btc_short"] = 1.0 - btc.at[exit_t, "open"] / btc.at[entry_t, "open"]
    # fixed horizons, no stop, price only
    for h in HORIZONS:
        t = entry_t + pd.Timedelta(minutes=h)
        rec[f"h{h}"] = 1.0 - k.at[t, "open"] / entry if t in k.index else np.nan
    # pre-event context
    pre = k.loc[:entry_t - pd.Timedelta(minutes=1)]
    rec["pre24h_ret"] = pre.close.iloc[-1] / pre.close.iloc[-1441] - 1.0 if len(pre) > 1441 else np.nan
    qv_daily = pre.quote_volume.resample("1D").sum()
    qv_daily = qv_daily.iloc[-31:-1]
    rec["qv30_median"] = qv_daily.median() if len(qv_daily) else np.nan
    return rec


def cum_path(ev, data, minutes=180):
    k, _, _ = data
    t0 = ev.release.ceil("1min")
    if t0 not in k.index:
        return None
    seg = k.loc[t0:t0 + pd.Timedelta(minutes=minutes)]
    base = seg.open.iloc[0]
    s = (1.0 - seg.open / base)
    s.index = ((s.index - t0).total_seconds() // 60).astype(int)
    return s.reindex(range(minutes + 1))


def cluster_boot(df, col, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    groups = [g[col].values for _, g in df.groupby("notice")]
    G = len(groups)
    means = []
    for _ in range(n):
        pick = rng.integers(0, G, G)
        v = np.concatenate([groups[i] for i in pick])
        means.append(v.mean())
    means = np.array(means)
    return np.percentile(means, [2.5, 97.5]), (means <= 0).mean()


def summarize(df, col="net10"):
    out = {}
    for name, g in [("all", df)] + list(df.groupby("period")):
        if len(g) == 0:
            continue
        r = {"n": len(g), "mean%": g[col].mean() * 100, "median%": g[col].median() * 100,
             "win%": (g[col] > 0).mean() * 100}
        if name == "all":
            ci, p = cluster_boot(g, col)
            r["ci95"] = f"[{ci[0]*100:+.2f}, {ci[1]*100:+.2f}]"
            r["P(mean<=0)"] = p
            r["drop_best3%"] = g[col].sort_values().iloc[:-3].mean() * 100 if len(g) > 3 else np.nan
            r["btc_short%"] = g["btc_short"].mean() * 100
            r["net20%"] = g["net20"].mean() * 100
            r["funding%"] = g["funding"].mean() * 100
        out[name] = r
    return pd.DataFrame(out).T


def main():
    ev = pd.read_csv(OUT / "events_parsed.csv", parse_dates=["release", "settle"])
    ev["release"] = pd.to_datetime(ev.release, utc=True)
    ev["settle"] = pd.to_datetime(ev.settle, utc=True)
    ev = ev[ev.release >= DATA_START].sort_values("release").reset_index(drop=True)
    btc = pd.read_parquet(B / "klines_1m/BTCUSDT.parquet").set_index("date")
    recs, paths, excl = [], [], []
    cache = {}
    for _, e in ev.iterrows():
        if e.symbol not in cache:
            cache[e.symbol] = load_symbol(e.symbol)
        data = cache[e.symbol]
        if data is None:
            excl.append({"symbol": e.symbol, "reason": "no_nonzero_volume_data"})
            continue
        for d in DELAYS:
            r = run_event(e, data, btc, d)
            if r.get("excluded"):
                excl.append({"symbol": e.symbol, "delay": d, "reason": r["excluded"]})
            else:
                recs.append(r)
        p = cum_path(e, data)
        if p is not None:
            paths.append(p.rename(e.symbol))
    df = pd.DataFrame(recs)
    df.to_parquet(OUT / "trades_all_delays.parquet")
    pd.DataFrame(excl).to_csv(OUT / "exclusions.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"events with data: {df.symbol.nunique()} symbols, {df.notice.nunique()} notices; exclusions: {len(excl)}")
    if excl:
        print(pd.DataFrame(excl).groupby(["reason"]).size().to_string())
    for d in DELAYS:
        g = df[df.delay == d]
        print(f"\n=== delay {d} min: n={len(g)}, exit reasons {g.exit_reason.value_counts().to_dict()}, "
              f"hold median {g.hold_h.median():.1f}h, funding events missing {g.fund_missing.sum()}")
        print(summarize(g).round(3).to_string())
        print("fixed horizons, price only, mean% / median%:",
              {f"h{h}": f"{g[f'h{h}'].mean()*100:+.2f}/{g[f'h{h}'].median()*100:+.2f}" for h in HORIZONS})
    P = pd.concat(paths, axis=1)
    P.to_csv(OUT / "cum_path_minutes.csv")
    print("\n=== mean cumulative SHORT return (%) at minute k after announcement (price only, no cost):")
    print((P.mean(axis=1) * 100).loc[[0, 1, 2, 3, 5, 10, 15, 30, 60, 120, 180]].round(2).to_string())
    print("median:")
    print((P.median(axis=1) * 100).loc[[0, 1, 2, 3, 5, 10, 15, 30, 60, 120, 180]].round(2).to_string())
    g = df[df.delay == 5].sort_values("release")
    print("\n=== per-event, delay 5 (net10 %, hold h, reason, funding %, btc short %):")
    print(g[["symbol", "period", "entry_t", "hold_h", "exit_reason", "gross", "funding", "net10", "btc_short", "pre24h_ret", "qv30_median"]]
          .assign(gross=lambda x: x.gross * 100, funding=lambda x: x.funding * 100, net10=lambda x: x.net10 * 100,
                  btc_short=lambda x: x.btc_short * 100, pre24h_ret=lambda x: x.pre24h_ret * 100)
          .round(2).to_string())


if __name__ == "__main__":
    main()
