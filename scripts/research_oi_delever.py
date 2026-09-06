"""C2 event study: does an extreme 30-minute price move revert when it is accompanied by a
drop in open interest (forced deleveraging) and continue when OI rises (new positioning)?

Frozen design:
- Panel: 5m bars from Binance public 1m klines + 5m `metrics` (sum_open_interest).
- Universe at event time: trailing 30-day median daily quote volume >= 10M USDT (also report >= 1M band).
- Event at 5m bar t (bar closed): over the last 6 bars (30 min)
    price move  m = mark_close[t] / mark_close[t-6] - 1,  |m| >= 3 * sigma,
    sigma = std of 30-min mark returns over the prior 7 days (2016 bars, shifted by 6),
    OI change   o = OI[t] / OI[t-6] - 1.
  Buckets: OI_down (o <= -X), OI_up (o >= +X), OI_flat otherwise; X in {5%, 10%}; crossed with sign(m).
- Position: FADE the move (short after up-move, long after down-move); entry at open of bar t+1
  (last price), exit at open of bar t+1+h for h in {5, 15, 30, 60, 240} min; arithmetic return on
  entry notional; cost 20 bp round trip; no stop (event study).
- Dedupe: first trigger per pair, then 60-minute cooldown per pair.
- Controls: BTC fade over the same window (BTC return * -sign(m)); funding ignored (holds < 4h,
  a settlement inside the window is possible but the median event is far shorter than 8h).
- Splits by event date: 2025 train / 2026-01..05 valid / 2026-06+ holdout. Cluster bootstrap by
  4-hour calendar block (cascades are market-wide).
Nothing is selected after seeing results: every (X, sign, bucket, horizon) cell is reported.
"""
from __future__ import annotations
import sys, glob, os
from multiprocessing import Pool
import numpy as np, pandas as pd

B = "/root/freqtrade/user_data/data/binance_public"
OUT = "/root/freqtrade/user_data/oi_delever_20260906"
K_SIGMA = 3.0
W = 6              # 30 minutes in 5m bars
SIG_BARS = 2016    # 7 days of 5m bars
HORIZONS = [1, 3, 6, 12, 48]   # bars: 5, 15, 30, 60, 240 min
X_LIST = [0.05, 0.10]
COOLDOWN = 12      # bars


def one_symbol(sym: str):
    try:
        mf = f"{B}/metrics/{sym}.parquet"
        if not os.path.exists(mf):
            return None
        k = pd.read_parquet(f"{B}/klines_1m/{sym}.parquet")
        nz = k[k.volume > 0]
        if len(nz) < 100000:
            return None
        k = k[(k.date >= nz.date.min()) & (k.date <= nz.date.max())].set_index("date")
        k5 = k.resample("5min").agg(open=("open", "first"), close=("close", "last"), quote_volume=("quote_volume", "sum"))
        m = pd.read_parquet(f"{B}/markprice_1m/{sym}.parquet").set_index("date")["mark_close"].resample("5min").last()
        oi = pd.read_parquet(mf).set_index("date")["sum_open_interest"]
        df = k5.join(m.rename("mark")).join(oi.rename("oi"), how="left")
        df = df[df.open.notna()]
        df["oi"] = df.oi.ffill(limit=2)
        df["mv"] = df.mark / df.mark.shift(W) - 1
        df["sig"] = df.mv.shift(W).rolling(SIG_BARS, min_periods=SIG_BARS // 2).std()
        df["oc"] = df.oi / df.oi.shift(W) - 1
        qv_d = df.quote_volume.resample("1D").sum()
        qv30 = qv_d.rolling(30, min_periods=20).median().shift(1)
        df["qv30"] = qv30.reindex(df.index, method="ffill")
        df["ret_next_open"] = df.open.shift(-1)
        for h in HORIZONS:
            df[f"r{h}"] = df.open.shift(-1 - h) / df.open.shift(-1) - 1   # long return from t+1 open to t+1+h open
        cand = df[(df.mv.abs() >= K_SIGMA * df.sig) & df.oc.notna() & df.sig.notna() & (df.qv30 >= 1e6)].copy()
        if cand.empty:
            return None
        # dedupe with cooldown
        keep, last = [], None
        for t in cand.index:
            if last is None or (t - last) >= pd.Timedelta(minutes=5 * COOLDOWN):
                keep.append(t); last = t
        cand = cand.loc[keep]
        cand["symbol"] = sym
        cand["sign"] = np.sign(cand.mv)
        cols = ["symbol", "mv", "sig", "oc", "qv30", "sign"] + [f"r{h}" for h in HORIZONS]
        return cand[cols].reset_index()
    except Exception as e:  # noqa
        print("ERR", sym, e, file=sys.stderr)
        return None


def bucket(o, x):
    return np.where(o <= -x, "OI_down", np.where(o >= x, "OI_up", "OI_flat"))


def cluster_boot(v, blocks, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    ub = np.unique(blocks)
    groups = [v[blocks == b] for b in ub]
    G = len(groups)
    means = np.array([np.concatenate([groups[i] for i in rng.integers(0, G, G)]).mean() for _ in range(n)])
    return np.percentile(means, [2.5, 97.5]), (means <= 0).mean()


def main():
    os.makedirs(OUT, exist_ok=True)
    syms = sorted(os.path.basename(f)[:-8] for f in glob.glob(f"{B}/metrics/*.parquet"))
    syms = [s for s in syms if s.endswith("USDT")]
    with Pool(16) as p:
        parts = [x for x in p.map(one_symbol, syms, chunksize=4) if x is not None]
    ev = pd.concat(parts, ignore_index=True)
    ev = ev[ev.date >= pd.Timestamp("2025-01-08", tz="UTC")]
    # BTC control
    kb = pd.read_parquet(f"{B}/klines_1m/BTCUSDT.parquet").set_index("date").open.resample("5min").first()
    for h in HORIZONS:
        ev[f"btc{h}"] = (kb.shift(-1 - h) / kb.shift(-1) - 1).reindex(ev.date).values
    ev["period"] = np.where(ev.date.dt.year == 2025, "2025_train", np.where(ev.date < pd.Timestamp("2026-06-01", tz="UTC"), "2026H1_valid", "2026Jun+_holdout"))
    ev["block"] = ev.date.dt.floor("4h").astype("int64")
    ev.to_parquet(f"{OUT}/events.parquet")
    pd.set_option("display.width", 250)
    print(f"events: {len(ev)} across {ev.symbol.nunique()} symbols; sign up {int((ev.sign > 0).sum())}, down {int((ev.sign < 0).sum())}")
    for qv_min, label in [(1e7, "qv30>=10M"), (1e6, "qv30>=1M")]:
        e = ev[ev.qv30 >= qv_min]
        for x in X_LIST:
            e = e.assign(bucket=bucket(e.oc.values, x))
            print(f"\n=== {label}, X={x:.0%}: FADE return (bps, net of 20bp) by bucket x sign x horizon; n / mean / median / CI95 / P(<=0) / btc-fade mean")
            rows = []
            for (bk, sg), g in e.groupby(["bucket", "sign"]):
                for h in HORIZONS:
                    fade = -sg * g[f"r{h}"].values * 1e4 - 20
                    ok = np.isfinite(fade)
                    if ok.sum() < 20:
                        continue
                    ci, pv = cluster_boot(fade[ok], g.block.values[ok])
                    btc = (-sg * g[f"btc{h}"].values * 1e4)[ok]
                    rows.append(dict(bucket=bk, sign="up" if sg > 0 else "down", h_min=5 * h, n=int(ok.sum()), mean=fade[ok].mean(), median=np.median(fade[ok]),
                                     lo=ci[0], hi=ci[1], p=pv, btc_fade=np.nanmean(btc),
                                     train=fade[ok][(g.period.values[ok] == "2025_train")].mean(), valid=fade[ok][(g.period.values[ok] == "2026H1_valid")].mean(),
                                     holdout=fade[ok][(g.period.values[ok] == "2026Jun+_holdout")].mean()))
            print(pd.DataFrame(rows).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
