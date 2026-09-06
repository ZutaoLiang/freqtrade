"""C5 event study: 1-minute basis (last vs mark) spikes on Binance USDT perps.

Frozen design:
- b[t] = close[t] / mark_close[t] - 1 on 1m bars (perp last price premium over mark).
  Note: Binance mark = index + EMA of basis, so b understates the true (last - index) premium;
  we test the observable proxy only.
- Event: |b[t]| >= 0.30% AND |b[t]| >= 5 * rolling 24h std of b (shifted 1 bar), trailing 30-day
  median daily quote volume >= 10M USDT (also report >= 1M).
- Position: fade the premium (b>0 -> short, b<0 -> long) at open of bar t+1, exit at open of
  bar t+1+h for h in {5, 15, 30, 60} minutes; arithmetic return on notional, 20 bp round trip.
- Dedupe: 60-minute cooldown per pair. Cluster bootstrap by 4-hour calendar block.
- Overlap: flag events where the pair had |funding| >= 40 bps at its most recent settlement
  (funding-skew territory) and report with/without.
- Splits: 2025 train / 2026-01..05 valid / 2026-06+ holdout. All cells reported.
"""
import glob, os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd

B = "/root/freqtrade/user_data/data/binance_public"
OUT = "/root/freqtrade/user_data/basis_spike_20260906"
ABS_MIN = 0.003
K = 5.0
HORIZONS = [5, 15, 30, 60]
COOLDOWN = 60


def one(sym):
    try:
        mf = f"{B}/markprice_1m/{sym}.parquet"
        if not os.path.exists(mf):
            return None
        k = pd.read_parquet(f"{B}/klines_1m/{sym}.parquet")
        nz = k[k.volume > 0]
        if len(nz) < 100000:
            return None
        k = k[(k.date >= nz.date.min()) & (k.date <= nz.date.max())].set_index("date")
        m = pd.read_parquet(mf).set_index("date")["mark_close"]
        df = k[["open", "close", "quote_volume"]].join(m.rename("mark"), how="left")
        df = df[df.mark.notna() & (df.open > 0)]
        df["b"] = df.close / df.mark - 1
        df["bstd"] = df.b.shift(1).rolling(1440, min_periods=720).std()
        qv30 = df.quote_volume.resample("1D").sum().rolling(30, min_periods=20).median().shift(1)
        df["qv30"] = qv30.reindex(df.index, method="ffill")
        f = pd.read_parquet(f"{B}/funding/{sym}.parquet").set_index("date")["funding_rate"]
        f = f[(f.index >= df.index[0]) & (f.index <= df.index[-1])]
        df["last_fr"] = f.reindex(df.index, method="ffill").abs()
        for h in HORIZONS:
            df[f"r{h}"] = df.open.shift(-1 - h) / df.open.shift(-1) - 1
        c = df[(df.b.abs() >= ABS_MIN) & (df.b.abs() >= K * df.bstd) & (df.qv30 >= 1e6)]
        if c.empty:
            return None
        keep, last = [], None
        for t in c.index:
            if last is None or (t - last) >= pd.Timedelta(minutes=COOLDOWN):
                keep.append(t); last = t
        c = c.loc[keep].copy()
        c["symbol"] = sym
        c["sign"] = np.sign(c.b)
        return c[["symbol", "b", "bstd", "qv30", "last_fr", "sign"] + [f"r{h}" for h in HORIZONS]].reset_index()
    except Exception as e:
        print("ERR", sym, e, file=sys.stderr)
        return None


def cluster_boot(v, blocks, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    ub = np.unique(blocks); groups = [v[blocks == b] for b in ub]; G = len(groups)
    means = np.array([np.concatenate([groups[i] for i in rng.integers(0, G, G)]).mean() for _ in range(n)])
    return np.percentile(means, [2.5, 97.5]), (means <= 0).mean()


def main():
    os.makedirs(OUT, exist_ok=True)
    syms = sorted(os.path.basename(f)[:-8] for f in glob.glob(f"{B}/markprice_1m/*USDT.parquet"))
    with Pool(16) as p:
        parts = [x for x in p.map(one, syms, chunksize=4) if x is not None]
    ev = pd.concat(parts, ignore_index=True)
    ev = ev[ev.date >= pd.Timestamp("2025-01-03", tz="UTC")]
    ev["period"] = np.where(ev.date.dt.year == 2025, "2025_train", np.where(ev.date < pd.Timestamp("2026-06-01", tz="UTC"), "2026H1_valid", "2026Jun+_holdout"))
    ev["block"] = ev.date.dt.floor("4h").astype("int64")
    ev["fs_overlap"] = ev.last_fr >= 0.004
    ev.to_parquet(f"{OUT}/events.parquet")
    pd.set_option("display.width", 250)
    print(f"events: {len(ev)} across {ev.symbol.nunique()} symbols; premium(+) {int((ev.sign > 0).sum())}, discount(-) {int((ev.sign < 0).sum())}; funding-skew overlap {ev.fs_overlap.mean():.1%}")
    for qv_min, label in [(1e7, "qv30>=10M"), (1e6, "qv30>=1M")]:
        for ov, ovl in [(None, "all"), (False, "no funding-skew overlap")]:
            e = ev[ev.qv30 >= qv_min]
            if ov is not None:
                e = e[~e.fs_overlap]
            rows = []
            for sg, g in e.groupby("sign"):
                for h in HORIZONS:
                    fade = -sg * g[f"r{h}"].values * 1e4 - 20
                    ok = np.isfinite(fade)
                    if ok.sum() < 20:
                        continue
                    ci, pv = cluster_boot(fade[ok], g.block.values[ok])
                    per = g.period.values[ok]
                    rows.append(dict(sign="premium->short" if sg > 0 else "discount->long", h_min=h, n=int(ok.sum()), mean=fade[ok].mean(), median=np.median(fade[ok]), lo=ci[0], hi=ci[1], p=pv,
                                     train=fade[ok][per == "2025_train"].mean(), valid=fade[ok][per == "2026H1_valid"].mean(), holdout=fade[ok][per == "2026Jun+_holdout"].mean(), win=(fade[ok] > 0).mean() * 100))
            print(f"\n=== {label}, {ovl}: FADE return bps net of 20bp")
            print(pd.DataFrame(rows).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
