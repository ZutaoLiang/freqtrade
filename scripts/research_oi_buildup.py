"""C3 event study: open interest builds up while price stays flat; direction from top-trader positioning.

Frozen design:
- 5m panel: klines (resampled) + metrics (sum_open_interest, sum_toptrader_long_short_ratio).
- Event at bar t: OI[t]/OI[t-48] - 1 >= +10% over 4h (also a 1h arm: OI[t]/OI[t-12] - 1 >= +5%)
  AND |close[t]/close[t-48] - 1| <= 0.5 * sigma4h, sigma4h = std of 4h returns over the prior 7 days
  (2016 bars, shifted). Universe: trailing 30d median daily quote volume >= 10M (also >= 1M).
- Direction: sign of the 1h change in the top-trader POSITION long/short ratio (ratio[t] - ratio[t-12]);
  rising -> long, falling -> short; zero change -> skipped. This is the only direction rule.
- Entry at open of t+1, exit at open of t+1+h, h in {12, 48} bars (1h, 4h). 20 bp round trip.
- Dedupe: 4h cooldown per pair. Cluster bootstrap by 4h calendar block. Splits as elsewhere.
- Also reported (diagnostic, not a strategy): mean |return| over the horizon for event bars vs all
  bars of the same pair, to see whether a build-up precedes a larger move at all.
"""
import glob, os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd

B = "/root/freqtrade/user_data/data/binance_public"
OUT = "/root/freqtrade/user_data/oi_buildup_20260906"
HORIZONS = [12, 48]
ARMS = {"4h_oi10": (48, 0.10), "1h_oi5": (12, 0.05)}


def one(sym):
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
        mt = pd.read_parquet(mf).set_index("date")[["sum_open_interest", "sum_toptrader_long_short_ratio"]]
        df = k5.join(mt, how="left")
        df = df[df.open.notna()]
        df["oi"] = df.sum_open_interest.ffill(limit=2)
        df["tt"] = df.sum_toptrader_long_short_ratio.ffill(limit=2)
        df["r4h"] = df.close / df.close.shift(48) - 1
        df["sig4h"] = df.r4h.shift(48).rolling(2016, min_periods=1000).std()
        qv30 = df.quote_volume.resample("1D").sum().rolling(30, min_periods=20).median().shift(1)
        df["qv30"] = qv30.reindex(df.index, method="ffill")
        df["dtt"] = df.tt - df.tt.shift(12)
        for h in HORIZONS:
            df[f"r{h}"] = df.open.shift(-1 - h) / df.open.shift(-1) - 1
        base_abs = {h: df[f"r{h}"].abs().mean() for h in HORIZONS}
        out = []
        for arm, (w, x) in ARMS.items():
            oc = df.oi / df.oi.shift(w) - 1
            c = df[(oc >= x) & (df.r4h.abs() <= 0.5 * df.sig4h) & df.dtt.notna() & (df.dtt != 0) & (df.qv30 >= 1e6)]
            keep, last = [], None
            for t in c.index:
                if last is None or (t - last) >= pd.Timedelta(hours=4):
                    keep.append(t); last = t
            c = c.loc[keep].copy()
            if c.empty:
                continue
            c["symbol"] = sym; c["arm"] = arm; c["dir"] = np.sign(c.dtt); c["oc"] = oc.loc[c.index]
            for h in HORIZONS:
                c[f"base_abs{h}"] = base_abs[h]
            out.append(c[["symbol", "arm", "dir", "oc", "qv30", "r4h", "sig4h"] + [f"r{h}" for h in HORIZONS] + [f"base_abs{h}" for h in HORIZONS]].reset_index())
        return pd.concat(out) if out else None
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
    syms = sorted(os.path.basename(f)[:-8] for f in glob.glob(f"{B}/metrics/*USDT.parquet"))
    with Pool(16) as p:
        parts = [x for x in p.map(one, syms, chunksize=4) if x is not None]
    ev = pd.concat(parts, ignore_index=True)
    ev = ev[ev.date >= pd.Timestamp("2025-01-08", tz="UTC")]
    ev["period"] = np.where(ev.date.dt.year == 2025, "2025_train", np.where(ev.date < pd.Timestamp("2026-06-01", tz="UTC"), "2026H1_valid", "2026Jun+_holdout"))
    ev["block"] = ev.date.dt.floor("4h").astype("int64")
    ev.to_parquet(f"{OUT}/events.parquet")
    pd.set_option("display.width", 250)
    for arm in ARMS:
        for qv_min, label in [(1e7, "qv30>=10M"), (1e6, "qv30>=1M")]:
            e = ev[(ev.arm == arm) & (ev.qv30 >= qv_min)]
            rows = []
            for h in HORIZONS:
                r = e[f"r{h}"].values; ok = np.isfinite(r)
                if ok.sum() < 20:
                    continue
                net = (e.dir.values * r * 1e4 - 20)[ok]
                ci, pv = cluster_boot(net, e.block.values[ok]); per = e.period.values[ok]
                rows.append(dict(h_min=5 * h, n=int(ok.sum()), long_share=(e.dir.values[ok] > 0).mean() * 100, mean=net.mean(), median=np.median(net), lo=ci[0], hi=ci[1], p=pv,
                                 train=net[per == "2025_train"].mean(), valid=net[per == "2026H1_valid"].mean(), holdout=net[per == "2026Jun+_holdout"].mean(),
                                 abs_move_bps=np.abs(r[ok]).mean() * 1e4, base_abs_bps=e[f"base_abs{h}"].values[ok].mean() * 1e4,
                                 uncond_long=(r[ok] * 1e4).mean()))
            print(f"\n=== {arm}, {label}: n events {len(e)}, symbols {e.symbol.nunique()}; direction = sign of 1h change in top-trader position ratio; net bps after 20bp")
            print(pd.DataFrame(rows).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
