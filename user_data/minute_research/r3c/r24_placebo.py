"""R24 matched placebo test.

For every real R24 trade (coin c, settlement T), placebo shorts are taken at settlements of the SAME coin within +-30 days
where the R24 condition is false; execution identical to R24 (short at the 5m open T+5 min, exit at the open 480 min later,
10 bp/side, funding settled in (entry, exit] credited to the short). Permutation: one placebo per real trade, 2000 draws;
report where the real mean sits in the placebo-mean distribution, per segment.
"""
import numpy as np, pandas as pd
import r24_metrics as M

DD = "/root/freqtrade/user_data/data/r3b/futures"
SEG = {"TRAIN": ("../r3/bt_r24_train", "2025-01-01", "2025-10-01"), "VALID-C": ("bt_r24_validC", "2025-12-01", "2026-03-01"),
       "HOLDOUT": ("../r3/bt_r24_holdout", "2026-03-01", "2026-09-01")}
rng = np.random.default_rng(7)
cache = {}


def coin(base):
    if base not in cache:
        k = pd.read_feather(f"{DD}/{base}_USDT_USDT-5m-futures.feather", columns=["date", "open"]).set_index("date").open
        f = pd.read_feather(f"{DD}/{base}_USDT_USDT-1h-funding_rate.feather", columns=["date", "open"]).set_index("date").open
        cond = (f >= 3e-4) & (f.shift(1) >= 3e-4) & (f.shift(2) >= 3e-4)
        cache[base] = (k, f, cond)
    return cache[base]


def short_ret(k, f, T):
    e, x = T + pd.Timedelta(minutes=5), T + pd.Timedelta(minutes=485)
    if e not in k.index or x not in k.index:
        return np.nan
    fund = f[(f.index > e) & (f.index <= x)].sum()
    return -(k[x] / k[e] - 1) + fund - 20e-4


def main():
    for seg, (d, a, b) in SEG.items():
        t = M.load(d); t = t[(t.t >= a) & (t.t < b)]
        real, pools = [], []
        for r in t.itertuples():
            k, f, cond = coin(r.base); T = pd.Timestamp(r.open_date).tz_convert("UTC") - pd.Timedelta(minutes=5)
            real.append(short_ret(k, f, T))
            cand = f.index[(~cond) & (f.index >= T - pd.Timedelta(days=30)) & (f.index <= T + pd.Timedelta(days=30))]
            pr = np.array([short_ret(k, f, c) for c in cand]); pr = pr[~np.isnan(pr)]
            pools.append(pr)
        real = np.array(real); ok = ~np.isnan(real) & np.array([len(p) > 0 for p in pools])
        real = real[ok]; pools = [p for p, o in zip(pools, ok) if o]
        draws = np.array([[p[rng.integers(len(p))] for p in pools] for _ in range(2000)]).mean(axis=1)
        pool_mean = np.mean([p.mean() for p in pools])
        print(f"{seg:8s} trades {len(real):4d} | R24 mean {real.mean()*1e4:7.1f} bp | matched placebo mean {pool_mean*1e4:7.1f} bp "
              f"| edge {(real.mean()-pool_mean)*1e4:7.1f} bp | placebo draws >= R24: {(draws >= real.mean()).mean():.3f} "
              f"| placebo 95% range {np.percentile(draws,2.5)*1e4:.0f}..{np.percentile(draws,97.5)*1e4:.0f} bp")


if __name__ == "__main__":
    main()
