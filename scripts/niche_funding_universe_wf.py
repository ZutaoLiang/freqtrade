"""Which universe actually delivers, once the portfolio constraint bites.

Per-event mean says liquid pairs are better, but the strategy can hold one trade per pair
and only `CAP` at a time, so a band that looks good per event can be unreachable. This
walks quarterly folds over the full 2025-01..2026-08 range and reports each candidate
universe under that constraint. Universe membership is decided by the pair's TRAILING
30-day volume at the event, so it is a live rule, not hindsight.
"""
import numpy as np
import pandas as pd

EVENTS = "/root/freqtrade/user_data/niche_work/universe_events.parquet"
OUT = "/root/freqtrade/user_data/niche_work/universe_wf.csv"
CAP = 6
COL = "h235_s15"
HOLD_MIN = 236
UNIVERSES = {
    "3M-60M (original band)": (3e6, 6e7),
    ">=3M  no upper limit": (3e6, np.inf),
    ">=10M no upper limit": (1e7, np.inf),
    ">=30M no upper limit": (3e7, np.inf),
    ">=60M no upper limit": (6e7, np.inf),
    "everything with volume": (0.0, np.inf),
}
RULE = dict(fr=40.0, streak=1)


def portfolio(df, cap=CAP, hold=HOLD_MIN):
    open_until, keep = {}, []
    for r in df.sort_values("date").itertuples():
        for k in [k for k, v in open_until.items() if v <= r.date]:
            del open_until[k]
        if len(open_until) >= cap or r.sym in open_until:
            continue
        open_until[r.sym] = r.date + pd.Timedelta(minutes=hold)
        keep.append(r.Index)
    return df.loc[keep]


def main():
    d = pd.read_parquet(EVENTS)
    d = d[d.trail_qv.notna()].reset_index(drop=True)
    d = d[(d.fr_bps.abs() >= RULE["fr"]) & (d[f"streak_{RULE['fr']:g}"] >= RULE["streak"])]
    quarters = sorted(pd.PeriodIndex(d.date, freq="Q").astype(str).unique())

    rows = []
    for name, (lo, hi) in UNIVERSES.items():
        u = d[(d.trail_qv >= lo) & (d.trail_qv < hi)].reset_index(drop=True)
        t = portfolio(u)
        t = t.assign(q=pd.PeriodIndex(t.date, freq="Q").astype(str))
        by_q = t.groupby("q")[COL].agg(n="size", tot="sum", mean="mean")
        rows.append(dict(universe=name, pairs=t.sym.nunique(), n=len(t),
                         tot=t[COL].sum(), mean_bps=1e4 * t[COL].mean(),
                         pos_q=int((by_q.tot > 0).sum()), n_q=len(by_q),
                         worst_q=100 * by_q.tot.min(),
                         **{f"q_{q}": 100 * by_q.tot.get(q, np.nan) for q in quarters}))
        print(f"{name:24s} pairs={t.sym.nunique():3d} trades={len(t):5d} "
              f"total={100*t[COL].sum():+8.1f}%  mean={1e4*t[COL].mean():+6.1f}bps  "
              f"quarters positive {int((by_q.tot>0).sum())}/{len(by_q)}  worst q {100*by_q.tot.min():+7.1f}%")

    r = pd.DataFrame(rows)
    r.to_csv(OUT, index=False)
    print("\nquarterly total, % of one stake:")
    print(r.set_index("universe")[[f"q_{q}" for q in quarters]].round(0).to_string())


if __name__ == "__main__":
    main()
