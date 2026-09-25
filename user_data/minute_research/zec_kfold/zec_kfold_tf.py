"""Timeframe-generic version of zec_kfold.py (see PREREGISTRATION.md addendum)."""
import itertools, sys
import numpy as np, pandas as pd
import zec_kfold as Z

TF_MIN = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60}
TARGETS = (0.5, 1.0, 2.0, 4.0)


def bars(sym, tf):
    k = pd.read_parquet(f"{Z.R}/klines_1m/{sym}.parquet").set_index("date")
    rule = f"{TF_MIN[tf]}min"
    h = k.resample(rule).agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
                              "quote_volume": "sum", "count": "sum", "taker_buy_quote_volume": "sum"}).dropna()
    return h[h.index < "2026-09-01"]


def build(tf):
    h = bars(Z.SYM, tf); step = pd.Timedelta(minutes=TF_MIN[tf])
    h["btc"] = bars("BTCUSDT", tf).close.reindex(h.index).ffill()
    f = pd.read_parquet(f"{Z.R}/funding/{Z.SYM}.parquet"); f["date"] = f.date.dt.round("h")
    f = f.drop_duplicates("date", keep="last").set_index("date").funding_rate
    h["fr_settle"] = f.reindex(h.index).fillna(0.0)
    close_t = h.index + step
    h["fr_last"] = f.reindex(close_t, method="ffill").to_numpy()
    h["fr_mean3"] = f.rolling(3).mean().reindex(close_t, method="ffill").to_numpy()
    m = pd.read_parquet(f"{Z.R}/markprice_1m/{Z.SYM}.parquet").set_index("date").mark_close
    h["mark"] = m.reindex(close_t - pd.Timedelta(minutes=1), method="ffill").to_numpy()   # mark of the bar's last minute
    o = pd.read_parquet(f"{Z.R}/metrics/{Z.SYM}.parquet").set_index("date").sort_index()
    o = o[~o.index.duplicated(keep="last")]
    for c in ("sum_open_interest_value", "sum_toptrader_long_short_ratio", "count_long_short_ratio"):
        h[c] = o[c].reindex(close_t, method="ffill").to_numpy()                          # last snapshot at or before close
    return h


def main():
    tfs = sys.argv[1:] or ["30m", "15m", "5m", "1m"]
    out = []
    for tf in tfs:
        h = build(tf); S = Z.families(h); bpd = 1440 / TF_MIN[tf]
        qs = [round(1 - t / bpd, 6) for t in TARGETS]
        combos, S2 = [], {}
        for f, (s, hold, lo) in S.items():
            for mode, hb in (("bars", hold), ("clock", int(hold * 60 / TF_MIN[tf]))):
                S2[f"{f}|{mode}"] = (s, hb, lo)
        win, mn = int(90 * bpd), int(10 * bpd)
        orig_signal = Z.signal
        def sig(s, q, lo, win=win, mn=mn):
            nz = s.where(s != 0)
            up = nz.rolling(win, min_periods=mn).quantile(q).shift(1); low = nz.rolling(win, min_periods=mn).quantile(1 - q).shift(1)
            x = np.where(s >= np.maximum(up, 0), 1, 0)
            if not lo: x = np.where(s <= np.minimum(low, 0), -1, x)
            return np.where(np.isnan(up) | s.isna(), 0, x).astype(np.int8)
        Z.signal = sig
        combos = [(k, q) for k in S2 for q in qs]
        fl = Z.fold_label(h.index); idx = h.index
        ent, sd, ret, cnt = Z.run(combos, h, S2, Z.COST); _, _, ret15, _ = Z.run(combos, h, S2, 15.0)
        Z.signal = orig_signal
        active = sum((pd.Timestamp(b) - pd.Timestamp(a)).days for _, a, b in Z.FOLDS)
        rows, daily = [], {}
        drift_cache = {}
        for k, (f, q) in enumerate(combos):
            e = ent[k, :cnt[k]]; s = sd[k, :cnt[k]]; r = ret[k, :cnt[k]]; r15 = ret15[k, :cnt[k]]
            lab = fl[e]; keep = lab != ""; e, s, r, r15, lab = e[keep], s[keep], r[keep], r15[keep], lab[keep]
            day = pd.Series(idx[e].floor("D")); st = Z.stats(r, day)
            hb = S2[f][1]
            if hb not in drift_cache:
                dr = Z.drift(h, hb).to_numpy(); drift_cache[hb] = {L: np.nanmean(dr[fl == L]) for L in ("F1", "F2", "F3", "F4", "F5", "F6")}
            adj = r - s * np.array([drift_cache[hb][L] for L in lab]) if len(r) else r
            rows.append(dict(tf=tf, family=f, q=q, hold_bars=hb, **st, per_day=len(r) / active,
                             folds_pos=sum(r[lab == L].sum() > 0 for L in ("F1", "F2", "F3", "F4", "F5", "F6")),
                             mean15_bp=r15.mean() * 1e4 if len(r15) else np.nan, adj_bp=adj.mean() * 1e4 if len(adj) else np.nan))
            daily[(f, q)] = (pd.Series(r).groupby(day.to_numpy()).sum(), pd.Series(lab).groupby(day.to_numpy()).first())
        df = pd.DataFrame(rows)
        df["pass_rel"] = (df.mean_bp > 0) & (df.pf >= 1.2) & (df.t >= 2.0) & (df.folds_pos >= 4) & (df.per_day >= 0.4) & (df.mean15_bp > 0) & (df.adj_bp > 0)
        df["pass_std"] = df.pass_rel & (df.per_day >= 1.0)
        pbo = Z.cscv_pbo(daily)
        print(f"== {tf}: combos {len(df)} | PBO {pbo:.2f} | pass relaxed {int(df.pass_rel.sum())} | standard {int(df.pass_std.sum())} | "
              f"median mean_bp {df.mean_bp.median():.1f} | share mean>0 {(df.mean_bp > 0).mean():.0%}", flush=True)
        cols = ["tf", "family", "q", "hold_bars", "n", "per_day", "mean_bp", "pf", "t", "folds_pos", "mean15_bp", "adj_bp"]
        print(df.sort_values("t", ascending=False)[cols].head(6).round(3).to_string(index=False), flush=True)
        df.to_csv(f"screen_folds_{tf}.csv", index=False); out.append(df)
    pd.concat(out).to_csv("screen_folds_alltf.csv", index=False)


if __name__ == "__main__":
    main()
