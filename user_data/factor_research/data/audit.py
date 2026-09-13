"""Data quality report over the resampled panel.

Answers the question phase 4 depends on: how many pairs, over how long, with
how few gaps, are actually usable -- and therefore how many factors can be
swept before the survivors are just noise.
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import MAX_WORKERS  # noqa: E402

RICH = "/root/freqtrade/user_data/data/binance_public/resampled"
OUT = "/root/freqtrade/user_data/data/binance_public/reports"
TF = os.environ.get("AUDIT_TF", "1h")
STEP = pd.Timedelta(TF.replace("m", "min").replace("d", "D"))


def check(path):
    sym = os.path.basename(path)[:-8]
    df = pd.read_parquet(path, columns=["date", "open", "high", "low", "close",
                                        "volume", "quote_volume"])
    if df.empty:
        return {"sym": sym, "bars": 0}
    d = df["date"]
    span = (d.iloc[-1] - d.iloc[0]) / STEP + 1
    gaps = d.diff().dropna()
    return {
        "sym": sym,
        "bars": len(df),
        "start": d.iloc[0],
        "end": d.iloc[-1],
        "days": (d.iloc[-1] - d.iloc[0]).total_seconds() / 86400,
        "completeness": len(df) / span,
        "n_gaps": int((gaps > STEP).sum()),
        "max_gap_bars": int(gaps.max() / STEP) if len(gaps) else 0,
        "zero_vol_pct": float((df["volume"] == 0).mean() * 100),
        "median_quote_vol": float(df["quote_volume"].median()),
        "nan_pct": float(df[["open", "high", "low", "close"]].isna().any(axis=1).mean() * 100),
    }


if __name__ == "__main__":
    src = os.path.join(RICH, TF)
    paths = sorted(os.path.join(src, f) for f in os.listdir(src) if f.endswith(".parquet"))
    print(f"auditing {len(paths)} symbols at {TF} ...", flush=True)
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        rows = list(ex.map(check, paths))
    df = pd.DataFrame(rows).sort_values("bars", ascending=False)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, f"quality_{TF}.csv"), index=False)

    live = df[df.bars > 0]
    print(f"\n=== {TF} universe ===")
    print(f"symbols with data      : {len(live)} / {len(df)}")
    print(f"bars  min/median/max   : {live.bars.min()} / {int(live.bars.median())} / {live.bars.max()}")
    print(f"range                  : {live.start.min()}  ->  {live.end.max()}")
    print(f"median completeness    : {live.completeness.median():.4f}")
    print(f"median zero-volume pct : {live.zero_vol_pct.median():.2f}%")
    print("\ncoverage tiers (calendar days of history):")
    for d in (30, 90, 180, 365, 540):
        print(f"  >= {d:4d}d : {(live.days >= d).sum():4d} symbols")
    print("\nliquidity tiers (median quote volume per bar, USDT):")
    for v in (1e4, 1e5, 1e6, 1e7):
        print(f"  >= {v:>10,.0f} : {(live.median_quote_vol >= v).sum():4d} symbols")
    print("\nworst 10 by completeness:")
    print(live.nsmallest(10, "completeness")[
        ["sym", "bars", "completeness", "n_gaps", "max_gap_bars"]].to_string(index=False))
