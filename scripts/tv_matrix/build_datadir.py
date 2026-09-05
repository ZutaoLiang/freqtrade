"""Assemble a dedicated freqtrade datadir for the TradingView matrix.

OHLCV files are symlinked from the public archive; funding rate and mark price
files are converted from the raw parquet archives into the layout freqtrade
expects (both at the 1h timeframe that the Binance exchange class asks for,
inner-merged on the settlement timestamps).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

SRC_OHLCV = Path("user_data/data/binance_public/freqtrade/futures")
SRC_FUNDING = Path("user_data/data/binance_public/funding")
SRC_MARK = Path("user_data/data/binance_public/markprice_1m")
TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "1d"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="user_data/research/tradingview/universe_2026-07.json")
    ap.add_argument("--out", default="user_data/data/tv_matrix/futures")
    args = ap.parse_args()

    uni = json.loads(Path(args.universe).read_text())
    pairs = sorted({p for g in uni["groups"].values() for p in g})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    report = {"pairs": len(pairs), "missing_funding": [], "missing_mark": [], "intervals": {}}
    for pair in pairs:
        base = pair.split("/")[0]
        stem = f"{base}_USDT_USDT"
        for tf in TIMEFRAMES:
            src = SRC_OHLCV / f"{stem}-{tf}-futures.feather"
            dst = out / f"{stem}-{tf}-futures.feather"
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(src.resolve())

        fsrc = SRC_FUNDING / f"{base}USDT.parquet"
        if not fsrc.exists():
            report["missing_funding"].append(pair)
        else:
            f = pd.read_parquet(fsrc).sort_values("date")
            report["intervals"][pair] = sorted(
                set(f["funding_interval_hours"].dropna().unique().tolist())
            )
            fr = pd.DataFrame(
                {
                    "date": f["date"],
                    "open": f["funding_rate"].astype("float64"),
                    "high": f["funding_rate"].astype("float64"),
                    "low": f["funding_rate"].astype("float64"),
                    "close": f["funding_rate"].astype("float64"),
                    "volume": 0.0,
                }
            )
            # Settlement stamps must sit on whole hours for the inner merge.
            fr = fr[fr["date"].dt.minute.eq(0) & fr["date"].dt.second.eq(0)]
            fr.reset_index(drop=True).to_feather(
                out / f"{stem}-1h-funding_rate.feather", compression="lz4"
            )

        msrc = SRC_MARK / f"{base}USDT.parquet"
        if msrc.exists():
            m = pd.read_parquet(msrc).sort_values("date").set_index("date")
            agg = m.resample("1h").agg(
                {"mark_open": "first", "mark_high": "max", "mark_low": "min", "mark_close": "last"}
            ).dropna()
        else:
            # Documented approximation: no mark archive for this pair, so the
            # pair's own last price stands in. Basis error on a perp is far
            # smaller than dropping funding fees to zero would be.
            report["missing_mark"].append(pair)
            m = pd.read_feather(SRC_OHLCV / f"{stem}-1m-futures.feather").set_index("date")
            agg = m.resample("1h").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last"}
            ).dropna()
            agg.columns = ["mark_open", "mark_high", "mark_low", "mark_close"]
        if True:
            mk = pd.DataFrame(
                {
                    "date": agg.index,
                    "open": agg["mark_open"].astype("float64"),
                    "high": agg["mark_high"].astype("float64"),
                    "low": agg["mark_low"].astype("float64"),
                    "close": agg["mark_close"].astype("float64"),
                    "volume": 0.0,
                }
            )
            mk.reset_index(drop=True).to_feather(
                out / f"{stem}-1h-mark.feather", compression="lz4"
            )

    Path(args.universe).with_suffix(".datadir_report.json").write_text(
        json.dumps(report, indent=2)
    )
    odd = {p: v for p, v in report["intervals"].items() if v != [8.0]}
    print(f"pairs={report['pairs']} missing_funding={len(report['missing_funding'])} "
          f"missing_mark={len(report['missing_mark'])}")
    print(f"non-8h funding intervals: {odd}")


if __name__ == "__main__":
    main()
