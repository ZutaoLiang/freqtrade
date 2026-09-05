"""Freeze the four 20-coin research groups for the TradingView matrix.

Selection window: the 30 complete UTC days preceding the backtest start.
No backtest-month data is used for selection.

Rules (skills/tradingview/SKILL.md section 3):
  * majors      - top 20 by median daily quote volume
  * high vol    - top 20 by median daily TR / prev close, majors excluded
  * low vol     - bottom 20 by the same metric, majors and high-vol excluded
  * mixed       - majors[:7] + high[:7] + low[:6], deduplicated

Engine eligibility (pre-declared): a pair must appear in freqtrade's bundled
Binance leverage-tier table, otherwise the futures backtest cannot run at all.

Pre-declared extra threshold (fixed before any backtest is run): the
volatility pools require a median daily quote volume of at least 1,000,000
USDT, so the low-volatility group cannot fill up with untradeable names.
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather

DATA = Path("user_data/data/binance_public/freqtrade/futures")
# Engine eligibility: freqtrade refuses to backtest a futures pair that has no
# leverage tiers, so a pair missing from the bundled Binance table cannot be a
# candidate. Applied before any backtest is run, never after seeing results.
LEVERAGE_TIERS = set(
    json.loads(Path("freqtrade/exchange/binance_leverage_tiers.json").read_text())
)
MIN_QUOTE_VOL = 1_000_000.0
STABLE_BASES = {
    "USDC", "FDUSD", "TUSD", "BUSD", "DAI", "USDP", "USDD", "EUR", "USTC",
    "AEUR", "XUSD", "USD1", "USDE", "SUSDE",
}
LEV_TOKEN = re.compile(r"(UP|DOWN|BULL|BEAR)$")


def pair_name(symbol_file: str) -> str:
    base = symbol_file.split("_USDT_USDT")[0]
    return f"{base}/USDT:USDT"


def scan_one(args) -> dict | None:
    stem, win_start, win_end, need_from, need_to = args
    base = stem.split("_USDT_USDT")[0]
    if base in STABLE_BASES or LEV_TOKEN.search(base):
        return None
    if pair_name(stem) not in LEVERAGE_TIERS:
        return None

    daily_path = DATA / f"{stem}_USDT_USDT-1d-futures.feather"
    minute_path = DATA / f"{stem}_USDT_USDT-1m-futures.feather"
    if not daily_path.exists() or not minute_path.exists():
        return None

    d = feather.read_feather(daily_path, columns=["date", "open", "high", "low", "close", "volume"])
    d = d.sort_values("date")
    # Coverage: warmup start through the end of the backtest month.
    if d["date"].min() > need_from or d["date"].max() < need_to:
        return None
    if (d[["open", "high", "low", "close"]] <= 0).any().any():
        return None
    if (d["high"] < d["low"]).any():
        return None

    win = d[(d["date"] >= win_start) & (d["date"] <= win_end)].copy()
    if len(win) < 30:
        return None
    prev_close = d["close"].shift(1)
    tr = pd.concat(
        [
            d["high"] - d["low"],
            (d["high"] - prev_close).abs(),
            (d["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    tr_ratio = (tr / prev_close)[(d["date"] >= win_start) & (d["date"] <= win_end)]
    if tr_ratio.isna().any():
        return None

    m = feather.read_feather(minute_path, columns=["date", "close", "volume"])
    m = m[(m["date"] >= win_start) & (m["date"] < win_end + pd.Timedelta(days=1))]
    if m.empty:
        return None
    quote = m["close"] * m["volume"]
    daily_quote = quote.groupby(m["date"].dt.floor("D")).sum()
    if len(daily_quote) < 30 or (daily_quote <= 0).any():
        return None

    return {
        "pair": pair_name(stem),
        "stem": stem,
        "quote_vol_median": float(daily_quote.median()),
        "tr_ratio_median": float(tr_ratio.median()),
        "daily_bars": int(len(win)),
        "minute_bars": int(len(m)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bt-start", default="2026-07-01")
    ap.add_argument("--bt-end", default="2026-08-01")
    ap.add_argument("--warmup-days", type=int, default=100)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="user_data/research/tradingview/universe_2026-07.json")
    args = ap.parse_args()

    bt_start = pd.Timestamp(args.bt_start, tz="UTC")
    bt_end = pd.Timestamp(args.bt_end, tz="UTC")
    win_end = bt_start - pd.Timedelta(days=1)
    win_start = bt_start - pd.Timedelta(days=30)
    need_from = bt_start - pd.Timedelta(days=args.warmup_days)
    need_to = bt_end - pd.Timedelta(days=1)

    stems = sorted(
        p.name.split("_USDT_USDT")[0]
        for p in DATA.glob("*_USDT_USDT-1d-futures.feather")
    )
    jobs = [(s, win_start, win_end, need_from, need_to) for s in stems]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(scan_one, jobs, chunksize=4):
            if res:
                rows.append(res)

    df = pd.DataFrame(rows)
    df.to_csv(Path(args.out).with_suffix(".ranking.csv"), index=False)

    majors = df.sort_values("quote_vol_median", ascending=False).head(20)
    rest = df[~df["pair"].isin(majors["pair"])]
    liquid = rest[rest["quote_vol_median"] >= MIN_QUOTE_VOL]
    high = liquid.sort_values("tr_ratio_median", ascending=False).head(20)
    low_pool = liquid[~liquid["pair"].isin(high["pair"])]
    low = low_pool.sort_values("tr_ratio_median", ascending=True).head(20)

    mixed = list(dict.fromkeys(
        list(majors["pair"][:7]) + list(high["pair"][:7]) + list(low["pair"][:6])
    ))

    out = {
        "backtest_start": args.bt_start,
        "backtest_end": args.bt_end,
        "selection_window": [str(win_start.date()), str(win_end.date())],
        "warmup_days": args.warmup_days,
        "min_quote_volume_for_vol_groups": MIN_QUOTE_VOL,
        "candidates_scanned": len(stems),
        "candidates_qualified": len(df),
        "groups": {
            "majors": list(majors["pair"]),
            "high_vol": list(high["pair"]),
            "low_vol": list(low["pair"]),
            "mixed": mixed,
        },
        "metrics": {
            r["pair"]: {
                "quote_vol_median": r["quote_vol_median"],
                "tr_ratio_median": r["tr_ratio_median"],
            }
            for r in rows
        },
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"qualified {len(df)} / scanned {len(stems)}")
    for name, pairs in out["groups"].items():
        print(f"{name:9s} n={len(pairs)} :: {', '.join(p.split('/')[0] for p in pairs)}")


if __name__ == "__main__":
    main()
