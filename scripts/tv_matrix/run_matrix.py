"""Run the 4 groups x 6 timeframes matrix for one TradingView strategy port.

Every unit is an independent freqtrade backtest with its own config, export
archive and log. Nothing is aggregated across units except for the summary
table; zero-trade and failed units are kept in the output.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from freqtrade.data.btanalysis import load_backtest_data, load_backtest_stats

TIMEFRAMES = ["1d", "1h", "30m", "15m", "5m", "1m"]


def unit_metrics(zip_path: Path, group: str, timeframe: str) -> dict:
    stats = load_backtest_stats(str(zip_path))
    name = list(stats["strategy"])[0]
    s = stats["strategy"][name]
    trades = load_backtest_data(str(zip_path))

    row = {
        "group": group,
        "timeframe": timeframe,
        "pairs": len(s["pairlist"]),
        "start": s["backtest_start"],
        "end": s["backtest_end"],
        "trades": len(trades),
        "longs": int((~trades["is_short"]).sum()) if len(trades) else 0,
        "shorts": int(trades["is_short"].sum()) if len(trades) else 0,
        "rejected_signals": s.get("rejected_signals"),
        "final_balance": s["final_balance"],
        "net_profit_abs": s["profit_total_abs"],
        "net_profit_pct": s["profit_total"] * 100,
        "winrate_pct": None,
        "profit_factor": s.get("profit_factor"),
        "sharpe": s.get("sharpe"),
        "sortino": s.get("sortino"),
        "calmar": s.get("calmar"),
        "max_drawdown_pct": s.get("max_drawdown_account", 0) * 100,
        "avg_duration_min": None,
        "fees_abs": None,
        "funding_abs": None,
        "gross_profit_abs": None,
        "avg_trade_net": None,
        "avg_trade_gross": None,
        "exit_reasons": {},
        "forced_exits": 0,
        "open_at_end": 0,
        "stoploss_exits": 0,
        "liquidations": 0,
        "same_candle_flips": 0,
    }
    if len(trades):
        fees = (
            trades["open_rate"] * trades["amount"] * trades["fee_open"]
            + trades["close_rate"] * trades["amount"] * trades["fee_close"]
        )
        funding = trades["funding_fees"] if "funding_fees" in trades else pd.Series(0.0, index=trades.index)
        gross = trades["profit_abs"] + fees + funding
        counts = trades["exit_reason"].value_counts().to_dict()
        row.update(
            {
                "winrate_pct": float((trades["profit_abs"] > 0).mean() * 100),
                "avg_duration_min": float(trades["trade_duration"].mean()),
                "fees_abs": float(fees.sum()),
                "funding_abs": float(funding.sum()),
                "gross_profit_abs": float(gross.sum()),
                "avg_trade_net": float(trades["profit_abs"].mean()),
                "avg_trade_gross": float(gross.mean()),
                "exit_reasons": {k: int(v) for k, v in counts.items()},
                "forced_exits": int(counts.get("force_exit", 0)),
                "stoploss_exits": int(counts.get("stop_loss", 0)),
                "liquidations": int(counts.get("liquidation", 0)),
            }
        )
        # A reversal that freqtrade managed to execute inside one candle:
        # a close and the opposite open on the same pair at the same time.
        closes = trades[["pair", "close_date"]].rename(columns={"close_date": "t"})
        opens = trades[["pair", "open_date"]].rename(columns={"open_date": "t"})
        row["same_candle_flips"] = int(len(closes.merge(opens, on=["pair", "t"])))
        row["open_at_end"] = int(row["forced_exits"])

        per_pair = (
            trades.assign(fees=fees, funding=funding, gross=gross)
            .groupby("pair")
            .agg(
                trades=("profit_abs", "size"),
                net_abs=("profit_abs", "sum"),
                gross_abs=("gross", "sum"),
                fees_abs=("fees", "sum"),
                funding_abs=("funding", "sum"),
                winrate_pct=("profit_abs", lambda x: float((x > 0).mean() * 100)),
                avg_duration_min=("trade_duration", "mean"),
            )
            .reset_index()
        )
    else:
        per_pair = pd.DataFrame(
            columns=["pair", "trades", "net_abs", "gross_abs", "fees_abs",
                     "funding_abs", "winrate_pct", "avg_duration_min"]
        )
    return {"row": row, "per_pair": per_pair}


def run_unit(args_tuple) -> dict:
    group, pairs, timeframe, cfg_base, outdir, strategy, datadir, timerange, detail = args_tuple
    cfg = json.loads(Path(cfg_base).read_text())
    cfg["timeframe"] = timeframe
    cfg["exchange"]["pair_whitelist"] = pairs
    resdir = outdir / f"res_{group}_{timeframe}"
    resdir.mkdir(parents=True, exist_ok=True)
    cfg["export"] = "trades"
    # freqtrade 2026.1 names the archive itself; only the directory is ours.
    cfg["exportdirectory"] = str(resdir)
    cfg_path = outdir / f"config_{group}_{timeframe}.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))

    log = outdir / f"log_{group}_{timeframe}.txt"
    cmd = [
        "freqtrade", "backtesting", "-c", str(cfg_path), "-s", strategy,
        "--datadir", datadir, "--timeframe", timeframe, "--timerange", timerange,
        "--cache", "none", "--export", "trades",
    ]
    # Intra-candle execution detail. It never changes the signal timeframe -
    # it only decides where inside the candle a stop can be filled, so it is
    # only worth paying for once a strategy actually carries a stop.
    if detail and detail != timeframe:
        cmd += ["--timeframe-detail", detail]
    # Every unit starts by loading exchange markets over the same local proxy.
    # Firing all workers at once makes that call fail with ExchangeNotAvailable,
    # which has nothing to do with the backtest itself - so stagger the starts
    # and retry a failed unit a couple of times before recording it as FAILED.
    time.sleep(random.uniform(0.0, 8.0))
    for attempt in range(3):
        proc = subprocess.run(cmd, capture_output=True, text=True)
        log.write_text(
            f"=== attempt {attempt + 1} ===\n" + proc.stdout + "\n===STDERR===\n" + proc.stderr
        )
        if proc.returncode == 0:
            break
        time.sleep(20.0 * (attempt + 1) + random.uniform(0.0, 10.0))

    zips = sorted(resdir.glob("*.zip"))
    if proc.returncode != 0 or not zips:
        return {
            "row": {"group": group, "timeframe": timeframe, "status": "FAILED",
                    "pairs": len(pairs), "trades": 0},
            "per_pair": pd.DataFrame(),
        }
    res = unit_metrics(max(zips, key=lambda p: p.stat().st_mtime), group, timeframe)
    res["row"]["status"] = "ok"
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="TrendShiftSignal")
    ap.add_argument("--universe", default="user_data/research/tradingview/universe_2026-07.json")
    ap.add_argument("--config", default="user_data/research/tradingview/config_base.json")
    ap.add_argument("--datadir", default="user_data/data/tv_matrix")
    ap.add_argument("--timerange", default="20260701-20260801")
    ap.add_argument("--outdir", default="user_data/research/tradingview/matrix_trendshift_2026-07")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeframes", default=",".join(TIMEFRAMES))
    ap.add_argument("--detail", default="")
    args = ap.parse_args()

    uni = json.loads(Path(args.universe).read_text())
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    timeframes = [tf.strip() for tf in args.timeframes.split(",") if tf.strip()]
    jobs = [
        (group, pairs, tf, args.config, outdir, args.strategy, args.datadir,
         args.timerange, args.detail)
        for group, pairs in uni["groups"].items()
        for tf in timeframes
    ]
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(run_unit, jobs):
            results.append(res)
            r = res["row"]
            print(f"{r['group']:9s} {r['timeframe']:3s} status={r.get('status')} "
                  f"trades={r.get('trades')} net={r.get('net_profit_abs')}", flush=True)

    summary = pd.DataFrame([r["row"] for r in results])
    summary.to_csv(outdir / "summary.csv", index=False)
    per_pair = pd.concat(
        [r["per_pair"].assign(group=r["row"]["group"], timeframe=r["row"]["timeframe"])
         for r in results if len(r["per_pair"])],
        ignore_index=True,
    ) if any(len(r["per_pair"]) for r in results) else pd.DataFrame()
    per_pair.to_csv(outdir / "per_pair.csv", index=False)
    print(f"\nwrote {outdir/'summary.csv'} and {outdir/'per_pair.csv'}")


if __name__ == "__main__":
    main()
