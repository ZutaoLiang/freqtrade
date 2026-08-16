#!/usr/bin/env python3
"""Backtest each 2026 calendar month independently at 1x and 2% risk.

The four monthly refresh anchors are run serially.  Every calendar-month run
starts with a fresh 500 USDT wallet and is exported separately so that sizing,
drawdown and end-of-window forced exits can be audited rather than inferred
from a continuously compounded backtest.
"""

from __future__ import annotations

from collections import defaultdict
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any
import zipfile

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "user_data/analysis/monthly-independent-riskbudget2pct-2026"
SELECTION_ROOT = (
    REPO / "user_data/analysis/monthly-incremental-top2-2026/selections"
)
DATADIR = (
    REPO / "user_data/data/binance-monthly-incremental-top2-2026-30m-windowed"
)
BASE_CONFIG = REPO / "user_data/config_high_volume_mainwave_v18_2026_02_08.json"
STRATEGY = "HighVolumeMainWaveMonthlyOffsetRiskBudget2PctV23"
ANCHORS = (1, 8, 15, 22)
MONTHS = (
    ("2026-02", "20260201-20260301"),
    ("2026-03", "20260301-20260401"),
    ("2026-04", "20260401-20260501"),
    ("2026-05", "20260501-20260601"),
    ("2026-06", "20260601-20260701"),
    ("2026-07", "20260701-20260801"),
    ("2026-08", "20260801-20260814"),
)
PARAMETERS = {
    "pretrend_short_minimum_rvol": 999.0,
    "persistent_mainwave_long_multiplier": 2.0,
    "strong_short_stake_multiplier": 1.75,
}
# The strategy itself hard-gates entries at four actual positions.  A larger
# engine setting keeps its pre-custom-stake Binance tier lookup below the
# recent-contract minimum tier limit when unlimited stake is in use.
ENGINE_MAX_OPEN_TRADES = 100
WALLET = 500
MINIMUM_AVAILABLE_KIB = 640 * 1024


def available_kib() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    raise RuntimeError("MemAvailable is unavailable")


def bounds(timerange: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_text, end_text = timerange.split("-")
    return (
        pd.Timestamp(start_text, tz="UTC"),
        pd.Timestamp(end_text, tz="UTC"),
    )


def selected_pairs(selection: Path, timerange: str) -> list[str]:
    start, end = bounds(timerange)
    payload = json.loads(selection.read_text())
    symbols = {
        symbol
        for period in payload["periods"]
        if pd.Timestamp(period["start"]) < end
        and pd.Timestamp(period["end_exclusive"]) > start
        for symbol in period["selected_symbols"]
    }
    return sorted(
        f"{symbol.removesuffix('USDT')}/USDT:USDT" for symbol in symbols
    )


def load_result(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        name = next(
            item
            for item in archive.namelist()
            if item.endswith(".json") and not item.endswith("_config.json")
        )
        payload = json.loads(archive.read(name))
    return payload["strategy"][STRATEGY]


def maximum_rss(log: Path) -> int:
    match = re.search(
        r"Maximum resident set size \(kbytes\): (\d+)",
        log.read_text(errors="replace"),
    )
    if not match:
        raise RuntimeError(f"Maximum RSS is absent from {log}")
    return int(match.group(1))


def maximum_concurrency(trades: list[dict[str, Any]]) -> int:
    opened: dict[int, int] = defaultdict(int)
    closed: dict[int, int] = defaultdict(int)
    for trade in trades:
        opened[int(trade["open_timestamp"])] += 1
        closed[int(trade["close_timestamp"])] += 1
    concurrent = 0
    maximum = 0
    for timestamp in sorted(set(opened) | set(closed)):
        # A trade closing on a candle frees its slot before a same-candle entry.
        concurrent -= closed[timestamp]
        concurrent += opened[timestamp]
        maximum = max(maximum, concurrent)
    return maximum


def summarize(
    anchor: int,
    month: str,
    timerange: str,
    artifact: Path,
    log: Path,
    selection: Path,
) -> dict[str, Any]:
    result = load_result(artifact)
    trades = result["trades"]
    forced = [trade for trade in trades if trade["exit_reason"] == "force_exit"]
    liquidations = [
        trade for trade in trades if trade["exit_reason"] == "liquidation"
    ]
    leverages = sorted({float(trade["leverage"]) for trade in trades})
    return {
        "anchor_day": anchor,
        "month": month,
        "timerange": timerange,
        "selection": str(selection.relative_to(REPO)),
        "union_pairs": len(selected_pairs(selection, timerange)),
        "trades": int(result["total_trades"]),
        "long_trades": int(result["trade_count_long"]),
        "short_trades": int(result["trade_count_short"]),
        "wins": int(result["wins"]),
        "draws": int(result["draws"]),
        "losses": int(result["losses"]),
        "winrate_pct": float(result["winrate"]) * 100,
        "profit_abs": float(result["profit_total_abs"]),
        "profit_pct": float(result["profit_total"]) * 100,
        "final_balance": float(result["final_balance"]),
        "profit_factor": float(result["profit_factor"]),
        "max_relative_drawdown_pct": float(result["max_relative_drawdown"]) * 100,
        "long_profit_abs": float(result["profit_total_long_abs"]),
        "short_profit_abs": float(result["profit_total_short_abs"]),
        "average_stake": float(result["avg_stake_amount"]),
        "total_volume": float(result["total_volume"]),
        "forced_exit_trades": len(forced),
        "forced_exit_profit_abs": sum(float(trade["profit_abs"]) for trade in forced),
        "liquidation_trades": len(liquidations),
        "actual_max_concurrent": maximum_concurrency(trades),
        "leverage_values": leverages,
        "maximum_rss_kib": maximum_rss(log),
        "artifact": str(artifact.relative_to(REPO)),
    }


def run(anchor: int, month: str, timerange: str) -> dict[str, Any]:
    anchor_name = f"offset-{anchor:02d}"
    run_name = f"{anchor_name}-{month}"
    selection = SELECTION_ROOT / f"{anchor_name}.json"
    pairs = selected_pairs(selection, timerange)
    artifact_dir = OUTPUT / "artifacts" / anchor_name / month
    log = OUTPUT / "logs" / anchor_name / f"{month}.log"
    config = OUTPUT / "configs" / anchor_name / f"{month}.json"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    config.parent.mkdir(parents=True, exist_ok=True)
    existing = next(iter(sorted(artifact_dir.glob("*.zip"))), None)
    if existing is None:
        free_kib = available_kib()
        if free_kib < MINIMUM_AVAILABLE_KIB:
            raise RuntimeError(
                f"MemAvailable is only {free_kib / 1024:.0f} MiB; "
                "refusing to start another backtest"
            )
        overlay = {
            "strategy": STRATEGY,
            "max_open_trades": ENGINE_MAX_OPEN_TRADES,
            "dry_run_wallet": WALLET,
            "v20_refit_parameters": PARAMETERS,
            "v20_monthly_offset_selection": str(
                selection.relative_to(REPO / "user_data/analysis")
            ),
            "bot_name": f"v20-independent-{run_name}",
        }
        config.write_text(json.dumps(overlay, ensure_ascii=False, indent=2) + "\n")
        command = [
            "/usr/bin/time",
            "-v",
            "env",
            "MALLOC_ARENA_MAX=2",
            "MALLOC_TRIM_THRESHOLD_=131072",
            "nice",
            "-n",
            "10",
            "python3",
            "-m",
            "freqtrade",
            "backtesting",
            "--config",
            str(BASE_CONFIG.relative_to(REPO)),
            "--config",
            str(config.relative_to(REPO)),
            "--datadir",
            str(DATADIR.relative_to(REPO)),
            "--timerange",
            timerange,
            "--max-open-trades",
            str(ENGINE_MAX_OPEN_TRADES),
            "--dry-run-wallet",
            str(WALLET),
            "--fee",
            "0.0007",
            "--cache",
            "none",
            "--export",
            "trades",
            "--backtest-directory",
            str(artifact_dir.relative_to(REPO)),
            "--pairs",
            *pairs,
        ]
        print(
            f"START {run_name} pairs={len(pairs)} "
            f"available={free_kib / 1024:.0f}MiB",
            flush=True,
        )
        with log.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                command,
                cwd=REPO,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode:
            raise RuntimeError(f"Backtest failed; see {log}")
        if "\n\tSwaps: 0\n" not in log.read_text(errors="replace"):
            raise RuntimeError(f"Swap activity or missing metrics; see {log}")
        existing = next(iter(sorted(artifact_dir.glob("*.zip"))), None)
        if existing is None:
            raise FileNotFoundError(f"No result archive in {artifact_dir}")
    row = summarize(anchor, month, timerange, existing, log, selection)
    if row["actual_max_concurrent"] > 4:
        raise RuntimeError(f"Actual concurrency exceeded four in {run_name}")
    if row["leverage_values"] not in ([], [1.0]):
        raise RuntimeError(f"Unexpected leverage in {run_name}: {row['leverage_values']}")
    print(
        f"DONE  {run_name}: {row['profit_pct']:+.2f}% "
        f"PF {row['profit_factor']:.2f} DD {row['max_relative_drawdown_pct']:.2f}% "
        f"trades {row['trades']} forced {row['forced_exit_trades']} "
        f"RSS {row['maximum_rss_kib'] / 1024:.0f}MiB",
        flush=True,
    )
    return row


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = Path("/tmp/monthly-independent-riskbudget2pct-2026.lock")
    rows: list[dict[str, Any]] = []
    with lock.open("w", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another independent monthly run is active") from exc
        handle.write(str(os.getpid()))
        handle.flush()
        for anchor in ANCHORS:
            for month, timerange in MONTHS:
                rows.append(run(anchor, month, timerange))
                (OUTPUT / "results.json").write_text(
                    json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
                )


if __name__ == "__main__":
    main()
