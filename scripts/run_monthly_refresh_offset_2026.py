#!/usr/bin/env python3
"""Run the four monthly refresh-day variants serially with bounded memory."""

from __future__ import annotations

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
OUTPUT = REPO / "user_data/analysis/monthly-refresh-offset-2026"
SELECTION_ROOT = OUTPUT / "selections"
DATADIR = REPO / "user_data/data/binance-monthly-refresh-offset-2026-30m-windowed"
BASE_CONFIG = REPO / "user_data/config_high_volume_mainwave_v18_2026_02_08.json"
STRATEGY = "HighVolumeMainWaveMonthlyOffsetRefitV20"
MAX_OPEN_TRADES = 4
TIMERANGE = "20260201-20260814"
START = pd.Timestamp("2026-02-01", tz="UTC")
END = pd.Timestamp("2026-08-14", tz="UTC")
TEST_START = pd.Timestamp("2026-06-01", tz="UTC")
ANCHORS = (1, 8, 15, 22)
PARAMETERS = {
    "pretrend_short_minimum_rvol": 999.0,
    "persistent_mainwave_long_multiplier": 2.0,
    "strong_short_stake_multiplier": 1.75,
}
EXTRA_CONFIG: dict[str, Any] = {}


def available_kib() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    raise RuntimeError("MemAvailable is unavailable")


def selected_pairs(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    symbols = {
        symbol
        for period in payload["periods"]
        if pd.Timestamp(period["start"]) < END
        and pd.Timestamp(period["end_exclusive"]) > START
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


def monthly_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    trades = pd.DataFrame(result["trades"])
    trades["close_date"] = pd.to_datetime(trades["close_date"], utc=True)
    trades["month"] = trades["close_date"].clip(
        upper=END - pd.Timedelta(microseconds=1)
    ).dt.strftime("%Y-%m")
    equity = float(result["starting_balance"])
    rows = []
    for month, group in trades.groupby("month", sort=True):
        profit = float(group["profit_abs"].sum())
        rows.append(
            {
                "month": month,
                "trades": len(group),
                "profit_abs": profit,
                "starting_equity": equity,
                "return_pct": profit / equity * 100 if equity else 0.0,
            }
        )
        equity += profit
    return rows


def summarize(anchor: int, artifact: Path, log: Path, selection: Path) -> dict[str, Any]:
    result = load_result(artifact)
    trades = pd.DataFrame(result["trades"])
    trades["close_date"] = pd.to_datetime(trades["close_date"], utc=True)
    train_profit = float(trades.loc[trades["close_date"] < TEST_START, "profit_abs"].sum())
    test_profit = float(trades.loc[trades["close_date"] >= TEST_START, "profit_abs"].sum())
    test_starting_equity = float(result["starting_balance"]) + train_profit
    payload = json.loads(selection.read_text())
    active_periods = [
        row
        for row in payload["periods"]
        if pd.Timestamp(row["start"]) < END
        and pd.Timestamp(row["end_exclusive"]) > START
    ]
    pool_sizes = [len(row["selected_symbols"]) for row in active_periods]
    return {
        "anchor_day": anchor,
        "timerange": TIMERANGE,
        "selection": str(selection.relative_to(REPO)),
        "periods": len(active_periods),
        "mean_pool_size": sum(pool_sizes) / len(pool_sizes),
        "minimum_pool_size": min(pool_sizes),
        "maximum_pool_size": max(pool_sizes),
        "union_pairs": len(selected_pairs(selection)),
        "trades": int(result["total_trades"]),
        "rejected_signals": int(result["rejected_signals"]),
        "profit_abs": float(result["profit_total_abs"]),
        "profit_pct": float(result["profit_total"]) * 100,
        "final_balance": float(result["final_balance"]),
        "profit_factor": float(result["profit_factor"]),
        "max_relative_drawdown_pct": float(result["max_relative_drawdown"]) * 100,
        "return_over_drawdown": (
            float(result["profit_total"]) * 100
            / (float(result["max_relative_drawdown"]) * 100)
        ),
        "long_profit_abs": float(result["profit_total_long_abs"]),
        "short_profit_abs": float(result["profit_total_short_abs"]),
        "train_profit_abs": train_profit,
        "train_return_pct": train_profit / float(result["starting_balance"]) * 100,
        "test_profit_abs": test_profit,
        "test_starting_equity": test_starting_equity,
        "test_continuous_return_pct": test_profit / test_starting_equity * 100,
        "monthly": monthly_rows(result),
        "maximum_rss_kib": maximum_rss(log),
        "artifact": str(artifact.relative_to(REPO)),
    }


def run(anchor: int) -> dict[str, Any]:
    name = f"offset-{anchor:02d}"
    selection = SELECTION_ROOT / f"{name}.json"
    pairs = selected_pairs(selection)
    artifact_dir = OUTPUT / "artifacts" / name
    log = OUTPUT / "logs" / f"{name}.log"
    config = OUTPUT / "configs" / f"{name}.json"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    config.parent.mkdir(parents=True, exist_ok=True)
    existing = next(iter(sorted(artifact_dir.glob("*.zip"))), None)
    if existing is None:
        if available_kib() < 640 * 1024:
            raise RuntimeError("MemAvailable is below the 640 MiB launch reserve")
        overlay = {
            "strategy": STRATEGY,
            "max_open_trades": MAX_OPEN_TRADES,
            "dry_run_wallet": 500,
            "v20_refit_parameters": PARAMETERS,
            "v20_monthly_offset_selection": str(
                selection.relative_to(REPO / "user_data/analysis")
            ),
            "bot_name": f"v20-{OUTPUT.name}-{name}",
            **EXTRA_CONFIG,
        }
        config.write_text(json.dumps(overlay, ensure_ascii=False, indent=2) + "\n")
        command = [
            "/usr/bin/time", "-v", "env", "MALLOC_ARENA_MAX=2",
            "MALLOC_TRIM_THRESHOLD_=131072", "nice", "-n", "10",
            "python3", "-m", "freqtrade", "backtesting",
            "--config", str(BASE_CONFIG.relative_to(REPO)),
            "--config", str(config.relative_to(REPO)),
            "--datadir", str(DATADIR.relative_to(REPO)),
            "--timerange", TIMERANGE,
            "--max-open-trades", str(MAX_OPEN_TRADES),
            "--dry-run-wallet", "500", "--fee", "0.0007",
            "--cache", "none", "--export", "trades", "--breakdown", "month",
            "--backtest-directory", str(artifact_dir.relative_to(REPO)),
            "--pairs", *pairs,
        ]
        print(
            f"START {name} pairs={len(pairs)} "
            f"available={available_kib() / 1024:.0f}MiB",
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
    row = summarize(anchor, existing, log, selection)
    print(
        f"DONE  {name}: {row['profit_pct']:.2f}% PF {row['profit_factor']:.2f} "
        f"DD {row['max_relative_drawdown_pct']:.2f}% trades {row['trades']} "
        f"RSS {row['maximum_rss_kib'] / 1024:.0f}MiB",
        flush=True,
    )
    return row


def main() -> None:
    lock = Path("/tmp/monthly-refresh-offset-2026.lock")
    rows: list[dict[str, Any]] = []
    with lock.open("w", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another monthly offset backtest is active") from exc
        handle.write(str(os.getpid()))
        handle.flush()
        for anchor in ANCHORS:
            rows.append(run(anchor))
            (OUTPUT / "results.json").write_text(
                json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()
