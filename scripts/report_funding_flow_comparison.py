"""Summarize funding-flow backtest archives into comparable CSV reports."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd


ORDER = {
    "FundingSkewBaseline1m": 0,
    "FundingSkewMom15Filter1m": 1,
    "FundingSkewMom15Scaled1m": 2,
    "FundingSkewBaseline5m": 3,
}


def load_archive(path: Path) -> dict[str, dict[str, Any]]:
    """Return all strategy payloads stored in a Freqtrade result archive."""
    with zipfile.ZipFile(path) as archive:
        candidates = [
            name
            for name in archive.namelist()
            if name.endswith(".json") and not name.endswith("_config.json")
        ]
        if len(candidates) != 1:
            raise ValueError(f"Expected one result JSON in {path}, found {candidates}")
        payload = json.loads(archive.read(candidates[0]))
    return payload["strategy"]


def trade_frame(result: dict[str, Any]) -> pd.DataFrame:
    trades = pd.DataFrame(result["trades"])
    if trades.empty:
        return trades
    for column in ("open_date", "close_date"):
        trades[column] = pd.to_datetime(trades[column], utc=True)
    trades["quarter"] = trades["close_date"].dt.tz_localize(None).dt.to_period("Q").astype(str)
    return trades


def overview_row(name: str, result: dict[str, Any], trades: pd.DataFrame) -> dict[str, Any]:
    funding = trades["funding_fees"] if not trades.empty else pd.Series(dtype=float)
    quarter_profit = trades.groupby("quarter")["profit_abs"].sum()
    return {
        "strategy": name,
        "timeframe": result["timeframe"],
        "timeframe_detail": result["timeframe_detail"] or "-",
        "trades": int(result["total_trades"]),
        "avg_stake_usdt": float(result["avg_stake_amount"]),
        "mean_bps": float(result["profit_mean"]) * 10_000,
        "win_rate_pct": float(result["winrate"]) * 100,
        "profit_usdt": float(result["profit_total_abs"]),
        "return_pct": float(result["profit_total"]) * 100,
        "profit_factor": float(result["profit_factor"]),
        "max_drawdown_usdt": float(result["max_drawdown_abs"]),
        "max_drawdown_pct": float(result["max_drawdown_account"]) * 100,
        "funding_usdt": float(funding.sum()),
        "funding_nonzero_trades": int(funding.ne(0).sum()),
        "positive_quarters": int(quarter_profit.gt(0).sum()),
        "quarters": len(quarter_profit),
    }


def quarter_rows(name: str, result: dict[str, Any], trades: pd.DataFrame) -> list[dict[str, Any]]:
    starting_balance = float(result["starting_balance"])
    rows: list[dict[str, Any]] = []
    for quarter, group in trades.groupby("quarter", sort=True):
        profit = float(group["profit_abs"].sum())
        rows.append(
            {
                "strategy": name,
                "quarter": quarter,
                "trades": len(group),
                "mean_bps": float(group["profit_ratio"].mean()) * 10_000,
                "win_rate_pct": float(group["profit_abs"].gt(0).mean()) * 100,
                "profit_usdt": profit,
                "initial_wallet_pct": profit / starting_balance * 100,
                "funding_usdt": float(group["funding_fees"].sum()),
            }
        )
    return rows


def tag_rows(name: str, trades: pd.DataFrame) -> list[dict[str, Any]]:
    if trades.empty:
        return []
    rows: list[dict[str, Any]] = []
    for tag, group in trades.groupby("enter_tag", dropna=False, sort=True):
        rows.append(
            {
                "strategy": name,
                "enter_tag": tag,
                "trades": len(group),
                "avg_stake_usdt": float(group["stake_amount"].mean()),
                "mean_bps": float(group["profit_ratio"].mean()) * 10_000,
                "profit_usdt": float(group["profit_abs"].sum()),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("archives", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results: dict[str, dict[str, Any]] = {}
    for archive in args.archives:
        archive_results = load_archive(archive)
        overlap = results.keys() & archive_results.keys()
        if overlap:
            raise ValueError(f"Duplicate strategies in archives: {sorted(overlap)}")
        results.update(archive_results)

    unknown = sorted(set(results) - set(ORDER))
    if unknown:
        raise ValueError(f"Unexpected strategies: {unknown}")

    overview: list[dict[str, Any]] = []
    quarters: list[dict[str, Any]] = []
    tags: list[dict[str, Any]] = []
    for name in sorted(results, key=ORDER.get):
        result = results[name]
        trades = trade_frame(result)
        overview.append(overview_row(name, result, trades))
        quarters.extend(quarter_rows(name, result, trades))
        tags.extend(tag_rows(name, trades))

    args.output.mkdir(parents=True, exist_ok=True)
    overview_frame = pd.DataFrame(overview)
    quarter_frame = pd.DataFrame(quarters)
    tag_frame = pd.DataFrame(tags)
    overview_frame.to_csv(args.output / "overview.csv", index=False)
    quarter_frame.to_csv(args.output / "quarters.csv", index=False)
    tag_frame.to_csv(args.output / "entry_tags.csv", index=False)

    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(overview_frame.round(2).to_string(index=False))
        print()
        print(quarter_frame.round(2).to_string(index=False))
        print()
        print(tag_frame.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
