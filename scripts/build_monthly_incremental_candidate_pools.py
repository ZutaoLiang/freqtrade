#!/usr/bin/env python3
"""Add at most two rising-rank weekly candidates to each monthly core pool.

The overlay uses only point-in-time selection artifacts.  At each weekly
snapshot it considers currently filtered Top20 symbols outside the active
monthly core, keeps symbols whose raw-volume rank improved versus the prior
snapshot (a new entrant is assigned prior rank 21), and admits the best two
until the next snapshot.  A monthly refresh resets the overlay because the
new core already incorporates that snapshot.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
BASE_ROOT = REPO / "user_data/analysis/monthly-refresh-offset-2026/selections"
WEEKLY = REPO / "user_data/analysis/weekly-volume-rotation-2026-02-08.json"
OUTPUT = REPO / "user_data/analysis/monthly-incremental-top2-2026/selections"
START = pd.Timestamp("2026-02-01", tz="UTC")
END = pd.Timestamp("2026-08-14", tz="UTC")
ANCHORS = (1, 8, 15, 22)
MAX_INCREMENTAL = 2


def timestamp(value: str) -> pd.Timestamp:
    return pd.Timestamp(value)


def load_snapshots() -> dict[pd.Timestamp, dict[str, Any]]:
    weekly = json.loads(WEEKLY.read_text())
    snapshots = {timestamp(row["start"]): row for row in weekly["periods"]}
    # January snapshots are needed only as the prior-rank reference at the
    # common February 1 boundary.
    for anchor in (8, 15, 22):
        payload = json.loads((BASE_ROOT / f"offset-{anchor:02d}.json").read_text())
        row = payload["periods"][0]
        snapshots[timestamp(row["start"])] = row
    return dict(sorted(snapshots.items()))


def active_core(
    periods: list[dict[str, Any]],
    moment: pd.Timestamp,
) -> dict[str, Any]:
    return next(
        row
        for row in periods
        if timestamp(row["start"]) <= moment < timestamp(row["end_exclusive"])
    )


def choose_incremental(
    current: dict[str, Any],
    previous: dict[str, Any],
    core_symbols: set[str],
) -> list[dict[str, Any]]:
    previous_ranks = {
        row["raw_symbol"]: int(row["volume_rank"])
        for row in previous["top_volume"]
    }
    candidates = []
    for row in current["selected"]:
        raw_symbol = row["raw_symbol"]
        if raw_symbol in core_symbols:
            continue
        current_rank = int(row["volume_rank"])
        previous_rank = previous_ranks.get(raw_symbol, 21)
        improvement = previous_rank - current_rank
        if improvement <= 0:
            continue
        candidates.append(
            {
                "raw_symbol": raw_symbol,
                "current_volume_rank": current_rank,
                "previous_volume_rank": (
                    previous_rank if raw_symbol in previous_ranks else None
                ),
                "rank_improvement": improvement,
                "new_raw_top20_entry": raw_symbol not in previous_ranks,
                "quote_volume_usdt": float(row["quote_volume_usdt"]),
            }
        )
    candidates.sort(
        key=lambda row: (
            -row["rank_improvement"],
            row["current_volume_rank"],
            row["raw_symbol"],
        )
    )
    return candidates[:MAX_INCREMENTAL]


def build_anchor(
    anchor: int,
    snapshots: dict[pd.Timestamp, dict[str, Any]],
) -> dict[str, Any]:
    base_path = BASE_ROOT / f"offset-{anchor:02d}.json"
    base = json.loads(base_path.read_text())
    core_periods = base["periods"]
    snapshot_times = sorted(snapshots)
    boundaries = {START, END}
    boundaries.update(moment for moment in snapshot_times if START <= moment < END)
    boundaries.update(
        max(START, timestamp(row["start"]))
        for row in core_periods
        if timestamp(row["end_exclusive"]) > START
        and timestamp(row["start"]) < END
    )
    boundaries.update(
        min(END, timestamp(row["end_exclusive"]))
        for row in core_periods
        if timestamp(row["end_exclusive"]) > START
        and timestamp(row["start"]) < END
    )
    ordered_boundaries = sorted(boundaries)
    overlay: list[dict[str, Any]] = []
    periods: list[dict[str, Any]] = []

    for index, start in enumerate(ordered_boundaries[:-1]):
        end = ordered_boundaries[index + 1]
        core = active_core(core_periods, start)
        core_start = timestamp(core["start"])
        core_symbols = list(core["selected_symbols"])
        if start == core_start:
            # A monthly refresh always resets the overlay, including on the
            # anchor days that do not coincide with a weekly snapshot.  Without
            # this the previous overlay would survive into a core that may
            # already contain the same symbols.
            overlay = []
        elif start in snapshots:
            position = snapshot_times.index(start)
            if position == 0:
                raise RuntimeError(f"No prior snapshot for {start}")
            overlay = choose_incremental(
                snapshots[start],
                snapshots[snapshot_times[position - 1]],
                set(core_symbols),
            )
        overlay_symbols = [row["raw_symbol"] for row in overlay]
        if set(core_symbols).intersection(overlay_symbols):
            raise AssertionError("incremental candidates overlap the active core")
        periods.append(
            {
                "start": start.isoformat(),
                "end_exclusive": end.isoformat(),
                "selection_time": start.isoformat(),
                "core_anchor_day": anchor,
                "core_selection_time": core["selection_time"],
                "core_selected_symbols": core_symbols,
                "incremental_selection_time": (
                    start.isoformat() if start in snapshots and start != core_start
                    else None
                ),
                "incremental_candidates": copy.deepcopy(overlay),
                "incremental_selected_symbols": overlay_symbols,
                "selected_symbols": core_symbols + overlay_symbols,
            }
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "monthly_core": str(base_path.relative_to(REPO)),
            "weekly_snapshots": str(WEEKLY.relative_to(REPO)),
        },
        "method": {
            "core_refresh": f"monthly at day {anchor:02d} 00:00 UTC",
            "incremental_refresh": "weekly point-in-time snapshots",
            "incremental_maximum_pairs": MAX_INCREMENTAL,
            "incremental_score": "prior raw-volume rank minus current raw-volume rank",
            "new_raw_top20_prior_rank": 21,
            "minimum_rank_improvement": 1,
            "incremental_ttl": "until next weekly snapshot",
            "monthly_refresh_resets_incremental_pool": True,
        },
        "range": {
            "start": START.isoformat(),
            "end_exclusive": END.isoformat(),
        },
        "anchor_day": anchor,
        "periods": periods,
    }


def main() -> None:
    snapshots = load_snapshots()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    payloads = {}
    for anchor in ANCHORS:
        payload = build_anchor(anchor, snapshots)
        target = OUTPUT / f"offset-{anchor:02d}.json"
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        payloads[anchor] = payload
        periods = payload["periods"]
        overlay_periods = sum(bool(row["incremental_selected_symbols"]) for row in periods)
        overlay_union = {
            symbol
            for row in periods
            for symbol in row["incremental_selected_symbols"]
        }
        print(
            f"wrote {target.relative_to(REPO)} periods={len(periods)} "
            f"overlay_periods={overlay_periods} overlay_union={len(overlay_union)}",
            flush=True,
        )

    combined = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "four monthly-core plus incremental-top2 selections",
        "method": {"purpose": "shared low-memory datadir"},
        "range": {"start": START.isoformat(), "end_exclusive": END.isoformat()},
        "periods": [
            {**copy.deepcopy(row), "source_anchor_day": anchor}
            for anchor, payload in payloads.items()
            for row in payload["periods"]
        ],
    }
    target = OUTPUT / "combined.json"
    target.write_text(json.dumps(combined, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {target.relative_to(REPO)}", flush=True)


if __name__ == "__main__":
    main()
