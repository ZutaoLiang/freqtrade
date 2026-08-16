#!/usr/bin/env python3
"""Run shared four-slot Top2 at 1x leverage with a 2% base risk budget."""

from pathlib import Path

import run_monthly_refresh_offset_2026 as runner


REPO = Path(__file__).resolve().parents[1]
runner.OUTPUT = (
    REPO / "user_data/analysis/monthly-incremental-top2-riskbudget2pct-2026"
)
runner.SELECTION_ROOT = (
    REPO / "user_data/analysis/monthly-incremental-top2-2026/selections"
)
runner.DATADIR = (
    REPO / "user_data/data/binance-monthly-incremental-top2-2026-30m-windowed"
)
runner.STRATEGY = "HighVolumeMainWaveMonthlyOffsetRiskBudget2PctV23"
# Keep the provisional unlimited-stake value below recent-contract leverage
# tier limits.  The strategy itself enforces the actual four-position limit.
runner.MAX_OPEN_TRADES = 100


if __name__ == "__main__":
    runner.main()
