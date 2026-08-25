#!/usr/bin/env python3
"""Event study for the 1h breakout signal, independent of the backtester.

The strategy family in ``user_data/strategies/HighVolumeFourMtfV1.py`` enters on
a 1h Donchian-20 breakout confirmed by relative volume and a same-direction 4h
EMA trend.  This script measures that raw condition directly: it evaluates the
signal on completed 1h candles inside the point-in-time pool windows and reports
the distribution of directional forward returns at several horizons.  No entry
tiering, position sizing, stop loss or slot contention is involved, so the
result answers only one question -- does the underlying condition predict
anything at all.

Forward returns start at the signal candle's close, the first price the strategy
could act on.  Two controls are reported:

* every same-side bar of the same pair-windows (the drift of the traded
  universe itself), and
* only the bars whose 4h regime matches the signal direction, which is the
  control the signal must actually beat.

Significance uses both a naive t-statistic and a bootstrap that resamples whole
``(pair, ISO week)`` blocks, because breakouts cluster inside a week and the
naive statistic overstates the effective sample size.  The reported ``boot p``
is the one-sided share of bootstrap means at or below zero.

Findings as of 2026-08 are recorded in
``.claude/skills/high-volume-trend-research/reference/2026-08-review-findings.md``.

Runtime is a few minutes and peak memory stays well under the host budget in
``AGENTS.md``; one pair is loaded at a time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]

DATASETS = {
    "2025": (
        REPO / "user_data/analysis/monthly-rotation-2025/selections/offset-01.json",
        REPO / "user_data/data/binance-monthly-rotation-2025-30m-windowed/futures",
    ),
    "2026": (
        REPO / "user_data/analysis/monthly-refresh-offset-28anchors-2026/selections/offset-01.json",
        REPO / "user_data/data/binance-monthly-incremental-top2-28anchors-2026-30m-windowed/futures",
    ),
}


def windows(selection: Path) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    """Map each raw symbol to the periods during which it was in the pool."""
    out: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for period in json.loads(Path(selection).read_text())["periods"]:
        start = pd.Timestamp(period["start"])
        end = pd.Timestamp(period["end_exclusive"])
        for symbol in period["selected_symbols"]:
            out.setdefault(symbol, []).append((start, end))
    return out


def indicators_1h(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["donchian_high20"] = df["high"].shift(1).rolling(20).max()
    df["donchian_low20"] = df["low"].shift(1).rolling(20).min()
    quote_volume = df["volume"] * df["close"]
    df["rvol20"] = quote_volume / quote_volume.shift(1).rolling(20).mean().replace(0, np.nan)
    df["up"] = df["close"] > df["open"]
    df["down"] = df["close"] < df["open"]
    return df


def trend_4h(df4: pd.DataFrame) -> pd.DataFrame:
    df4 = df4.copy()
    df4["ema20"] = df4["close"].ewm(span=20, adjust=False).mean()
    df4["ema50"] = df4["close"].ewm(span=50, adjust=False).mean()
    df4["bull"] = (df4["close"] > df4["ema50"]) & (df4["ema20"] > df4["ema50"])
    df4["bear"] = (df4["close"] < df4["ema50"]) & (df4["ema20"] < df4["ema50"])
    # merge_informative_pair shifts the informative frame by one full candle, so
    # a 4h candle is only usable once the next one has opened.
    df4["avail"] = (df4["date"] + pd.Timedelta(hours=4)).astype("datetime64[ms, UTC]")
    return df4[["avail", "bull", "bear"]].rename(columns={"avail": "date"})


def collect(
    selection: Path,
    root: Path,
    horizons: list[int],
    rvol_min: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    signals: list[pd.DataFrame] = []
    controls: list[pd.DataFrame] = []
    for symbol, periods in windows(selection).items():
        base = f"{symbol.removesuffix('USDT')}_USDT_USDT"
        path_1h = root / f"{base}-1h-futures.feather"
        path_4h = root / f"{base}-4h-futures.feather"
        if not path_1h.exists() or not path_4h.exists():
            continue
        frame = pd.merge_asof(
            indicators_1h(pd.read_feather(path_1h)).sort_values("date"),
            trend_4h(pd.read_feather(path_4h)).sort_values("date"),
            on="date",
            direction="backward",
        )
        close = frame["close"].to_numpy()
        for horizon in horizons:
            frame[f"fwd{horizon}"] = np.concatenate(
                [close[horizon:] / close[:-horizon] - 1, np.full(horizon, np.nan)]
            )
        active = np.zeros(len(frame), dtype=bool)
        for start, end in periods:
            active |= (frame["date"] >= start).to_numpy() & (frame["date"] < end).to_numpy()
        frame = frame.loc[active]

        long_signal = (
            (frame["close"] > frame["donchian_high20"])
            & (frame["rvol20"] >= rvol_min)
            & frame["up"]
            & frame["bull"]
        )
        short_signal = (
            (frame["close"] < frame["donchian_low20"])
            & (frame["rvol20"] >= rvol_min)
            & frame["down"]
            & frame["bear"]
        )
        for mask, side in ((long_signal, 1.0), (short_signal, -1.0)):
            hits = frame.loc[mask.fillna(False)]
            if not len(hits):
                continue
            row = hits[["date"]].copy()
            row["pair"] = symbol
            row["side"] = side
            for horizon in horizons:
                row[f"fwd{horizon}"] = hits[f"fwd{horizon}"] * side
            signals.append(row)

        # One control row per bar per side.  Mixing both sides into a single
        # control makes the two halves cancel and the control mean collapse to
        # zero, which is why the sides are kept apart here.
        for side, regime in ((1.0, frame["bull"]), (-1.0, frame["bear"])):
            row = frame[["date"]].copy()
            row["pair"] = symbol
            row["side"] = side
            row["regime"] = regime.fillna(False).to_numpy()
            for horizon in horizons:
                row[f"fwd{horizon}"] = frame[f"fwd{horizon}"] * side
            controls.append(row)

    return (
        pd.concat(signals, ignore_index=True) if signals else pd.DataFrame(),
        pd.concat(controls, ignore_index=True),
    )


def block_bootstrap(
    frame: pd.DataFrame,
    column: str,
    rounds: int,
    seed: int = 0,
) -> tuple[float, float]:
    """Resample whole (pair, ISO week) blocks; signals cluster inside a week."""
    rng = np.random.default_rng(seed)
    frame = frame.dropna(subset=[column])
    if frame.empty:
        return float("nan"), float("nan")
    key = frame["pair"] + "|" + frame["date"].dt.strftime("%G-%V")
    groups = [group[column].to_numpy() for _, group in frame.groupby(key, sort=False)]
    count = len(groups)
    means = np.empty(rounds)
    for index in range(rounds):
        pick = rng.integers(0, count, count)
        means[index] = np.concatenate([groups[j] for j in pick]).mean()
    return float(np.mean(means <= 0)), float(np.std(means))


def report(name: str, horizons: list[int], rvol_min: float, rounds: int) -> None:
    selection, root = DATASETS[name]
    signals, controls = collect(selection, root, horizons, rvol_min)
    print(f"\n===== {name} =====")
    print(
        f"信号 {len(signals)} (多 {int((signals['side'] > 0).sum())} "
        f"空 {int((signals['side'] < 0).sum())}), 对照 bar {len(controls) // 2}"
    )
    for side, label in ((1.0, "多头"), (-1.0, "空头")):
        side_signals = signals[signals["side"] == side]
        side_controls = controls[controls["side"] == side]
        regime_controls = side_controls[side_controls["regime"]]
        print(
            f"\n--- {label}  信号 n={len(side_signals)}  "
            f"同向趋势对照 bar={len(regime_controls)} ---"
        )
        print(
            f"{'H':>4s} {'信号均值%':>10s} {'全体对照%':>10s} {'同趋势对照%':>12s} "
            f"{'超额(对同趋势)%':>15s} {'胜率%':>7s} {'朴素t':>7s} {'boot p':>8s}"
        )
        for horizon in horizons:
            column = f"fwd{horizon}"
            observed = side_signals[column].dropna()
            if len(observed) < 3:
                continue
            control_all = side_controls[column].dropna().mean()
            control_regime = regime_controls[column].dropna().mean()
            t_stat = observed.mean() / (observed.std(ddof=1) / np.sqrt(len(observed)))
            p_value, _ = block_bootstrap(side_signals, column, rounds)
            print(
                f"{horizon:4d} {observed.mean() * 100:10.4f} {control_all * 100:10.4f} "
                f"{control_regime * 100:12.4f} "
                f"{(observed.mean() - control_regime) * 100:15.4f} "
                f"{(observed > 0).mean() * 100:7.2f} {t_stat:7.2f} {p_value:8.4f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "datasets",
        nargs="*",
        choices=list(DATASETS),
        help="which pool/datadir pairs to evaluate (default: all)",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[1, 4, 12, 24, 48, 72],
        help="forward return horizons in 1h candles",
    )
    parser.add_argument(
        "--rvol-min",
        type=float,
        default=1.0,
        help="minimum 20-bar relative quote volume required by the signal",
    )
    parser.add_argument(
        "--bootstrap-rounds",
        type=int,
        default=2000,
        help="number of block-bootstrap resamples",
    )
    args = parser.parse_args()
    for name in args.datasets or list(DATASETS):
        report(name, args.horizons, args.rvol_min, args.bootstrap_rounds)


if __name__ == "__main__":
    main()
