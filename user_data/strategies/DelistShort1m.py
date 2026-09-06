"""DelistShort1m -- short a Binance USDT perpetual after a delisting announcement.

Two event sources, both produced by scripts/delist_announcement_poller.py into
user_data/delist/events.json (and the matching RemotePairList file pairlist.json):

  kind "futures": "Binance Futures Will Delist ... Perpetual Contract(s)" -- signal on the 1m candle
                  at ceil(notice + 4 min), filled at the next open (= notice + 5 min); 15% stop.
  kind "spot":    "Binance Will Delist A, B, C on DATE" -- the coin's perp keeps trading; the first
                  15 minutes whipsaw, so signal at ceil(notice + 14 min), fill at +15 min; 25% stop.
                  Since 2025-12 the perp settlement is announced only inside this notice
                  ("Futures" section) -> `futures_settle`.

Exit for both: candle-open (backtest) / ticker (live) price >= entry * (1 + stop) -> "close_stop";
otherwise forced "settle_exit" at min(settle, futures_settle) - exit_before_min (default 60),
because the final hour before settlement has no insurance-fund support and only IOC liquidation.
A last-resort "settle_emergency" fires at settlement - 20 min if the earlier exit failed.

One trade per event: confirm_trade_entry refuses a pair that already traded after the notice.
Trades remember their own settlement in custom data, so exits keep working even if events.json
is later rewritten or the poller dies.

Research: skills/fable/delisting-short-research-20260906.md, spot-delist-perp-research-20260906.md.
Regime note: strong 2025Q3..2026Q1, weak 2024 and 2026Q2+; stop after two quarters <= 0.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from freqtrade.persistence import Trade
from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy


logger = logging.getLogger(__name__)
DELIST_DIR = Path(__file__).resolve().parents[1] / "delist"
EVENTS_PATH = Path(os.environ.get("DELIST_EVENTS_PATH", DELIST_DIR / "events.json"))
PAIRLIST_PATH = DELIST_DIR / "pairlist.json"
STALE_POLLER_MIN = 10


class DelistShort1m(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1m"
    can_short = True
    startup_candle_count = 0
    minimal_roi = {"0": 100.0}
    # Catastrophe floor only. Custom exits do the real work; -0.5 sits well above the in-sample
    # worst adverse excursion (32%) and, with stoploss_on_exchange in the live config, protects
    # the position while the bot is down. Liquidation at 1x is near +100%.
    stoploss = -0.50
    process_only_new_candles = True
    use_exit_signal = True   # freqtrade evaluates custom_exit only when this is True
    order_types = {"entry": "market", "exit": "market", "stoploss": "market", "stoploss_on_exchange": False}

    # futures-notice events
    delay_min = IntParameter(1, 60, default=4, space="buy", optimize=False)
    close_stop = DecimalParameter(0.05, 0.30, default=0.15, decimals=2, space="sell", optimize=False)
    # spot-notice events
    spot_delay_min = IntParameter(1, 120, default=14, space="buy", optimize=False)
    spot_close_stop = DecimalParameter(0.05, 0.50, default=0.25, decimals=2, space="sell", optimize=False)
    # shared
    entry_window_min = IntParameter(1, 60, default=15, space="buy", optimize=False)
    exit_before_min = IntParameter(30, 360, default=60, space="sell", optimize=False)
    emergency_before_min = IntParameter(5, 59, default=20, space="sell", optimize=False)

    _events: dict[str, list[dict]] = {}
    _events_mtime: float = 0.0
    _stale_warned_at: datetime | None = None

    # ------------------------------------------------------------------ events
    def _load_events(self, force: bool = False) -> None:
        try:
            mtime = EVENTS_PATH.stat().st_mtime
        except FileNotFoundError:
            if force:
                logger.warning("no events file at %s -- nothing will trade until the poller writes it", EVENTS_PATH)
            self._events = {}
            self._events_mtime = 0.0
            return
        if not force and mtime == self._events_mtime:
            return
        ev: dict[str, list[dict]] = {}
        n = 0
        for e in json.loads(EVENTS_PATH.read_text()):
            if e.get("form", "plain") != "plain":
                continue   # "Delist and Update the Leverage & Margin Tiers" notices: no edge, skipped
            e = dict(
                e,
                kind=e.get("kind", "futures"),
                release=pd.Timestamp(e["release"]).tz_convert("UTC"),
                settle=pd.Timestamp(e["settle"]).tz_convert("UTC"),
                futures_settle=pd.Timestamp(e["futures_settle"]).tz_convert("UTC") if e.get("futures_settle") else None,
            )
            e["end"] = min(t for t in (e["settle"], e["futures_settle"]) if t is not None)
            ev.setdefault(e["pair"], []).append(e)
            n += 1
        self._events = ev
        self._events_mtime = mtime
        logger.info("loaded %d delisting events for %d pairs from %s", n, len(ev), EVENTS_PATH)

    def _check_poller(self, now: datetime) -> None:
        """Warn (rate-limited) when the poller has not written the pairlist file recently."""
        try:
            updated = pd.Timestamp(json.loads(PAIRLIST_PATH.read_text())["updated"])
        except Exception:  # noqa: BLE001
            updated = None
        stale = updated is None or (pd.Timestamp(now) - updated) > pd.Timedelta(minutes=STALE_POLLER_MIN)
        if stale and (self._stale_warned_at is None or (now - self._stale_warned_at).total_seconds() > 600):
            logger.warning("delist poller looks stale (pairlist updated %s); new announcements are NOT being picked up", updated)
            self._stale_warned_at = now

    def bot_start(self, **kwargs) -> None:
        self._load_events(force=True)

    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        # Live/dry-run: pick up new announcements. Backtest: file frozen at bot_start so a poller
        # run on the same machine cannot change a running backtest.
        if self.config["runmode"].value in ("live", "dry_run"):
            self._load_events()
            self._check_poller(current_time)

    def _params(self, e: dict) -> tuple[int, float]:
        if e["kind"] == "spot":
            return int(self.spot_delay_min.value), float(self.spot_close_stop.value)
        return int(self.delay_min.value), float(self.close_stop.value)

    def _event_for(self, pair: str, when: pd.Timestamp) -> dict | None:
        """Most recently released event whose [release, end) window contains `when`."""
        best = None
        for e in self._events.get(pair, []):
            if e["release"] <= when < e["end"] and (best is None or e["release"] > best["release"]):
                best = e
        return best

    def _end_for(self, pair: str, opened: pd.Timestamp) -> pd.Timestamp | None:
        """Earliest settlement (spot delisting or perp settlement) of any known event after the trade opened."""
        ends = [t for e in self._events.get(pair, []) for t in (e["settle"], e["futures_settle"]) if t is not None and t > opened]
        return min(ends) if ends else None

    # ----------------------------------------------------------------- signals
    def populate_indicators(self, dataframe, metadata):
        return dataframe

    def populate_entry_trend(self, dataframe, metadata):
        dataframe["enter_short"] = 0
        for e in self._events.get(metadata["pair"], []):
            delay, _ = self._params(e)
            t0 = (e["release"] + pd.Timedelta(minutes=delay)).ceil("1min")
            t1 = min(t0 + pd.Timedelta(minutes=int(self.entry_window_min.value)),
                     e["end"] - pd.Timedelta(minutes=int(self.exit_before_min.value)))
            mask = (dataframe.date >= t0) & (dataframe.date < t1) & (dataframe.volume > 0)
            dataframe.loc[mask, ["enter_short", "enter_tag"]] = (1, f"delist_{e['kind']}")
        return dataframe

    def populate_exit_trend(self, dataframe, metadata):
        dataframe["exit_short"] = 0
        return dataframe

    def confirm_trade_entry(self, pair, order_type, amount, rate, time_in_force, current_time, entry_tag, side, **kwargs) -> bool:
        if side != "short":
            return False
        e = self._event_for(pair, pd.Timestamp(current_time))
        if e is None:
            return False
        # never past the exchange's no-new-orders cutoff (30 min before settlement) or our own exit
        if pd.Timestamp(current_time) >= e["end"] - pd.Timedelta(minutes=max(int(self.exit_before_min.value), 30)):
            return False
        # one trade per event
        for t in Trade.get_trades_proxy(pair=pair):
            if pd.Timestamp(t.open_date_utc) >= e["release"]:
                return False
        return True

    # ------------------------------------------------------------------- exits
    def _remember(self, trade: Trade, pair: str) -> tuple[pd.Timestamp | None, float]:
        """Settlement time and stop for this trade, persisted in trade custom data on first use."""
        end_s = trade.get_custom_data("delist_end")
        stop = trade.get_custom_data("delist_stop")
        if end_s and stop:
            return pd.Timestamp(end_s), float(stop)
        opened = pd.Timestamp(trade.open_date_utc)
        e = self._event_for(pair, opened)
        end = self._end_for(pair, opened)
        stop = self._params(e)[1] if e else float(self.close_stop.value)
        if end is not None:
            trade.set_custom_data("delist_end", end.isoformat())
            trade.set_custom_data("delist_stop", stop)
            trade.set_custom_data("delist_kind", e["kind"] if e else "unknown")
        return end, stop

    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        end, stop = self._remember(trade, pair)
        if current_rate >= trade.open_rate * (1 + stop):
            return "close_stop"
        if end is None:
            logger.warning("%s: open delisting trade without a known settlement -- exiting", pair)
            return "no_event"
        now = pd.Timestamp(current_time)
        if now >= end - pd.Timedelta(minutes=int(self.emergency_before_min.value)):
            return "settle_emergency"
        if now >= end - pd.Timedelta(minutes=int(self.exit_before_min.value)):
            return "settle_exit"
        return None

    def leverage(self, pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, **kwargs) -> float:
        return 1.0
