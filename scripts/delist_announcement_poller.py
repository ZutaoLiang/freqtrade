"""Binance futures delisting announcement poller.

Reads the Binance CMS "Delisting" catalog, parses every futures-delisting notice into
(symbol, release time, settlement time), and writes two files the DelistShort1m strategy
and the RemotePairList consume:

  user_data/delist/events.json    [{"pair": "XXX/USDT:USDT", "symbol": "XXXUSDT",
                                    "release": iso, "settle": iso, "notice": title}, ...]
  user_data/delist/pairlist.json  {"pairs": [...active pairs...], "refresh_period": 60}

Active = now in [release, settle - exit_before_min). The strategy decides entry timing itself.

Modes:
  --once                 fetch, write, exit
  --loop --interval 60   poll forever (live/dry-run companion process); RemotePairList refreshes every 60 s too
  --from-research CSV    build events.json from user_data/delist_short_20260906/events_parsed.csv
                         (backtest replay; no network)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import requests

LIST_URL = "https://www.binance.com/bapi/composite/v1/public/cms/article/list/query"
DETAIL_URL = "https://www.binance.com/bapi/composite/v1/public/cms/article/detail/query"
CATALOG_DELISTING = 161
DT = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\s*\(UTC\)"
TITLE_RE = re.compile(r"futures.*delist|delist.*perpetual|perpetual.*delist", re.I)
# "Delist and Update the Leverage & Margin Tiers" notices are risk-control delistings of old contracts;
# 17 such events (2024-01..2025-10) averaged about -3% for the short. Flagged and not traded.
TIER_FORM_RE = re.compile(r"update the leverage|leverage\s*&\s*margin tiers|leverage and margin tiers", re.I)
# Second event source: SPOT delisting notices ("Binance Will Delist A, B, C on DATE", and the
# "Vote to Delist Results and Will Delist ..." batches). The coin's perp keeps trading for weeks;
# research: skills/fable/spot-delist-perp-research-20260906.md (delay 15 min, 25% stop).
SPOT_RE = re.compile(r"^Binance Will Delist |Vote to Delist Results and Will Delist ")
SPOT_COINS_RE = re.compile(r"Will Delist ([A-Z0-9, &]+?)(?: on \d|$| and Support)")
SPOT_TIME_RE = re.compile(r"cease trading on all trading pairs for .*? at " + DT, re.S)
# Spot notices carry a "Futures" section: either "trading is not affected" or an automatic settlement
# of the coins' perps a few days later (since 2025-12 that is the ONLY place the perp settlement is announced).
SPOT_FUT_SETTLE_RE = re.compile(r"automatic settlement on the contracts of the aforementioned token\(?s?\)? at " + DT)
OUT_DIR = Path("/root/freqtrade/user_data/delist")


def _text(node, out):
    if node.get("node") == "text":
        out.append(node["text"])
    for c in node.get("child", []):
        _text(c, out)
    if node.get("tag") in ("p", "li", "tr", "br"):
        out.append("\n")


def parse_body(body: str) -> list[tuple[str, str]]:
    """Return [(symbol, 'YYYY-MM-DD HH:MM'), ...] from the settlement block of a notice."""
    body = body.replace("&nbsp;", " ")
    i = body.find("automatic settlement")
    if i < 0:
        return []
    seg = body[i:]
    m = re.search(r"not allowed to open|not be able to open|will be delisted after|will be removed after", seg)
    seg = seg[: m.start()] if m else seg[:800]
    out = []
    for line in seg.split("\n"):
        if not re.search(DT, line):
            continue
        t = re.findall(DT, line)[0]
        for s in re.findall(r"\b([A-Z0-9]+USDT)\b", line):
            out.append((s, t))
    for sent in re.split(r"(?<=\.)\s", seg.replace("\n", " ")):
        mm = re.search(r"on (.*?) at " + DT, sent)
        if mm:
            for s in re.findall(r"\b([A-Z0-9]+USDT)\b", mm.group(1)):
                out.append((s, mm.group(2)))
    return list(dict.fromkeys(out))


def fetch_events(proxies: dict | None, lookback_days: int) -> list[dict]:
    since_ms = int((datetime.now(UTC) - timedelta(days=lookback_days)).timestamp() * 1000)
    arts = []
    for page in range(1, 20):
        r = requests.get(LIST_URL, params={"type": 1, "catalogId": CATALOG_DELISTING, "pageNo": page, "pageSize": 50},
                         proxies=proxies, timeout=30)
        r.raise_for_status()
        a = r.json()["data"]["catalogs"][0]["articles"]
        arts += a
        if not a or a[-1]["releaseDate"] < since_ms:
            break
    events = []
    for x in arts:
        title = x["title"]
        is_fut, is_spot = bool(TITLE_RE.search(title)), bool(SPOT_RE.search(title)) and "Futures" not in title
        if x["releaseDate"] < since_ms or not (is_fut or is_spot):
            continue
        d = requests.get(DETAIL_URL, params={"articleCode": x["code"]}, proxies=proxies, timeout=30).json()["data"]
        out = []
        _text(json.loads(d["body"]), out)
        body = "".join(out).replace("&nbsp;", " ")
        rel = datetime.fromtimestamp(x["releaseDate"] / 1000, UTC)
        if is_fut:
            for sym, t in parse_body(body):
                settle = datetime.strptime(t, "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
                events.append({"pair": f"{sym[:-4]}/USDT:USDT", "symbol": sym, "kind": "futures", "release": rel.isoformat(),
                               "settle": settle.isoformat(), "notice": title,
                               "form": "tier_update" if TIER_FORM_RE.search(title) else "plain"})
        else:
            m = SPOT_COINS_RE.search(title)
            tm = SPOT_TIME_RE.search(body) or re.search(DT, body)
            if m and tm:
                settle = datetime.strptime(tm.group(1), "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
                fs = SPOT_FUT_SETTLE_RE.search(body)
                fut_settle = datetime.strptime(fs.group(1), "%Y-%m-%d %H:%M").replace(tzinfo=UTC).isoformat() if fs else None
                per_coin = {sym: t for sym, t in parse_body(body)}   # older notices name the contracts explicitly
                for coin in re.findall(r"\b[A-Z0-9]{2,10}\b", m.group(1)):
                    sym = f"{coin}USDT"
                    fsx = datetime.strptime(per_coin[sym], "%Y-%m-%d %H:%M").replace(tzinfo=UTC).isoformat() if sym in per_coin else fut_settle
                    events.append({"pair": f"{coin}/USDT:USDT", "symbol": sym, "kind": "spot", "release": rel.isoformat(),
                                   "settle": settle.isoformat(), "futures_settle": fsx, "notice": title, "form": "plain"})
        time.sleep(0.2)
    return events


def write_outputs(events: list[dict], exit_before_min: int, now: datetime | None = None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    now = now or datetime.now(UTC)
    # keep the latest settlement per symbol (postponements re-announce)
    by_sym: dict[tuple, dict] = {}
    for e in sorted(events, key=lambda e: e["release"]):
        by_sym[(e["symbol"], e.get("kind", "futures"))] = e
    events = sorted(by_sym.values(), key=lambda e: e["release"])
    active = [e["pair"] for e in events
              if e.get("form", "plain") == "plain"
              and datetime.fromisoformat(e["release"]) <= now < datetime.fromisoformat(e["settle"]) - timedelta(minutes=exit_before_min)]
    tmp = OUT_DIR / "events.json.tmp"
    tmp.write_text(json.dumps(events, indent=1))
    tmp.replace(OUT_DIR / "events.json")
    tmp = OUT_DIR / "pairlist.json.tmp"
    tmp.write_text(json.dumps({"pairs": active, "refresh_period": 60, "updated": now.isoformat(),
                               "events_total": len(events), "kinds": {"futures": sum(e.get("kind", "futures") == "futures" for e in events),
                                                                       "spot": sum(e.get("kind") == "spot" for e in events)}}))
    tmp.replace(OUT_DIR / "pairlist.json")
    return events, active


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--lookback-days", type=int, default=14)
    ap.add_argument("--exit-before-min", type=int, default=60)
    ap.add_argument("--proxy", default="http://127.0.0.1:10811")
    ap.add_argument("--from-research", default=None, help="events_parsed.csv -> events.json, no network")
    ap.add_argument("--spot-research", default=None, help="spot_delist_notices.json to add as spot events (backtest)")
    a = ap.parse_args()
    proxies = {"http": a.proxy, "https": a.proxy} if a.proxy else None
    if a.from_research:
        import pandas as pd
        df = pd.read_csv(a.from_research)
        events = [{"pair": f"{r.symbol[:-4]}/USDT:USDT", "symbol": r.symbol,
                   "release": pd.Timestamp(r.release).isoformat(), "settle": pd.Timestamp(r.settle).isoformat(), "notice": r.title,
                   "form": "tier_update" if TIER_FORM_RE.search(r.title) else "plain", "kind": "futures"}
                  for r in df.itertuples() if r.symbol.endswith("USDT")]
        if a.spot_research:
            for n in json.load(open(a.spot_research)):
                if not n["delist_utc"]:
                    continue
                fs = SPOT_FUT_SETTLE_RE.search(n["body"])
                fut_settle = pd.Timestamp(fs.group(1), tz="UTC").isoformat() if fs else None
                per_coin = {sym: t for sym, t in parse_body(n["body"])}
                for coin in n["coins"]:
                    sym = f"{coin}USDT"
                    fsx = pd.Timestamp(per_coin[sym], tz="UTC").isoformat() if sym in per_coin else fut_settle
                    events.append({"pair": f"{coin}/USDT:USDT", "symbol": sym, "kind": "spot", "release": pd.Timestamp(n["release"]).isoformat(),
                                   "settle": pd.Timestamp(n["delist_utc"], tz="UTC").isoformat(), "futures_settle": fsx, "notice": n["title"], "form": "plain"})
        ev, act = write_outputs(events, a.exit_before_min, now=datetime(2000, 1, 1, tzinfo=UTC))
        print(f"wrote {len(ev)} historical events to {OUT_DIR}/events.json")
        return
    while True:
        try:
            events = fetch_events(proxies, a.lookback_days)
            ev, act = write_outputs(events, a.exit_before_min)
            print(f"{datetime.now(UTC).isoformat()} events={len(ev)} active={act}", flush=True)
        except Exception as e:  # noqa
            print(f"{datetime.now(UTC).isoformat()} ERROR {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        if not a.loop:
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
