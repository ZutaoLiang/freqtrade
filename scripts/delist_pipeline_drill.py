"""End-to-end drill for the delisting pipeline WITHOUT waiting for a real notice.

Injects a synthetic event for a liquid pair into user_data/delist/events.json + pairlist.json,
so a dry-run bot should: add the pair via RemotePairList within a minute, enter short at
notice + 5 min (futures kind) or + 15 min (spot kind), and exit at settlement - exit_before.
Stop the poller while drilling (it would overwrite the files), and restore afterwards with
`--restore` (or by running the poller once).

  .venv/bin/python scripts/delist_pipeline_drill.py --pair DOGE/USDT:USDT --kind futures --settle-in-min 80
  .venv/bin/freqtrade trade -c config-delist-dryrun.json --strategy DelistShort1m   # watch the log
  .venv/bin/python scripts/delist_pipeline_drill.py --restore
"""
import argparse
import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

D = Path("/root/freqtrade/user_data/delist")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="DOGE/USDT:USDT")
    ap.add_argument("--kind", choices=["futures", "spot"], default="futures")
    ap.add_argument("--settle-in-min", type=int, default=80, help="synthetic settlement time from now; exit fires at settle - exit_before_min")
    ap.add_argument("--release-ago-min", type=int, default=1, help="pretend the notice came this many minutes ago")
    ap.add_argument("--restore", action="store_true")
    a = ap.parse_args()
    if a.restore:
        for f in ("events.json", "pairlist.json"):
            b = D / f"{f}.drill_backup"
            if b.exists():
                shutil.move(b, D / f)
                print("restored", f)
        return
    for f in ("events.json", "pairlist.json"):
        if (D / f).exists() and not (D / f"{f}.drill_backup").exists():
            shutil.copy(D / f, D / f"{f}.drill_backup")
    now = datetime.now(UTC)
    rel = now - timedelta(minutes=a.release_ago_min)
    settle = now + timedelta(minutes=a.settle_in_min)
    sym = a.pair.split("/")[0] + "USDT"
    ev = {"pair": a.pair, "symbol": sym, "kind": a.kind, "release": rel.isoformat(), "settle": settle.isoformat(),
          "futures_settle": settle.isoformat() if a.kind == "spot" else None, "notice": "DRILL synthetic event", "form": "plain"}
    (D / "events.json").write_text(json.dumps([ev], indent=1))
    (D / "pairlist.json").write_text(json.dumps({"pairs": [a.pair], "refresh_period": 60, "updated": now.isoformat(), "drill": True}))
    delay = 5 if a.kind == "futures" else 15
    print(f"drill event written: {a.pair} kind={a.kind} release={rel:%H:%M:%S} -> expected entry ~{(rel + timedelta(minutes=delay)):%H:%M} UTC, "
          f"settle={settle:%H:%M} -> settle_exit at {(settle - timedelta(minutes=60)):%H:%M} UTC. Start the dry-run bot now; restore with --restore when done.")


if __name__ == "__main__":
    main()
