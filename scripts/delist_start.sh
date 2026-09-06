#!/bin/bash
# Start the delisting-short stack: announcement poller (every 60 s) + freqtrade.
# Usage: scripts/delist_start.sh [config]   (default config-delist-dryrun.json)
set -u
cd /root/freqtrade
CFG=${1:-config-delist-dryrun.json}
mkdir -p user_data/delist/logs
if [ -f user_data/delist/poller.pid ] && kill -0 "$(cat user_data/delist/poller.pid)" 2>/dev/null; then
  echo "poller already running (pid $(cat user_data/delist/poller.pid))"
else
  nohup bash -c 'while true; do .venv/bin/python scripts/delist_announcement_poller.py --loop --interval 60 --proxy http://127.0.0.1:10811; echo "$(date -u +%FT%TZ) poller exited, restarting in 30s" >&2; sleep 30; done' \
    >> user_data/delist/logs/poller.log 2>&1 &
  echo $! > user_data/delist/poller.pid
  echo "poller started (pid $!)"
fi
# wait for the first pairlist write so RemotePairList does not start on a missing file
for i in $(seq 1 30); do [ -f user_data/delist/pairlist.json ] && break; sleep 2; done
if [ -f user_data/delist/freqtrade.pid ] && kill -0 "$(cat user_data/delist/freqtrade.pid)" 2>/dev/null; then
  echo "freqtrade already running (pid $(cat user_data/delist/freqtrade.pid))"
else
  nohup .venv/bin/freqtrade trade -c "$CFG" --strategy DelistShort1m --logfile user_data/delist/logs/freqtrade.log \
    > user_data/delist/logs/freqtrade.stdout 2>&1 &
  echo $! > user_data/delist/freqtrade.pid
  echo "freqtrade started (pid $!) with $CFG"
fi
