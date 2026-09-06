#!/bin/bash
# Stop the delisting-short stack by pid file (never pkill -f: the pattern matches this shell).
cd /root/freqtrade
for name in freqtrade poller; do
  f=user_data/delist/$name.pid
  if [ -f "$f" ]; then
    pid=$(cat "$f")
    if kill -0 "$pid" 2>/dev/null; then
      pkill -TERM -P "$pid" 2>/dev/null   # children of the restart loop / freqtrade
      kill -TERM "$pid" 2>/dev/null
      echo "stopped $name (pid $pid)"
    fi
    rm -f "$f"
  fi
done
