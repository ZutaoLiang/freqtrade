#!/usr/bin/env bash
set -e

export RUN_SEG="VALID"
DIR="/root/freqtrade/user_data/minute_research/r3"

echo "Launching 9 batches in parallel on VALID..."

python3 $DIR/batch1_f1_f2.py > $DIR/batch1_valid.log 2>&1 &
PID1=$!

python3 $DIR/batch3_f5_f6.py > $DIR/batch3_valid.log 2>&1 &
PID3=$!

python3 $DIR/batch4_f7_f8.py > $DIR/batch4_valid.log 2>&1 &
PID4=$!

python3 $DIR/batch5_f9_f10.py > $DIR/batch5_valid.log 2>&1 &
PID5=$!

python3 $DIR/exp500_batch1_squeeze.py > $DIR/exp500_batch1_valid.log 2>&1 &
PID6=$!

python3 $DIR/exp500_batch2_orderflow.py > $DIR/exp500_batch2_valid.log 2>&1 &
PID7=$!

python3 $DIR/exp500_batch3_oi.py > $DIR/exp500_batch3_valid.log 2>&1 &
PID8=$!

python3 $DIR/exp500_batch4_funding.py > $DIR/exp500_batch4_valid.log 2>&1 &
PID9=$!

python3 $DIR/exp500_batch5_hybrids.py > $DIR/exp500_batch5_valid.log 2>&1 &
PID10=$!

echo "PIDs: $PID1 $PID3 $PID4 $PID5 $PID6 $PID7 $PID8 $PID9 $PID10"

for p in $PID1 $PID3 $PID4 $PID5 $PID6 $PID7 $PID8 $PID9 $PID10; do
    wait $p
    echo "Process $p finished."
done

echo "All 9 parallel batches finished successfully!"
