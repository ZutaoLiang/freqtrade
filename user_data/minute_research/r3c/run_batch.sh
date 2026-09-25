#!/bin/bash
s=$1; seg=$2
cd /root/freqtrade/user_data/minute_research/r3c
source /root/freqtrade/.venv/bin/activate
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=8 RUN_SEG=$seg
start=$(date +%s)
timeout 7200 /usr/bin/time -f "PEAK %M" python $s > logs/${s%.py}_$seg.log 2>&1
echo "$s $seg exit=$? sec=$(( $(date +%s)-start ))" >> logs/_batch_done.txt
