#!/bin/bash
# run one TRAIN script in r3c, log stdout, record time and exit code
s=$1
cd /root/freqtrade/user_data/minute_research/r3c
source /root/freqtrade/.venv/bin/activate
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 RUN_SEG=TRAIN SEG=TRAIN
start=$(date +%s)
timeout 3600 python $s > logs/${s%.py}.log 2>&1
echo "$s exit=$? sec=$(( $(date +%s)-start ))" >> logs/_done.txt
