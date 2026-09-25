#!/bin/bash
# run every screened framework for one coin; usage: run_coin.sh COINUSDT PARALLEL
C=$1; P=${2:-6}; B=/root/freqtrade/user_data/minute_research/zec_screen/$C
mkdir -p $B/logs; : > $B/logs/_jobs.txt
for s in batch1_f1_f2 batch2_f3_f4 batch3_f5_f6 batch4_f7_f8 batch5_f9_f10 exp500_batch1_squeeze exp500_batch2_orderflow exp500_batch3_oi exp500_batch4_funding exp500_batch5_hybrids; do
  for seg in TRAIN VALID; do echo "r3c $s.py $seg" >> $B/logs/_jobs.txt; done; done
echo "r4_mtf run_300_rounds.py -" >> $B/logs/_jobs.txt
echo "r5_clean_mtf run_300_clean_mtf.py -" >> $B/logs/_jobs.txt
echo "r6_orthogonal_alphas run_300_orthogonal.py -" >> $B/logs/_jobs.txt
echo "r7_confluence_study run_confluence_study.py -" >> $B/logs/_jobs.txt
echo "r8_bear_adaptive run_300_bear_adaptive.py -" >> $B/logs/_jobs.txt
: > $B/logs/_done.txt
cat $B/logs/_jobs.txt | xargs -P $P -L 1 bash -c 'd=$0; s=$1; seg=$2; cd '"$B"'/$d; source /root/freqtrade/.venv/bin/activate; export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=4 SCREEN_COINS='"$C"' RUN_SEG=$seg; t0=$(date +%s); /usr/bin/time -f "PEAK %M" timeout 5400 python -W ignore $s > '"$B"'/logs/${d}_${s%.py}_$seg.log 2>&1; echo "$d $s $seg exit=$? sec=$(( $(date +%s)-t0 ))" >> '"$B"'/logs/_done.txt'
echo ALLDONE >> $B/logs/_done.txt
