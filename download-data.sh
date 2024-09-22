#!/bin/bash

time_range=$1
time_frames=(5m 15m 30m 1h 4h 6h 8h 12h 1d)

for time_frame in "${time_frames[@]}"; do
  bash ./run.sh download-data --timeframe $time_frame --timerange $time_range

  echo "Download $time_frame completed."
done

