timeframe=$1
timerange=$2

echo "Backtesting $strategy on timeframe:$timeframe in timerange:$timerange"

whitelist=$(cat config.json | tr -d '\n' | grep -oP '(?<="pair_whitelist": \[)[^\]]*' | tr -d '"' | tr ',' ' ')

./run.sh download-data --timeframe $timeframe --timerange $timerange --pairs $whitelist
