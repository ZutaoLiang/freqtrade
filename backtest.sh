timeframe=$1
timerange=$2
strategy=$3

echo "Backtesting $strategy on timeframe:$timeframe in timerange:$timerange"

# whitelist=$(cat config.json | tr -d '\n' | grep -oP '(?<="pair_whitelist": \[)[^\]]*' | tr -d '"' | tr ',' ' ')
# ./run.sh download-data --timeframe $timeframe --timerange $timerange --pairs $whitelist

./run.sh backtesting --timeframe $timeframe --timerange $timerange -s $strategy --enable-protections

