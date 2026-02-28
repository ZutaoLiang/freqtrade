timeframe=$1
timerange=$2
strategy=$3
config="user_data/config/config-$strategy.json"

if [ ! -f "$config" ]; then
    config="config.json"
fi

echo "Downloading backtest data for $strategy on timeframe:$timeframe in timerange:$timerange with config:$config"

whitelist=$(cat $config | tr -d '\n' | grep -oP '(?<="pair_whitelist": \[)[^\]]*' | tr -d '"' | tr ',' ' ')

extra_args="${@:4}"
./run.sh download-data --timeframe $timeframe --timerange $timerange --pairs $whitelist $extra_args
