timeframe=$1
timerange=$2
strategy=$3
config="user_data/config/config-$strategy.json"

if [ ! -f "$config" ]; then
    config="config.json"
fi

echo "Backtesting $strategy on timeframe:$timeframe in timerange:[$timerange] with config:$config"

# ./download-backtest-data.sh $timeframe $timerange $strategy

./run.sh backtesting --timeframe $timeframe --timerange $timerange -s $strategy -c $config --enable-protections --cache none

