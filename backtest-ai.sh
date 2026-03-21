timeframe=$1
timerange=$2
strategy=$3
model=$4
extra_args="${@:5}"

config="user_data/config/config-$strategy.json"

if [ ! -f "$config" ]; then
    config="config.json"
fi

echo "Backtesting $strategy on timeframe:$timeframe in timerange:[$timerange] with config:$config"

# ./download-backtest-data.sh $timeframe $timerange $strategy
echo "./run.sh backtesting --timerange $timerange -s $strategy -c $config --freqaimodel $model --enable-protections --cache none $extra_args"
./run.sh backtesting --timeframe $timeframe --timerange $timerange -s $strategy -c $config --freqaimodel $model --enable-protections --cache none $extra_args

