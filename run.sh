script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $script_dir
proxychains4 .venv/bin/python freqtrade/main.py $@

