#!/usr/bin/env python3
"""LongShortV3 basket-combination experiments.

Pairs are screened from 2025H2 klines (out-of-sample vs 2026 evaluation).
Each basket is backtested on IS (Jan-May 2026) and OOS (Jun-Aug 2026) separately.
"""
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path('/root/freqtrade')
DATADIR = str(ROOT / 'user_data/data_full/binance')
BASE_CFG = str(ROOT / 'user_data' / 'basket_exp' / 'base-config-bt.json')
OUTDIR = ROOT / 'user_data' / 'basket_exp'
OUTDIR.mkdir(exist_ok=True)

def usdt(p):  # alt symbol -> freqtrade pair
    return f"{p}/USDT:USDT"

EXPERIMENTS = {
    # name: (main_pair, [alt legs])
    'E0_baseline': ('ETH', ['LINK', 'SUI', 'UNI', 'ADA', 'SKY']),
    'E1_majors':   ('ETH', ['BNB', 'SOL', 'XRP', 'ADA', 'DOGE']),
    'E2_lowvr':    ('ETH', ['ONDO', 'RENDER', 'ENS', 'ETC', 'SAND']),
    'E3_defi':     ('ETH', ['UNI', 'CRV', 'LDO', 'PENDLE', 'AAVE']),
    'E4_l2cluster':('ETH', ['INJ', 'NEAR', 'ARB', 'TIA', 'SEI']),
    'E5_mainbtc':  ('BTC', ['ETH', 'SOL', 'XRP', 'ADA', 'BNB']),
    'E6_control':  ('ETH', ['HBAR', 'DOT', 'LTC', 'APT', 'OP']),
}

WINDOWS = {
    'IS':  '20260101-20260601',
    'OOS': '20260601-20260817',
    'FULL': '20260101-20260817',
}

METRIC_KEYS = [
    'profit_total_pct', 'trade_count', 'winrate', 'sharpe', 'sortino',
    'calmar', 'max_drawdown_account', 'profit_factor', 'cagr',
]

def make_config(name, main, alts, window, extra=None):
    cfg = json.loads(Path(BASE_CFG).read_text())
    pairs = [usdt(main)] + [usdt(a) for a in alts]
    cfg['exchange']['pair_whitelist'] = pairs
    cfg['exchange']['pair_blacklist'] = []
    cfg['main_pairs'] = usdt(main)
    cfg['_exp_name'] = name  # ignored, informational
    if extra:
        cfg.update(extra)
    p = OUTDIR / f'cfg_{name}_{window}.json'
    p.write_text(json.dumps(cfg, indent=1))
    return str(p)

def run_one(name, main, alts, window, tr, extra=None):
    cfg = make_config(name, main, alts, window, extra)
    log = OUTDIR / f'log_{name}_{window}.txt'
    cmd = [
        '/root/freqtrade/.venv/bin/freqtrade', 'backtesting', '--config', cfg, '--strategy', 'LongShortV3',
        '--datadir', DATADIR, '--timerange', tr, '--timeframe', '5m',
    ]
    with log.open('w') as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    text = log.read_text()
    m = re.search(r'(?P<k>[\w ]+?)\s*\│', text)
    # parse summary table lines "│ Metric │ Value │"
    out = {'exp': name, 'window': window, 'rc': r.returncode}
    for line in text.splitlines():
        if '│' not in line:
            continue
        cells = [c.strip() for c in line.split('│')[1:-1]]
        if len(cells) == 2:
            out[cells[0]] = cells[1]
    if 'Total profit %' not in out:
        out['ERROR'] = 'no result'
    return out

def main():
    extra = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    exps = json.loads(sys.argv[2]) if len(sys.argv) > 2 else EXPERIMENTS
    outname = sys.argv[3] if len(sys.argv) > 3 else 'results_b1.json'
    jobs = []
    for name, (mainp, alts) in exps.items():
        for window, tr in WINDOWS.items():
            jobs.append((name, mainp, alts, window, tr, extra))
    results = []
    with ThreadPoolExecutor(4) as ex:
        for res in ex.map(lambda j: run_one(*j), jobs):
            results.append(res)
            print(res.get('exp'), res.get('window'), res.get('Total profit %'),
                  res.get('Sharpe'), res.get('Absolute drawdown'), flush=True)
    json.dump(results, (OUTDIR / outname).open('w'), indent=1, default=str)

if __name__ == '__main__':
    main()
