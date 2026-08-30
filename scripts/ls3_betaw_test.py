#!/usr/bin/env python3
"""Test beta-weighted alt-leg staking (LongShortV3BetaW) vs known results."""
import json
import subprocess
import re
from pathlib import Path

ROOT = Path('/root/freqtrade')
DATADIR = str(ROOT / 'user_data/data_full/binance')
BASE_CFG = str(ROOT / 'user_data/basket_exp/base-config-bt.json')
OUT = ROOT / 'user_data/basket_exp'
FT = '/root/freqtrade/.venv/bin/freqtrade'

W = {'FULL': '20260101-20260817', 'IS': '20260101-20260601', 'OOS': '20260601-20260817',
     'Y25': '20250101-20260101'}
B = {'C1': ['RENDER', 'SKY', 'LDO', 'ARB', 'AVAX'],
     'R3b': ['LINK', 'UNI', 'SKY', 'SOL', 'DOGE'],
     'C4': ['AVAX', 'LTC', 'DOGE', 'LINK', 'UNI'],
     'E0': ['LINK', 'SUI', 'UNI', 'ADA', 'SKY']}
JOBS = [('C1', 'FULL'), ('C1', 'IS'), ('C1', 'OOS'), ('C1', 'Y25'),
        ('R3b', 'FULL'), ('C4', 'FULL'), ('E0', 'FULL')]


def run(name, win):
    cfg = json.loads(Path(BASE_CFG).read_text())
    cfg['exchange']['pair_whitelist'] = ['ETH/USDT:USDT'] + [f'{a}/USDT:USDT' for a in B[name]]
    cfg['exchange']['pair_blacklist'] = []
    cfg['main_pairs'] = 'ETH/USDT:USDT'
    cfg['enable_dynamic_stake'] = True
    p = OUT / f'cfg_bw_{name}_{win}.json'
    p.write_text(json.dumps(cfg, indent=1))
    log = OUT / f'log_bw_{name}_{win}.txt'
    cmd = [FT, 'backtesting', '--config', str(p), '--strategy', 'LongShortV3BetaW',
           '--datadir', DATADIR, '--timerange', W[win], '--timeframe', '5m']
    with log.open('w') as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    out = {'name': f'{name}_{win}'}
    for line in log.read_text().splitlines():
        if '│ LongShortV3BetaW │' in line:
            c = [x.strip() for x in line.split('│')[1:-1]]
            out.update(trades=int(c[1]), profit_usdt=float(c[3].replace(',', '')),
                       profit_pct=float(c[4].replace(',', '')))
            break
    return out


def main():
    from concurrent.futures import ThreadPoolExecutor
    res = []
    with ThreadPoolExecutor(4) as ex:
        for r in ex.map(lambda j: run(*j), JOBS):
            res.append(r)
            print(r, flush=True)
    json.dump(res, (OUT / 'results_betaw.json').open('w'), indent=1)


if __name__ == '__main__':
    main()
