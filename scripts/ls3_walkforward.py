#!/usr/bin/env python3
"""Monthly walk-forward basket selection for LongShortV3.

For each test month M (Apr..Aug-16 2026):
  screen on months [M-3, M-1] using 1h resampled from data_full 5m.
  - WF_COINT: filter |beta-1|<=0.25, rank by ADF stat asc, top 5 alts
  - CTRL_BETA: same filter, rank by |beta-1| asc (no cointegration info)
Backtest each on month M only. Report USDT P&L per month and total.
"""
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

ROOT = Path('/root/freqtrade')
DATADIR = str(ROOT / 'user_data/data_full/binance')
BASE_CFG = str(ROOT / 'user_data/basket_exp/base-config-bt.json')
OUTDIR = ROOT / 'user_data' / 'basket_exp'
FT = '/root/freqtrade/.venv/bin/freqtrade'

TESTS = [  # (label, timerange, screen_start, screen_end)
    ('apr', '20260401-20260501', '2026-01-01', '2026-03-31'),
    ('may', '20260501-20260601', '2026-02-01', '2026-04-30'),
    ('jun', '20260601-20260701', '2026-03-01', '2026-05-31'),
    ('jul', '20260701-20260801', '2026-04-01', '2026-06-30'),
    ('aug', '20260801-20260817', '2026-05-01', '2026-07-31'),
]


def r1h(path):
    d = pd.read_feather(path, columns=['date', 'close'])
    d['date'] = pd.to_datetime(d['date'], utc=True)
    return d.set_index('date')['close'].resample('1h', label='left', closed='left').last().dropna()


F = Path(DATADIR) / 'futures'
ETH = r1h(F / 'ETH_USDT_USDT-5m-futures.feather')
SYMS = sorted(p.stem.replace('_USDT_USDT-5m-futures', '') for p in F.glob('*-5m-futures.feather'))
SYMS = [s for s in SYMS if s != 'ETH']
_cache = {}


def series(sym):
    if sym not in _cache:
        _cache[sym] = r1h(F / f'{sym}_USDT_USDT-5m-futures.feather')
    return _cache[sym]


def screen(a, b):
    """Return DataFrame sym,beta,adf ranked. Window [a,b]."""
    rows = []
    ethw = ETH[a:b]
    for s in SYMS:
        y = series(s)[a:b]
        j = pd.concat([y.rename('y'), ethw.rename('x')], axis=1, join='inner').dropna()
        if len(j) < 1500:
            continue
        ols = sm.OLS(np.log(j['y']), sm.add_constant(np.log(j['x']))).fit()
        resid = ols.resid
        adf = adfuller(resid, regression='c', autolag='AIC')[0]
        rows.append((s, ols.params[1], adf))
    df = pd.DataFrame(rows, columns=['sym', 'beta', 'adf'])
    return df[df['beta'].between(0.75, 1.25)]


def bt(name, alts, tr):
    cfg = json.loads(Path(BASE_CFG).read_text())
    cfg['exchange']['pair_whitelist'] = [f'ETH/USDT:USDT'] + [f'{a}/USDT:USDT' for a in alts]
    cfg['exchange']['pair_blacklist'] = []
    cfg['main_pairs'] = 'ETH/USDT:USDT'
    p = OUTDIR / f'cfg_wf_{name}.json'
    p.write_text(json.dumps(cfg, indent=1))
    log = OUTDIR / f'log_wf_{name}.txt'
    cmd = [FT, 'backtesting', '--config', str(p), '--strategy', 'LongShortV3',
           '--datadir', DATADIR, '--timerange', tr, '--timeframe', '5m']
    with log.open('w') as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    text = log.read_text()
    out = {'name': name, 'profit': None, 'pct': None}
    for line in text.splitlines():
        if '│ LongShortV3 │' in line:
            c = [x.strip() for x in line.split('│')[1:-1]]
            out['trades'] = int(c[1])
            out['profit'] = float(c[3].replace(',', ''))
            out['pct'] = float(c[4].replace(',', ''))
            break
    return out


def main():
    jobs, picks = [], {}
    for lab, tr, sa, sb in TESTS:
        df = screen(sa, sb).sort_values('adf')
        coint = list(df['sym'].head(5))
        beta = list(df.assign(d=(df['beta'] - 1).abs()).sort_values('d')['sym'].head(5))
        picks[lab] = {'coint': coint, 'beta': beta, 'adf_top': list(zip(df['sym'], df['adf'].round(2)))[:8]}
        jobs.append((f'coint_{lab}', coint, tr))
        jobs.append((f'beta_{lab}', beta, tr))
    print(json.dumps(picks, indent=1))
    results = []
    with ThreadPoolExecutor(4) as ex:
        for r in ex.map(lambda j: bt(*j), jobs):
            results.append(r)
            print(r['name'], r['profit'], r['pct'], r.get('trades'), flush=True)
    json.dump({'picks': picks, 'results': results},
              (OUTDIR / 'results_walkforward.json').open('w'), indent=1, default=str)


if __name__ == '__main__':
    main()
