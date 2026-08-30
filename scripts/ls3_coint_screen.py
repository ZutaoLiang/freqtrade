#!/usr/bin/env python3
"""Engle-Granger cointegration screen: log(alt) ~ log(ETH), ADF on residual.

Windows:
  S1 = 2025-02-01..2025-12-31 (1h feathers in data dir; limited universe)
  S2 = 2026-01-01..2026-05-31 (resampled from data_full 5m; full 42-pair universe)
S2 is the primary screen -> Jun-Aug 2026 stays untouched for OOS validation.
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

DATA = Path('/root/freqtrade/user_data/data/binance/futures')
FULL = Path('/root/freqtrade/user_data/data_full/binance/futures')
OUT = Path('/root/freqtrade/user_data/basket_exp/coint_screen.csv')


def load_feather(path):
    d = pd.read_feather(path)
    d = d[['date', 'close']].dropna()
    d['date'] = pd.to_datetime(d['date'], utc=True)
    return d.set_index('date')['close']


def resample_1h(path):
    d = pd.read_feather(path)
    d['date'] = pd.to_datetime(d['date'], utc=True)
    s = d.set_index('date')['close'].resample('1h', label='left', closed='left').last().dropna()
    return s


def eg_test(y, x):
    """Both pd.Series, aligned. Returns dict of cointegration stats."""
    ly, lx = np.log(y.values), np.log(x.values)
    X = sm.add_constant(lx)
    ols = sm.OLS(ly, X).fit()
    beta, resid = ols.params[1], ols.resid
    adf = adfuller(resid, regression='c', autolag='AIC')
    adf_stat, adf_p = adf[0], adf[1]
    # OU half-life: de_t = gamma * e_{t-1}
    de = np.diff(resid)
    e_lag = resid[:-1]
    g = sm.OLS(de, sm.add_constant(e_lag)).fit().params[1]
    hl_h = -np.log(2) / g if g < 0 else np.inf
    # spread scale in log units, and z-score tradability: std of resid
    return dict(beta=beta, adf=adf_stat, p=adf_p, hl_days=hl_h / 24.0,
                resid_std=float(np.std(resid)))


def window_1h(full: bool):
    if full:
        eth = resample_1h(FULL / 'ETH_USDT_USDT-5m-futures.feather')
        eth = eth['2026-01-01':'2026-05-31']
        get = lambda s: resample_1h(FULL / f'{s}_USDT_USDT-5m-futures.feather')['2026-01-01':'2026-05-31']
        syms = sorted(p.stem.replace('_USDT_USDT-5m-futures', '') for p in FULL.glob('*-5m-futures.feather'))
    else:
        eth = load_feather(DATA / 'ETH_USDT_USDT-1h-futures.feather')['2025-02-01':'2025-12-31']
        get = lambda s: load_feather(DATA / f'{s}_USDT_USDT-1h-futures.feather')['2025-02-01':'2025-12-31']
        syms = sorted(p.stem.replace('_USDT_USDT-1h-futures', '') for p in DATA.glob('*-1h-futures.feather'))
    syms = [s for s in syms if s != 'ETH']
    rows = []
    for s in syms:
        p = (FULL if full else DATA) / f'{s}_USDT_USDT-{"5m" if full else "1h"}-futures.feather'
        if not p.exists():
            continue
        try:
            y = get(s)
        except Exception:
            continue
        j = pd.concat([y.rename('y'), eth.rename('x')], axis=1, join='inner').dropna()
        if len(j) < 3000:
            continue
        r = eg_test(j['y'], j['x'])
        r.update(sym=s, n=len(j))
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    s2 = window_1h(True).add_prefix('s2_').rename(columns={'s2_sym': 'sym'})
    s1 = window_1h(False).add_prefix('s1_')
    m = s2.merge(s1[['s1_sym', 's1_adf', 's1_p', 's1_hl_days', 's1_resid_std']],
                 left_on='sym', right_on='s1_sym', how='left')
    # composite rank: lower adf better; hl<20d and resid_std<0.5 = usable
    m['rank_adf'] = m['s2_adf'].rank()
    m['stable'] = ((m['s1_adf'] < -3.0) & (m['s2_adf'] < -3.0)).astype(int)
    m = m.sort_values('s2_adf')
    OUT.parent.mkdir(exist_ok=True)
    m.to_csv(OUT, index=False, float_format='%.4f')
    pd.set_option('display.width', 200)
    cols = ['sym', 's2_adf', 's2_p', 's2_hl_days', 's2_resid_std', 's2_beta', 's2_n',
            's1_adf', 's1_hl_days', 'stable']
    print(m[cols].to_string(index=False))


if __name__ == '__main__':
    main()
