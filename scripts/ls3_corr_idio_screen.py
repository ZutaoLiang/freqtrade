"""Correlation/idio-vol screen for LongShortV3 basket selection.

Premise: strategy profits from portfolio-level TP/TS on alts that co-move with ETH,
so select for high correlation, low idiosyncratic vol, stable beta, thick liquidity.
IS = 2025-01..10 (selection), OOS = 2025-11..2026-08 (persistence check).
"""
import numpy as np
import pandas as pd

OUT = '/root/freqtrade/user_data/basket_exp'
IS = ('2025-01-01', '2025-10-31')
OOS = ('2025-11-01', '2026-08-16')

close = pd.read_pickle(f'{OUT}/close_1h.pkl')
ret = np.log(close).diff()

eth = ret['ETHUSDT']


def metrics(name, rng):
    r = ret[name].loc[rng[0]:rng[1]]
    e = eth.loc[rng[0]:rng[1]]
    m = pd.concat([r, e], axis=1).dropna()
    if len(m) < 2000:
        return None
    y, x = m.iloc[:, 0], m.iloc[:, 1]
    beta = x.cov(y) / x.var()
    resid = y - beta * x
    corr = x.corr(y)
    if not np.isfinite(beta) or y.var() <= 0 or x.var() <= 0:
        return None
    idio_share = resid.var() / y.var()
    # rolling beta stability: 30d windows
    rb = x.rolling(24 * 30).cov(y) / x.rolling(24 * 30).var()
    rbm = rb.dropna()
    rbstd = rbm.std() / abs(rbm.mean()) if abs(rbm.mean()) > 1e-9 else np.inf
    # spread drift: std of daily log-spread change (lower = slow-wandering spread)
    ls = (np.log(close[name]) - beta * np.log(close['ETHUSDT'])).loc[rng[0]:rng[1]]
    spread_vol = ls.diff().dropna().std()
    return dict(beta=beta, corr=corr, idio=idio_share, rbstd=rbstd,
                spread_vol=spread_vol, n=len(m))


rows = []
for name in close.columns:
    if name == 'ETHUSDT':
        continue
    liq_mask = None
    mi, mo = metrics(name, IS), metrics(name, OOS)
    if not mi or not mo:
        continue
    rows.append(dict(pair=name, **{f'{k}_is': v for k, v in mi.items()},
                     **{f'{k}_oos': v for k, v in mo.items()}))

df = pd.DataFrame(rows)
# liquidity from the 5m feathers
import glob, os
liq = {}
for f in glob.glob('/root/freqtrade/user_data/data/screen_5m/*-5m.feather'):
    n = os.path.basename(f).replace('-5m.feather', '')
    d = pd.read_feather(f, columns=['date', 'quote_volume'])
    liq[n] = d.set_index('date')['quote_volume'].resample('1D').sum().median()
df['liq'] = df['pair'].map(liq)
df.to_csv(f'{OUT}/corr_idio_results.csv', index=False)
print('pairs tested:', len(df))
print('liq>$20M:', (df.liq > 2e7).sum())
