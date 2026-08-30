"""Walk-forward validation for LongShortV3 basket selection.

Each month M: select top-5 alts using ONLY trailing 6m data (corr/idio/beta-stability
rank, liquidity>$20M/d, corr>0.5, complete local data), backtest month M with ETH+picks.
Fixed portfolios (R3b, original whitelist) run over identical monthly segments.
Outputs configs + schedule CSV; backtests are run by wf_run.sh.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

OUT = '/root/freqtrade/user_data/basket_exp'
DATA = '/root/freqtrade/user_data/data/binance/futures'
close = pd.read_pickle(f'{OUT}/close_1h.pkl')
ret = np.log(close).diff()
eth = ret['ETHUSDT']

# universe: complete local data (5m + funding + mark), dedupe USDC variants
fut = {os.path.basename(f).split('-')[0] for f in glob.glob(f'{DATA}/*-5m-futures.feather')}
fund = {os.path.basename(f).split('-')[0] for f in glob.glob(f'{DATA}/*-funding_rate.feather')}
mark = {os.path.basename(f).split('-')[0] for f in glob.glob(f'{DATA}/*-mark.feather')}
universe = fut & fund & mark
# dedupe: keep USDT variant of each base
bases = {}
for s in universe:
    base = s.replace('USDT', '').replace('USDC', '')
    prefer = s.endswith('USDT') or base not in bases
    if prefer or base not in bases:
        if s.endswith('USDT') or base not in bases:
            bases.setdefault(base, s)
        if s.endswith('USDT'):
            bases[base] = s
universe = set(bases.values()) - {'ETHUSDT'}
# normalize: futures file symbol 'DOGE_USDT_USDT' -> data symbol 'DOGEUSDT', exchange 'DOGE/USDT:USDT'
name_map = {}
for s in list(universe):
    parts = s.split('_')
    if len(parts) == 3:
        base, quote = parts[0], parts[1]
        data_name = base + quote
        if data_name in close.columns:
            name_map[s] = data_name
universe = set(name_map) - {'ETH_USDT_USDT', 'BTC_USDT_USDT'}  # ETH is the signal anchor; BTC blacklisted in config
# liquidity (full-period daily median quote volume; 5m feathers first, screen_5m fallback)
liq = {}
for s in universe:
    ds = name_map[s]
    val = None
    for f in (f'{DATA}/{s}-5m-futures.feather', f'/root/freqtrade/user_data/data/screen_5m/{ds}-5m.feather'):
        if os.path.exists(f):
            try:
                d = pd.read_feather(f, columns=['date', 'quote_volume'])
                val = d.set_index('date')['quote_volume'].resample('1D').sum().median()
                break
            except Exception:
                continue
    liq[s] = val if val is not None else 0


def metrics(sym, rng):
    r = ret[sym].loc[rng[0]:rng[1]]
    e = eth.loc[rng[0]:rng[1]]
    m = pd.concat([r, e], axis=1).dropna()
    if len(m) < 24 * 120:  # at least ~120 days of overlap
        return None
    y, x = m.iloc[:, 0], m.iloc[:, 1]
    if x.var() <= 0 or y.var() <= 0:
        return None
    beta = x.cov(y) / x.var()
    if not np.isfinite(beta):
        return None
    resid = y - beta * x
    corr = x.corr(y)
    rb = x.rolling(24 * 30).cov(y) / x.rolling(24 * 30).var()
    rbm = rb.dropna()
    rbstd = rbm.std() / abs(rbm.mean()) if abs(rbm.mean()) > 1e-9 else np.inf
    ls = (np.log(close[sym]) - beta * np.log(close['ETHUSDT'])).loc[rng[0]:rng[1]]
    spread_vol = ls.diff().dropna().std()
    return dict(corr=corr, idio=resid.var() / y.var(), rbstd=rbstd, spread_vol=spread_vol)


months = pd.period_range('2025-07', '2026-08', freq='M')
schedule = []
for pm in months:
    m_start = pm.start_time.tz_localize('UTC')
    m_end = (pm + 1).start_time.tz_localize('UTC')
    seg = f'{pm.start_time.strftime("%Y%m%d")}-{m_end.strftime("%Y%m%d")}'
    t_end = m_start - pd.Timedelta(hours=1)
    t_start = t_end - pd.DateOffset(months=6)
    rng = (str(t_start.date()), str(t_end.date()))
    scored = []
    for s in universe:
        if liq.get(s, 0) < 2e7:
            continue
        r = metrics(name_map[s], rng)
        if r and r['corr'] > 0.5:
            scored.append((s, r))
    if len(scored) < 5:
        print(f'{pm}: only {len(scored)} candidates, skip')
        continue
    df = pd.DataFrame({s: r for s, r in scored}).T
    rank = df['corr'].rank(ascending=False) + df['idio'].rank() + df['rbstd'].rank() + df['spread_vol'].rank()
    picks = rank.sort_values().head(5).index.tolist()
    cfg_path = f'{OUT}/wf_cfg/cfg_WF_{seg}.json'
    os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
    base = json.load(open('/root/freqtrade/user_data/config/LongShortV3-config.json'))
    base['exchange']['ccxt_config']['timeout'] = 30000
    base['exchange']['ccxt_async_config']['timeout'] = 30000
    base['exchange']['ccxt_async_config']['aiohttp_proxy'] = 'http://127.0.0.1:10811'
    base['exchange']['pair_whitelist'] = ['ETH/USDT:USDT'] + [f"{s.split('_')[0]}/USDT:USDT" for s in picks]
    json.dump(base, open(cfg_path, 'w'), indent=4)
    # fixed portfolios over same segment
    for name, pairs in {
        'R3b': ['LINK/USDT:USDT', 'UNI/USDT:USDT', 'SKY/USDT:USDT', 'SOL/USDT:USDT', 'DOGE/USDT:USDT'],
        'ORIG': ['LINK/USDT:USDT', 'SUI/USDT:USDT', 'UNI/USDT:USDT', 'ADA/USDT:USDT', 'SKY/USDT:USDT'],
    }.items():
        c = json.loads(json.dumps(base))
        c['exchange']['pair_whitelist'] = ['ETH/USDT:USDT'] + pairs
        json.dump(c, open(f'{OUT}/wf_cfg/cfg_{name}_{seg}.json', 'w'), indent=4)
    schedule.append(dict(seg=seg, picks=','.join(picks)))
    print(pm, 'picks:', picks, flush=True)

pd.DataFrame(schedule).to_csv(f'{OUT}/wf_schedule.csv', index=False)
print('schedule written:', len(schedule), 'months')
