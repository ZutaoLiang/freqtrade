"""Prove the backtest only ever traded the coins the screen allowed.

The whitelist holds every coin that was ever selected, so freqtrade itself
places no restriction on when a coin may be traded -- the restriction lives in
the strategy, which zeroes its position whenever the precomputed gate is false.
This checks that claim against the trades the engine actually made, hour by
hour, and measures the one-candle overshoot the next-open fill implies.
"""
import glob
import json
import sys
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, "/root/freqtrade/scripts/ma_harness")
import trend_lab as tl  # noqa: E402


def main():
    z = sorted(glob.glob(
        "/root/freqtrade/user_data/backtest_results/vsd_audit/*.zip"))[-1]
    with zipfile.ZipFile(z) as f:
        name = [n for n in f.namelist()
                if n.endswith(".json") and "meta" not in n and "config" not in n][0]
        d = json.load(f.open(name))
    ft = pd.DataFrame(d["strategy"][list(d["strategy"])[0]]["trades"])
    ft["open_date"] = pd.to_datetime(ft["open_date"], utc=True)
    ft["close_date"] = pd.to_datetime(ft["close_date"], utc=True)
    print(f"审计对象: {z.split('/')[-1]}  {len(ft)} 笔")

    sel = tl.selection_mask(3, 3, top_vol=30)
    p = tl.load()["p"]
    gate = pd.read_parquet(
        "/root/freqtrade/user_data/research/ma_harness/gate.parquet")
    # the raw panel carries USDC twins and index products that map onto the
    # same freqtrade pair name, so keep only the symbols the screen can pick
    non_coin = set(json.load(open(
        "/root/freqtrade/user_data/research/ma_harness/non_coin_symbols.json")))
    keep = [i for i, s in enumerate(p.symbols)
            if s.isascii() and s.endswith("USDT") and s not in non_coin]
    basket = pd.DataFrame(sel[:, keep], index=p.index,
                          columns=[f"{p.symbols[i][:-4]}/USDT:USDT" for i in keep])
    basket = basket.loc[gate.index, gate.columns]

    pos_in, pos_tot = 0, 0
    gate_in = 0
    entry_ok, entry_gate_ok = 0, 0
    outside_runs = []
    for r in ft.itertuples():
        hours = pd.date_range(r.open_date, r.close_date - pd.Timedelta(hours=1),
                              freq="1h")
        hours = hours[hours.isin(basket.index)]
        if len(hours) == 0:
            continue
        inb = basket.loc[hours, r.pair].to_numpy()
        ing = gate.loc[hours, r.pair].to_numpy()
        pos_tot += len(hours)
        pos_in += int(inb.sum())
        gate_in += int(ing.sum())
        if r.open_date in basket.index:
            entry_ok += int(bool(basket.loc[r.open_date, r.pair]))
            entry_gate_ok += int(bool(gate.loc[r.open_date, r.pair]))
        outside_runs.append(int((~inb).sum()))

    outside_runs = np.array(outside_runs)
    print(f"\n持仓小时总数 {pos_tot}")
    print(f"  在当期 30 币池内: {pos_in} ({100*pos_in/pos_tot:.3f}%)")
    print(f"  同时满足资金费门槛: {gate_in} ({100*gate_in/pos_tot:.3f}%)")
    print(f"入场那一根在币池内的比例 {100*entry_ok/len(ft):.3f}%，"
          f"同时门槛为真 {100*entry_gate_ok/len(ft):.3f}%")
    print(f"每笔在池外的小时数: 0 小时的占 {100*(outside_runs==0).mean():.1f}%，"
          f"1 小时的占 {100*(outside_runs==1).mean():.1f}%，"
          f"最多 {outside_runs.max()} 小时")

    # 并发币数是否越界
    hours = basket.index
    held = pd.DataFrame(False, index=hours, columns=gate.columns)
    for r in ft.itertuples():
        h = pd.date_range(r.open_date, r.close_date - pd.Timedelta(hours=1),
                          freq="1h")
        h = h[h.isin(hours)]
        if len(h):
            held.loc[h, r.pair] = True
    n = held.sum(axis=1)
    print(f"\n同时持仓币数: 均值 {n.mean():.2f}, 最大 {int(n.max())}, "
          f"超过 30 的小时数 {int((n > 30).sum())}")
    both = (held.to_numpy() & ~basket.to_numpy()).sum(axis=1)
    print(f"任一小时持有池外币的数量: 最大 {int(both.max())}, "
          f"出现过的小时数 {int((both > 0).sum())} / {len(hours)} "
          f"({100*(both>0).mean():.2f}%)")


if __name__ == "__main__":
    main()
