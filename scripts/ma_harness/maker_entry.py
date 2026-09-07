"""Would a resting limit entry fill, and would the fills be worth having?

The engine currently buys the open of the candle after the signal, paying the
taker fee. The alternative is to rest a post-only order at the signal candle's
close for that hour: it fills at a better price and the maker fee, but only when
price comes back to it -- which is the definition of adverse selection on a
breakout. This measures both halves: how often it fills, and what the fills earn
against what the crossing entries earn.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/root/freqtrade/scripts/ma_harness")
import trend_lab as tl  # noqa: E402

TAKER, MAKER = 0.0005, 0.0002


def main():
    sel = tl.selection_mask(3, 3, top_vol=30)
    d = tl.load()
    close, high, low = d["close"], d["high"], d["low"]
    op = np.asarray(d["p"]["open"], dtype=np.float32)
    fund = d["fund"]
    raw = tl.donchian_state(close, 96, 12, high, low, 2.0, 24)
    f = pd.DataFrame(fund).rolling(3 * 24).mean().shift(1).to_numpy()
    af = np.abs(f)
    thr = np.nanquantile(np.where(np.isfinite(af), af, np.nan), 0.8,
                         axis=1, keepdims=True)
    pos = np.where(sel & (af >= thr), raw, 0.0)

    prev = np.vstack([np.zeros((1, pos.shape[1])), pos[:-1]])
    entries = np.argwhere((prev == 0) & (pos != 0))
    print(f"入场事件 {len(entries)} 次")

    rows = []
    T = pos.shape[0]
    for t, c in entries:
        if t + 1 >= T:
            continue
        side = pos[t, c]
        sig_close = float(close[t, c])          # 信号 K 线收盘
        nxt_open = float(op[t + 1, c])          # 现模型的成交价
        nxt_low, nxt_high = float(low[t + 1, c]), float(high[t + 1, c])
        if not np.isfinite([sig_close, nxt_open, nxt_low, nxt_high]).all():
            continue
        # 挂在信号收盘价：多头要价格回落到该价位，空头要反弹上来
        filled = (nxt_low <= sig_close) if side > 0 else (nxt_high >= sig_close)
        # 持有到状态机自己出场，出场价用那一根的开盘（与现模型一致）
        j = t + 1
        while j < T and pos[j, c] == side:
            j += 1
        if j >= T:
            continue
        exit_px = float(op[j, c])
        if not np.isfinite(exit_px):
            continue
        fnd = float(np.nansum(fund[t + 1:j + 1, c])) / 8.0 * side
        gross_taker = side * (exit_px / nxt_open - 1.0)
        gross_maker = side * (exit_px / sig_close - 1.0)
        rows.append(dict(
            filled=bool(filled),
            taker=gross_taker - 2 * TAKER - fnd,
            maker=gross_maker - MAKER - TAKER - fnd,
            gap=side * (nxt_open / sig_close - 1.0)))
    df = pd.DataFrame(rows)
    fill = df.filled.mean()
    print(f"限价挂在信号收盘价、有效期 1 小时：成交率 {100*fill:.1f}%")
    print(f"跳空（下一根开盘相对信号收盘，顺方向为正）: "
          f"均值 {100*df.gap.mean():+.3f}%  中位 {100*df.gap.median():+.3f}%")
    print(f"\n{'情形':<34}{'笔数':>7}{'平均每笔':>10}{'合计':>10}")
    print(f"{'现模型：全部下一根开盘吃单':<34}{len(df):>7}{100*df.taker.mean():>+9.3f}%{100*df.taker.sum():>+9.1f}pp")
    fl = df[df.filled]
    print(f"{'只做 maker：成交的那些（挂单价成交）':<34}{len(fl):>7}{100*fl.maker.mean():>+9.3f}%{100*fl.maker.sum():>+9.1f}pp")
    print(f"{'  同一批单子若改为吃单':<34}{len(fl):>7}{100*fl.taker.mean():>+9.3f}%{100*fl.taker.sum():>+9.1f}pp")
    un = df[~df.filled]
    print(f"{'被放弃的（挂单未成交）若吃单能赚':<34}{len(un):>7}{100*un.taker.mean():>+9.3f}%{100*un.taker.sum():>+9.1f}pp")
    print(f"\n混合：挂单成交则 maker，未成交则下一根开盘吃单")
    mix = np.where(df.filled, df.maker, df.taker)
    print(f"{'':<34}{len(df):>7}{100*mix.mean():>+9.3f}%{100*mix.sum():>+9.1f}pp")


if __name__ == "__main__":
    main()
