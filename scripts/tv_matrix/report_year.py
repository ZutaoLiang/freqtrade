"""Build the group x timeframe tables for a full-year matrix run."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TF_ORDER = ["1d", "1h", "30m", "15m", "5m", "1m"]
GROUP_ORDER = ["majors", "high_vol", "low_vol", "mixed"]


def pivot(df: pd.DataFrame, value: str, agg: str = "mean") -> pd.DataFrame:
    p = df.pivot_table(index="group", columns="timeframe", values=value, aggfunc=agg)
    return p.reindex(index=GROUP_ORDER, columns=TF_ORDER)


def md_table(p: pd.DataFrame, fmt: str) -> str:
    head = "| 组 \\ 周期 | " + " | ".join(TF_ORDER) + " |"
    sep = "| --- | " + " | ".join(["---:"] * len(TF_ORDER)) + " |"
    rows = [
        "| " + g + " | " + " | ".join(
            ("-" if pd.isna(p.loc[g, c]) else format(p.loc[g, c], fmt)) for c in TF_ORDER
        ) + " |"
        for g in GROUP_ORDER
    ]
    return "\n".join([head, sep] + rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="user_data/research/tradingview/year_trendshift_2026/summary.csv")
    ap.add_argument("--exclude-months", default="", help="comma separated, e.g. 2026-08")
    args = ap.parse_args()

    df = pd.read_csv(args.summary)
    df = df[df["status"] == "ok"].copy()
    excl = {m for m in args.exclude_months.split(",") if m}
    full = df[~df["month"].isin(excl)]

    out = []
    out.append(f"单元总数 {len(df)}，其中完整月单元 {len(full)}；"
               f"月份 {sorted(df['month'].unique())}\n")

    out.append("### 月度收益率均值 %（每组 1000 USDT 起始资金）\n")
    out.append(md_table(pivot(full, "net_profit_pct"), ".2f"))
    out.append("\n### 月度收益率中位数 %\n")
    out.append(md_table(pivot(full, "net_profit_pct", "median"), ".2f"))
    out.append("\n### 夏普均值（freqtrade 年化口径，单月窗口噪声大）\n")
    out.append(md_table(pivot(full, "sharpe"), ".2f"))
    out.append("\n### 最大回撤均值 %\n")
    out.append(md_table(pivot(full, "max_drawdown_pct"), ".2f"))
    out.append("\n### 最大回撤最差值 %\n")
    out.append(md_table(pivot(full, "max_drawdown_pct", "max"), ".2f"))
    out.append("\n### 盈利月份数 / 总月份数\n")
    win = full.assign(w=(full["net_profit_pct"] > 0).astype(int))
    p_win = win.pivot_table(index="group", columns="timeframe", values="w", aggfunc="sum")
    p_n = win.pivot_table(index="group", columns="timeframe", values="w", aggfunc="count")
    p = (p_win.astype(int).astype(str) + " / " + p_n.astype(int).astype(str))
    p = p.reindex(index=GROUP_ORDER, columns=TF_ORDER)
    head = "| 组 \\ 周期 | " + " | ".join(TF_ORDER) + " |"
    sep = "| --- | " + " | ".join(["---:"] * len(TF_ORDER)) + " |"
    out.append("\n".join([head, sep] + [
        "| " + g + " | " + " | ".join(str(p.loc[g, c]) for c in TF_ORDER) + " |"
        for g in GROUP_ORDER
    ]))
    out.append("\n### 交易数合计\n")
    out.append(md_table(pivot(full, "trades", "sum"), ".0f"))
    out.append("\n### 手续费合计 USDT\n")
    out.append(md_table(pivot(full, "fees_abs", "sum"), ".0f"))
    out.append("\n### 毛利合计 USDT（净利 + 手续费 + 资金费）\n")
    out.append(md_table(pivot(full, "gross_profit_abs", "sum"), ".0f"))

    text = "\n".join(out)
    Path(args.summary).with_name("tables.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
