"""Causal liquidity ranking, so a screen can be re-run on a tradeable universe.

The whitelist that came out of the round-3 shortlist was topped by `4/USDT`,
`GUA/USDT` and `BTR/USDT` -- micro-caps where the flat 5 bps slippage the cost
model assumes is not credible. Rather than measure their true impact first
(`bookDepth`, 13 GiB), it is cheaper and more decisive to ask whether the edge
exists at all among names where 5 bps *is* credible. If it does not, no
slippage estimate can save it; if it does, slippage stops being the binding
question.

The rank here uses the same trailing 7-day median quote volume as the universe
mask, so it is causal at every bar: rank 0 is the most liquid symbol tradeable
at that bar. Symbols outside the tradeable universe get -1.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from panel import Panel, CACHE  # noqa: E402
from universe import LIQUIDITY_WINDOW_DAYS, _bars_per_day  # noqa: E402

# Cumulative, not disjoint. The decision this feeds is "which universe do I
# trade", and that is answered by nesting: if top-50 works and the whole set
# works, the edge is not a micro-cap artefact. The one disjoint tier, `tail`,
# is the diagnostic that says whether it is *only* a micro-cap artefact.
CUMULATIVE = (50, 100, 200)
TAIL_FROM = 200


def build(tf, window_days=LIQUIDITY_WINDOW_DAYS):
    p = Panel(tf)
    bpd = _bars_per_day(tf)
    mask = np.asarray(np.load(os.path.join(CACHE, tf, "universe_mask.npy"),
                              mmap_mode="r"))
    qv = np.asarray(p["quote_volume"], dtype="float64")
    win = max(1, window_days * bpd)
    med = pd.DataFrame(qv).rolling(win, min_periods=win // 2).median().to_numpy()

    # Rank only inside the tradeable set; anything else must never receive a
    # rank, or a dead pair with a stale volume median would displace a live one.
    med = np.where(mask & np.isfinite(med), med, -np.inf)
    order = np.argsort(-med, axis=1, kind="stable")
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(order.shape[1])[None, :], axis=1)
    rank = np.where(med > -np.inf, rank, -1).astype("int16")

    np.save(os.path.join(CACHE, tf, "liq_rank.npy"), rank)
    return rank


def load(tf):
    path = os.path.join(CACHE, tf, "liq_rank.npy")
    if not os.path.exists(path):
        build(tf)
    return np.load(path, mmap_mode="r")


def tier_masks(tf, base_mask, rank=None):
    """`{name: mask}` for every tier, including the full universe as `all`."""
    r = np.asarray(load(tf) if rank is None else rank)
    valid = base_mask & (r >= 0)
    out = {"all": base_mask}
    for n in CUMULATIVE:
        out[f"top{n}"] = valid & (r < n)
    out["tail"] = valid & (r >= TAIL_FROM)
    return out


if __name__ == "__main__":
    for tf in (sys.argv[1:] or ["1h"]):
        mask = np.asarray(np.load(os.path.join(CACHE, tf, "universe_mask.npy"),
                                  mmap_mode="r"))
        r = build(tf)
        p = Panel(tf)
        print(f"\n=== {tf} liquidity tiers ===")
        for name, m in tier_masks(tf, mask, r).items():
            w = m.sum(axis=1)
            print(f"{name:>6}: width median {np.median(w):.0f}  max {w.max()}")
        # Name the top of the book so the tiers can be sanity-checked by eye.
        last = np.asarray(r[-1])
        top = [p.symbols[i] for i in np.argsort(np.where(last >= 0, last, 1 << 14))[:15]]
        print("most liquid at last bar:", ", ".join(top))
