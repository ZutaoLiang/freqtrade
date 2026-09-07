"""Export the screen and the funding gate so a freqtrade strategy can replay them.

The vectorised research runs the coin screen and the funding gate as panel
operations across all 860 symbols at once. A freqtrade strategy sees one pair at
a time and has no funding column, so both are precomputed here and shipped as a
lookup table. What the native backtest then re-derives on its own is exactly the
part under test: the Donchian channels, the ATR trail, order fills, fees and
funding accounting.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/root/freqtrade/scripts/ma_harness")
import trend_lab as tl  # noqa: E402

OUT = Path("/root/freqtrade/user_data/research/ma_harness")


def main():
    sel = tl.selection_mask(3, 3, top_vol=30)
    d = tl.load()
    p = d["p"]
    f = pd.DataFrame(d["fund"]).rolling(3 * 24).mean().shift(1).to_numpy()
    af = np.abs(f)
    thr = np.nanquantile(np.where(np.isfinite(af), af, np.nan), 0.8,
                         axis=1, keepdims=True)
    gate = sel & (af >= thr)

    live = np.flatnonzero(sel.any(axis=1))
    lo, hi = int(live[0]), int(live[-1])
    cols = np.flatnonzero(gate[lo:hi + 1].any(axis=0))
    idx = p.index[lo:hi + 1]
    pairs = [f"{p.symbols[c][:-4]}/USDT:USDT" for c in cols]

    allow = pd.DataFrame(gate[lo:hi + 1][:, cols], index=idx, columns=pairs)
    allow.index.name = "date"
    allow.to_parquet(OUT / "gate.parquet")
    (OUT / "universe_vsd.json").write_text(json.dumps(
        {"groups": {"vsd": pairs}}, indent=1))
    print(f"{len(pairs)} pairs ever tradable, {len(idx)} bars "
          f"{idx[0]} .. {idx[-1]}")
    print(f"bars with at least one pair open to entry: "
          f"{100*allow.any(axis=1).mean():.1f}%, "
          f"mean pairs allowed per bar {allow.sum(axis=1).mean():.2f}")
    print("wrote", OUT / "gate.parquet", "and", OUT / "universe_vsd.json")


if __name__ == "__main__":
    main()
