"""Base series for panels that lack count / taker / mark / OI fields (e.g. 1h_hist).

Same definitions as base.build for every series it can compute; series that need missing
fields are simply absent, and factors.for_timeframe drops specs that reference them.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import operators as ops  # noqa: E402
from base import BARS_PER_DAY, _div, cache_dir  # noqa: E402
from config import atomic_save  # noqa: E402
from panel import Panel  # noqa: E402


def build(panel):
    bpd = BARS_PER_DAY[panel.tf.split("_")[0]]
    day = max(2, bpd)
    close = np.asarray(panel["close"], dtype="float64")
    open_ = np.asarray(panel["open"], dtype="float64")
    high = np.asarray(panel["high"], dtype="float64")
    low = np.asarray(panel["low"], dtype="float64")
    volume = np.asarray(panel["volume"], dtype="float64")
    qv = np.asarray(panel["quote_volume"], dtype="float64")
    prev_close = ops.ts_delay(close, 1)
    ret1 = _div(close, prev_close) - 1.0
    vwap = _div(qv, volume)
    s = {
        "ret1": ret1,
        "gap": _div(open_, prev_close) - 1.0,
        "body": _div(close - open_, open_),
        "hl_range": _div(high - low, close),
        "upper_wick": _div(high - np.maximum(open_, close), close),
        "lower_wick": _div(np.minimum(open_, close) - low, close),
        "close_loc": _div(close - low, high - low),
        "vwap_dev": _div(close, vwap) - 1.0,
        "log_qv": ops.sign_log(qv),
        "vol_ratio": _div(volume, ops.ts_mean(volume, day)),
        "qv_ratio": _div(qv, ops.ts_mean(qv, day)),
        "amihud": _div(np.abs(ret1), qv) * 1e9,
        "close": close,
    }
    fields = set(panel.meta.get("fields", []))
    if "funding_rate" in fields:
        fr = np.asarray(panel["funding_rate"], dtype="float64")
        # the hist panel stores the rate only on settlement bars (0 elsewhere); carry it forward
        # like the main panel does. A genuine 0 rate is treated as missing -- rare and harmless.
        if (fr != 0).mean() < 0.5:
            import pandas as pd
            fr = pd.DataFrame(np.where(fr != 0, fr, np.nan)).ffill(limit=8 * bpd).to_numpy()
        s["funding_rate"] = fr
        s["funding_annual"] = fr * 3.0 * 365.0   # 8h assumed where the interval field is absent
        s["funding_z"] = ops.ts_zscore(fr, 7 * day)
    return {k: v.astype("float32") for k, v in s.items()}


if __name__ == "__main__":
    for tf in sys.argv[1:] or ["1h_hist"]:
        p = Panel(tf)
        series = build(p)
        d = cache_dir(tf)
        os.makedirs(d, exist_ok=True)
        for k, v in series.items():
            atomic_save(os.path.join(d, f"{k}.npy"), lambda f, v=v: np.save(f, v))
        atomic_save(os.path.join(d, "index.json"), lambda f: f.write(json.dumps(sorted(series), indent=2).encode()))
        print(f"{p!r}: {len(series)} base series cached to {d}")
