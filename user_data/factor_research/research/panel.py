"""Aligned panel layout for factor research.

Every field becomes one float32 matrix of shape (time, symbol) sharing a single
date index and symbol order. Time-series operators then run as one vectorised
call down axis=0 across the whole universe instead of a per-pair loop, and the
same layout serves cross-sectional work later without a second data path.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import atomic_save  # noqa: E402

RICH = "/root/freqtrade/user_data/data/binance_public/resampled"
CACHE = "/root/freqtrade/user_data/data/binance_public/panels"
FIELDS = ["open", "high", "low", "close", "volume",
          "quote_volume", "count", "taker_buy_volume"]


def _step(tf):
    # "1h_hist" and similar variants share the base timeframe before the underscore
    tf = tf.split("_")[0]
    return pd.Timedelta(tf.replace("m", "min").replace("d", "D"))


def build(tf, fields=FIELDS, symbols=None):
    """Assemble panels for one timeframe and cache them to disk."""
    src = os.path.join(RICH, tf)
    files = sorted(f for f in os.listdir(src) if f.endswith(".parquet"))
    if symbols is not None:
        keep = set(symbols)
        files = [f for f in files if f[:-8] in keep]

    frames = {}
    for f in files:
        df = pd.read_parquet(os.path.join(src, f), columns=["date"] + list(fields))
        if len(df):
            frames[f[:-8]] = df.set_index("date")

    syms = sorted(frames)
    start = min(d.index[0] for d in frames.values())
    end = max(d.index[-1] for d in frames.values())
    index = pd.date_range(start, end, freq=_step(tf), tz="UTC")

    out = os.path.join(CACHE, tf)
    os.makedirs(out, exist_ok=True)
    for field in fields:
        # Concat once per field rather than reindexing each symbol separately:
        # one aligned join is far cheaper than N individual reindex calls.
        wide = pd.concat({s: frames[s][field] for s in syms}, axis=1)
        wide = wide.reindex(index)[syms]
        arr = wide.to_numpy(dtype="float32")
        atomic_save(os.path.join(out, f"{field}.npy"), lambda f: np.save(f, arr))
        del wide, arr

    meta = {"tf": tf, "symbols": syms, "fields": list(fields),
            "start": str(index[0]), "end": str(index[-1]), "n_bars": len(index)}
    atomic_save(os.path.join(out, "meta.json"),
                lambda f: f.write(json.dumps(meta, indent=2).encode()))
    return meta


def verify(tf):
    """Check every field matches the size meta.json implies.

    meta.json is small enough to survive a crash that truncates a 560 MiB
    field, so its presence alone does not mean the panel is intact.
    """
    d = os.path.join(CACHE, tf)
    meta = json.load(open(os.path.join(d, "meta.json")))
    expect = meta["n_bars"] * len(meta["symbols"]) * 4
    bad = []
    for field in meta["fields"]:
        path = os.path.join(d, f"{field}.npy")
        if not os.path.exists(path):
            bad.append((field, "missing", 0))
        else:
            actual = os.path.getsize(path) - 128  # npy header
            if abs(actual - expect) > 4096:
                bad.append((field, "truncated", actual))
    return meta, bad


class Panel:
    """Lazy accessor over the cached matrices; fields load on first use."""

    def __init__(self, tf, mmap=True):
        self.tf = tf
        self.dir = os.path.join(CACHE, tf)
        self.meta = json.load(open(os.path.join(self.dir, "meta.json")))
        self.symbols = self.meta["symbols"]
        self.index = pd.date_range(self.meta["start"], self.meta["end"],
                                   freq=_step(tf), tz="UTC")
        self._mmap = "r" if mmap else None
        self._cache = {}

    def __getitem__(self, field):
        if field not in self._cache:
            path = os.path.join(self.dir, f"{field}.npy")
            self._cache[field] = np.load(path, mmap_mode=self._mmap)
        return self._cache[field]

    @property
    def shape(self):
        return len(self.index), len(self.symbols)

    def frame(self, field):
        """One field as a DataFrame, for inspection rather than compute."""
        return pd.DataFrame(np.asarray(self[field]), index=self.index,
                            columns=self.symbols)

    def __repr__(self):
        t, n = self.shape
        return f"<Panel {self.tf} {t} bars x {n} symbols {self.index[0].date()}..{self.index[-1].date()}>"


if __name__ == "__main__":
    for tf in (sys.argv[1:] or ["1h"]):
        meta = build(tf)
        _, bad = verify(tf)
        status = "OK" if not bad else f"CORRUPT {bad}"
        print(f"{tf}: {meta['n_bars']} bars x {len(meta['symbols'])} symbols  "
              f"{meta['start']} -> {meta['end']}  [{status}]")
