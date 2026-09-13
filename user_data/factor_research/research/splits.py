"""Fixed train / validation / holdout time splits.

Written once to disk and read back thereafter. The point is that the holdout
boundary cannot drift: if it were recomputed per run, every re-run against a
longer dataset would quietly shift it and the holdout would stop being one.
"""
import json
import os

import pandas as pd

from panel import CACHE

SPLIT_FILE = os.path.join(CACHE, "splits.json")
TRAIN_FRAC = 0.60
VALID_FRAC = 0.20  # holdout takes the remainder


def create(tf="1h", train_frac=TRAIN_FRAC, valid_frac=VALID_FRAC, force=False):
    if os.path.exists(SPLIT_FILE) and not force:
        return load()
    meta = json.load(open(os.path.join(CACHE, tf, "meta.json")))
    start, end = pd.Timestamp(meta["start"]), pd.Timestamp(meta["end"])
    total = end - start
    splits = {
        "created_from": tf,
        "train": [str(start), str(start + total * train_frac)],
        "valid": [str(start + total * train_frac),
                  str(start + total * (train_frac + valid_frac))],
        "holdout": [str(start + total * (train_frac + valid_frac)), str(end)],
    }
    os.makedirs(CACHE, exist_ok=True)
    with open(SPLIT_FILE, "w") as f:
        json.dump(splits, f, indent=2)
    return splits


def load():
    return json.load(open(SPLIT_FILE))


def slice_mask(index, name):
    """Boolean selector over a DatetimeIndex for one named split."""
    lo, hi = load()[name]
    lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
    return (index >= lo) & (index < hi)


if __name__ == "__main__":
    s = create()
    for name in ("train", "valid", "holdout"):
        lo, hi = s[name]
        days = (pd.Timestamp(hi) - pd.Timestamp(lo)).days
        print(f"{name:8s} {lo[:10]} -> {hi[:10]}  ({days} days)")
    print(f"\nwritten to {SPLIT_FILE}")
    print("holdout must not be touched until the pipeline has run end to end once.")
