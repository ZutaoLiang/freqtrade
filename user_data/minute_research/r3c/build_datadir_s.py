"""Datadir r3s = r3b (U162, symlinked) + 5m/1h-funding/1h-mark for every other pair in seasoned_universe.parquet."""
import os, sys
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
sys.path.insert(0, "/root/freqtrade/user_data/minute_research/r3c")
import build_datadir_b as B            # reuse w() / one() with OUT redirected

SRC, OUT = "/root/freqtrade/user_data/data/r3b/futures", "/root/freqtrade/user_data/data/r3s/futures"
B.OUT = OUT

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(SRC):
        if "-5m-futures" in f or "funding_rate" in f or "-1h-mark" in f:
            dst = f"{OUT}/{f}"
            if not os.path.exists(dst):
                os.symlink(f"{SRC}/{f}", dst)
    pairs = pd.read_parquet("/root/freqtrade/user_data/data/r3s/seasoned_universe.parquet").pair.unique()
    have = {f.split("_USDT_USDT")[0] for f in os.listdir(OUT) if f.endswith("-5m-futures.feather")}
    todo = [p.split("/")[0] for p in pairs if p.split("/")[0] not in have]
    todo = [b for b in todo if os.path.exists(f"/root/freqtrade/user_data/data/binance_public/funding/{b}USDT.parquet")
            and os.path.exists(f"/root/freqtrade/user_data/data/binance_public/markprice_1m/{b}USDT.parquet")]
    print("building", len(todo))
    with ProcessPoolExecutor(16) as ex:
        for r in ex.map(B.one, todo):
            pass
    print("5m files:", sum(f.endswith("-5m-futures.feather") for f in os.listdir(OUT)))
