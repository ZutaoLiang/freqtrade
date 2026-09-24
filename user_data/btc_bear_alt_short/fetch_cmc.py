"""Download weekly point-in-time CoinMarketCap listing snapshots (top 1000 by market cap)."""
import json, os, sys, time, datetime as dt, urllib.request
from concurrent.futures import ThreadPoolExecutor

OUT = os.path.join(os.path.dirname(__file__), "cmc")
URL = ("https://api.coinmarketcap.com/data-api/v3/cryptocurrency/listings/historical"
       "?date={d}&limit=1000&start=1&convertId=2781")

def fetch(d):
    path = os.path.join(OUT, f"{d}.json")
    if os.path.exists(path):
        return d, "cached"
    for attempt in range(5):
        try:
            req = urllib.request.Request(URL.format(d=d), headers={"User-Agent": "Mozilla/5.0"})
            raw = urllib.request.urlopen(req, timeout=60).read()
            rows = json.loads(raw)["data"]
            slim = [{"sym": r["symbol"], "slug": r["slug"], "rank": r["cmcRank"],
                     "mcap": r["quotes"][0]["marketCap"]} for r in rows]
            if len(slim) < 500:
                raise ValueError(f"only {len(slim)} rows")
            tmp = path + ".tmp"
            json.dump(slim, open(tmp, "w"))
            os.replace(tmp, path)
            return d, len(slim)
        except Exception as e:
            err = e
            time.sleep(3 * (attempt + 1))
    return d, f"FAIL {err}"

if __name__ == "__main__":
    start, end = dt.date(2022, 10, 2), dt.date(2026, 8, 16)
    days = []
    d = start
    while d <= end:
        days.append(d.isoformat()); d += dt.timedelta(days=7)
    with ThreadPoolExecutor(4) as ex:
        for d, r in ex.map(fetch, days):
            print(d, r, flush=True)
