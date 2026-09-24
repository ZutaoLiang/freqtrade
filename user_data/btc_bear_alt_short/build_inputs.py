"""Build 4h inputs for the BTC-bear alt-short backtest.

Outputs (inputs.npz next to this file), aligned to the 4h_long panel grid:
  open/high/low/close   4h OHLC per symbol (from panels/4h_long)
  liq                   4h_long universe mask (>=30d history, 7d median daily quote volume >= 2.4M)
  fund                  sum of funding settled inside each 4h bar (long pays +)
  mcap                  point-in-time CoinMarketCap market cap (latest weekly snapshot <= bar - 1 day), NaN if unmatched
  btc_d                 BTC daily closes (Binance USDT-M, from 2021-06) for the regime filters

Funding: 1h_hist stores the settled rate on settlement bars only (0 elsewhere), so it is
summed per bar; the 1h panel carries the last rate forward, so it is charged pro rata
(rate / interval_hours per hour), matching factor_research/research/cost.py.
"""
import glob, json, os, urllib.request
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PANELS = os.path.join(HERE, "..", "data", "binance_public", "panels")

STABLE = {"USDC", "FDUSD", "TUSD", "BUSD", "USDP", "DAI", "USDE", "PYUSD", "USD1", "RLUSD", "EUR", "EURI", "USDS", "AEUR"}
EXCLUDE = {"BTC", "ETH", "WBTC", "WETH", "STETH", "WBETH", "BTCDOM", "PAXG", "XAUT", "XAU", "XAG"} | STABLE
PREFIXES = ("1000000", "1000", "1M")


def load_panel(name, field):
    d = os.path.join(PANELS, name)
    meta = json.load(open(os.path.join(d, "meta.json")))
    idx = pd.date_range(meta["start"], meta["end"], freq={"4h_long": "4h", "1h_hist": "1h", "1h": "1h"}[name])
    return pd.DataFrame(np.load(os.path.join(d, f"{field}.npy")), index=idx, columns=meta["symbols"])


def base_of(sym):
    b = sym[:-4] if sym.endswith("USDT") else sym
    for p in PREFIXES:
        if b.startswith(p) and len(b) > len(p):
            return b[len(p):]
    return b


def funding_4h(grid, syms):
    hist = load_panel("1h_hist", "funding_rate").astype("float64")
    cur = load_panel("1h", "funding_rate").astype("float64")
    iv = load_panel("1h", "funding_interval_hours").astype("float64")
    cut = pd.Timestamp("2026-01-01", tz="UTC")
    hist = hist[hist.index < cut].fillna(0.0)                      # settlement-only storage
    iv = iv.where(np.isfinite(iv) & (iv > 0), 8.0)
    cur = (cur / iv)[cur.index >= cut].fillna(0.0)                 # carried storage, pro rata per hour
    fr = pd.concat([hist.reindex(columns=syms), cur.reindex(columns=syms)]).fillna(0.0)
    return fr.resample("4h").sum().reindex(grid).fillna(0.0)


def mcap_4h(grid, syms):
    files = sorted(glob.glob(os.path.join(HERE, "cmc", "*.json")))
    dates, rows = [], []
    for f in files:
        snap = json.load(open(f))
        best = {}
        for r in snap:
            s, m = r["sym"].upper(), r["mcap"] or 0.0
            if m > best.get(s, 0.0):          # ticker collisions: keep the larger coin
                best[s] = m
        dates.append(pd.Timestamp(os.path.basename(f)[:10], tz="UTC"))
        rows.append([best.get(base_of(s), np.nan) for s in syms])
    snap = pd.DataFrame(rows, index=dates, columns=syms)
    # a snapshot dated D is usable from D + 1 day (it is the D 00:00 listing; add a day of slack)
    snap.index = snap.index + pd.Timedelta("1D")
    out = snap.reindex(grid, method="ffill")
    return out


def btc_daily():
    path = os.path.join(HERE, "btc_1d.csv")
    if not os.path.exists(path):
        rows, start = [], int(pd.Timestamp("2021-06-01", tz="UTC").timestamp() * 1000)
        while True:
            url = f"https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=1d&startTime={start}&limit=1500"
            k = json.loads(urllib.request.urlopen(url, timeout=60).read())
            if not k:
                break
            rows += k
            start = k[-1][0] + 86400000
            if len(k) < 1500:
                break
        df = pd.DataFrame([(r[0], float(r[4])) for r in rows], columns=["ts", "close"])
        df["date"] = pd.to_datetime(df.ts, unit="ms", utc=True)
        df[["date", "close"]].to_csv(path, index=False)
    df = pd.read_csv(path, parse_dates=["date"])
    return df.set_index("date")["close"]


def main():
    px = {f: load_panel("4h_long", f) for f in ["open", "high", "low", "close"]}
    liq = load_panel("4h_long", "universe_mask")
    grid, syms = px["close"].index, list(px["close"].columns)
    keep = [s for s in syms if s.endswith("USDT") and s.isascii() and base_of(s) not in EXCLUDE]
    fund = funding_4h(grid, keep)
    mcap = mcap_4h(grid, keep)
    btc = btc_daily()
    np.savez_compressed(
        os.path.join(HERE, "inputs.npz"),
        grid=grid.asi8, syms=np.array(keep),
        open=px["open"][keep].to_numpy("float64"), high=px["high"][keep].to_numpy("float64"),
        low=px["low"][keep].to_numpy("float64"), close=px["close"][keep].to_numpy("float64"),
        liq=liq[keep].to_numpy(bool), fund=fund.to_numpy("float64"), mcap=mcap.to_numpy("float64"),
        btc_d_idx=btc.index.asi8, btc_d=btc.to_numpy("float64"))
    elig = (mcap.to_numpy() >= 2e8) & liq[keep].to_numpy(bool)
    per_day = pd.Series(elig.sum(1), index=grid).resample("1D").last()
    matched = np.isfinite(mcap.to_numpy()).any(0).sum()
    print(f"symbols kept {len(keep)} (matched to CMC {matched}); eligible (>=200M & liquid) per day: "
          f"median {per_day.median():.0f}, min {per_day.min():.0f}, max {per_day.max():.0f}")
    print(per_day.resample("QE").median().to_string())


if __name__ == "__main__":
    main()
