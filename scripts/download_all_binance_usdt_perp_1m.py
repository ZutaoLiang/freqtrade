#!/usr/bin/env python3
"""Download and aggregate Binance USD-M crypto/USDT perpetual 1m archives.

The universe is the union of two sources so that contracts delisted inside the
requested window are still screened:

* Binance ``exchangeInfo`` filtered to quoteAsset == USDT, contractType ==
  PERPETUAL and underlyingType == COIN.
* Every USDT symbol directory published under the public archive, minus the
  symbols ``exchangeInfo`` classifies as a non-COIN product and minus the
  reviewed index products in ``DELISTED_NON_CRYPTO``.

Taking the universe from ``exchangeInfo`` alone would silently drop any
contract Binance delisted before the download ran, biasing a historical
volume screen toward survivors.  The archive listing is point-in-time, so the
union keeps delisted contracts eligible while still excluding dated futures,
USDC/BUSD contracts, indexes, and all TradFi/stock perpetuals.

Official monthly archives are used through July 2026 and daily archives for
2026-08-01 through 2026-08-13.  Every ZIP is checked against Binance's SHA-256
CHECKSUM before it is converted to Freqtrade feather files.

The process is resumable: valid ZIP files and completed feather files are reused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import threading
import time
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests


START = date(2026, 1, 1)
END = date(2026, 8, 13)
EXCHANGE_INFO = "https://fapi.binance.com/fapi/v1/exchangeInfo"
S3_ENDPOINT = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
DOWNLOAD_BASE = "https://data.binance.vision"
S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

# Delisted archive directories that were index or basket products rather than a
# single coin.  ``exchangeInfo`` no longer describes them, so the non-COIN
# filter cannot reach them and they are listed explicitly.
DELISTED_NON_CRYPTO = {
    "BLUEBIRDUSDT",
    "DOTECOUSDT",
    "FOOTBALLUSDT",
}

RAW_ROOT = Path("user_data/data/binance-public")
FEATHER_ROOT = Path("user_data/data/binance/futures")
MANIFEST = RAW_ROOT / "futures/um/klines/usdt-coin-perpetual-2026-through-08-13.json"
REPORT = RAW_ROOT / "futures/um/klines/usdt-coin-perpetual-2026-through-08-13-report.json"

CSV_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]
AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
TIMEFRAMES = {
    "5m": ("5min", 5),
    "15m": ("15min", 15),
    "30m": ("30min", 30),
    "1h": ("1h", 60),
    "4h": ("4h", 240),
    "1d": ("1D", 1440),
}

_thread_local = threading.local()


def session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        _thread_local.session = requests.Session()
    return _thread_local.session


def request(url: str, *, params: dict[str, str] | None = None, stream: bool = False) -> requests.Response:
    error: Exception | None = None
    for attempt in range(6):
        try:
            response = session().get(url, params=params, stream=stream, timeout=(15, 120))
            if response.status_code >= 500 or response.status_code == 429:
                response.close()
                raise requests.HTTPError(f"HTTP {response.status_code}: {response.url}")
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            error = exc
            if attempt == 5:
                break
            time.sleep(min(2**attempt, 15))
    assert error is not None
    raise error


def archived_usdt_symbols() -> set[str]:
    """Return every USDT symbol directory published under the public archive."""
    symbols: set[str] = set()
    for interval in ("monthly", "daily"):
        prefix = f"data/futures/um/{interval}/klines/"
        root = ET.fromstring(
            request(
                S3_ENDPOINT,
                params={
                    "list-type": "2",
                    "prefix": prefix,
                    "delimiter": "/",
                    "max-keys": "1000",
                },
            ).content
        )
        if root.findtext("s3:IsTruncated", namespaces=S3_NS) == "true":
            raise RuntimeError(f"archive symbol listing for {interval} is truncated")
        for item in root.findall("s3:CommonPrefixes", S3_NS):
            symbol = item.findtext("s3:Prefix", namespaces=S3_NS)[len(prefix) :].strip("/")
            if symbol.endswith("USDT") and "_" not in symbol:
                symbols.add(symbol)
    return symbols


def crypto_usdt_perpetuals() -> list[str]:
    symbols = request(EXCHANGE_INFO).json()["symbols"]
    listed = {
        item["symbol"]
        for item in symbols
        if item.get("quoteAsset") == "USDT"
        and item.get("contractType") == "PERPETUAL"
        and item.get("underlyingType") == "COIN"
        and "_" not in item["symbol"]
    }
    # exchangeInfo describes only currently listed contracts.  Anything it
    # classifies as a non-COIN product stays excluded; anything it no longer
    # knows about is a delisted contract that the screen must still see.
    known = {item["symbol"] for item in symbols}
    delisted = archived_usdt_symbols() - known - DELISTED_NON_CRYPTO
    return sorted(listed | delisted)


def list_objects(prefix: str) -> list[dict[str, Any]]:
    root = ET.fromstring(
        request(
            S3_ENDPOINT,
            params={"list-type": "2", "prefix": prefix, "max-keys": "1000"},
        ).content
    )
    return [
        {
            "key": item.findtext("s3:Key", namespaces=S3_NS),
            "size": int(item.findtext("s3:Size", namespaces=S3_NS) or 0),
        }
        for item in root.findall("s3:Contents", S3_NS)
    ]


def archives_for_symbol(symbol: str) -> tuple[str, list[dict[str, Any]]]:
    monthly_prefix = (
        f"data/futures/um/monthly/klines/{symbol}/1m/{symbol}-1m-{START.year}-"
    )
    daily_prefix = f"data/futures/um/daily/klines/{symbol}/1m/{symbol}-1m-2026-08-"
    wanted_monthly = {f"{symbol}-1m-2026-{month:02d}.zip" for month in range(1, 8)}
    wanted_daily = {f"{symbol}-1m-2026-08-{day:02d}.zip" for day in range(1, 14)}
    objects = []
    for item in list_objects(monthly_prefix):
        if item["key"].rsplit("/", 1)[-1] in wanted_monthly:
            objects.append(item)
    for item in list_objects(daily_prefix):
        if item["key"].rsplit("/", 1)[-1] in wanted_daily:
            objects.append(item)
    return symbol, sorted(objects, key=lambda item: item["key"])


def scan(workers: int) -> dict[str, Any]:
    universe = crypto_usdt_perpetuals()
    found: dict[str, list[dict[str, Any]]] = {}
    errors: list[str] = []
    total_bytes = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(archives_for_symbol, symbol): symbol for symbol in universe}
        for done, future in enumerate(as_completed(futures), 1):
            symbol = futures[future]
            try:
                _, objects = future.result()
                if objects:
                    found[symbol] = objects
                    total_bytes += sum(item["size"] for item in objects)
            except Exception as exc:
                errors.append(f"{symbol}: {exc!r}")
            if done % 50 == 0 or done == len(futures):
                print(
                    f"scan {done}/{len(futures)} symbols={len(found)} "
                    f"compressed={total_bytes / 1024**3:.2f}GiB errors={len(errors)}",
                    flush=True,
                )
    if errors:
        raise RuntimeError("archive scan failed:\n" + "\n".join(errors[:20]))
    manifest = {
        "source": DOWNLOAD_BASE,
        "filters": {
            "quoteAsset": "USDT",
            "contractType": "PERPETUAL",
            "underlyingType": "COIN",
        },
        "start": START.isoformat(),
        "end": END.isoformat(),
        "exchange_info_symbols": len(universe),
        "symbols": found,
        "archive_count": sum(len(items) for items in found.values()),
        "compressed_bytes": total_bytes,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST.read_text())


def local_path(key: str) -> Path:
    if not key.startswith("data/"):
        raise ValueError(f"unexpected archive key: {key}")
    return RAW_ROOT / key.removeprefix("data/")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(item: dict[str, Any]) -> tuple[str, int, bool]:
    key = item["key"]
    target = local_path(key)
    checksum_target = target.with_name(target.name + ".CHECKSUM")
    checksum_content = request(f"{DOWNLOAD_BASE}/{key}.CHECKSUM").content
    expected = checksum_content.decode().split()[0].lower()
    target.parent.mkdir(parents=True, exist_ok=True)
    checksum_target.write_bytes(checksum_content)
    if target.exists() and target.stat().st_size == item["size"] and sha256(target) == expected:
        return key, target.stat().st_size, True

    part = target.with_name(target.name + ".part")
    response = request(f"{DOWNLOAD_BASE}/{key}", stream=True)
    digest = hashlib.sha256()
    with part.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
                digest.update(chunk)
    response.close()
    if part.stat().st_size != item["size"]:
        raise IOError(f"size mismatch for {key}: {part.stat().st_size} != {item['size']}")
    if digest.hexdigest().lower() != expected:
        raise IOError(f"SHA-256 mismatch for {key}")
    part.replace(target)
    return key, target.stat().st_size, False


def check_disk(manifest: dict[str, Any]) -> None:
    remaining = sum(
        item["size"]
        for items in manifest["symbols"].values()
        for item in items
        if not local_path(item["key"]).exists()
    )
    # Empirically, seven compressed feather files are near the ZIP total.  Reserve an
    # additional GiB so the workspace is not filled to 100% during atomic writes.
    estimated_feather = int(manifest["compressed_bytes"] * 1.1)
    required = remaining + estimated_feather + 1024**3
    free = shutil.disk_usage(RAW_ROOT.parent).free
    print(
        f"disk free={free / 1024**3:.2f}GiB remaining_download={remaining / 1024**3:.2f}GiB "
        f"estimated_feather={estimated_feather / 1024**3:.2f}GiB",
        flush=True,
    )
    if required > free:
        raise RuntimeError(
            f"insufficient disk: need about {required / 1024**3:.2f}GiB, "
            f"only {free / 1024**3:.2f}GiB free"
        )


def download(manifest: dict[str, Any], workers: int, selected: set[str] | None) -> None:
    items = [
        item
        for symbol, archives in manifest["symbols"].items()
        if selected is None or symbol in selected
        for item in archives
    ]
    errors: list[str] = []
    downloaded = reused = total_bytes = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_archive, item): item["key"] for item in items}
        for done, future in enumerate(as_completed(futures), 1):
            key = futures[future]
            try:
                _, size, was_reused = future.result()
                total_bytes += size
                reused += int(was_reused)
                downloaded += int(not was_reused)
            except Exception as exc:
                errors.append(f"{key}: {exc!r}")
            if done % 100 == 0 or done == len(futures):
                print(
                    f"download {done}/{len(futures)} new={downloaded} reused={reused} "
                    f"verified={total_bytes / 1024**3:.2f}GiB errors={len(errors)}",
                    flush=True,
                )
    if errors:
        raise RuntimeError("downloads failed:\n" + "\n".join(errors[:20]))


def read_archive(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        frame = pd.read_csv(
            archive.open(archive.namelist()[0]),
            header=None,
            names=CSV_COLUMNS,
            usecols=list(range(6)),
        )
    if str(frame.iloc[0]["open_time"]) == "open_time":
        frame = frame.iloc[1:]
    timestamp = pd.to_numeric(frame.pop("open_time"), errors="raise").astype("int64")
    timestamp = timestamp.where(timestamp <= 10**14, timestamp // 1000)
    frame["date"] = pd.to_datetime(timestamp, unit="ms", utc=True).astype("datetime64[ms, UTC]")
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("float64")
    return frame[["date", "open", "high", "low", "close", "volume"]]


def atomic_feather(frame: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".part")
    frame.to_feather(temporary)
    temporary.replace(target)


def convert_symbol(symbol: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    paths = [local_path(item["key"]) for item in items]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{symbol}: {len(missing)} archives are missing")
    minute = pd.concat([read_archive(path) for path in paths], ignore_index=True)
    minute = minute.sort_values("date").reset_index(drop=True)
    duplicate_count = int(minute["date"].duplicated().sum())
    if duplicate_count:
        raise ValueError(f"{symbol}: {duplicate_count} duplicate timestamps")
    invalid = (
        (minute["high"] < minute[["open", "close", "low"]].max(axis=1))
        | (minute["low"] > minute[["open", "close", "high"]].min(axis=1))
        | (minute[["open", "high", "low", "close", "volume"]] < 0).any(axis=1)
    )
    if invalid.any():
        raise ValueError(f"{symbol}: {int(invalid.sum())} invalid OHLCV rows")

    base = symbol.removesuffix("USDT")
    target_prefix = FEATHER_ROOT / f"{base}_USDT_USDT"
    atomic_feather(minute, target_prefix.with_name(target_prefix.name + "-1m-futures.feather"))
    indexed = minute.set_index("date")
    counts: dict[str, int] = {"1m": len(minute)}
    for timeframe, (rule, expected) in TIMEFRAMES.items():
        grouped = indexed.resample(rule, label="left", closed="left")
        sizes = grouped.size()
        output = grouped.agg(AGG)
        output = output.loc[sizes == expected].dropna(subset=["open", "high", "low", "close"])
        output = output.reset_index()[["date", "open", "high", "low", "close", "volume"]]
        output["date"] = output["date"].astype("datetime64[ms, UTC]")
        for column in ["open", "high", "low", "close", "volume"]:
            output[column] = output[column].astype("float64")
        atomic_feather(
            output,
            target_prefix.with_name(target_prefix.name + f"-{timeframe}-futures.feather"),
        )
        counts[timeframe] = len(output)

    intervals = minute["date"].diff().dropna() / pd.Timedelta(minutes=1)
    gaps = int((intervals - 1).clip(lower=0).sum())
    return {
        "symbol": symbol,
        "first": minute["date"].iloc[0].isoformat(),
        "last": minute["date"].iloc[-1].isoformat(),
        "gaps_1m": gaps,
        "rows": counts,
    }


def convert(
    manifest: dict[str, Any],
    workers: int,
    selected: set[str] | None,
    report_path: Path = REPORT,
) -> dict[str, Any]:
    work = {
        symbol: items
        for symbol, items in manifest["symbols"].items()
        if selected is None or symbol in selected
    }
    results: dict[str, Any] = {}
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(convert_symbol, symbol, items): symbol for symbol, items in work.items()
        }
        for done, future in enumerate(as_completed(futures), 1):
            symbol = futures[future]
            try:
                results[symbol] = future.result()
            except Exception as exc:
                errors.append(f"{symbol}: {exc!r}")
            if done % 20 == 0 or done == len(futures):
                print(
                    f"convert {done}/{len(futures)} successful={len(results)} errors={len(errors)}",
                    flush=True,
                )
    report = {
        "start": START.isoformat(),
        "end": END.isoformat(),
        "successful": len(results),
        "failed": len(errors),
        "errors": errors,
        "symbols": results,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if errors:
        raise RuntimeError("conversion failed:\n" + "\n".join(errors[:20]))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["scan", "download", "convert", "all"], default="all")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--convert-workers", type=int, default=2)
    parser.add_argument(
        "--report",
        type=Path,
        default=REPORT,
        help="Conversion report path (useful for bounded partial runs).",
    )
    parser.add_argument("--refresh-manifest", action="store_true")
    parser.add_argument("--symbols", nargs="*", help="Optional raw Binance symbols for a partial run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.refresh_manifest or not MANIFEST.exists():
        manifest = scan(args.workers)
    else:
        manifest = load_manifest()
    print(
        f"manifest symbols={len(manifest['symbols'])} archives={manifest['archive_count']} "
        f"compressed={manifest['compressed_bytes'] / 1024**3:.2f}GiB",
        flush=True,
    )
    if args.phase == "scan":
        return
    selected = set(args.symbols) if args.symbols else None
    unknown = selected - set(manifest["symbols"]) if selected else set()
    if unknown:
        raise ValueError(f"symbols absent from manifest: {sorted(unknown)}")
    if selected is None:
        check_disk(manifest)
    if args.phase in {"download", "all"}:
        download(manifest, args.workers, selected)
    if args.phase in {"convert", "all"}:
        convert(manifest, args.convert_workers, selected, args.report)


if __name__ == "__main__":
    main()
