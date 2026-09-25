"""Re-judge every candidate that reached VALID/HOLDOUT in r3/r4/r5 under the scheme-C split:
TRAIN 2025-01-01..10-01 | EXCLUDED 2025-10-01..12-01 | VALID 2025-12-01..2026-03-01 | HOLDOUT 2026-03-01..
Trades entered in the excluded window are dropped. Output: one row per candidate with the SKILL §4 items on VALID,
plus the event-gate frequency (>= 0.4/day). HOLDOUT is only reported where it had already been read (r4) or where the
candidate passes VALID (then it is read once here and flagged)."""
import sys
sys.path.insert(0, "scripts/minute_research/r3"); sys.path.insert(0, "scripts/minute_research/r4")
import numpy as np, pandas as pd

T0, EX0, V0, H0 = (pd.Timestamp(x, tz="UTC") for x in ("2025-01-01", "2025-10-01", "2025-12-01", "2026-03-01"))


def segC(ts):
    ts = pd.DatetimeIndex(pd.to_datetime(ts, utc=True))
    # BUGFIX 2026-09-25: TRAIN has a lower bound. Before this fix anything earlier than 2025-10-01 counted as TRAIN, so panels
    # starting in 2024-11 (p7_panel*) silently put 2024-11/12 into TRAIN.
    return np.where(ts < T0, "PRE", np.where(ts < EX0, "TRAIN", np.where(ts < V0, "EXCL", np.where(ts < H0, "VALID", "HOLDOUT"))))


def stats(x, cost_col="cost"):
    if len(x) == 0:
        return {"n": 0}
    days = (x.t.max() - x.t.min()).days + 1
    dm = x.groupby(x.t.dt.floor("D")).ret.sum()
    g, b = x.ret[x.ret > 0].sum(), -x.ret[x.ret < 0].sum()
    prof = x.ret.sum()
    mon = x.groupby(x.t.dt.tz_localize(None).dt.to_period("M")).ret.sum()
    stress = (x.ret - 0.5 * 2 * x[cost_col]).sum() if cost_col in x else np.nan
    return {"n": len(x), "days": int(dm.size), "per_day": round(len(x) / 90 if x.t.min() >= V0 and x.t.max() < H0 else len(x) / days, 2),
            "net_bp": round(x.ret.mean() * 1e4, 1), "pf": round(g / b, 2) if b else np.inf,
            "t": round(dm.mean() / dm.std() * np.sqrt(len(dm)), 2) if dm.std() > 0 else np.nan,
            "mon+": f"{int((mon > 0).sum())}/{len(mon)}", "top_pair": round(x.groupby("pair").ret.sum().max() / prof, 2) if prof > 0 else np.nan,
            "top_day": round(dm.max() / prof, 2) if prof > 0 else np.nan, "stress1.5>0": bool(stress > 0) if stress == stress else None,
            "L_bp": round(x.ret[x.side == 1].mean() * 1e4, 1) if (x.side == 1).any() else np.nan,
            "S_bp": round(x.ret[x.side == -1].mean() * 1e4, 1) if (x.side == -1).any() else np.nan}


def verdict(tr, va):
    if va.get("n", 0) == 0:
        return "no VALID trades"
    std = {"B": va["n"] >= 300 and va["days"] >= 60 and va["per_day"] >= 1, "C": va["net_bp"] > 0 and va["pf"] >= 1.2 and tr.get("pf", 0) >= 1.1,
           "E": (int(va["mon+"].split("/")[0]) / int(va["mon+"].split("/")[1]) >= 0.6) and (va["top_pair"] <= 0.3) and (va["top_day"] <= 0.2) if va["net_bp"] > 0 else False,
           "F": bool(va["stress1.5>0"]), "t>=2(VALID only)": va["t"] >= 2}
    ev = {"B'": va["per_day"] >= 0.4 and va["n"] >= 36, "C": std["C"], "F": std["F"], "t>=2": std["t>=2(VALID only)"]}
    s = "PASS" if all(std.values()) else "fail " + ",".join(k for k, v in std.items() if not v)
    e = "PASS" if all(ev.values()) else "fail " + ",".join(k for k, v in ev.items() if not v)
    return f"§4: {s} | event: {e}"


def judge(name, t, old=""):
    t = t.copy(); t["t"] = pd.to_datetime(t.t, utc=True); t["seg"] = segC(t.t)
    if "cost" not in t:
        t["cost"] = 10e-4
    tr, va = stats(t[t.seg == "TRAIN"]), stats(t[t.seg == "VALID"])
    ex = t[t.seg == "EXCL"]
    row = {"candidate": name, "old_VALID": old, "TRAIN": f"n{tr.get('n')} {tr.get('net_bp')}bp PF{tr.get('pf')} t{tr.get('t')}",
           "EXCL_dropped": f"n{len(ex)} {round(ex.ret.mean()*1e4,1) if len(ex) else '-'}bp",
           "VALID_C": {k: va.get(k) for k in ("n", "days", "per_day", "net_bp", "pf", "t", "mon+", "top_pair", "top_day", "stress1.5>0", "L_bp", "S_bp")},
           "verdict": verdict(tr, va)}
    print(f"\n### {name}\n  old VALID: {old}\n  TRAIN: {row['TRAIN']}   excluded Oct-Nov: {row['EXCL_dropped']}\n  VALID-C: {row['VALID_C']}\n  -> {row['verdict']}", flush=True)
    return row, t
