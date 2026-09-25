"""Unify all screened frameworks for one coin and apply SKILL §4-style gates on TRAIN and VALID-C only."""
import glob, sys
import numpy as np, pandas as pd

def load(coin):
    B = f"{coin}"; rows = []
    for n in ["batch1", "batch2", "batch3", "batch4", "batch5", "exp500_batch1", "exp500_batch2", "exp500_batch3", "exp500_batch4", "exp500_batch5"]:
        try:
            tr = pd.read_csv(f"{B}/r3c/{n}_results.csv"); va = pd.read_csv(f"{B}/r3c/{n}_valid_results.csv")
        except FileNotFoundError:
            continue
        m = tr.merge(va, on=["round", "name"], suffixes=("_tr", "_va"))
        rows.append(pd.DataFrame({"fw": "engine200", "id": m["round"], "name": m["name"], "tr_n": m.n_tr, "tr_pf": m.pf_tr,
                                  "tr_mean": m.mean_bp_tr, "tr_t": m.day_t_tr, "va_n": m.n_va, "va_pf": m.pf_va,
                                  "va_mean": m.mean_bp_va, "va_t": m.day_t_va}))
    for d, f in [("r4_mtf", "r751_r1050_trainvalidC.csv"), ("r5_clean_mtf", "r1051_r1350_trainvalidC.csv"),
                 ("r6_orthogonal_alphas", "r1351_r1650_trainvalidC.csv"), ("r8_bear_adaptive", "r1651_r1950_trainvalidC.csv"),
                 ("r7_confluence_study", "confluence_study_results.csv")]:
        p = glob.glob(f"{B}/{d}/{f}")
        if not p: continue
        x = pd.read_csv(p[0])
        g = lambda a, b: x[a] if a in x else x[b]
        rows.append(pd.DataFrame({"fw": d, "id": x["round"] if "round" in x else x["theme"] + "/" + x["level"].astype(str),
                                  "name": x["name"], "tr_n": g("tr_trades", "tr_n"), "tr_pf": x.tr_pf, "tr_mean": x.tr_mean_bp,
                                  "tr_t": x.tr_t, "va_n": g("va_trades", "va_n"), "va_pf": x.va_pf, "va_mean": x.va_mean_bp, "va_t": x.va_t}))
    t = pd.concat(rows, ignore_index=True)
    t["tr_pd"], t["va_pd"] = t.tr_n / 273, t.va_n / 90
    return t

def gates(t):
    tr_ok = (t.tr_mean > 0) & (t.tr_pf >= 1.1) & (t.tr_t >= 1.5)
    va_ok = (t.va_mean > 0) & (t.va_pf >= 1.2) & (t.va_t >= 2.0)
    t["std"] = tr_ok & (t.tr_pd >= 1) & va_ok & (t.va_n >= 300) & (t.va_pd >= 1)
    t["rel"] = tr_ok & (t.tr_pd >= 0.4) & va_ok & (t.va_n >= 36) & (t.va_pd >= 0.4)
    t["tr_only_rel"] = tr_ok & (t.tr_pd >= 0.4)
    return t

if __name__ == "__main__":
    coin = sys.argv[1]
    t = gates(load(coin))
    print(coin, "rounds:", len(t), "| per framework:", t.fw.value_counts().to_dict())
    print("TRAIN-pass (relaxed freq):", int(t.tr_only_rel.sum()), "| full pass standard:", int(t["std"].sum()), "| full pass relaxed:", int(t.rel.sum()))
    cols = ["fw", "id", "name", "tr_n", "tr_pf", "tr_t", "va_n", "va_pf", "va_mean", "va_t"]
    print(t[t.rel].sort_values("va_t", ascending=False)[cols].round(2).head(25).to_string(index=False))
    t.to_csv(f"{coin}/screen_{coin}.csv", index=False)
