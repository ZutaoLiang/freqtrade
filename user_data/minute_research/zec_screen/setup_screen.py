"""Copy the six panel research frameworks into zec_screen/<COIN>/ and point them at one coin.
Patches (copies only): output paths -> the copy; universe = SCREEN_COINS env instead of U162;
r4_mtf np.diff lookahead fixed (returns padded so index k = bar k)."""
import os, re, shutil, sys
SRC = "/root/freqtrade/user_data/minute_research"
DIRS = ["r3c", "r4_mtf", "r5_clean_mtf", "r6_orthogonal_alphas", "r7_confluence_study", "r8_bear_adaptive"]
coin = sys.argv[1]
dst_root = f"{SRC}/zec_screen/{coin}"
for d in DIRS:
    dst = f"{dst_root}/{d}"; os.makedirs(dst, exist_ok=True)
    for f in os.listdir(f"{SRC}/{d}"):
        if f.endswith(".py") or f in ("universe_rank.csv", "U60.txt"):
            shutil.copy(f"{SRC}/{d}/{f}", dst)
for d in DIRS:
    for f in os.listdir(f"{dst_root}/{d}"):
        if not f.endswith(".py"): continue
        p = f"{dst_root}/{d}/{f}"; s = o = open(p).read()
        for dd in DIRS + ["r3"]:
            s = s.replace(f"/minute_research/{dd}/", f"/minute_research/zec_screen/{coin}/{dd if dd != 'r3' else 'r3c'}/")
            s = s.replace(f"/minute_research/{dd}\"", f"/minute_research/zec_screen/{coin}/{dd if dd != 'r3' else 'r3c'}\"")
        s = re.sub(r"(\n(\s*)self\.u162_indices = np\.array\(\[i for i, s in enumerate\(self\.symbols\) if s in u162_set\], dtype=np\.int32\))",
                   r'\1\n\2if os.environ.get("SCREEN_COINS"): self.u162_indices = np.array([i for i, s in enumerate(self.symbols) if s in os.environ["SCREEN_COINS"].split(",")], dtype=np.int32)', s)
        if "SCREEN_COINS" in s and "import os" not in s:
            s = "import os\n" + s
        if d == "r4_mtf" and f == "run_300_rounds.py":
            s = s.replace("btc_ret_1d = np.diff(np.log(np.maximum(btc_c1d, 1e-8)), axis=0)",
                          "btc_ret_1d = np.vstack([np.full((1, btc_c1d.shape[1]), np.nan), np.diff(np.log(np.maximum(btc_c1d, 1e-8)), axis=0)])")
            s = s.replace("ret_4h_alt = np.diff(np.log(np.maximum(c4h, 1e-8)), axis=0)",
                          "ret_4h_alt = np.vstack([np.full((1, c4h.shape[1]), np.nan), np.diff(np.log(np.maximum(c4h, 1e-8)), axis=0)])")
        if s != o: open(p, "w").write(s)
n = sum(open(f"{dst_root}/{d}/{f}").read().count("SCREEN_COINS") > 0 for d in DIRS for f in os.listdir(f"{dst_root}/{d}") if f.endswith(".py"))
print(coin, "files with universe override:", n, "| leftover original paths:",
      sum(open(f"{dst_root}/{d}/{f}").read().count(f"minute_research/{dd}/") for d in DIRS for f in os.listdir(f"{dst_root}/{d}") if f.endswith(".py") for dd in DIRS if f"zec_screen" not in dd) )
