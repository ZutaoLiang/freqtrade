"""Consolidate all VALID results for R51-R250 and R251-R750."""
import glob
import pandas as pd

DIR = "/root/freqtrade/user_data/minute_research/r3"

# 1. R51-R250
b_files = [f"{DIR}/batch{i}_valid_results.csv" for i in range(1, 6)]
dfs_200 = [pd.read_csv(f) for f in b_files]
df_200 = pd.concat(dfs_200, ignore_index=True).sort_values("round")
df_200.to_csv(f"{DIR}/r51_r250_valid_master.csv", index=False)
print(f"R51-R250 VALID Master saved: {len(df_200)} rounds.")

# 2. R251-R750
e_files = [f"{DIR}/exp500_batch{i}_valid_results.csv" for i in range(1, 6)]
dfs_500 = [pd.read_csv(f) for f in e_files]
df_500 = pd.concat(dfs_500, ignore_index=True).sort_values("round")
df_500.to_csv(f"{DIR}/r251_r750_valid_master.csv", index=False)
print(f"R251-R750 VALID Master saved: {len(df_500)} rounds.")

# 3. Overall All 700
df_all = pd.concat([df_200, df_500], ignore_index=True)
df_all.to_csv(f"{DIR}/all_700_valid_master.csv", index=False)
print(f"All 700 VALID Master saved: {len(df_all)} rounds.")
