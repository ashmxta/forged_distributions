import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau
# Not debugged yet


# 1. Load the CSVs
df1 = pd.read_csv("compo_res/compo_res_ranked.csv")
df2 = pd.read_csv("compo_res_rm100/compo_res_rm_ranked.csv")

# 2. Assign ranks (0 = lowest cost)
df1 = df1.sort_values(by="Total Privacy Cost").reset_index(drop=True)
df1["rank"] = np.arange(len(df1))

df2 = df2.sort_values(by="Total Privacy Cost").reset_index(drop=True)
df2["rank"] = np.arange(len(df2))

# 3. Ignore the top 100 points from df1
df1_900 = df1.iloc[100:]

# 4. Find common points
common_points = set(df1_900["point"]).intersection(df2["point"])
df1_common = df1_900[df1_900["point"].isin(common_points)]
df2_common = df2[df2["point"].isin(common_points)]

# 5. Merge to align ranks
merged = pd.merge(
    df1_common[["point", "rank"]],
    df2_common[["point", "rank"]],
    on="point",
    suffixes=("_orig", "_new")
)

# 6. Compute metrics
spearman, _ = spearmanr(merged["rank_orig"], merged["rank_new"])
kendall, _ = kendalltau(merged["rank_orig"], merged["rank_new"])
mean_shift = np.mean(np.abs(merged["rank_orig"] - merged["rank_new"]))

print(f"Compared {len(merged)} points")
print(f"Spearman's rho: {spearman:.4f}")
print(f"Kendall's tau: {kendall:.4f}")
print(f"Mean rank shift: {mean_shift:.2f}")
