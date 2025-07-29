import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau

df1 = pd.read_csv("/h/321/ashmita/forged_distributions/sensitivity/compo_res/compo_res_ranked.csv")
df2 = pd.read_csv("/h/321/ashmita/forged_distributions/sensitivity/compo_res_rm100/compo_res_rm_ranked.csv")


df1 = df1.sort_values(by="Total Privacy Cost").reset_index(drop=True)
df1["rank"] = np.arange(1, len(df1) + 1)

df2 = df2.sort_values(by="Total Privacy Cost").reset_index(drop=True)
df2["rank"] = np.arange(1, len(df2) + 1)

df1_900 = df1.iloc[100:].copy().reset_index(drop=True)
df1_900["rank"] = np.arange(1, len(df1_900) + 1)

common_points = set(df1_900["point"]).intersection(df2["point"])
df1_common = df1_900[df1_900["point"].isin(common_points)]
df2_common = df2[df2["point"].isin(common_points)]

merged = pd.merge(
    df1_common[["point", "rank"]],
    df2_common[["point", "rank"]],
    on="point",
    suffixes=("_orig", "_new")
)

spearman, _ = spearmanr(merged["rank_orig"], merged["rank_new"])
kendall, _ = kendalltau(merged["rank_orig"], merged["rank_new"])
mean_shift = np.mean(np.abs(merged["rank_orig"] - merged["rank_new"]))

print(f"Compared {len(merged)} points")
print(f"Spearman's rho: {spearman:.4f}")
print(f"Kendall's tau: {kendall:.4f}")
print(f"Mean rank shift (adjusted for top 100 removal): {mean_shift:.2f}")
