import pandas as pd

infile = "res_concat_rm.csv"
outfile = "compo_res_rm_ranked.csv"

df = pd.read_csv(infile)
ranked = df.groupby("point")["Privacy cost"].sum().sort_values()
ranked.to_csv(outfile, header=["Total Privacy Cost"])

print("Saved ranked result to", outfile)
print("Lowest 10 privacy cost points:\n", ranked.head(10))
print("Highest 10 privacy cost points:\n", ranked.tail(10))