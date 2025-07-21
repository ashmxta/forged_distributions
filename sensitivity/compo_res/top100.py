#!/usr/bin/env python3
import pandas as pd
import os

def main():
    # locate this script’s folder, so paths are always relative to it
    here = os.path.dirname(os.path.realpath(__file__))

    # 1) load your ranked results CSV
    csv_path = os.path.join(here, "compo_res_ranked.csv")
    df = pd.read_csv(csv_path)

    # 2) grab the 100 points with the smallest Total Privacy Cost
    #    (make sure this matches your actual column name)
    top100 = df.nsmallest(100, "Total Privacy Cost")["point"].astype(int).tolist()

    # 3) write them out as a space-separated list
    outpath = os.path.join(here, "removed_points.txt")
    with open(outpath, "w") as f:
        f.write(" ".join(map(str, top100)))

    print(f"Wrote {len(top100)} indices to {outpath}")

if __name__ == "__main__":
    main()
