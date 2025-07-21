import pandas as pd
import os
import sys
import glob

def average_distance_for_point(point_number):
    folder = os.path.join(os.path.dirname(__file__), "res_per_point")
    pattern = os.path.join(folder, f"run_run*_point{point_number}.csv")
    files = glob.glob(pattern)

    if not files:
        print(f"No matching files found for point {point_number}")
        sys.exit(1)

    dfs = []
    for f in files:
        df = pd.read_csv(f)[["step", "distance (sum)", "point"]]
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    # Group by step and point, then average distance
    averaged = combined.groupby(["step", "point"], as_index=False)["distance (sum)"].mean()

    output_file = os.path.join(folder, f"res_runA_point{point_number}.csv")
    averaged.to_csv(output_file, index=False)
    print(f"Averages saved to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python res_aver.py <point_number>")
        sys.exit(1)

    point = int(sys.argv[1])
    average_distance_for_point(point)
