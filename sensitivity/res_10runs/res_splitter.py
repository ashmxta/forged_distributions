import pandas as pd
import os
import sys
import re

def split_csv_by_point(csv_path):
    # Extract run number from file name, e.g., res1.csv -> 1
    match = re.search(r'res(\d+)', os.path.basename(csv_path))
    if not match:
        raise ValueError("Filename must be of the form resX.csv where X is a number.")
    run_number = match.group(1)

    # Read CSV
    df = pd.read_csv(csv_path)

    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), "res_per_point")
    os.makedirs(output_dir, exist_ok=True)

    # Split by unique points
    for point in df["point"].unique():
        df_point = df[df["point"] == point]
        output_path = os.path.join(output_dir, f"run_run{run_number}_point{int(point)}.csv")
        df_point.to_csv(output_path, index=False)
        print(f"Wrote: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python res_splitter.py <resX.csv>")
        sys.exit(1)
    split_csv_by_point(sys.argv[1])
