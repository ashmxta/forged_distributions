import pandas as pd
import os

# Directory containing the CSVs
RES_DIR = os.path.dirname(__file__)
OUTPUT_FILE = os.path.join(RES_DIR, "res_concat.csv")

# Collect all filenames
file_list = [os.path.join(RES_DIR, f"res{i}.csv") for i in range(1, 11)]

# Read and concatenate
df_all = pd.concat([pd.read_csv(file) for file in file_list], ignore_index=True)

# Save to output
df_all.to_csv(OUTPUT_FILE, index=False)

print(f"Saved concatenated CSV to {OUTPUT_FILE}")


