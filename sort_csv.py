import os
import pandas as pd

# 🔹 Target folder to process
target_dir = r"E:\Garry\tase"

print("Sorting CSV files in:", target_dir)

for file in os.listdir(target_dir):
    if file.endswith(".csv") and not file.endswith("_sorted.csv"):
        print(f"\nProcessing: {file}")

        file_path = os.path.join(target_dir, file)

        # Skip first 2 rows
        df = pd.read_csv(file_path, sep=",", skiprows=2)

        # Clean column names
        df.columns = df.columns.str.strip()

        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
            df = df.dropna(subset=["Date"])
            df = df.sort_values("Date")

            new_name = file.replace(".csv", "_sorted.csv")
            new_path = os.path.join(target_dir, new_name)

            df.to_csv(new_path, sep=",", index=False)

            print(f"Created: {new_name}")
        else:
            print(f"No 'Date' column found in {file}")

print("\nDone.")