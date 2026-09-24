from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FILE = ROOT / "datasets" / "Piecuch_CNC" / "FeatureAndMetadata_Milling.csv"

print("=" * 70)
print("MECHGUARD - PIECuch CSV INSPECTION")
print("=" * 70)

print(f"File: {FILE}")
print(f"Size: {FILE.stat().st_size / (1024 * 1024):.2f} MB")

print("\n--- RAW FIRST 10 LINES ---")

with open(FILE, "r", encoding="utf-8", errors="replace") as f:
    for i in range(10):
        line = f.readline()

        if not line:
            break

        print(f"{i + 1}: {line.rstrip()}")

print("\n--- TRYING CSV DELIMITERS ---")

for delimiter in [",", ";", "\t", "|"]:

    try:

        df = pd.read_csv(
            FILE,
            sep=delimiter,
            nrows=5
        )

        print(
            f"\nDelimiter {repr(delimiter)} "
            f"-> {len(df.columns)} columns"
        )

        print(
            list(df.columns)
        )

    except Exception as e:

        print(
            f"\nDelimiter {repr(delimiter)} "
            f"-> ERROR: {e}"
        )

print("\nInspection complete.")