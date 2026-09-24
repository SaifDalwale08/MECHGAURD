from pathlib import Path
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat


# ============================================================
# MECHGUARD - DATASET INSPECTION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "datasets"


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# 1. PROJECT / DATASET STRUCTURE
# ============================================================

section("MECHGUARD DATASET STRUCTURE")

print(f"Project root : {ROOT}")
print(f"Dataset root : {DATASETS}")

if not DATASETS.exists():
    print("ERROR: datasets folder not found.")
    raise SystemExit(1)

for item in sorted(DATASETS.iterdir()):

    if item.is_dir():

        print(f"\n[DIR] {item.name}")

        for child in sorted(item.iterdir()):

            if child.is_dir():

                print(f"    [DIR] {child.name}")

            else:

                size_mb = child.stat().st_size / (1024 * 1024)

                print(
                    f"    {child.name} "
                    f"({size_mb:.2f} MB)"
                )


# ============================================================
# 2. CWRU DATASET
# ============================================================

section("CWRU DATASET")

CWRU = DATASETS / "CWRU"

if CWRU.exists():

    mat_files = sorted(CWRU.glob("*.mat"))

    if not mat_files:

        print("No .mat files found in CWRU.")

    else:

        for file in mat_files:

            print(f"\n--- {file.name} ---")

            try:

                data = loadmat(file)

                keys = [
                    key
                    for key in data.keys()
                    if not key.startswith("__")
                ]

                for key in keys:

                    value = data[key]

                    if isinstance(value, np.ndarray):

                        print(
                            f"{key}: "
                            f"shape={value.shape}, "
                            f"dtype={value.dtype}"
                        )

                    else:

                        print(
                            f"{key}: "
                            f"type={type(value).__name__}"
                        )

            except Exception as e:

                print(
                    f"ERROR reading {file.name}: {e}"
                )

else:

    print("CWRU folder not found.")


# ============================================================
# 3. PADERBORN DATASET
# ============================================================

section("PADERBORN DATASET")

PADERBORN = DATASETS / "Paderborn"

if PADERBORN.exists():

    total_paderborn_files = 0

    for root, dirs, files in os.walk(PADERBORN):

        root_path = Path(root)

        for file in sorted(files):

            path = root_path / file

            size_mb = path.stat().st_size / (1024 * 1024)

            relative_path = path.relative_to(PADERBORN)

            print(
                f"{relative_path} "
                f"({size_mb:.2f} MB)"
            )

            total_paderborn_files += 1

    print(
        f"\nTotal Paderborn files: "
        f"{total_paderborn_files}"
    )

else:

    print("Paderborn folder not found.")


# ============================================================
# 4. PIECuch CNC DATASET
# ============================================================

section("PIECUCH CNC DATASET")

PIECUCH = DATASETS / "Piecuch_CNC"

if PIECUCH.exists():

    csv_files = sorted(PIECUCH.glob("*.csv"))

    if not csv_files:

        print("No CSV files found in Piecuch_CNC.")

    else:

        for file in csv_files:

            print(f"\n--- {file.name} ---")

            try:

                # Read the CSV
                df = pd.read_csv(file)

                print(
                    f"Rows    : {len(df):,}"
                )

                print(
                    f"Columns : {len(df.columns)}"
                )

                # ------------------------------------------------
                # Column names
                # ------------------------------------------------

                print("\nColumns:")

                for column in df.columns:

                    print(f"  - {column}")

                # ------------------------------------------------
                # Data types
                # ------------------------------------------------

                print("\nData types:")

                print(
                    df.dtypes.to_string()
                )

                # ------------------------------------------------
                # First rows
                # ------------------------------------------------

                print("\nFirst 5 rows:")

                print(
                    df.head().to_string()
                )

                # ------------------------------------------------
                # Missing values
                # ------------------------------------------------

                print("\nMissing values:")

                missing = df.isnull().sum()

                missing = missing[
                    missing > 0
                ]

                if len(missing) > 0:

                    print(
                        missing.to_string()
                    )

                else:

                    print(
                        "No missing values."
                    )

                # ------------------------------------------------
                # Basic statistics
                # ------------------------------------------------

                print(
                    "\nNumeric column summary:"
                )

                numeric_columns = (
                    df.select_dtypes(
                        include=np.number
                    )
                )

                if not numeric_columns.empty:

                    print(
                        numeric_columns.describe()
                        .transpose()
                        .to_string()
                    )

                else:

                    print(
                        "No numeric columns found."
                    )

            except Exception as e:

                print(
                    f"ERROR reading {file.name}: {e}"
                )

else:

    print(
        "Piecuch_CNC folder not found."
    )


# ============================================================
# 5. NASA MILLING DATASET
# ============================================================

section("NASA MILLING DATASET")

NASA = DATASETS / "NASA_Milling"

if NASA.exists():

    total_nasa_files = 0

    for root, dirs, files in os.walk(NASA):

        root_path = Path(root)

        for file in sorted(files):

            path = root_path / file

            size_mb = (
                path.stat().st_size
                / (1024 * 1024)
            )

            relative_path = (
                path.relative_to(NASA)
            )

            print(
                f"{relative_path} "
                f"({size_mb:.2f} MB)"
            )

            total_nasa_files += 1

    print(
        f"\nTotal NASA files: "
        f"{total_nasa_files}"
    )

else:

    print(
        "NASA_Milling folder not found."
    )


# ============================================================
# 6. FINAL SUMMARY
# ============================================================

section("INSPECTION COMPLETE")

print(
    """
Dataset inspection finished successfully.

The next stage will be:
1. Analyze dataset structure
2. Identify usable signals/features
3. Identify labels/targets
4. Design preprocessing
5. Build training datasets
6. Train and evaluate MECHGUARD models
"""
)