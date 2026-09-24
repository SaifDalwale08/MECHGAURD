from pathlib import Path
import pandas as pd

# ============================================================
# MECHGUARD - PIECuch SCHEMA INSPECTION
# ============================================================

CSV_PATH = Path(
    r"D:\MECHGUARD\datasets\Piecuch_CNC\FeatureAndMetadata_Milling.csv"
)

TARGET = "CycleToFailureNormalized"

print("=" * 80)
print("MECHGUARD - PIECuch SCHEMA INSPECTION")
print("=" * 80)

# ------------------------------------------------------------
# 1. CHECK FILE
# ------------------------------------------------------------

if not CSV_PATH.exists():
    print("\nERROR: Dataset file not found!")
    print(f"Expected path:\n{CSV_PATH}")
    raise SystemExit(1)

print(f"\nDataset:")
print(CSV_PATH)

# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

try:
    df = pd.read_csv(
        CSV_PATH,
        sep=";",
        skiprows=1
    )
except Exception as e:
    print("\nERROR while loading dataset:")
    print(e)
    raise SystemExit(1)

print("\n" + "=" * 80)
print("1. DATASET SHAPE")
print("=" * 80)

print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

# ------------------------------------------------------------
# 3. COLUMN TYPES
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("2. COLUMN TYPES")
print("=" * 80)

for i, col in enumerate(df.columns, start=1):

    try:
        unique_count = df[col].nunique(dropna=False)
    except Exception:
        unique_count = "ERROR"

    print(
        f"{i:3d}. {col:<65} "
        f"dtype={str(df[col].dtype):<12} "
        f"unique={unique_count}"
    )

# ------------------------------------------------------------
# 4. NON-NUMERIC COLUMNS
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("3. NON-NUMERIC COLUMNS")
print("=" * 80)

non_numeric = df.select_dtypes(exclude="number").columns.tolist()

print(f"Count: {len(non_numeric)}")

for col in non_numeric:

    print(f"\n--- {col} ---")

    try:
        values = df[col].drop_duplicates().head(30)

        for value in values:
            print(repr(value))

    except Exception as e:
        print(f"Could not inspect column: {e}")

# ------------------------------------------------------------
# 5. TARGET
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("4. TARGET")
print("=" * 80)

if TARGET not in df.columns:

    print(f"ERROR: Target column '{TARGET}' was not found.")

else:

    print(f"Target column : {TARGET}")
    print(f"Original dtype: {df[TARGET].dtype}")

    # The dataset uses comma decimal notation:
    # Example:
    # 0,979591837
    #
    # Convert comma decimal -> dot decimal.

    target_numeric = (
        df[TARGET]
        .astype(str)
        .str.strip()
        .str.replace(",", ".", regex=False)
    )

    target_numeric = pd.to_numeric(
        target_numeric,
        errors="coerce"
    )

    print(f"Converted dtype: {target_numeric.dtype}")
    print(f"Min           : {target_numeric.min()}")
    print(f"Max           : {target_numeric.max()}")
    print(f"Mean          : {target_numeric.mean()}")
    print(f"Median        : {target_numeric.median()}")
    print(f"Std           : {target_numeric.std()}")
    print(f"Missing       : {target_numeric.isna().sum()}")

    print("\nTarget sample:")

    print(
        target_numeric
        .head(30)
        .to_string(index=False)
    )

# ------------------------------------------------------------
# 6. POSSIBLE ID / METADATA COLUMNS
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("5. POSSIBLE ID / METADATA COLUMNS")
print("=" * 80)

keywords = [
    "file",
    "name",
    "experiment",
    "sequence",
    "cycle",
    "tool",
    "target",
    "life",
    "failure",
    "time",
    "index",
    "id",
    "material",
    "hardness",
    "holder",
    "adoc",
    "rdoc",
    "rpm",
    "feed",
    "speed",
]

for col in df.columns:

    low = col.lower()

    if any(k in low for k in keywords):

        print(
            f"{col:<70} "
            f"dtype={df[col].dtype}"
        )

# ------------------------------------------------------------
# 7. ACCELEROMETER / VIBRATION CANDIDATES
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("6. ACCELEROMETER / VIBRATION CANDIDATES")
print("=" * 80)

accel_cols = [
    col
    for col in df.columns
    if any(
        k in col.lower()
        for k in [
            "accelerometer",
            "accel",
            "vibration",
            "vib",
        ]
    )
]

print(f"Count: {len(accel_cols)}")

for col in accel_cols:
    print(col)

# ------------------------------------------------------------
# 8. CURRENT CANDIDATES
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("7. CURRENT CANDIDATES")
print("=" * 80)

current_cols = [
    col
    for col in df.columns
    if any(
        k in col.lower()
        for k in [
            "current",
            "curr",
            "amp",
        ]
    )
]

print(f"Count: {len(current_cols)}")

for col in current_cols:
    print(col)

# ------------------------------------------------------------
# 9. TEMPERATURE CANDIDATES
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("8. TEMPERATURE CANDIDATES")
print("=" * 80)

temp_cols = [
    col
    for col in df.columns
    if any(
        k in col.lower()
        for k in [
            "temp",
            "temperature",
        ]
    )
]

print(f"Count: {len(temp_cols)}")

if temp_cols:

    for col in temp_cols:
        print(col)

else:

    print("No temperature-related columns found.")

# ------------------------------------------------------------
# 10. PROCESS PARAMETER CANDIDATES
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("9. PROCESS PARAMETER CANDIDATES")
print("=" * 80)

process_keywords = [
    "rpm",
    "speed",
    "feed",
    "depth",
    "cut",
    "force",
    "load",
    "power",
    "torque",
    "adoc",
    "rdoc",
]

process_cols = [
    col
    for col in df.columns
    if any(
        k in col.lower()
        for k in process_keywords
    )
]

print(f"Count: {len(process_cols)}")

for col in process_cols:
    print(col)

# ------------------------------------------------------------
# 11. LOW-VARIANCE / CONSTANT COLUMNS
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("10. LOW-VARIANCE / CONSTANT COLUMNS")
print("=" * 80)

numeric_cols = df.select_dtypes(
    include="number"
).columns.tolist()

low_variance_found = False

for col in numeric_cols:

    try:

        nunique = df[col].nunique(dropna=False)

        if nunique <= 3:

            low_variance_found = True

            values = (
                df[col]
                .drop_duplicates()
                .tolist()
            )

            print(
                f"{col:<70} "
                f"unique={nunique} "
                f"values={values}"
            )

    except Exception as e:

        print(
            f"{col:<70} "
            f"ERROR: {e}"
        )

if not low_variance_found:
    print("No columns with <= 3 unique numeric values.")

# ------------------------------------------------------------
# 12. NUMERIC CORRELATION WITH TARGET
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("11. TOP NUMERIC CORRELATIONS WITH TARGET")
print("=" * 80)

if TARGET in df.columns:

    # Convert target safely.
    target_for_corr = (
        df[TARGET]
        .astype(str)
        .str.strip()
        .str.replace(",", ".", regex=False)
    )

    target_for_corr = pd.to_numeric(
        target_for_corr,
        errors="coerce"
    )

    numeric_df = df.select_dtypes(
        include="number"
    ).copy()

    numeric_df[TARGET] = target_for_corr

    correlations = (
        numeric_df
        .corr(numeric_only=True)[TARGET]
        .drop(TARGET)
        .dropna()
        .sort_values(
            key=lambda x: x.abs(),
            ascending=False
        )
    )

    print(
        f"Numeric features evaluated: "
        f"{len(correlations)}"
    )

    print("\nTop 30 by absolute correlation:\n")

    for col, signed_corr in correlations.head(30).items():

        print(
            f"{col:<65} "
            f"corr={signed_corr:+.6f} "
            f"abs={abs(signed_corr):.6f}"
        )

else:

    print("Target not found. Correlation skipped.")

# ------------------------------------------------------------
# 13. UNIQUE COUNTS FOR IMPORTANT METADATA
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("12. IMPORTANT METADATA DISTRIBUTION")
print("=" * 80)

metadata_candidates = [
    "FileName",
    "NumberOfCycle",
    "SampleIndex",
    "TollIndex",
    "MillingToolType",
    "ADOC",
    "RDOC",
    "HardnessMean",
    "ToolHolderLength",
    "CycleToFailure",
    "CycleToFailureNormalized",
]

for col in metadata_candidates:

    if col not in df.columns:
        continue

    print(f"\n--- {col} ---")
    print(f"dtype  : {df[col].dtype}")
    print(f"unique : {df[col].nunique(dropna=False)}")

    try:

        values = (
            df[col]
            .drop_duplicates()
            .head(20)
            .tolist()
        )

        print(f"values : {values}")

    except Exception as e:

        print(f"Could not display values: {e}")

# ------------------------------------------------------------
# 14. SAMPLE ROW
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("13. SAMPLE ROW")
print("=" * 80)

try:

    sample = df.iloc[0]

    for col, value in sample.items():

        print(
            f"{col:<65} : {value}"
        )

except Exception as e:

    print(f"Could not display sample row: {e}")

# ------------------------------------------------------------
# 15. BASIC DATA QUALITY CHECK
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("14. DATA QUALITY CHECK")
print("=" * 80)

print(f"Total rows           : {len(df)}")
print(f"Total columns        : {len(df.columns)}")
print(f"Duplicate rows       : {df.duplicated().sum()}")
print(
    f"Total missing cells  : "
    f"{df.isna().sum().sum()}"
)

# ------------------------------------------------------------
# 16. TARGET QUALITY CHECK
# ------------------------------------------------------------

if TARGET in df.columns:

    print("\n" + "=" * 80)
    print("15. TARGET QUALITY CHECK")
    print("=" * 80)

    print(
        f"Target numeric valid : "
        f"{target_numeric.notna().sum()}"
    )

    print(
        f"Target missing       : "
        f"{target_numeric.isna().sum()}"
    )

    print(
        f"Target < 0          : "
        f"{(target_numeric < 0).sum()}"
    )

    print(
        f"Target > 1          : "
        f"{(target_numeric > 1).sum()}"
    )

    print(
        f"Target == 0         : "
        f"{(target_numeric == 0).sum()}"
    )

    print(
        f"Target == 1         : "
        f"{(target_numeric == 1).sum()}"
    )

# ------------------------------------------------------------
# COMPLETE
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("SCHEMA INSPECTION COMPLETE")
print("=" * 80)

print("\nNo model training was performed.")
print("No dataset files were modified.")
print("=" * 80)