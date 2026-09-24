from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# MECHGUARD - PIECuch CNC DATASET PROFILING
# ============================================================

DATASET_PATH = Path(
    r"D:\MECHGUARD\datasets\Piecuch_CNC\FeatureAndMetadata_Milling.csv"
)


def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main():

    section("MECHGUARD - PIECuch CNC DATASET PROFILING")

    print(f"Dataset:")
    print(DATASET_PATH)

    if not DATASET_PATH.exists():
        print("\nERROR: Dataset file does not exist.")
        return

    print(
        f"\nFile size: "
        f"{DATASET_PATH.stat().st_size / (1024 * 1024):.2f} MB"
    )

    # ========================================================
    # 1. LOAD DATASET
    # ========================================================

    section("1. LOADING DATASET")

    try:

        # The first row contains generic names:
        # Column1, Column2, ..., Column131
        #
        # The SECOND row contains the actual feature names.
        #
        # Therefore:
        # skiprows=1
        # makes the second physical row the header.

        df = pd.read_csv(
            DATASET_PATH,
            sep=";",
            skiprows=1,
            low_memory=False
        )

        print("Dataset loaded successfully.")

    except Exception as e:

        print("ERROR while loading dataset:")
        print(e)

        return

    # ========================================================
    # 2. CLEAN COLUMN NAMES
    # ========================================================

    section("2. COLUMN NAMES")

    df.columns = [
        str(col).strip()
        for col in df.columns
    ]

    print(f"Total columns: {len(df.columns)}")

    for i, col in enumerate(df.columns, start=1):
        print(f"{i:3d}. {col}")

    # ========================================================
    # 3. BASIC SHAPE
    # ========================================================

    section("3. DATASET SHAPE")

    print(f"Rows    : {df.shape[0]:,}")
    print(f"Columns : {df.shape[1]:,}")

    # ========================================================
    # 4. CONVERT NUMERIC COLUMNS
    # ========================================================

    section("4. NUMERIC CONVERSION")

    # Columns that are definitely metadata/text
    text_columns = [
        "FileName",
    ]

    for col in df.columns:

        if col in text_columns:
            continue

        # Try converting everything else to numeric.
        converted = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        # If most values successfully convert,
        # keep the numeric version.
        valid_ratio = converted.notna().mean()

        if valid_ratio >= 0.90:

            df[col] = converted

    # CycleToFailureNormalized contains comma-decimal
    # values in this dataset, e.g. 0,979591837.
    if "CycleToFailureNormalized" in df.columns:

        df["CycleToFailureNormalized"] = (
            df["CycleToFailureNormalized"]
            .astype(str)
            .str.replace(",", ".", regex=False)
        )

        df["CycleToFailureNormalized"] = pd.to_numeric(
            df["CycleToFailureNormalized"],
            errors="coerce"
        )

    print("Numeric conversion completed.")

    # ========================================================
    # 5. DATA TYPES
    # ========================================================

    section("5. DATA TYPES")

    print(df.dtypes.to_string())

    # ========================================================
    # 6. MISSING VALUES
    # ========================================================

    section("6. MISSING VALUES")

    missing = df.isna().sum()

    missing = missing[
        missing > 0
    ].sort_values(
        ascending=False
    )

    if missing.empty:

        print("No missing values found.")

    else:

        print(missing.to_string())

    # ========================================================
    # 7. DUPLICATE ROWS
    # ========================================================

    section("7. DUPLICATE ROWS")

    duplicate_count = df.duplicated().sum()

    print(
        f"Duplicate rows: "
        f"{duplicate_count:,}"
    )

    # ========================================================
    # 8. NUMERIC / NON-NUMERIC
    # ========================================================

    section("8. NUMERIC VS NON-NUMERIC FEATURES")

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    non_numeric_columns = df.select_dtypes(
        exclude=[np.number]
    ).columns.tolist()

    print(
        f"Numeric columns    : "
        f"{len(numeric_columns)}"
    )

    print(
        f"Non-numeric columns: "
        f"{len(non_numeric_columns)}"
    )

    print("\nNon-numeric columns:")

    for col in non_numeric_columns:
        print(f" - {col}")

    # ========================================================
    # 9. IMPORTANT METADATA
    # ========================================================

    section("9. IMPORTANT METADATA COLUMNS")

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

        print(f"\n{col}")

        print(
            f"  dtype  : "
            f"{df[col].dtype}"
        )

        print(
            f"  unique : "
            f"{df[col].nunique(dropna=False):,}"
        )

        if pd.api.types.is_numeric_dtype(df[col]):

            print(
                f"  min    : "
                f"{df[col].min()}"
            )

            print(
                f"  max    : "
                f"{df[col].max()}"
            )

            print(
                f"  mean   : "
                f"{df[col].mean()}"
            )

    # ========================================================
    # 10. MILLING TOOL INFORMATION
    # ========================================================

    section("10. MILLING TOOL INFORMATION")

    if "MillingToolType" in df.columns:

        print(
            df["MillingToolType"]
            .value_counts(
                dropna=False
            )
            .sort_index()
            .to_string()
        )

    else:

        print(
            "MillingToolType column not found."
        )

    # ========================================================
    # 11. CYCLE INFORMATION
    # ========================================================

    section("11. CYCLE INFORMATION")

    cycle_columns = [
        "NumberOfCycle",
        "SampleIndex",
        "CycleToFailure",
        "CycleToFailureNormalized",
    ]

    for col in cycle_columns:

        if col not in df.columns:
            continue

        print(
            f"\n--- {col} ---"
        )

        print(
            f"Unique values : "
            f"{df[col].nunique(dropna=False):,}"
        )

        if pd.api.types.is_numeric_dtype(df[col]):

            print(
                f"Minimum       : "
                f"{df[col].min()}"
            )

            print(
                f"Maximum       : "
                f"{df[col].max()}"
            )

            print(
                f"Mean          : "
                f"{df[col].mean()}"
            )

            print(
                f"Median        : "
                f"{df[col].median()}"
            )

    # ========================================================
    # 12. NUMERIC DESCRIPTIVE STATISTICS
    # ========================================================

    section("12. NUMERIC DESCRIPTIVE STATISTICS")

    if len(numeric_columns) > 0:

        stats = (
            df[numeric_columns]
            .describe()
            .T
        )

        print(
            stats.to_string()
        )

    else:

        print(
            "WARNING: No numeric columns "
            "were detected."
        )

    # ========================================================
    # 13. SENSOR FEATURE GROUPS
    # ========================================================

    section("13. SENSOR FEATURE GROUPS")

    acceleration_columns = [
        col
        for col in df.columns
        if "Accelerometer" in col
    ]

    current_columns = [
        col
        for col in df.columns
        if "Current" in col
    ]

    print(
        f"Accelerometer columns : "
        f"{len(acceleration_columns)}"
    )

    print(
        f"Current columns       : "
        f"{len(current_columns)}"
    )

    print("\nACCELEROMETER FEATURES:")

    for col in acceleration_columns:
        print(f" - {col}")

    print("\nCURRENT FEATURES:")

    for col in current_columns:
        print(f" - {col}")

    # ========================================================
    # 14. FEATURE STATISTIC TYPES
    # ========================================================

    section("14. FEATURE STATISTIC TYPES")

    statistic_keywords = [
        "min",
        "max",
        "mean",
        "std",
        "skew",
        "kurtosis",
    ]

    for stat in statistic_keywords:

        matching = [
            col
            for col in df.columns
            if col.lower().endswith(
                f" - {stat}"
            )
        ]

        print(
            f"{stat:10s}: "
            f"{len(matching)} columns"
        )

    # ========================================================
    # 15. TARGET INFORMATION
    # ========================================================

    section("15. TARGET INFORMATION")

    target_candidates = [
        "CycleToFailure",
        "CycleToFailureNormalized",
    ]

    for col in target_candidates:

        if col not in df.columns:
            continue

        print(f"\n--- {col} ---")

        if pd.api.types.is_numeric_dtype(
            df[col]
        ):

            print(
                df[col]
                .describe()
                .to_string()
            )

        else:

            print(
                df[col]
                .value_counts(
                    dropna=False
                )
                .head(20)
                .to_string()
            )

    # ========================================================
    # 16. POTENTIAL DATA LEAKAGE
    # ========================================================

    section("16. POTENTIAL DATA LEAKAGE")

    leakage_candidates = [
        "CycleToFailure",
        "CycleToFailureNormalized",
        "NumberOfCycle",
        "SampleIndex",
    ]

    for col in leakage_candidates:

        if col in df.columns:

            print(
                f"[LEAKAGE CHECK] {col}"
            )

            print(
                "Do NOT automatically use "
                "this column as a model input."
            )

    # ========================================================
    # 17. INITIAL MODEL EXCLUSION LIST
    # ========================================================

    section("17. INITIAL MODEL EXCLUSION LIST")

    exclusion_columns = [
        "FileName",
        "NumberOfCycle",
        "SampleIndex",
        "TollIndex",
        "CycleToFailure",
        "CycleToFailureNormalized",
    ]

    for col in exclusion_columns:

        if col in df.columns:
            print(f" - {col}")

    # ========================================================
    # 18. INITIAL SENSOR FEATURE CANDIDATES
    # ========================================================

    section("18. INITIAL SENSOR FEATURE CANDIDATES")

    sensor_features = []

    for col in df.columns:

        if (
            "Accelerometer" in col
            or "Current" in col
        ):

            sensor_features.append(col)

    print(
        f"Initial sensor feature count: "
        f"{len(sensor_features)}"
    )

    for col in sensor_features:

        print(
            f" - {col}"
        )

    # ========================================================
    # 19. CORRELATION WITH TARGET
    # ========================================================

    section(
        "19. FEATURE CORRELATION "
        "WITH CYCLE-TO-FAILURE"
    )

    if (
        "CycleToFailureNormalized"
        in df.columns
        and len(numeric_columns) > 1
    ):

        numeric_features = (
            df.select_dtypes(
                include=[np.number]
            )
        )

        correlations = (
            numeric_features
            .corr()[
                "CycleToFailureNormalized"
            ]
            .drop(
                "CycleToFailureNormalized",
                errors="ignore"
            )
            .sort_values()
        )

        print(
            correlations.to_string()
        )

    else:

        print(
            "Target or numeric features "
            "not available."
        )

    # ========================================================
    # 20. FIRST FIVE DATA ROWS
    # ========================================================

    section("20. FIRST 5 DATA ROWS")

    print(
        df.head(5).to_string()
    )

    # ========================================================
    # 21. FINAL SUMMARY
    # ========================================================

    section("21. DATASET AUDIT SUMMARY")

    print(
        f"Rows                    : "
        f"{len(df):,}"
    )

    print(
        f"Columns                 : "
        f"{len(df.columns):,}"
    )

    print(
        f"Numeric columns         : "
        f"{len(numeric_columns):,}"
    )

    print(
        f"Non-numeric columns     : "
        f"{len(non_numeric_columns):,}"
    )

    print(
        f"Accelerometer features  : "
        f"{len(acceleration_columns):,}"
    )

    print(
        f"Current features        : "
        f"{len(current_columns):,}"
    )

    print(
        f"Duplicate rows          : "
        f"{duplicate_count:,}"
    )

    if missing.empty:

        print(
            "Missing values          : NONE"
        )

    else:

        print(
            f"Columns with missing    : "
            f"{len(missing)}"
        )

    print(
        "\nDataset profiling complete."
    )

    print(
        "\nDO NOT TRAIN YET."
        "\nFirst finalize feature selection, "
        "target construction, and split strategy."
    )


if __name__ == "__main__":
    main()