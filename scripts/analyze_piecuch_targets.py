from pathlib import Path
import pandas as pd
import numpy as np


DATASET_PATH = Path(
    r"D:\MECHGUARD\datasets\Piecuch_CNC\FeatureAndMetadata_Milling.csv"
)


def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main():

    section("MECHGUARD - PIECuch TARGET ANALYSIS")

    # ============================================================
    # LOAD
    # ============================================================

    df = pd.read_csv(
        DATASET_PATH,
        sep=";",
        skiprows=1,
        low_memory=False
    )

    df.columns = [
        str(col).strip()
        for col in df.columns
    ]

    # Convert numeric columns
    for col in df.columns:

        if col == "FileName":
            continue

        converted = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        if converted.notna().mean() >= 0.90:
            df[col] = converted

    # Explicitly handle normalized target
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

    # ============================================================
    # BASIC INFO
    # ============================================================

    section("1. BASIC DATASET INFO")

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns):,}")

    # ============================================================
    # FILE / TOOL INFORMATION
    # ============================================================

    section("2. FILE / TOOL INFORMATION")

    for col in [
        "FileName",
        "MillingToolType",
        "NumberOfCycle",
        "SampleIndex",
        "TollIndex"
    ]:

        if col not in df.columns:
            print(f"{col}: NOT FOUND")
            continue

        print(f"\n--- {col} ---")

        print(
            df[col]
            .value_counts(dropna=False)
            .head(50)
            .to_string()
        )

    # ============================================================
    # UNIQUE FILES
    # ============================================================

    section("3. UNIQUE FILES")

    if "FileName" in df.columns:

        files = sorted(
            df["FileName"]
            .dropna()
            .astype(str)
            .unique()
        )

        print(
            f"Unique FileName values: "
            f"{len(files)}"
        )

        for name in files:
            print(f" - {name}")

    # ============================================================
    # CYCLE TO FAILURE
    # ============================================================

    section("4. CYCLE TO FAILURE")

    if "CycleToFailure" in df.columns:

        print(
            df["CycleToFailure"]
            .describe()
            .to_string()
        )

        print("\nFirst 30 values:")

        print(
            df[
                [
                    "FileName",
                    "NumberOfCycle",
                    "CycleToFailure"
                ]
            ]
            .head(30)
            .to_string(index=False)
        )

    # ============================================================
    # NORMALIZED CYCLE TO FAILURE
    # ============================================================

    section("5. CYCLE TO FAILURE NORMALIZED")

    if "CycleToFailureNormalized" in df.columns:

        print(
            df["CycleToFailureNormalized"]
            .describe()
            .to_string()
        )

        print("\nFirst 30 values:")

        print(
            df[
                [
                    "FileName",
                    "NumberOfCycle",
                    "CycleToFailureNormalized"
                ]
            ]
            .head(30)
            .to_string(index=False)
        )

    # ============================================================
    # TOOL-BY-TOOL STATISTICS
    # ============================================================

    section("6. TOOL-BY-TOOL STATISTICS")

    if (
        "FileName" in df.columns
        and "CycleToFailureNormalized" in df.columns
    ):

        grouped = (
            df
            .groupby("FileName")[
                "CycleToFailureNormalized"
            ]
            .agg(
                [
                    "count",
                    "min",
                    "max",
                    "mean",
                    "median"
                ]
            )
            .sort_index()
        )

        print(
            grouped.to_string()
        )

    # ============================================================
    # NORMALIZED HEALTH BANDS
    # ============================================================

    section("7. NORMALIZED VALUE BANDS")

    if "CycleToFailureNormalized" in df.columns:

        x = df[
            "CycleToFailureNormalized"
        ]

        bands = pd.cut(
            x,
            bins=[
                -np.inf,
                0.20,
                0.40,
                0.60,
                0.80,
                np.inf
            ],
            labels=[
                "0.00 - 0.20",
                "0.20 - 0.40",
                "0.40 - 0.60",
                "0.60 - 0.80",
                "0.80 - 1.00+"
            ]
        )

        print(
            bands
            .value_counts(
                sort=False,
                dropna=False
            )
            .to_string()
        )

    # ============================================================
    # POTENTIAL BINARY TARGETS
    # ============================================================

    section("8. POTENTIAL BINARY TARGETS")

    if "CycleToFailureNormalized" in df.columns:

        x = df[
            "CycleToFailureNormalized"
        ]

        thresholds = [
            0.20,
            0.30,
            0.40,
            0.50,
            0.60,
            0.70
        ]

        for threshold in thresholds:

            degraded = (
                x < threshold
            )

            normal_count = (
                (~degraded)
                .sum()
            )

            degraded_count = (
                degraded
                .sum()
            )

            print(
                f"\nThreshold < {threshold:.2f}"
            )

            print(
                f"Normal / not degraded : "
                f"{normal_count}"
            )

            print(
                f"Degraded              : "
                f"{degraded_count}"
            )

    # ============================================================
    # PROPOSED 3-STATE TARGET
    # ============================================================

    section("9. POTENTIAL THREE-STATE TARGET")

    if "CycleToFailureNormalized" in df.columns:

        x = df[
            "CycleToFailureNormalized"
        ]

        states = pd.cut(
            x,
            bins=[
                -np.inf,
                0.30,
                0.60,
                np.inf
            ],
            labels=[
                "DEGRADED",
                "WARNING",
                "NORMAL"
            ]
        )

        print(
            states
            .value_counts(
                sort=False,
                dropna=False
            )
            .to_string()
        )

    # ============================================================
    # CYCLE DISTRIBUTION PER FILE
    # ============================================================

    section("10. CYCLE RANGE PER FILE")

    if (
        "FileName" in df.columns
        and "NumberOfCycle" in df.columns
    ):

        cycle_summary = (
            df
            .groupby("FileName")[
                "NumberOfCycle"
            ]
            .agg(
                [
                    "count",
                    "min",
                    "max"
                ]
            )
            .sort_index()
        )

        print(
            cycle_summary.to_string()
        )

    # ============================================================
    # PROCESS PARAMETERS
    # ============================================================

    section("11. PROCESS PARAMETERS")

    for col in [
        "ADOC",
        "RDOC",
        "HardnessMean",
        "ToolHolderLength"
    ]:

        if col not in df.columns:
            continue

        print(f"\n--- {col} ---")

        print(
            df[col]
            .describe()
            .to_string()
        )

        print(
            "\nUnique values:"
        )

        print(
            df[col]
            .value_counts(
                dropna=False
            )
            .head(30)
            .to_string()
        )

    # ============================================================
    # FINAL
    # ============================================================

    section("12. ANALYSIS COMPLETE")

    print(
        "Do NOT train yet."
    )

    print(
        "Use the above distribution to "
        "choose the target definition and "
        "validation strategy."
    )


if __name__ == "__main__":
    main()