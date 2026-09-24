from pathlib import Path
import pandas as pd


DATASET_PATH = Path(
    r"D:\MECHGUARD\datasets\Piecuch_CNC\FeatureAndMetadata_Milling.csv"
)


def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main():

    section("MECHGUARD - PIECuch GROUP ANALYSIS")

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

    # ============================================================
    # TARGET CONVERSION
    # ============================================================

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
    # PARSE FILENAME
    # ============================================================

    parts = (
        df["FileName"]
        .astype(str)
        .str.split("_", expand=True)
    )

    df["Experiment"] = parts[0]
    df["Sequence"] = parts[1]
    df["Cycle"] = parts[2]

    # ============================================================
    # BASIC GROUP COUNTS
    # ============================================================

    section("1. GROUP COUNTS")

    print(
        f"Total rows       : {len(df):,}"
    )

    print(
        f"Unique experiments: "
        f"{df['Experiment'].nunique():,}"
    )

    print(
        f"Unique sequences : "
        f"{df['Sequence'].nunique():,}"
    )

    print(
        f"Unique cycles    : "
        f"{df['Cycle'].nunique():,}"
    )

    # ============================================================
    # EXPERIMENT DISTRIBUTION
    # ============================================================

    section("2. EXPERIMENT SAMPLE COUNTS")

    experiment_counts = (
        df["Experiment"]
        .value_counts()
        .sort_index()
    )

    print(
        experiment_counts.to_string()
    )

    # ============================================================
    # EXPERIMENT SUMMARY
    # ============================================================

    section("3. EXPERIMENT TARGET SUMMARY")

    experiment_summary = (
        df
        .groupby("Experiment")
        .agg(
            samples=(
                "CycleToFailureNormalized",
                "count"
            ),
            target_min=(
                "CycleToFailureNormalized",
                "min"
            ),
            target_max=(
                "CycleToFailureNormalized",
                "max"
            ),
            target_mean=(
                "CycleToFailureNormalized",
                "mean"
            ),
            target_median=(
                "CycleToFailureNormalized",
                "median"
            ),
            sequences=(
                "Sequence",
                "nunique"
            ),
        )
        .sort_index()
    )

    print(
        experiment_summary.to_string()
    )

    # ============================================================
    # SEQUENCE DISTRIBUTION
    # ============================================================

    section("4. SEQUENCE DISTRIBUTION")

    sequence_counts = (
        df["Sequence"]
        .value_counts()
        .sort_index()
    )

    print(
        sequence_counts.to_string()
    )

    # ============================================================
    # EXPERIMENT + SEQUENCE
    # ============================================================

    section("5. EXPERIMENT + SEQUENCE COUNTS")

    exp_seq = (
        df
        .groupby(
            [
                "Experiment",
                "Sequence"
            ]
        )
        .size()
        .reset_index(
            name="samples"
        )
    )

    print(
        exp_seq.to_string(
            index=False
        )
    )

    # ============================================================
    # TOOL TYPE
    # ============================================================

    section("6. TOOL TYPE BY EXPERIMENT")

    if "MillingToolType" in df.columns:

        tool_distribution = pd.crosstab(
            df["Experiment"],
            df["MillingToolType"]
        )

        print(
            tool_distribution.to_string()
        )

    # ============================================================
    # TARGET RANGE PER EXPERIMENT
    # ============================================================

    section("7. TARGET RANGE PER EXPERIMENT")

    target_range = (
        df
        .groupby("Experiment")
        ["CycleToFailureNormalized"]
        .agg(
            [
                "count",
                "min",
                "max",
                "mean",
                "std"
            ]
        )
        .sort_index()
    )

    print(
        target_range.to_string()
    )

    # ============================================================
    # SAMPLE ORDER
    # ============================================================

    section("8. FIRST 100 FILE NAMES")

    print(
        df[
            [
                "FileName",
                "Experiment",
                "Sequence",
                "Cycle",
                "CycleToFailureNormalized"
            ]
        ]
        .head(100)
        .to_string(index=False)
    )

    # ============================================================
    # CHECK WHETHER EXPERIMENT IS A GOOD GROUP
    # ============================================================

    section("9. GROUPING CHECK")

    print(
        "The Experiment column is derived from "
        "the first filename component."
    )

    print(
        "\nWe will inspect whether each experiment "
        "contains a coherent degradation trajectory."
    )

    for experiment in sorted(
        df["Experiment"].unique()
    ):

        subset = df[
            df["Experiment"] == experiment
        ]

        values = (
            subset[
                "CycleToFailureNormalized"
            ]
            .dropna()
            .sort_values()
            .values
        )

        if len(values) == 0:
            continue

        print(
            f"{experiment}: "
            f"{len(values)} samples | "
            f"target {values.min():.4f}"
            f" -> {values.max():.4f}"
        )

    # ============================================================
    # FINAL
    # ============================================================

    section("10. ANALYSIS COMPLETE")

    print(
        "Do NOT train yet."
    )

    print(
        "Use this output to determine the "
        "correct group-aware train/test split."
    )


if __name__ == "__main__":
    main()