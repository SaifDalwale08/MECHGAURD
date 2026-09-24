from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(r"D:\MECHGUARD")

DATA_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "Piecuch_CNC"
    / "FeatureAndMetadata_Milling.csv"
)

MODEL_DIR = PROJECT_ROOT / "models" / "piecuch_rul"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "CycleToFailureNormalized"

# Metadata / leakage columns
EXCLUDE_COLUMNS = {
    "FileName",
    "NumberOfCycle",
    "SampleIndex",
    "TollIndex",
    "CycleToFailure",
    "CycleToFailureNormalized",
    "MillingToolType",
    "ADOC",
    "ToolHolderLength",
    "RDOC",
    "HardnessMean",
}


# ============================================================
# HELPERS
# ============================================================

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate_model(name, model, X_test, y_test):

    predictions = model.predict(X_test)

    # RUL is normalized between 0 and 1.
    # Keep predictions inside the valid range.
    predictions = np.clip(predictions, 0.0, 1.0)

    mae = mean_absolute_error(y_test, predictions)
    score_rmse = rmse(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(f"MAE  : {mae:.6f}")
    print(f"RMSE : {score_rmse:.6f}")
    print(f"R²   : {r2:.6f}")

    return {
        "model": name,
        "mae": mae,
        "rmse": score_rmse,
        "r2": r2,
        "predictions": predictions,
    }


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("MECHGUARD - PIECUCH CNC RUL TRAINING V2")
print("=" * 70)

print()
print("V2 PURPOSE:")
print("Train using SENSOR FEATURES ONLY.")
print("48 accelerometer features + 72 current features = 120 features.")
print()


# ============================================================
# LOAD DATASET
# ============================================================

print("[1/9] Loading dataset...")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}"
    )

df = pd.read_csv(
    DATA_PATH,
    sep=";",
    skiprows=1,
    low_memory=False,
)

print(f"Dataset shape: {df.shape}")


# ============================================================
# TARGET
# ============================================================

print()
print("[2/9] Preparing target...")

df[TARGET] = (
    df[TARGET]
    .astype(str)
    .str.replace(",", ".", regex=False)
    .astype(float)
)

if df[TARGET].isna().any():
    raise ValueError("Target contains missing values.")

if not df[TARGET].between(0.0, 1.0).all():
    raise ValueError("Target contains values outside [0,1].")

print(f"Target: {TARGET}")
print(f"Min   : {df[TARGET].min():.6f}")
print(f"Max   : {df[TARGET].max():.6f}")
print(f"Mean  : {df[TARGET].mean():.6f}")


# ============================================================
# EXPERIMENT GROUP
# ============================================================

print()
print("[3/9] Creating experiment groups...")

df["Experiment"] = (
    df["FileName"]
    .astype(str)
    .str.split("_")
    .str[0]
)

groups = df["Experiment"]

print(f"Unique experiment groups: {groups.nunique()}")

print()
print(groups.value_counts().sort_index().to_string())


# ============================================================
# SENSOR FEATURE SELECTION
# ============================================================

print()
print("[4/9] Selecting ONLY sensor features...")

# Explicitly select the actual accelerometer and current
# features instead of selecting every numeric column.

sensor_columns = [
    col
    for col in df.columns
    if (
        str(col).startswith("Accelerometer -")
        or str(col).startswith("Current -")
    )
]

# Remove anything unexpected.
sensor_columns = [
    col
    for col in sensor_columns
    if col not in EXCLUDE_COLUMNS
]

# Safety checks
accelerometer_features = [
    col
    for col in sensor_columns
    if str(col).startswith("Accelerometer -")
]

current_features = [
    col
    for col in sensor_columns
    if str(col).startswith("Current -")
]

print()
print(f"Accelerometer features: {len(accelerometer_features)}")
print(f"Current features      : {len(current_features)}")
print(f"TOTAL SENSOR FEATURES : {len(sensor_columns)}")

if len(accelerometer_features) != 48:
    raise RuntimeError(
        f"Expected 48 accelerometer features, "
        f"found {len(accelerometer_features)}."
    )

if len(current_features) != 72:
    raise RuntimeError(
        f"Expected 72 current features, "
        f"found {len(current_features)}."
    )

if len(sensor_columns) != 120:
    raise RuntimeError(
        f"Expected exactly 120 sensor features, "
        f"found {len(sensor_columns)}."
    )

print()
print("✓ Exactly 120 sensor features selected.")


# ============================================================
# FEATURE MATRIX
# ============================================================

X = df[sensor_columns].copy()
y = df[TARGET].copy()

if X.isna().any().any():
    raise ValueError("Sensor features contain NaN values.")

if not np.isfinite(X.to_numpy(dtype=float)).all():
    raise ValueError("Sensor features contain infinite values.")

print()
print("Sensor feature matrix:")
print(f"Rows    : {X.shape[0]}")
print(f"Features: {X.shape[1]}")


# ============================================================
# GROUP-AWARE SPLIT
# ============================================================

print()
print("[5/9] Creating group-aware train/test split...")

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

train_idx, test_idx = next(
    splitter.split(
        X,
        y,
        groups=groups,
    )
)

X_train = X.iloc[train_idx].copy()
X_test = X.iloc[test_idx].copy()

y_train = y.iloc[train_idx].copy()
y_test = y.iloc[test_idx].copy()

groups_train = groups.iloc[train_idx]
groups_test = groups.iloc[test_idx]

print(f"Training samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")

print()
print(f"Training groups: {groups_train.nunique()}")
print(f"Testing groups : {groups_test.nunique()}")

print()
print("TEST GROUPS:")
print(sorted(groups_test.unique()))

overlap = set(groups_train.unique()) & set(groups_test.unique())

if overlap:
    raise RuntimeError(
        f"GROUP LEAKAGE DETECTED: {sorted(overlap)}"
    )

print()
print("✓ Group leakage check passed.")


# ============================================================
# MODELS
# ============================================================

print()
print("[6/9] Preparing models...")

models = {}

models["Dummy Mean"] = DummyRegressor(
    strategy="mean"
)

models["Random Forest"] = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
)

models["HistGradientBoosting"] = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_iter=300,
    max_leaf_nodes=31,
    l2_regularization=0.1,
    random_state=42,
)

try:

    from xgboost import XGBRegressor

    models["XGBoost"] = XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    print("✓ XGBoost available.")

except ImportError:

    print("! XGBoost not installed.")


# ============================================================
# TRAIN
# ============================================================

print()
print("[7/9] Training models...")

results = []
trained_models = {}

for name, model in models.items():

    print()
    print("-" * 70)
    print(f"Training: {name}")
    print("-" * 70)

    model.fit(
        X_train,
        y_train,
    )

    result = evaluate_model(
        name,
        model,
        X_test,
        y_test,
    )

    results.append(result)
    trained_models[name] = model


# ============================================================
# COMPARISON
# ============================================================

print()
print("[8/9] Comparing models...")

results_df = pd.DataFrame(results)

results_df = results_df[
    ["model", "mae", "rmse", "r2"]
].sort_values(
    by="mae",
    ascending=True,
)

print()
print("=" * 70)
print("MODEL COMPARISON - V2")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ============================================================
# BEST MODEL
# ============================================================

best_name = results_df.iloc[0]["model"]

best_model = trained_models[best_name]

best_row = results_df.iloc[0]

print()
print("=" * 70)
print("SELECTED MODEL - V2")
print("=" * 70)

print(f"Model: {best_name}")
print(f"MAE  : {best_row['mae']:.6f}")
print(f"RMSE : {best_row['rmse']:.6f}")
print(f"R²   : {best_row['r2']:.6f}")


# ============================================================
# SAVE
# ============================================================

print()
print("[9/9] Saving V2 artifacts...")

model_path = MODEL_DIR / "piecuch_rul_model_v2.joblib"

feature_path = MODEL_DIR / "feature_columns_v2.json"

metrics_path = MODEL_DIR / "metrics_v2.json"

comparison_path = MODEL_DIR / "model_comparison_v2.csv"

prediction_path = MODEL_DIR / "test_predictions_v2.csv"


joblib.dump(
    best_model,
    model_path,
)


with open(
    feature_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        sensor_columns,
        f,
        indent=2,
    )


metrics_to_save = {
    "dataset": str(DATA_PATH),
    "task": "CNC/VMC RUL regression",
    "target": TARGET,
    "feature_type": "sensor_only",
    "accelerometer_features": 48,
    "current_features": 72,
    "total_sensor_features": 120,
    "group_column": "Experiment",
    "n_samples": int(len(df)),
    "train_samples": int(len(X_train)),
    "test_samples": int(len(X_test)),
    "train_groups": sorted(
        [str(x) for x in groups_train.unique()]
    ),
    "test_groups": sorted(
        [str(x) for x in groups_test.unique()]
    ),
    "selected_model": best_name,
    "mae": float(best_row["mae"]),
    "rmse": float(best_row["rmse"]),
    "r2": float(best_row["r2"]),
}


with open(
    metrics_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metrics_to_save,
        f,
        indent=2,
    )


results_df.to_csv(
    comparison_path,
    index=False,
)


predictions = best_model.predict(X_test)

predictions = np.clip(
    predictions,
    0.0,
    1.0,
)


prediction_df = pd.DataFrame({
    "Experiment": groups_test.to_numpy(),
    "Actual": y_test.to_numpy(),
    "Predicted": predictions,
    "AbsoluteError": np.abs(
        y_test.to_numpy() - predictions
    ),
})


prediction_df.to_csv(
    prediction_path,
    index=False,
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("V2 TRAINING COMPLETE")
print("=" * 70)

print()
print("Saved artifacts:")

print(model_path)
print(feature_path)
print(metrics_path)
print(comparison_path)
print(prediction_path)

print()
print("✓ 48 accelerometer features")
print("✓ 72 current features")
print("✓ 120 sensor features total")
print("✓ No process metadata used")
print("✓ No target/progression columns used")
print("✓ Group-aware evaluation")
print("✓ Model saved")
print()
print("NEXT: compare V2 against V1 before live integration.")