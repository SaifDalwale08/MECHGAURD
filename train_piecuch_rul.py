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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

# Metadata / leakage columns that must NOT be used as predictors.
EXCLUDE_COLUMNS = {
    "FileName",
    "NumberOfCycle",
    "SampleIndex",
    "TollIndex",
    "CycleToFailure",
    "CycleToFailureNormalized",
}


# ============================================================
# HELPERS
# ============================================================

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate_model(name, model, X_test, y_test):
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    score_rmse = rmse(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print()
    print("=" * 70)
    print(f"{name}")
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
# LOAD DATA
# ============================================================

print("=" * 70)
print("MECHGUARD - PIECUCH CNC RUL TRAINING")
print("=" * 70)

print()
print(f"[1/8] Loading dataset:")
print(DATA_PATH)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_PATH}\n"
        "Check the dataset path."
    )

df = pd.read_csv(
    DATA_PATH,
    sep=";",
    skiprows=1,
    low_memory=False,
)

print(f"Loaded shape: {df.shape}")


# ============================================================
# TARGET CONVERSION
# ============================================================

print()
print("[2/8] Preparing target...")

# Dataset uses comma decimal notation in this column.
df[TARGET] = (
    df[TARGET]
    .astype(str)
    .str.replace(",", ".", regex=False)
    .astype(float)
)

# Safety checks
if df[TARGET].isna().any():
    raise ValueError("Target contains NaN values.")

if not df[TARGET].between(0.0, 1.0).all():
    raise ValueError("Target contains values outside [0, 1].")

print(f"Target: {TARGET}")
print(f"Target min : {df[TARGET].min():.6f}")
print(f"Target max : {df[TARGET].max():.6f}")
print(f"Target mean: {df[TARGET].mean():.6f}")


# ============================================================
# CREATE EXPERIMENT GROUP
# ============================================================

print()
print("[3/8] Creating trajectory groups...")

if "FileName" not in df.columns:
    raise ValueError("FileName column is required.")

# Example:
# P002_F01_C1
# P003_F02_C1
#
# We use the Pxxx prefix as the group.
df["Experiment"] = (
    df["FileName"]
    .astype(str)
    .str.split("_")
    .str[0]
)

groups = df["Experiment"]

print(f"Unique experiment groups: {groups.nunique()}")

print()
print("Experiment distribution:")
print(groups.value_counts().sort_index().to_string())


# ============================================================
# SELECT SENSOR FEATURES
# ============================================================

print()
print("[4/8] Selecting sensor features...")

numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

feature_columns = [
    col
    for col in numeric_columns
    if col not in EXCLUDE_COLUMNS
]

# Never allow Experiment into X.
feature_columns = [
    col for col in feature_columns
    if col != "Experiment"
]

if len(feature_columns) == 0:
    raise ValueError("No usable sensor features found.")

X = df[feature_columns].copy()
y = df[TARGET].copy()

# Safety check: no NaN / inf
if X.isna().any().any():
    raise ValueError("Feature matrix contains NaN values.")

if not np.isfinite(X.to_numpy(dtype=float)).all():
    raise ValueError("Feature matrix contains infinite values.")

print(f"Number of selected features: {len(feature_columns)}")

print()
print("Selected features:")
for i, feature in enumerate(feature_columns, start=1):
    print(f"{i:03d}. {feature}")


# ============================================================
# GROUP-AWARE TRAIN / TEST SPLIT
# ============================================================

print()
print("[5/8] Creating group-aware train/test split...")

# IMPORTANT:
# Random row splitting would allow samples from the same
# degradation trajectory to appear in both train and test.
#
# Instead, entire experiments are kept together.

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
print("TRAIN groups:")
print(sorted(groups_train.unique()))

print()
print("TEST groups:")
print(sorted(groups_test.unique()))

# Verify no group leakage.
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
print("[6/8] Training models...")

models = {}


# ------------------------------------------------------------
# BASELINE
# ------------------------------------------------------------

models["Dummy Mean"] = DummyRegressor(
    strategy="mean"
)


# ------------------------------------------------------------
# RANDOM FOREST
# ------------------------------------------------------------

models["Random Forest"] = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
)


# ------------------------------------------------------------
# HISTOGRAM GRADIENT BOOSTING
# ------------------------------------------------------------

models["HistGradientBoosting"] = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_iter=300,
    max_leaf_nodes=31,
    l2_regularization=0.1,
    random_state=42,
)


# ------------------------------------------------------------
# XGBOOST
# ------------------------------------------------------------

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
    print("! XGBoost not installed. Skipping XGBoost.")


# ============================================================
# TRAIN + EVALUATE
# ============================================================

results = []
trained_models = {}

for name, model in models.items():

    print()
    print("-" * 70)
    print(f"Training: {name}")
    print("-" * 70)

    # Tree models do not need scaling.
    # Dummy also does not need scaling.
    model.fit(X_train, y_train)

    result = evaluate_model(
        name,
        model,
        X_test,
        y_test,
    )

    results.append(result)
    trained_models[name] = model


# ============================================================
# RESULTS TABLE
# ============================================================

print()
print("[7/8] Comparing models...")

results_df = pd.DataFrame(results)

results_df = results_df[
    ["model", "mae", "rmse", "r2"]
].sort_values(
    by="mae",
    ascending=True,
)

print()
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ============================================================
# SELECT MODEL
# ============================================================

# Lowest MAE is used as the primary selection criterion.
best_name = results_df.iloc[0]["model"]

best_model = trained_models[best_name]

best_metrics = results_df.iloc[0].to_dict()

print()
print("=" * 70)
print("SELECTED MODEL")
print("=" * 70)

print(f"Model: {best_name}")
print(f"MAE  : {best_metrics['mae']:.6f}")
print(f"RMSE : {best_metrics['rmse']:.6f}")
print(f"R²   : {best_metrics['r2']:.6f}")


# ============================================================
# SAVE MODEL
# ============================================================

print()
print("[8/8] Saving model artifacts...")

model_path = MODEL_DIR / "piecuch_rul_model.joblib"

joblib.dump(
    best_model,
    model_path,
)

# Save feature list
feature_path = MODEL_DIR / "feature_columns.json"

with open(feature_path, "w", encoding="utf-8") as f:
    json.dump(
        feature_columns,
        f,
        indent=2,
    )


# Save metrics
metrics_path = MODEL_DIR / "metrics.json"

metrics_to_save = {
    "dataset": str(DATA_PATH),
    "target": TARGET,
    "task": "RUL regression",
    "group_column": "Experiment",
    "n_samples": int(len(df)),
    "n_features": int(len(feature_columns)),
    "train_samples": int(len(X_train)),
    "test_samples": int(len(X_test)),
    "train_groups": sorted(
        [str(x) for x in groups_train.unique()]
    ),
    "test_groups": sorted(
        [str(x) for x in groups_test.unique()]
    ),
    "selected_model": best_name,
    "mae": float(best_metrics["mae"]),
    "rmse": float(best_metrics["rmse"]),
    "r2": float(best_metrics["r2"]),
}

with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(
        metrics_to_save,
        f,
        indent=2,
    )


# Save complete comparison
comparison_path = MODEL_DIR / "model_comparison.csv"

results_df.to_csv(
    comparison_path,
    index=False,
)


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

predictions = best_model.predict(X_test)

prediction_df = pd.DataFrame({
    "Actual": y_test.to_numpy(),
    "Predicted": predictions,
    "AbsoluteError": np.abs(
        y_test.to_numpy() - predictions
    ),
})

prediction_path = MODEL_DIR / "test_predictions.csv"

prediction_df.to_csv(
    prediction_path,
    index=False,
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print()
print("Saved files:")

print(model_path)
print(feature_path)
print(metrics_path)
print(comparison_path)
print(prediction_path)

print()
print("✓ Model trained.")
print("✓ Group-aware evaluation completed.")
print("✓ Model artifact saved.")
print("✓ Feature list saved.")
print("✓ Metrics saved.")
print()
print("Next step: integrate the model into MECHGUARD.")