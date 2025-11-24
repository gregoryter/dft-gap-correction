#!/usr/bin/env python3
# train_bandgap_optimized.py
#
# Trains an XGBoost model to predict the *correction factor* (Delta E)
# to map the DFT gap (PBE) to the experimental gap (Expt).
# ----------------------------------------------------------------------

import pathlib
import pandas as pd
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from joblib import dump
import yaml
from sklearn.model_selection import GroupKFold

# FIX: Required for IterativeImputer (a robust way to handle missing feature data)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# ── 1. Load data and setup target correction ──────────────────────────
DATA_PATH = pathlib.Path("final.csv")
# Assuming 'final.csv' is available in the run directory
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_PATH}. Please ensure final.csv is available.")
    exit(1)


# We predict the difference (correction) between the experimental and DFT gap
# Delta E = Expt_Gap - DFT_Gap
df['delta_gap'] = df['expt_gap'] - df['gap pbe']

# Filter data where both Expt and PBE gaps are available
mask = df["gap pbe"].notna() & df["expt_gap"].notna()
df_clean = df.loc[mask].copy()

# Columns to exclude from features (these are the targets/IDs/duplicates)
drop_cols = ["expt_gap", "delta_gap", "likely_mpid", "mpid", "formula_y"]
# Retain 'formula_x' for grouping
group_col = "formula_x"

# X features: all compositional features + the 'gap pbe' feature
X_all = df_clean.drop(columns=drop_cols).set_index(group_col)
# Y target: the correction factor (Delta E)
y_all = df_clean["delta_gap"].values

# Groups: Used for GroupKFold cross-validation (GroupKFold is crucial 
# when multiple rows correspond to the same material/composition).
groups_all = df_clean[group_col].values

print(f"Loaded {X_all.shape[0]} samples with {X_all.shape[1]} features.")
print(f"Target is the correction factor (Delta E) in the range [{y_all.min():.2f}, {y_all.max():.2f}] eV.")
print(f"Number of unique compositions (groups) for CV: {len(set(groups_all))}")

# ── 2. Model config ───────────────────────────────────────────────────
# Increased regularization (reg_alpha/reg_lambda) is recommended for
# limited data with many features to combat overfitting.
xgb_params = {
    "n_estimators":          806,
    "learning_rate":         0.02024,
    "max_depth":             5,
    "min_child_weight":      2.18,
    "subsample":             0.968,
    "colsample_bytree":      0.74,
    "gamma":                 0.0022,
    "reg_alpha":             1.5,  # L1 regularization
    "reg_lambda":            7.0,  # L2 regularization
    "objective":             "reg:squarederror",
    "eval_metric":           "rmse",
    "n_jobs":                -1,
    "random_state":          42,
    "verbosity":             0,
}

final_model = xgb.XGBRegressor(**xgb_params)

# ── 3. Pipeline with Imputation ───────────────────────────────────────
# IterativeImputer (MICE) replaces the crude .fillna(0)
pipeline = Pipeline([
    ("imputer", IterativeImputer(random_state=42, max_iter=10)),
    ("scaler", StandardScaler()),
    ("xgb", final_model)
])

# ── 4. Train ──────────────────────────────────────────────────────────
print("Training on full dataset (target = Delta E correction) …")
# The final model is trained on all data for deployment.
pipeline.fit(X_all, y_all)
print("Training complete.")

# ── 5. Save pipeline ──────────────────────────────────────────────────
MODEL_FILENAME = "bandgap_correction_model_xgb.joblib"
dump(pipeline, MODEL_FILENAME)
print(f"Model saved as {MODEL_FILENAME}")

# ── 6. Metadata ───────────────────────────────────────────────────────
metadata = {
    "model_version":     "4.1.0 (Correction Model, Imputation Fixed)",
    "description":       "XGBoost regressor predicting the DELTA (Expt - DFT) band gap correction.",
    "target":            "delta_gap (Expt - PBE)",
    "cv_scheme":         "REQUIRES GroupKFold (composition-held-out) validation on 'groups_all'",
    "best_cv_rmse_eV":   "TBD (run GroupKFold validation)",
    "hyperparameters":   xgb_params,
    "training_samples":  int(X_all.shape[0]),
    "feature_names":     list(X_all.columns),
    "feature_count":     int(X_all.shape[1]),
}

with open("bandgap_model_metadata_v4.yaml", "w") as f:
    yaml.safe_dump(metadata, f, sort_keys=False)
print("Metadata saved as bandgap_model_metadata_v4.yaml")