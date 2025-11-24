#!/usr/bin/env python3
# bandgap_model_api_v2.py
#
# Predicts the *absolute experimental* band gap (eV) for an arbitrary
# composition using the Correction Model (predicts Delta E).
# ----------------------------------------------------------------------

import pathlib
import warnings
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import load
from pymatgen.core.composition import Composition
from matminer.featurizers.composition import ElementProperty

# ────────────────────────────────────────────────────────────────────
# Silence benign warnings from matminer / sklearn
warnings.filterwarnings("ignore", message="MagpieData.*")
warnings.filterwarnings("ignore", message="X does not have valid feature names.*")

# ────────────────────────────────────────────────────────────────────
# 0 ▸ Load trained pipeline
# ────────────────────────────────────────────────────────────────────
# Loading the new correction model filename
MODEL_FILENAME = pathlib.Path(__file__).with_name("bandgap_correction_model_xgb.joblib")

try:
    _pipeline = load(MODEL_FILENAME)
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_FILENAME}. Please run 'train_bandgap_optimized.py' first.")
    exit(1)

# Extract feature names for alignment
try:
    FEATURE_NAMES = _pipeline.named_steps["xgb"].feature_names_in_
except AttributeError:
    FEATURE_NAMES = getattr(_pipeline, 'feature_names_in_', None)

if FEATURE_NAMES is None:
     print("Warning: Could not extract feature names from the loaded model.")


# ────────────────────────────────────────────────────────────────────
# 1 ▸ Featurizer
# ────────────────────────────────────────────────────────────────────
class Featurizer:
    """Generate Magpie composition descriptors."""
    def __init__(self):
        self._f = ElementProperty.from_preset("magpie")

    def featurize_formula(self, formula: str) -> pd.Series:
        """Featurize a single chemical formula."""
        try:
            comp = Composition(formula)
            vec  = self._f.featurize(comp)
            lab  = self._f.feature_labels()
            return pd.Series(vec, index=lab)
        except Exception as e:
            print(f"Error featurizing formula '{formula}': {e}")
            return pd.Series(dtype=float)


# ────────────────────────────────────────────────────────────────────
# 2 ▸ Prediction function
# ────────────────────────────────────────────────────────────────────
def predict_gap(formula: str, dft_gap: float) -> Tuple[float, Optional[float]]:
    """
    Predict the *experimental* band gap (eV) using the correction model.

    Parameters
    ----------
    formula : str
        Chemical formula (e.g. "GaN").
    dft_gap : float
        Scalar PBE band gap (eV) to be used as an input feature.

    Returns
    -------
    expt_gap_pred : float
        Predicted experimental band gap (eV).
    uncertainty   : float | None
        1-σ estimate from tree-wise standard deviation, or None if unavailable.
    """
    # 1 ▸ Featurize composition
    feats = Featurizer().featurize_formula(formula)
    
    if feats.empty:
        return np.nan, None

    # 2 ▸ Retain PBE gap as a feature in the input vector
    feats["gap pbe"] = dft_gap

    # 3 ▸ Align to model feature order, fill missing with NaN (Imputer in pipeline handles this)
    feats = feats.reindex(FEATURE_NAMES, fill_value=np.nan)
    X = feats.values.reshape(1, -1)

    # 4 ▸ Predict the correction factor (Delta E)
    # The pipeline automatically runs Imputer -> Scaler -> XGBoost.
    delta_pred = float(_pipeline.predict(X)[0])

    # 5. NEW LOGIC: Calculate the final predicted experimental gap
    # E_expt = E_DFT + Delta_E_pred
    expt_pred = dft_gap + delta_pred

    # 6 ▸ Tree-wise uncertainty estimate (optional)
    std_err = None
    try:
        # Get the XGBoost model step from the pipeline
        xgb_model = _pipeline.named_steps["xgb"]
        booster   = xgb_model.get_booster()
        
        # Manually run data through the preprocessors (Imputer and Scaler) for DMatrix creation
        X_imputed = _pipeline.named_steps["imputer"].transform(X)
        X_transformed = _pipeline.named_steps["scaler"].transform(X_imputed)

        n_trees = getattr(booster, "best_iteration", xgb_model.n_estimators) + 1
        dm = xgb.DMatrix(X_transformed, feature_names=FEATURE_NAMES)

        tree_preds = [
            booster.predict(dm, iteration_range=(i, i + 1))
            for i in range(n_trees)
        ]
        # Calculate standard deviation across all tree predictions
        std_err = float(np.std(np.vstack(tree_preds), axis=0)[0])
    except Exception:
        pass

    return expt_pred, std_err


# ────────────────────────────────────────────────────────────────────
# 3 ▸ CLI test
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test case: GaN (PBE gap ~1.8 eV, expected Expt gap ~3.4 eV)
    test_formula = "GaN"
    test_pbe_gap = 1.8
    pred, unc = predict_gap(test_formula, test_pbe_gap)

    if unc is None:
        print(f"Input {test_formula} (DFT Gap: {test_pbe_gap:.2f} eV)")
        print(f"Predicted Experimental Gap = {pred:.2f} eV")
    else:
        print(f"Input {test_formula} (DFT Gap: {test_pbe_gap:.2f} eV)")
        print(f"Predicted Experimental Gap = {pred:.2f} ± {unc:.2f} eV")
