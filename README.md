# DFT $\to$ Experimental Band Gap Correction Model
### Machine-learning pipeline for predicting realistic semiconductor band gaps

This repository provides a reproducible machine-learning framework that maps **DFT-calculated (PBE) band gaps** to **experimental band gaps** using an XGBoost-based **correction model**.

It includes:
- Data preprocessing and feature engineering
- **Magpie** composition descriptors (via `matminer`)
- A correction-based training strategy
- Optional uncertainty estimation from tree variance
- A clean Python API for standalone predictions

---

## 🔍 Overview & Motivation

Density Functional Theory (DFT), particularly with the PBE functional, is well-known to **systematically underestimate semiconductor band gaps**. Yet this error is **not uniform, not linear, and not governed by a simple empirical trend**. Instead, the discrepancy depends on subtle, composition-specific electronic effects.

**Why Analytical Correction Fails:**
- Different chemical families exhibit *very different* magnitudes of error.
- No single scaling factor or linear regression captures all materials.
- Orbital character, bonding environment, and electronegativity contrast introduce **highly nonlinear corrections**.

### The Correction Strategy

Instead of predicting the experimental gap directly, this model learns a **correction term** $\Delta E$:

$$
\Delta E = E_{\text{experimental}} - E_{\text{DFT(PBE)}}
$$

Then a corrected prediction is computed as:

$$
E_{\text{predicted}} = E_{\text{DFT}} + \Delta E_{\text{model}}
$$

This makes the model more robust and easier to generalize across diverse compositions.

---

## 🗂 Repository Structure

```text
bandgapfix/
├── data/
│   └── final.csv                       # Training dataset
├── examples/
│   └── bandgapfix_benchmark_results.csv # Sample predictions / benchmarking
├── metadata/
│   └── bandgap_model_metadata_v4.yaml   # Model config + training notes
├── models/
│   └── bandgap_correction_model_xgb.joblib  # Trained pipeline (Imputer+Scaler+XGB)
├── src/
│   ├── train_bandgap_optimized.py       # Training script
│   └── bandgap_model_api_v2.py          # Prediction API
├── requirements.txt
├── README.md
└── LICENSE
```

-----

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/dft-gap-correction.git
cd dft-gap-correction
pip install -r requirements.txt
```

> **Recommendation:** It is highly suggested to use a clean `conda` or `venv` environment to avoid version conflicts with `matminer` or `scikit-learn`.

-----

## 🚀 Quickstart: Predict a Corrected Band Gap

You can use the Python API to correct a PBE gap. You only need:

1.  A chemical formula (string)
2.  A DFT/PBE gap value (float, eV)

### Python Example

```python
from src.bandgap_model_api_v2 import predict_gap

# Example: Silicon with a PBE gap of ~0.6 eV
pred_gap, unc = predict_gap("Si", 0.6)

print(f"Predicted experimental gap: {pred_gap:.2f} eV")
print(f"Uncertainty estimate:       {unc:.2f} eV")
```

**Output:**

```text
Predicted experimental gap: 1.10 eV
Uncertainty estimate:       0.15 eV
```

### CLI Test

You can also run the API module directly from the command line to test the installation (defaults to GaN):

```bash
python src/bandgap_model_api_v2.py
```

-----

## 🏋️ Retrain the Model (Optional)

If you want to retrain the model from scratch using the provided data:

```bash
python src/train_bandgap_optimized.py
```

**This script will:**

1.  Read `data/final.csv`.
2.  Build the correction target.
3.  Train the pipeline (IterativeImputer → StandardScaler → XGBoost).
4.  Save the model to the `models/` directory.
5.  Update metadata in `metadata/`.

-----

## 📌 Dataset

The training dataset (`data/final.csv`) includes:

  * PBE band gaps (`gap pbe`)
  * Experimental band gaps (`expt_gap`)
  * Magpie composition descriptors
  * Composition IDs used for GroupKFold splitting

-----

## 🧠 Model Details

  * **Algorithm:** XGBoost Regressor
  * **Target:** $\Delta E$ correction (Expt − PBE)
  * **Pipeline:** IterativeImputer → StandardScaler → XGB
  * **Validation:** GroupKFold (composition-held-out)
  * **Output:** $E_{\text{pred}} = E_{\text{dft}} + \Delta E_{\text{pred}}$

Full hyperparameters and training metadata are stored in `metadata/bandgap_model_metadata_v4.yaml`.

-----

## ⚠️ Large Model File Note

The file `models/bandgap_correction_model_xgb.joblib` is approximately **76 MB**.
GitHub allows this, but recommends LFS for files >50 MB.

If you want to use Git LFS:

```bash
git lfs install
git lfs track "*.joblib"
git add .gitattributes models/bandgap_correction_model_xgb.joblib
git commit -m "Track model with Git LFS"
git push
```

-----

## 📜 License

This project is released under the **MIT License**.
See `LICENSE` for details.

-----

## 👤 Author

Maintained by **Grigoris [Your Last Name]**  
*Computational Materials Science • DFT • ML for materials*

If you use or adapt this, feel free to cite this repository!
