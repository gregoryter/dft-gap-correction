# DFT → Experimental Band Gap Correction Model  
### Machine-learning pipeline for predicting realistic semiconductor band gaps

This repository provides a reproducible machine-learning framework that maps  
**DFT-calculated (PBE) band gaps** to **experimental band gaps** using an XGBoost-based correction model.

It includes:

- Data preprocessing and feature engineering  
- Magpie composition descriptors  
- A correction-based training strategy  
- Uncertainty estimation from tree variance  
- A clean Python API for standalone predictions

---

## 🔍 Overview & Motivation

Density Functional Theory (DFT), particularly with the PBE functional, is well-known to **systematically underestimate semiconductor band gaps**.  
Yet this error is **not uniform, not linear, and not governed by a simple empirical trend**. Instead, the discrepancy depends on subtle, composition-specific electronic effects.

Some examples of why the underestimation is difficult to correct analytically:

- Different chemical families exhibit *very different* magnitudes of error  
- No single scaling factor or linear regression captures all materials  
- Orbital character, bonding environment, and electronegativity contrast introduce **highly nonlinear corrections**  
- Even materials with similar DFT band gaps can have **vastly different experimental values**

Because of this, data-driven models provide a natural path forward.  
This project trains a machine-learning model to predict a **correction term**:

\[
\Delta E = E_\text{experimental} - E_\text{DFT(PBE)}
\]

Once trained, the model predicts:

\[
E_\text{predicted} = E_\text{DFT} + \Delta E_\text{model}
\]

The outcome is:

- A **small, deployable ML model**  
- A **simple API** to correct band gaps for arbitrary compositions  
- A transparent workflow suitable for research or teaching

---

## 🗂 Repository Structure

