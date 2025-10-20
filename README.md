# Melting Point Prediction

A machine learning project for predicting melting points (Tm) of chemical compounds using molecular descriptors and fingerprints with XGBoost regression.

## 📋 Project Overview

This project uses advanced molecular features and XGBoost regression to predict the melting points of chemical compounds from their SMILES representations. The model combines multiple feature types including molecular descriptors, Morgan fingerprints, and MACCS keys for improved prediction accuracy.

## 🛠️ Features

- **Molecular Descriptors**: Comprehensive RDKit molecular descriptors
- **Feature Engineering**: Custom interaction features between molecular properties
- **Molecular Fingerprints**: 
  - Morgan fingerprints (1024-bit, radius 2)
  - MACCS keys (167-bit)
- **Model Optimization**: Hyperparameter tuning using Optuna
- **Cross-Validation**: 5-fold cross-validation for robust evaluation

## 📦 Dependencies

This project uses `uv` for dependency management. All dependencies are specified in `pyproject.toml`:

- `pandas>=2.3.3` - Data manipulation and analysis
- `numpy>=2.3.4` - Numerical computing
- `optuna>=4.5.0` - Hyperparameter optimization
- `rdkit>=2025.9.1` - Molecular descriptor and fingerprint computation
- `scikit-learn>=1.7.2` - Machine learning utilities and preprocessing
- `xgboost>=3.1.0` - Gradient boosting model

## 🚀 Installation

### Using uv (Recommended)

```bash
# Install dependencies
uv sync

# Run the script
uv run python XGBOOST.py
```

### Using pip

```bash
pip install pandas numpy optuna rdkit scikit-learn xgboost
python XGBOOST.py
```

## 📊 Dataset

The project expects the following CSV files:

- `train.csv` - Training dataset with SMILES and Tm (melting point) columns
- `test.csv` - Test dataset for prediction

## 🔬 Methodology

### Feature Engineering

1. **Basic Molecular Descriptors**:
   - Molecular Weight (MolWt)
   - Number of Hydrogen Donors (NumHDonors)
   - Number of Hydrogen Acceptors (NumHAcceptors)
   - Topological Polar Surface Area (TPSA)

2. **Interaction Features**:
   - MolWt × NumHDonors
   - MolWt × NumHAcceptors
   - TPSA / MolWt Ratio
   - NumDonors / NumAcceptors Ratio
   - And more...

3. **Molecular Fingerprints**:
   - Morgan Fingerprints (1024 features)
   - MACCS Keys (167 features)

### Model Training

- **Algorithm**: XGBoost Regressor
- **Optimization**: Optuna with 30 trials
- **Validation**: 5-fold cross-validation
- **Preprocessing**: Z-score standardization (StandardScaler)
- **Evaluation Metric**: Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)

### Hyperparameters Tuned

- `n_estimators`: 100-500
- `max_depth`: 3-12
- `learning_rate`: 0.005-0.3 (log scale)
- `subsample`: 0.6-1.0
- `colsample_bytree`: 0.6-1.0
- `min_child_weight`: 1-10
- `gamma`: 0-5

## 📈 Usage

```bash
# Run the complete pipeline
uv run python XGBOOST.py
```

The script will:
1. Load and process training data
2. Extract molecular features and fingerprints
3. Perform hyperparameter optimization
4. Train the final model
5. Generate predictions for test data
6. Save results to `submission.csv`

## 📤 Output

- `submission.csv` - Contains predictions with columns: `id`, `Tm`

## 🔧 Project Structure

```
melting-point/
├── XGBOOST.py          # Main training and prediction script
├── train.csv           # Training dataset
├── test.csv            # Test dataset
├── submission.csv      # Prediction results
├── pyproject.toml      # Project dependencies
├── uv.lock             # Locked dependency versions
└── README.md           # This file
```