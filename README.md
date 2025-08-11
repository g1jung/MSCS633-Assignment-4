# Fraud Detection with PyOD AutoEncoder

This project uses **PyOD**'s AutoEncoder to perform anomaly detection on the Kaggle dataset:
- Kaggle dataset: https://www.kaggle.com/datasets/whenamancodes/fraud-detection
- PyOD AutoEncoder docs: https://pyod.readthedocs.io/en/latest/pyod.models.html#module-pyod.models.auto_encoder

## Quick Start

1. Create a virtual environment and install requirements:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. Download the Kaggle dataset locally (CSV) and note its path, e.g., `./data/fraud.csv`.

3. Run the script (adjust paths/labels as needed):
```bash
python fraud_pyod_autoencoder.py --data ./data/fraud.csv --label_col Class --positive_label 1 --output_dir ./outputs
```

4. Outputs:
- `outputs/metrics.txt` — ROC AUC, PR AUC, confusion matrix, classification report
- `outputs/precision_recall.png` and `outputs/roc_curve.png`
- `outputs/predictions.csv` — anomaly scores and predicted labels for test set

## Method Summary

We treat fraud detection as **unsupervised anomaly detection**. We:
- Split the dataset into train/test with stratification
- Train the AutoEncoder **only on the majority (non-fraud) class**
- Score all test samples; higher scores = more anomalous
- Use PyOD's learned threshold to assign anomaly labels
- Evaluate with ROC AUC and Average Precision (PR AUC)

## Reproducibility

- Seeded split (`random_state=42`)
- StandardScaler fit on normal training data only
- Model hyperparameters are configurable via CLI
