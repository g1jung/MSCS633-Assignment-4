"""
Fraud Detection with PyOD AutoEncoder (Best Practices Version)

- Uses an unsupervised AutoEncoder from PyOD to learn normal transaction patterns.
- Trains on the non-fraud (majority) class only to minimize leakage and reflect anomaly detection.
- Evaluates with ROC AUC and Average Precision; saves PR/ROC curves.
- Command-line interface for reproducibility; configurable hyperparameters.
- Logging via print statements; could be extended to Python's `logging`.
- Compatible with IDEs like VS Code and managed notebook services like SageMaker Studio Lab.

Author: Your Name
GitHub: https://github.com/your-username/fraud-pyod-ae  # replace with your repository
"""


#!/usr/bin/env python3
"""
Fraud Detection with PyOD AutoEncoder

This script trains an unsupervised AutoEncoder (PyOD) on a fraud dataset.
- Trains on the majority (assumed "non-fraud") class only
- Scores all samples and classifies top anomalies as fraud
- Evaluates with ROC AUC, PR AUC, confusion matrix, and classification report
- Saves figures and a CSV of predictions

USAGE:
    python fraud_pyod_autoencoder.py --data path/to/data.csv --label_col Class --positive_label 1 --output_dir ./outputs

If your dataset columns differ, set --label_col and --positive_label accordingly.
"""

import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# PyOD
from pyod.models.auto_encoder import AutoEncoder

def load_dataset(path, label_col='Class', positive_label=1):
    df = pd.read_csv(path)
    # Try to coerce label col if missing
    if label_col not in df.columns:
        # Heuristic: try to find a binary column
        candidates = [c for c in df.columns if df[c].dropna().nunique() == 2]
        if len(candidates) == 0:
            raise ValueError(f"Label column '{label_col}' not found and no binary column could be inferred. Available columns: {list(df.columns)}")
        label_col = candidates[0]
        print(f"[INFO] label_col not found; inferred '{label_col}' as label.")
    y = (df[label_col] == positive_label).astype(int).values
    X = df.drop(columns=[label_col]).select_dtypes(include=[np.number]).values
    if X.size == 0:
        raise ValueError("No numeric feature columns found after dropping label. Please provide a numeric feature dataset.")
    return X, y, label_col

def train_autoencoder(X_train, hidden_neurons=(64, 32, 32, 64), epochs=30, batch_size=128, contamination=0.01, verbose=0):
    model = AutoEncoder(hidden_neurons=hidden_neurons,
                        epochs=epochs,
                        batch_size=batch_size,
                        contamination=contamination,
                        verbose=verbose)
    model.fit(X_train)
    return model

def plot_pr(y_true, scores, out_path):
    precision, recall, _ = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)
    plt.figure(figsize=(6,4))
    plt.step(recall, precision, where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (AP={ap:.3f})')
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close()

def plot_roc(y_true, scores, out_path):
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1],'--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (AUC={auc:.3f})')
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV dataset (Kaggle fraud-detection).")
    parser.add_argument("--label_col", default="Class", help="Name of label column (default: Class).")
    parser.add_argument("--positive_label", type=int, default=1, help="Value that denotes fraud/positive class (default: 1).")
    parser.add_argument("--test_size", type=float, default=0.3, help="Test size for holdout (default: 0.3).")
    parser.add_argument("--contamination", type=float, default=0.01, help="Expected fraction of anomalies (default: 0.01).")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs for AE (default: 30).")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for AE (default: 128).")
    parser.add_argument("--output_dir", default="./outputs", help="Directory to save outputs.")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    X, y, label_col_used = load_dataset(args.data, args.label_col, args.positive_label)

    # Split (note: unsupervised training uses normal class only)
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=args.test_size, stratify=y, random_state=42)
    normal_mask = y_train_full == 0
    X_train = X_train_full[normal_mask]

    # Scale (fit on train normals only; apply to all)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Train AE
    model = train_autoencoder(X_train_s, epochs=args.epochs, batch_size=args.batch_size, contamination=args.contamination)

    # Score
    test_scores = model.decision_function(X_test_s)  # higher means more abnormal in PyOD
    y_pred = (test_scores > model.threshold_).astype(int)

    # Metrics
    roc = roc_auc_score(y_test, test_scores)
    ap = average_precision_score(y_test, test_scores)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)

    # Save metrics
    metrics_txt = outdir / "metrics.txt"
    with open(metrics_txt, "w") as f:
        f.write(f"Label column: {label_col_used}\n")
        f.write(f"ROC AUC: {roc:.4f}\n")
        f.write(f"Average Precision (PR AUC): {ap:.4f}\n")
        f.write("Confusion Matrix (y_true rows x y_pred cols [0,1]):\n")
        f.write(str(cm) + "\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")
    print(f"[OK] Metrics saved to {metrics_txt}")

    # Save curves
    pr_path = outdir / "precision_recall.png"
    roc_path = outdir / "roc_curve.png"
    plot_pr(y_test, test_scores, pr_path)
    plot_roc(y_test, test_scores, roc_path)
    print(f"[OK] Curves saved: {pr_path}, {roc_path}")

    # Save predictions
    preds_df = pd.DataFrame({
        "score": test_scores,
        "y_true": y_test,
        "y_pred": y_pred
    })
    preds_csv = outdir / "predictions.csv"
    preds_df.to_csv(preds_csv, index=False)
    print(f"[OK] Predictions saved to {preds_csv}")

if __name__ == "__main__":
    main()
