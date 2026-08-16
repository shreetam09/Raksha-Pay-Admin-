"""
SOAIDEATHON-S40 — train_isolation_forest.py  (SS-4)
Trains the Isolation Forest teacher on normal-only features, then scores
the FULL dataset (all labels) to produce anomaly scores for downstream
distillation / evaluation.

Run after feature_engineering.py.
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
ARTIFACTS_DIR = os.path.join(PROCESSED_DIR, "artifacts")

# actual fraud rate in the unified set is ~0.41% (29,368 / 7,237,967)
CONTAMINATION = 0.005


def main():
    print("Loading features...")
    X_train = pd.read_parquet(os.path.join(PROCESSED_DIR, "X_train_normal.parquet"))
    X_full = pd.read_parquet(os.path.join(PROCESSED_DIR, "X_full.parquet"))
    meta = pd.read_parquet(os.path.join(PROCESSED_DIR, "meta_full.parquet"))
    print(f"  train (normal-only): {X_train.shape}")
    print(f"  full (all labels):   {X_full.shape}")

    print(f"\nTraining IsolationForest (contamination={CONTAMINATION})...")
    t0 = time.time()
    model = IsolationForest(
        n_estimators=200,
        max_samples=256,          # subsampling per tree, standard IF choice — keeps this fast at 7M+ rows
        contamination=CONTAMINATION,
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )
    model.fit(X_train)
    print(f"  trained in {time.time() - t0:.1f}s")

    print("\nScoring full dataset...")
    # decision_function: higher = more normal. Flip sign so higher = more anomalous,
    # matching the convention the Autoencoder's reconstruction error will also use.
    raw_scores = model.decision_function(X_full)
    anomaly_score = -raw_scores
    pred_label = (model.predict(X_full) == -1).astype(int)  # -1 = anomaly, 1 = normal

    meta = meta.copy()
    meta["if_anomaly_score"] = anomaly_score
    meta["if_pred_label"] = pred_label

    auc = roc_auc_score(meta["label"], meta["if_anomaly_score"])
    ap = average_precision_score(meta["label"], meta["if_anomaly_score"])
    print(f"\nEval vs true label (sanity check only -- teacher trained unsupervised):")
    print(f"  ROC-AUC: {auc:.4f}")
    print(f"  PR-AUC (average precision): {ap:.4f}")

    print("\nPer-source AUC breakdown:")
    for source in meta["source"].unique():
        sub = meta[meta["source"] == source]
        if sub["label"].nunique() < 2:
            print(f"  {source}: skipped (only one class present)")
            continue
        sub_auc = roc_auc_score(sub["label"], sub["if_anomaly_score"])
        print(f"  {source}: ROC-AUC {sub_auc:.4f}")

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(ARTIFACTS_DIR, "isolation_forest.joblib"))
    meta.to_parquet(os.path.join(PROCESSED_DIR, "if_scores.parquet"), index=False)

    print(f"\nSaved model -> {os.path.join(ARTIFACTS_DIR, 'isolation_forest.joblib')}")
    print(f"Saved scores -> {os.path.join(PROCESSED_DIR, 'if_scores.parquet')}")


if __name__ == "__main__":
    main()
