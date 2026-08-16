"""
SOAIDEATHON-S40 — feature_engineering.py
Builds the numeric feature matrix used by BOTH teacher models
(Isolation Forest, Autoencoder) from processed/unified_transactions.csv.

Design choices:
  - timestamp is dropped as a feature: ULB uses seconds-from-start, PaySim
    uses hour-steps, IEEE uses seconds-from-a-reference-point. None are
    comparable across sources, so including it raw would just teach the
    model "which source is this" via a back door.
  - balance_before/balance_after are null for ULB (no such concept) and
    IEEE (not in schema). Rather than imputing fake balances, we add a
    `has_balance_data` flag and fill the numeric columns with 0 -- the
    model gets an honest signal instead of fabricated magnitude.
  - transaction_type / device_type / source are LOW-cardinality categoricals
    (a handful of values each) -- one-hot encoded, no false ordinality.
  - device_info is HIGH-cardinality (thousands of distinct raw device
    strings from IEEE, e.g. "SM-G935F Build/NRD90M", on top of our
    synthetic buckets). One-hot encoding this blows up to 1000+ columns
    and tries to allocate tens of GB for a 7M-row frame. Frequency-encoded
    instead: each value maps to how common it is in the training set,
    collapsing it to a single numeric column -- standard practice for
    high-cardinality categoricals in fraud detection (device/IP/email-domain
    features in most IEEE-CIS competition solutions use this same trick).
  - Teachers train on label==0 rows only (unsupervised anomaly framing:
    learn what "normal" looks like, flag deviation). Full frame (all
    labels) is still saved separately for scoring/eval.

Run from `ml/` directory, after data_cleaning.py.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
UNIFIED_PATH = os.path.join(PROCESSED_DIR, "unified_transactions.csv")
ARTIFACTS_DIR = os.path.join(PROCESSED_DIR, "artifacts")

BOOL_COLS = ["device_new", "payee_new", "location_change", "call_active_during_txn"]
NUMERIC_COLS = ["amount", "balance_before", "balance_after", "account_age_days", "device_info_freq"]
LOW_CARD_CATS = ["transaction_type", "device_type", "source"]   # few distinct values -> one-hot
HIGH_CARD_COL = "device_info"                                    # thousands of distinct values -> frequency encoding


def load_unified() -> pd.DataFrame:
    df = pd.read_csv(UNIFIED_PATH, low_memory=False)
    return df


def build_features(df: pd.DataFrame, fit: bool, scaler: StandardScaler = None,
                    ohe_columns: list = None, freq_map: pd.Series = None):
    """
    fit=True: fit a new scaler, one-hot column layout, and device_info
    frequency map -- used for the training set. fit=False: reuse saved
    artifacts, used for eval/inference, so unseen categories don't silently
    create new columns or leak future frequency info into the eval score.
    """
    work = df.copy()

    # honest missingness flag instead of fabricated balances
    work["has_balance_data"] = (~work["balance_before"].isna()).astype(int)
    work["balance_before"] = work["balance_before"].fillna(0)
    work["balance_after"] = work["balance_after"].fillna(0)

    for c in BOOL_COLS:
        work[c] = work[c].astype(int)

    # high-cardinality device_info -> frequency encoding (single numeric column)
    if fit:
        freq_map = work[HIGH_CARD_COL].value_counts(normalize=True)
    work["device_info_freq"] = work[HIGH_CARD_COL].map(freq_map).fillna(0.0)

    # low-cardinality categoricals -> one-hot
    cat = pd.get_dummies(work[LOW_CARD_CATS], prefix=LOW_CARD_CATS, dtype=int)

    if fit:
        ohe_columns = list(cat.columns)
    else:
        # align to training-time columns: drop unseen, add missing as 0
        cat = cat.reindex(columns=ohe_columns, fill_value=0)

    numeric = work[NUMERIC_COLS].copy()
    if fit:
        scaler = StandardScaler()
        numeric_scaled = scaler.fit_transform(numeric)
    else:
        numeric_scaled = scaler.transform(numeric)
    numeric_scaled = pd.DataFrame(numeric_scaled, columns=NUMERIC_COLS, index=work.index)

    features = pd.concat(
        [numeric_scaled, work[BOOL_COLS], work[["has_balance_data"]], cat],
        axis=1,
    )

    return features, scaler, ohe_columns, freq_map


def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    print("Loading unified_transactions.csv...")
    df = load_unified()
    print(f"  {len(df):,} rows")
    print(f"  device_info cardinality: {df['device_info'].nunique():,} distinct values")

    normal = df[df["label"] == 0].reset_index(drop=True)
    print(f"  {len(normal):,} label==0 rows for teacher training")

    print("Building training features (fit scaler + one-hot layout + freq map)...")
    X_train, scaler, ohe_columns, freq_map = build_features(normal, fit=True)
    print(f"  Feature matrix: {X_train.shape}")

    print("Building full-frame features (all labels, for scoring/eval)...")
    X_full, _, _, _ = build_features(
        df, fit=False, scaler=scaler, ohe_columns=ohe_columns, freq_map=freq_map
    )

    # persist
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, "scaler.joblib"))
    joblib.dump(ohe_columns, os.path.join(ARTIFACTS_DIR, "ohe_columns.joblib"))
    joblib.dump(freq_map, os.path.join(ARTIFACTS_DIR, "device_info_freq_map.joblib"))
    joblib.dump(NUMERIC_COLS, os.path.join(ARTIFACTS_DIR, "numeric_cols.joblib"))
    joblib.dump(BOOL_COLS, os.path.join(ARTIFACTS_DIR, "bool_cols.joblib"))

    X_train.to_parquet(os.path.join(PROCESSED_DIR, "X_train_normal.parquet"), index=False)
    X_full.to_parquet(os.path.join(PROCESSED_DIR, "X_full.parquet"), index=False)
    df[["record_id", "source", "label"]].to_parquet(
        os.path.join(PROCESSED_DIR, "meta_full.parquet"), index=False
    )

    print(f"\nSaved:")
    print(f"  {os.path.join(PROCESSED_DIR, 'X_train_normal.parquet')}  {X_train.shape}")
    print(f"  {os.path.join(PROCESSED_DIR, 'X_full.parquet')}          {X_full.shape}")
    print(f"  {os.path.join(PROCESSED_DIR, 'meta_full.parquet')}       (record_id, source, label)")
    print(f"  scaler + column layout + freq map -> {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()