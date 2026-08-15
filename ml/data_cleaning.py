"""
SOAIDEATHON-S40 — data_cleaning.py
Merges ULB, PaySim, and IEEE-CIS (now with train_transaction.csv available)
into one unified schema, with rule-based synthetic backfill for
coercion/device signals that don't exist in any raw source.

Expected project structure (run this from the `ml/` directory):

    ml/
      .venv/
      datasets/
        creditcard.csv
        paysim dataset.csv
        sample_submission.csv
        test_identity.csv
        test_transaction.csv
        train_identity.csv
        train_transaction.csv
      processed/         <- output written here, sibling of datasets/
      data_cleaning.py   <- this file
      README.md
"""

import os
import pandas as pd
import numpy as np

np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed")

PATHS = {
    "ulb": os.path.join(DATASETS_DIR, "creditcard.csv"),
    "paysim": os.path.join(DATASETS_DIR, "paysim dataset.csv"),
    "ieee_train_identity": os.path.join(DATASETS_DIR, "train_identity.csv"),
    "ieee_train_transaction": os.path.join(DATASETS_DIR, "train_transaction.csv"),
    "ieee_test_identity": os.path.join(DATASETS_DIR, "test_identity.csv"),
    "ieee_test_transaction": os.path.join(DATASETS_DIR, "test_transaction.csv"),
    "sample_submission": os.path.join(DATASETS_DIR, "sample_submission.csv"),
}

TARGET_COLUMNS = [
    "record_id", "source", "amount", "timestamp", "transaction_type",
    "balance_before", "balance_after", "device_type", "device_info",
    "device_new", "payee_new", "location_change", "call_active_during_txn",
    "account_age_days", "label", "synthetic_fields",
]


def synth_coercion_signals(df: pd.DataFrame, label_col: str, amount_col: str) -> pd.DataFrame:
    """
    Rule-based synthetic backfill for fields no raw dataset contains:
    location_change, call_active_during_txn.

    Probabilities are DELIBERATELY correlated with the fraud label and with
    high amounts, to mimic documented scam patterns (urgent, high-value,
    pressured transfers). This is a heuristic construction, not learned or
    observed behavior -- every backfilled column is named in
    `synthetic_fields` so it's never mistaken for real signal downstream.
    """
    n = len(df)
    is_fraud = df[label_col].fillna(0).astype(int).values
    amount_pctile = df[amount_col].rank(pct=True).fillna(0.5).values

    base_call_active = np.where(is_fraud == 1, 0.35, 0.03)
    base_location_change = np.where(is_fraud == 1, 0.25, 0.02)

    call_active_p = np.clip(base_call_active + (amount_pctile * is_fraud * 0.3), 0, 0.9)
    location_change_p = np.clip(base_location_change + (amount_pctile * is_fraud * 0.2), 0, 0.8)

    df["call_active_during_txn"] = np.random.random(n) < call_active_p
    df["location_change"] = np.random.random(n) < location_change_p
    return df


def clean_ulb(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = pd.DataFrame()
    out["record_id"] = "ulb_" + df.index.astype(str)
    out["source"] = "ulb"
    out["amount"] = df["Amount"]
    out["timestamp"] = df["Time"]
    out["transaction_type"] = "unknown"
    out["balance_before"] = np.nan
    out["balance_after"] = np.nan
    out["device_type"] = "unknown"
    out["device_info"] = np.nan
    out["label"] = df["Class"].astype(int)

    out["device_new"] = np.random.random(len(out)) < np.where(out["label"] == 1, 0.4, 0.05)
    out["payee_new"] = np.random.random(len(out)) < np.where(out["label"] == 1, 0.5, 0.08)
    out["account_age_days"] = np.nan
    out = synth_coercion_signals(out, "label", "amount")

    out["synthetic_fields"] = "device_type,device_new,payee_new,location_change,call_active_during_txn,account_age_days,transaction_type,balance_before,balance_after"
    return out[TARGET_COLUMNS]


def clean_paysim(path: str, nrows: int = None) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=nrows)
    out = pd.DataFrame()
    out["record_id"] = "paysim_" + df.index.astype(str)
    out["source"] = "paysim"
    out["amount"] = df["amount"]
    out["timestamp"] = df["step"]  # NOTE: units = hours, not comparable to ULB's seconds
    out["transaction_type"] = df["type"]
    out["balance_before"] = df["oldbalanceOrg"]
    out["balance_after"] = df["newbalanceOrig"]
    out["device_type"] = "unknown"
    out["device_info"] = np.nan
    out["label"] = df["isFraud"].astype(int)
    out["account_age_days"] = np.nan

    # real derived signal: is this the first time nameOrig -> nameDest pair appears
    first_seen = ~df.duplicated(subset=["nameOrig", "nameDest"], keep="first")
    out["payee_new"] = first_seen.values  # REAL, derived from actual transaction history

    out["device_new"] = np.random.random(len(out)) < np.where(out["label"] == 1, 0.4, 0.05)
    out = synth_coercion_signals(out, "label", "amount")

    out["synthetic_fields"] = "device_type,device_new,location_change,call_active_during_txn,account_age_days"
    return out[TARGET_COLUMNS]


def clean_ieee(identity_path: str, transaction_path: str) -> pd.DataFrame:
    """
    Real join now that train_transaction.csv is available.
    identity file is optional per row -- not every TransactionID has one,
    so this is a LEFT join from transaction -> identity.
    """
    txn = pd.read_csv(transaction_path)
    ident = pd.read_csv(identity_path)

    merged = txn.merge(ident, on="TransactionID", how="left")

    out = pd.DataFrame()
    out["record_id"] = "ieee_" + merged["TransactionID"].astype(str)
    out["source"] = "ieee"
    out["amount"] = merged["TransactionAmt"]
    out["timestamp"] = merged["TransactionDT"]  # NOTE: seconds from a reference point, not wall-clock -- own unit, not comparable across sources
    out["transaction_type"] = merged["ProductCD"]
    out["balance_before"] = np.nan   # not present in IEEE-CIS schema
    out["balance_after"] = np.nan    # not present in IEEE-CIS schema
    out["device_type"] = merged["DeviceType"].fillna("unknown").str.lower()
    out["device_info"] = merged["DeviceInfo"]
    out["label"] = merged["isFraud"].astype(int)
    out["account_age_days"] = np.nan  # not present in IEEE-CIS schema

    # id_35 is a real session/device-consistency flag per Kaggle's own field docs
    if "id_35" in merged.columns:
        out["device_new"] = merged["id_35"].map({"T": False, "F": True})
    else:
        out["device_new"] = np.nan

    out["payee_new"] = np.nan  # IEEE-CIS has no payee-history field to derive this from
    out = synth_coercion_signals(out, "label", "amount")

    synthetic_cols = ["location_change", "call_active_during_txn", "payee_new", "balance_before", "balance_after", "account_age_days"]
    out["synthetic_fields"] = ",".join(synthetic_cols)
    return out[TARGET_COLUMNS]


def run_source_leakage_check(merged: pd.DataFrame, sample_size: int = 100_000):
    """Sanity check: can a classifier trivially guess `source` from features alone?"""
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    sample = merged.sample(n=min(sample_size, len(merged)), random_state=42).copy()
    features = sample[["amount", "device_new", "payee_new", "location_change", "call_active_during_txn"]].fillna(-1)

    results = {}
    for source in sample["source"].unique():
        sample[f"is_{source}"] = (sample["source"] == source).astype(int)
        Xtr, Xte, ytr, yte = train_test_split(features, sample[f"is_{source}"], test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=30, max_depth=6, random_state=42, n_jobs=2)
        clf.fit(Xtr, ytr)
        acc = accuracy_score(yte, clf.predict(Xte))
        results[source] = acc

    return results


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Cleaning ULB...")
    ulb = clean_ulb(PATHS["ulb"])
    print(f"  {len(ulb):,} rows")

    print("Cleaning PaySim (full file, may take a moment)...")
    paysim = clean_paysim(PATHS["paysim"])
    print(f"  {len(paysim):,} rows")

    print("Cleaning + joining IEEE-CIS (identity + transaction)...")
    ieee = clean_ieee(PATHS["ieee_train_identity"], PATHS["ieee_train_transaction"])
    print(f"  {len(ieee):,} rows")

    merged = pd.concat([ulb, paysim, ieee], ignore_index=True)
    del ulb, paysim, ieee

    out_path = os.path.join(OUTPUT_DIR, "unified_transactions.csv")
    merged.to_csv(out_path, index=False)
    print(f"\nSaved {len(merged):,} unified rows -> {out_path}")
    print(f"\nRows per source:\n{merged['source'].value_counts()}")
    print(f"\nLabel balance:\n{merged['label'].value_counts()}")

    print("\nRunning source-leakage check on a 100k-row sample...")
    leakage = run_source_leakage_check(merged)
    for source, acc in leakage.items():
        print(f"  classifier guesses '{source}' vs. rest with {acc:.1%} accuracy")
    print("(high accuracy across structurally different sources is expected;")
    print(" the useful signal is whether this DROPS as real overlapping features")
    print(" like amount/device_type get used more heavily downstream)")