import os
import pandas as pd
import numpy as np
from unittest.mock import patch
from ml.data_cleaning import (
    synth_coercion_signals,
    clean_ulb,
    clean_paysim,
    clean_ieee,
    backfill_account_age_days,
    backfill_device_info,
    backfill_device_payee_new,
    TARGET_COLUMNS
)

def test_synth_coercion_signals():
    df = pd.DataFrame({
        "label": [0, 1, 0, 1],
        "amount": [10, 1000, 5, 2000]
    })
    res = synth_coercion_signals(df.copy(), "label", "amount")
    assert "call_active_during_txn" in res.columns
    assert "location_change" in res.columns
    assert res["call_active_during_txn"].dtype == bool
    assert res["location_change"].dtype == bool

def test_clean_ulb(tmp_path, dummy_ulb_df):
    p = tmp_path / "ulb.csv"
    dummy_ulb_df.to_csv(p, index=False)
    res = clean_ulb(str(p))
    assert list(res.columns) == TARGET_COLUMNS
    assert len(res) == 2
    assert (res["source"] == "ulb").all()
    assert (res["transaction_type"] == "unknown").all()

def test_clean_paysim(tmp_path, dummy_paysim_df):
    p = tmp_path / "paysim.csv"
    dummy_paysim_df.to_csv(p, index=False)
    res = clean_paysim(str(p))
    assert list(res.columns) == TARGET_COLUMNS
    assert len(res) == 2
    assert (res["source"] == "paysim").all()

def test_clean_ieee(tmp_path, dummy_ieee_txn_df, dummy_ieee_ident_df):
    txn_p = tmp_path / "txn.csv"
    ident_p = tmp_path / "ident.csv"
    dummy_ieee_txn_df.to_csv(txn_p, index=False)
    dummy_ieee_ident_df.to_csv(ident_p, index=False)
    res = clean_ieee(str(ident_p), str(txn_p))
    assert list(res.columns) == TARGET_COLUMNS
    assert len(res) == 2
    assert (res["source"] == "ieee").all()
    assert res.loc[0, "device_type"] == "mobile"
    assert res.loc[1, "device_type"] == "unknown"

def test_backfill_account_age_days():
    df = pd.DataFrame({
        "label": [0, 1],
        "account_age_days": [np.nan, np.nan],
        "synthetic_fields": ["", ""]
    })
    res = backfill_account_age_days(df.copy())
    assert not res["account_age_days"].isna().any()
    assert res.loc[0, "synthetic_fields"] == "account_age_days"
    assert res.loc[1, "synthetic_fields"] == "account_age_days"

def test_backfill_device_info():
    df = pd.DataFrame({
        "device_info": [np.nan, "known_device"],
        "synthetic_fields": ["", ""]
    })
    res = backfill_device_info(df.copy())
    assert not res["device_info"].isna().any()
    assert res.loc[0, "synthetic_fields"] == "device_info"
    assert res.loc[1, "synthetic_fields"] == ""

def test_backfill_device_payee_new():
    df = pd.DataFrame({
        "label": [0, 1],
        "device_new": [np.nan, True],
        "payee_new": [False, np.nan],
        "synthetic_fields": ["", ""]
    })
    res = backfill_device_payee_new(df.copy())
    assert not res["device_new"].isna().any()
    assert not res["payee_new"].isna().any()
    assert res.loc[0, "synthetic_fields"] == "device_new"
    assert res.loc[1, "synthetic_fields"] == "payee_new"
