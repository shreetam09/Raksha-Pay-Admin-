import os
import pandas as pd
import numpy as np
from ml.feature_engineering import build_features

def test_build_features_fit_true():
    df = pd.DataFrame({
        "balance_before": [100.0, np.nan],
        "balance_after": [50.0, np.nan],
        "device_new": [True, False],
        "payee_new": [False, True],
        "location_change": [False, False],
        "call_active_during_txn": [True, False],
        "amount": [10.0, 20.0],
        "account_age_days": [100, 200],
        "device_info": ["dev_a", "dev_b"],
        "transaction_type": ["type1", "type2"],
        "device_type": ["mobile", "desktop"],
        "source": ["ulb", "paysim"]
    })
    features, scaler, ohe_cols, freq_map = build_features(df.copy(), fit=True)
    
    assert features is not None
    assert scaler is not None
    assert ohe_cols is not None
    assert freq_map is not None
    
    assert len(features) == 2
    assert features["device_new"].dtype in [int, np.int32, np.int64]
    
    assert "device_info_freq" in features.columns
    # device_info_freq is scaled by StandardScaler. 
    # For ["dev_a", "dev_b"], frequencies are [0.5, 0.5], variance is 0, so scaled to 0.0.
    assert features["device_info_freq"].iloc[0] == 0.0
    
    assert features["has_balance_data"].iloc[0] == 1
    assert features["has_balance_data"].iloc[1] == 0

def test_build_features_fit_false():
    df_fit = pd.DataFrame({
        "balance_before": [100.0, 50.0],
        "balance_after": [50.0, 20.0],
        "device_new": [True, False],
        "payee_new": [False, True],
        "location_change": [False, False],
        "call_active_during_txn": [True, False],
        "amount": [10.0, 20.0],
        "account_age_days": [100, 200],
        "device_info": ["dev_a", "dev_a"],
        "transaction_type": ["type1", "type1"],
        "device_type": ["mobile", "mobile"],
        "source": ["ulb", "ulb"]
    })
    _, scaler, ohe_cols, freq_map = build_features(df_fit, fit=True)
    
    df_test = pd.DataFrame({
        "balance_before": [np.nan],
        "balance_after": [np.nan],
        "device_new": [True],
        "payee_new": [False],
        "location_change": [False],
        "call_active_during_txn": [True],
        "amount": [15.0],
        "account_age_days": [150],
        "device_info": ["dev_unseen"],
        "transaction_type": ["type_unseen"],
        "device_type": ["mobile"],
        "source": ["ulb"]
    })
    
    features, out_scaler, out_ohe, out_freq = build_features(
        df_test, fit=False, scaler=scaler, ohe_columns=ohe_cols, freq_map=freq_map
    )
    
    # device_info_freq for unseen is 0.0. Scaler was fit on ["dev_a", "dev_a"] -> freq 1.0 (var 0, std=1.0 per scikit-learn).
    # (0.0 - 1.0) / 1.0 = -1.0
    assert features["device_info_freq"].iloc[0] == -1.0
    
    for col in ohe_cols:
        assert col in features.columns
    
    assert "transaction_type_type_unseen" not in features.columns
