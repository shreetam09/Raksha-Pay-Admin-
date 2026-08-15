import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def dummy_ulb_df():
    return pd.DataFrame({
        "Time": [0, 1],
        "Amount": [10.0, 1000.0],
        "Class": [0, 1]
    })

@pytest.fixture
def dummy_paysim_df():
    return pd.DataFrame({
        "step": [1, 2],
        "type": ["PAYMENT", "TRANSFER"],
        "amount": [9839.64, 181.00],
        "nameOrig": ["C123", "C456"],
        "oldbalanceOrg": [170136.0, 181.0],
        "newbalanceOrig": [160296.36, 0.0],
        "nameDest": ["M123", "C789"],
        "oldbalanceDest": [0.0, 0.0],
        "newbalanceDest": [0.0, 0.0],
        "isFraud": [0, 1]
    })

@pytest.fixture
def dummy_ieee_txn_df():
    return pd.DataFrame({
        "TransactionID": [3000000, 3000001],
        "TransactionDT": [86400, 86401],
        "TransactionAmt": [68.5, 29.0],
        "ProductCD": ["W", "W"],
        "isFraud": [0, 1]
    })

@pytest.fixture
def dummy_ieee_ident_df():
    return pd.DataFrame({
        "TransactionID": [3000000],
        "DeviceType": ["mobile"],
        "DeviceInfo": ["rv:11.0"],
        "id_35": ["T"]
    })
