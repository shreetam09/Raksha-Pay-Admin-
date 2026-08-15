
# Unified Schema — SOAIDEATHON-S40

## Target schema

| Field                      | Type  | Meaning                                                                                                                            |
| -------------------------- | ----- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `record_id`              | str   | Unique row identifier (source-prefixed)                                                                                            |
| `source`                 | str   | `ulb` / `paysim` / `ieee`                                                                                                    |
| `amount`                 | float | Transaction amount                                                                                                                 |
| `timestamp`              | float | Seconds/steps since a reference point (NOT wall-clock — units differ by source, kept relative)                                    |
| `transaction_type`       | str   | PAYMENT / TRANSFER / CASH_OUT / CASH_IN / DEBIT / unknown                                                                          |
| `balance_before`         | float | Sender balance before transaction (NaN if unavailable)                                                                             |
| `balance_after`          | float | Sender balance after transaction (NaN if unavailable)                                                                              |
| `device_type`            | str   | mobile / desktop / unknown                                                                                                         |
| `device_info`            | str   | Raw device string if available                                                                                                     |
| `device_new`             | bool  | Is this a device not seen before for this account (**synthetic for ULB/PaySim; real signal available for IEEE once joined**) |
| `payee_new`              | bool  | Is the payee new/first-time (**synthetic for all three — none of the raw sources track payee history**)                     |
| `location_change`        | bool  | Sudden geographic/IP change (**fully synthetic — no source has this**)                                                      |
| `call_active_during_txn` | bool  | Active call at time of transaction (**fully synthetic — no source has this; this is the coercion signal**)                  |
| `account_age_days`       | float | Age of account (**unavailable in all three — left NaN, flagged**)                                                           |
| `label`                  | int   | 0 = legitimate, 1 = fraud                                                                                                          |
| `synthetic_fields`       | str   | Comma-separated list of which fields on this row are synthetic, not observed                                                       |

## Per-source mapping

### ULB (`creditcard.csv`)

- `Time` → `timestamp` (seconds from first transaction in the dataset — already relative, not wall-clock)
- `Amount` → `amount`
- `Class` → `label`
- `V1`-`V28` → dropped from the unified table (PCA-anonymized, not reconstructable to real features; kept in a separate `ulb_pca_features` table if needed later for the anomaly model directly, but not part of the cross-source schema)
- No device, payee, location, or call fields at all → **all backfilled synthetically**

### PaySim (`paysim_dataset.csv`)

- `step` → `timestamp` (1 step = 1 hour, per PaySim's own documentation — different unit than ULB's seconds, kept as relative only, never compared directly across sources)
- `type` → `transaction_type`
- `amount` → `amount`
- `oldbalanceOrg` → `balance_before`
- `newbalanceOrig` → `balance_after`
- `isFraud` → `label`
- `nameDest` starting with `M` = merchant destination, `C` = customer — used to derive a rough `payee_new` proxy (first time this `nameDest` appears for this `nameOrig`) — this is a **real derived signal**, not synthetic, since it comes from actual transaction history in the data
- No device, location, or call fields → **backfilled synthetically**

### IEEE-CIS (`train_identity.csv`, joins to `train_transaction.csv` — not yet available)

- `DeviceType` → `device_type`
- `DeviceInfo` → `device_info`
- `id_35`, `id_36` (boolean-style match flags per Kaggle's own field documentation) → contribute to a **real** `device_new`-style signal once joined to transaction history (a session/device consistency flag)
- **Blocked without `train_transaction.csv`**: `TransactionID`, `isFraud`, `TransactionDT`, `TransactionAmt`, `ProductCD`, `card1`-`card6`, `addr1`/`addr2`, `P_emaildomain`/`R_emaildomain`, `M1`-`M9` are all in the transaction file per the competition's public schema, not the identity file. Until that file is supplied, IEEE rows can't carry `amount`, `label`, or `timestamp` — they're cleaned and held separately, ready to join on `TransactionID` the moment the transaction file is added.

## Fields with no real source at all (fully synthetic, all three sources)

`payee_new` (ULB/IEEE only — PaySim has a real proxy), `location_change`, `call_active_during_txn`, `account_age_days`. These are generated by the rule-based backfill logic, correlated with `label` to mimic documented scam patterns (e.g. fraud rows get a higher probability of `call_active_during_txn=True` when amount is high and device is new), and every row carries a `synthetic_fields` tag naming exactly which columns were fabricated — so nothing downstream can quietly treat a synthetic field as observed data.
