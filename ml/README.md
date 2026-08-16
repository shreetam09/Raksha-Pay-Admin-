
# Unified Schema — SOAIDEATHON-S40

## Status

`train_transaction.csv` is now available and joined — IEEE-CIS carries real `amount`,
`label`, and `timestamp` like the other two sources. Post-merge backfill
(`account_age_days`, `device_info`, leftover `device_new`/`payee_new`) has been
implemented and run. Feature engineering (frequency encoding, one-hot, scaling) has
been added as a downstream step and is documented separately at the bottom of this
file — it does not change the unified CSV schema itself.

## Target schema

| Field                      | Type  | Meaning                                                                                                                    |
| -------------------------- | ----- | -------------------------------------------------------------------------------------------------------------------------- |
| `record_id`              | str   | Unique row identifier (source-prefixed)                                                                                    |
| `source`                 | str   | `ulb` / `paysim` / `ieee`                                                                                            |
| `amount`                 | float | Transaction amount                                                                                                         |
| `timestamp`              | float | Seconds/steps since a reference point (NOT wall-clock — units differ by source, kept relative)                            |
| `transaction_type`       | str   | PAYMENT / TRANSFER / CASH_OUT / CASH_IN / DEBIT / unknown / IEEE ProductCD                                                 |
| `balance_before`         | float | Sender balance before transaction (NaN if unavailable — see note below, NOT backfilled)                                   |
| `balance_after`          | float | Sender balance after transaction (NaN if unavailable — see note below, NOT backfilled)                                    |
| `device_type`            | str   | mobile / desktop / unknown                                                                                                 |
| `device_info`            | str   | Raw device string (IEEE, real) or backfilled bucket (ULB/PaySim, synthetic — see below)                                   |
| `device_new`             | bool  | Is this a device not seen before for this account (real for PaySim/IEEE where derivable, synthetic elsewhere — see below) |
| `payee_new`              | bool  | Is the payee new/first-time (real for PaySim; synthetic for ULB/IEEE — see below)                                         |
| `location_change`        | bool  | Sudden geographic/IP change (**fully synthetic — no source has this**)                                              |
| `call_active_during_txn` | bool  | Active call at time of transaction (**fully synthetic — no source has this; this is the coercion signal**)          |
| `account_age_days`       | float | Age of account (**unavailable in all three raw sources; fully backfilled — see below**)                             |
| `label`                  | int   | 0 = legitimate, 1 = fraud                                                                                                  |
| `synthetic_fields`       | str   | Comma-separated list of which fields on this row are synthetic, not observed (per-row, not per-source — see below)        |

## Per-source mapping

### ULB (`creditcard.csv`)

- `Time` → `timestamp` (seconds from first transaction in the dataset — already relative, not wall-clock)
- `Amount` → `amount`
- `Class` → `label`
- `V1`-`V28` → dropped from the unified table (PCA-anonymized, not reconstructable to real features; not part of the cross-source schema)
- No device, payee, location, call, or account-age fields at all → **all backfilled**, either rule-based-synthetic (coercion signals, `account_age_days`) or bucketed-synthetic (`device_info`)

### PaySim (`paysim_dataset.csv`)

- `step` → `timestamp` (1 step = 1 hour, per PaySim's own documentation — different unit than ULB's seconds, kept as relative only, never compared directly across sources)
- `type` → `transaction_type`
- `amount` → `amount`
- `oldbalanceOrg` → `balance_before`
- `newbalanceOrig` → `balance_after`
- `isFraud` → `label`
- `nameDest` starting with `M` = merchant destination, `C` = customer — used to derive `payee_new` (first time this `nameDest` appears for this `nameOrig`) — **real derived signal**, not synthetic, from actual transaction history
- No device, location, call, or account-age fields → **backfilled** (device signals rule-based-synthetic; `device_info` bucketed-synthetic; `account_age_days` synthetic)

### IEEE-CIS (`train_identity.csv` LEFT-joined to `train_transaction.csv` on `TransactionID`)

- `TransactionAmt` → `amount`
- `TransactionDT` → `timestamp` (seconds from a competition-internal reference point — own unit, not comparable across sources)
- `ProductCD` → `transaction_type`
- `isFraud` → `label`
- `DeviceType` → `device_type`
- `DeviceInfo` → `device_info` (real where present; ~98% null pre-backfill since most IEEE rows have no identity-table match — backfilled with a bucket label where missing, same as ULB/PaySim)
- `id_35` (real session/device-consistency flag per Kaggle's field docs) → `device_new` where present; backfilled (rule-based-synthetic) for the rows without an identity-table match
- No payee-history field in the IEEE-CIS schema → `payee_new` **fully synthetic** for this source
- No `balance_before`/`balance_after` concept in this schema → **left NaN, not backfilled** (see note below)
- No account-age field → **backfilled** (synthetic)

## Fields with no real source at all (fully synthetic, all three sources)

`location_change`, `call_active_during_txn`, `account_age_days` (backfilled post-merge).
`payee_new` is synthetic for ULB/IEEE only — PaySim has a real derived proxy.
`device_new` is synthetic for ULB, and for the subset of IEEE rows with no `id_35`
value — PaySim (synthetic, no raw signal exists) and the rest of IEEE (real, from
`id_35`) differ. These are generated by rule-based backfill logic correlated with
`label` (e.g. fraud rows get a higher probability of `call_active_during_txn=True`
when amount is high and device is new), and `synthetic_fields` is set **per row**,
not per source — a cell only gets tagged if it was actually fabricated for that
specific row, since coverage of `device_new`/`payee_new`/`device_info` varies row
by row even within one source (e.g. IEEE rows with vs. without an identity match).

## `balance_before` / `balance_after` — deliberately NOT backfilled

ULB (card-present transactions) and IEEE-CIS have no real balance concept in their
raw schema — only PaySim does. Every other field above got a synthetic backfill;
these two didn't, on purpose: fabricating a plausible-looking balance number would
be worse than an honest null, since there's no real-world basis to derive it from
(unlike, say, `account_age_days`, where a label-correlated distribution is at least
a defensible heuristic). Downstream feature engineering handles this with a
`has_balance_data` flag (1 for PaySim rows, 0 for ULB/IEEE) instead of imputing
values — see below.

## Downstream: feature engineering (not part of the unified CSV)

`feature_engineering.py` builds the numeric matrix used for teacher-model training
(Isolation Forest, Autoencoder) from the unified CSV above. Not stored back into
`unified_transactions.csv` — kept as a separate step so the unified table stays a
clean, source-of-truth schema independent of any one model's feature needs.

- `timestamp` dropped as a feature (units incompatible across sources, see above)
- `balance_before`/`balance_after` nulls filled with 0 + `has_balance_data` flag added
- `transaction_type`, `device_type`, `source` — low cardinality → one-hot encoded
- `device_info` — high cardinality (1,790 distinct values once IEEE's real device
  strings are included) → **frequency-encoded** to a single `device_info_freq`
  numeric column instead of one-hot (one-hot on this column tried to allocate
  ~94GB for a 7.2M-row frame — standard practice for high-cardinality categoricals
  in fraud detection is frequency/count encoding, not one-hot)
- All numeric columns scaled with `StandardScaler`, fit on `label==0` rows only
