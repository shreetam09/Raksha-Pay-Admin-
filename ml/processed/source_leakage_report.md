# Source-Leakage Sanity Check — Verdict

Threshold: a source is flagged FAIL if real, shared features (amount) account for
more than 70% of what the classifier used to guess dataset origin.
Leakage driven by synthetic fields (device_new, payee_new, location_change,
call_active_during_txn) is expected and does not by itself fail this check,
since those columns are intentionally generated differently per source.

| Source | Accuracy | Real-feature importance (amount) | Verdict |
|---|---|---|---|
| paysim | 99.8% | 26.7% | **PASS** |
| ulb | 99.8% | 27.6% | **PASS** |
| ieee | 100.0% | 9.3% | **PASS** |

## Feature importance breakdown

**paysim**
- `payee_new`: 57.0%
- `amount`: 26.7%
- `device_new`: 16.4%
- `location_change`: 0.0%
- `call_active_during_txn`: 0.0%

**ulb**
- `payee_new`: 60.8%
- `amount`: 27.6%
- `device_new`: 11.4%
- `call_active_during_txn`: 0.1%
- `location_change`: 0.1%

**ieee**
- `payee_new`: 55.9%
- `device_new`: 34.8%
- `amount`: 9.3%
- `location_change`: 0.0%
- `call_active_during_txn`: 0.0%

## Reading this
If `amount` is doing most of the work, that's expected right now -- ULB, PaySim, and IEEE have genuinely different amount distributions by construction (different currencies/scales, different transaction types). This becomes a real concern only once feature engineering starts normalizing amount across sources (e.g. amount percentile within source) -- rerun this check after that step, not just once, at the start.