# Source-Leakage Sanity Check — Verdict

Threshold: a source is flagged FAIL if real, shared features (amount) account for
more than 70% of what the classifier used to guess dataset origin.
Leakage driven by synthetic fields (device_new, payee_new, location_change,
call_active_during_txn) is expected and does not by itself fail this check,
since those columns are intentionally generated differently per source.

| Source | Accuracy | Real-feature importance (amount) | Verdict |
|---|---|---|---|
| paysim | 99.6% | 38.6% | **PASS** |
| ulb | 97.8% | 65.0% | **PASS** |
| ieee | 97.5% | 47.9% | **PASS** |

## Feature importance breakdown

**paysim**
- `payee_new`: 61.0%
- `amount`: 38.6%
- `device_new`: 0.3%
- `location_change`: 0.1%
- `call_active_during_txn`: 0.0%

**ulb**
- `amount`: 65.0%
- `payee_new`: 30.4%
- `device_new`: 4.4%
- `call_active_during_txn`: 0.1%
- `location_change`: 0.1%

**ieee**
- `payee_new`: 50.0%
- `amount`: 47.9%
- `device_new`: 1.8%
- `call_active_during_txn`: 0.2%
- `location_change`: 0.1%

## Reading this
If `amount` is doing most of the work, that's expected right now -- ULB, PaySim, and IEEE have genuinely different amount distributions by construction (different currencies/scales, different transaction types). This becomes a real concern only once feature engineering starts normalizing amount across sources (e.g. amount percentile within source) -- rerun this check after that step, not just once, at the start.