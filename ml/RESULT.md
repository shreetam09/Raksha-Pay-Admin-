
# SOAIDEATHON-S40 — ML Pipeline Results

**Session date:** August 18, 2026
**Scope:** SS-4 (Isolation Forest) → SS-5 (Autoencoder) → SS-6 (Ensemble) → SS-7 (Student distillation)
**Hardware:** NVIDIA GeForce RTX 2050 (4GB VRAM, CUDA 12.6), local Windows machine

All four stages ran end-to-end today. Full sweep table, per-stage numbers, and environment gotchas below.

---

## Environment setup — issues hit and fixes

Worth keeping for next time / onboarding:

| Issue                                                                                         | Cause                                                                                                                                                                                                                            | Fix                                                                                                                                                                                                                           |
| --------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `uv pip install pytorch==1.0.2` fails to build                                              | Wrong package name — PyPI's`pytorch` isn't the real package                                                                                                                                                                   | Use`torch`, not `pytorch`, in `requirements.txt`                                                                                                                                                                        |
| `ModuleNotFoundError: No module named 'torch'` despite `uv pip list` showing it installed | Ran`python3` instead of `python` — Windows venvs don't ship a `python3.exe`, so `python3` silently escaped the venv to a system interpreter                                                                             | Always use`python`, not `python3`, on Windows. Verify with `Get-Command python3` vs `Get-Command python` if unsure which resolves where                                                                               |
| Torch installed but`torch.cuda.is_available()` → `False`                                 | `uv pip install torch --index-url .../cu126` saw torch already satisfied from a prior CPU-only install and skipped reinstalling it                                                                                             | Add`--reinstall`: `uv pip install torch --index-url https://download.pytorch.org/whl/cu126 --reinstall`                                                                                                                   |
| `feature_engineering.py` — `ArrayMemoryError: Unable to allocate 93.7 GiB`               | One-hot encoding`device_info` (1,790 distinct values) on a 7.2M-row frame instead of frequency-encoding it                                                                                                                     | Fixed in`feature_engineering.py` — switched `device_info` to frequency encoding (single numeric column), kept one-hot only for the genuinely low-cardinality columns (`transaction_type`, `device_type`, `source`) |
| Occasional`KeyboardInterrupt` mid-`DataLoader` iteration on GPU runs                      | Transient — looks like a stalled/interrupted first attempt, not a reproducible bug (both`train_autoencoder.py` and `train_student.py` hit this once each, then completed cleanly on the very next run with no code changes) | Just rerun if it happens once; investigate further only if it recurs                                                                                                                                                          |

---

## SS-4 — Isolation Forest

`train_isolation_forest.py`, trained on `X_train_normal.parquet` (label==0 only), `contamination=0.005`, `n_estimators=200`.

- **Trained in 88.5s**
- **ROC-AUC: 0.9521**
- **PR-AUC: 0.1005**

Per-source ROC-AUC:

| source | ROC-AUC |
| ------ | ------- |
| ulb    | 0.8598  |
| paysim | 0.9391  |
| ieee   | 0.8160  |

Artifacts: `processed/artifacts/isolation_forest.joblib`, `processed/if_scores.parquet`

---

## SS-5 — Autoencoder

`train_autoencoder.py`, PyTorch MLP `27 → 64 → 32 → 16 (latent) → 32 → 64 → 27`, trained on normal-only rows.

- **Trained in 1884.3s** (31.4 min, CPU run — GPU install wasn't fixed yet at this point in the session)
- Early stopped at epoch 14, best val_loss at epoch 4 (0.001931)
- **ROC-AUC: 0.8982**
- **PR-AUC: 0.0690**

Per-source ROC-AUC:

| source | ROC-AUC |
| ------ | ------- |
| ulb    | 0.8895  |
| paysim | 0.8843  |
| ieee   | 0.8284  |

Weaker than IF alone on every metric — expected for AE vs tree-based anomaly detection on tabular data without heavy tuning; this is the reason for ensembling, not a bug.

Artifacts: `processed/artifacts/autoencoder.pt`, `processed/ae_scores.parquet`, `processed/plots/autoencoder/` (7 plots: loss curve, ROC, PR, score distribution, confusion matrix, per-source AUC, combined dashboard)

---

## SS-6 — Ensemble

`combine_ensemble.py`, merges IF + AE scores via rank/percentile normalization, produces 4 variants.

| variant                        | ROC-AUC          | PR-AUC           |
| ------------------------------ | ---------------- | ---------------- |
| if_score_norm (teacher alone)  | 0.9521           | 0.1005           |
| ae_score_norm (teacher alone)  | 0.8982           | 0.0690           |
| ensemble_simple_avg            | 0.9471           | **0.1614** |
| ensemble_weighted_avg          | 0.9471           | 0.1614           |
| ensemble_max                   | 0.9404           | 0.1009           |
| **ensemble_logreg_meta** | **0.9564** | 0.1569           |

**`ensemble_logreg_meta` beats standalone IF on ROC-AUC** (0.9564 vs 0.9521) — the ensemble genuinely adds value, not just averaging down. `simple_avg`/`weighted_avg` post the best PR-AUC (0.1614) if precision-at-low-recall matters more than overall ranking.

Per-source ROC-AUC (logreg_meta):

| source | ROC-AUC |
| ------ | ------- |
| ulb    | 0.8956  |
| paysim | 0.9378  |
| ieee   | 0.8394  |

`ensemble_logreg_meta` selected as SS-7's distillation target — it's already a calibrated probability (unlike the rank-based variants, which only mean something relative to a reference population and don't translate to a single transaction hitting an API in isolation).

Artifacts: `processed/ensemble_scores.parquet`, `processed/artifacts/ensemble_meta_model.joblib`, `processed/plots/ensemble/` (7 plots)

---

## SS-7 — Student distillation (capacity sweep)

`train_student.py`, distilled `ensemble_logreg_meta` into a small MLP student. Loss = `0.5·BCE(hard label) + 0.5·BCE(soft target)`. 6 architectures trained identically on full 5.2M-row training set (GPU), evaluated on a held-out 1.09M-row test split.

| arch                       | params          | size             | test ROC-AUC     | test PR-AUC      | CPU latency        |
| -------------------------- | --------------- | ---------------- | ---------------- | ---------------- | ------------------ |
| tiny_4                     | 125             | 0.5 KB           | 0.9936           | 0.8451           | 0.099 ms           |
| tiny_8                     | 249             | 1.0 KB           | 0.9949           | 0.8555           | 0.130 ms           |
| small_16                   | 497             | 1.9 KB           | 0.9948           | 0.8566           | 0.062 ms           |
| small_16_8                 | 641             | 2.5 KB           | 0.9958           | 0.8466           | 0.138 ms           |
| **medium_32_16**     | **1,537** | **6.0 KB** | **0.9976** | **0.8776** | **0.077 ms** |
| large_64_32 (ceiling ref.) | 4,097           | 16.0 KB          | 0.9973           | 0.8792           | 0.091 ms           |

**Selected: `medium_32_16`** — smallest architecture within tolerance (ROC-AUC ≤0.002, PR-AUC ≤0.003) of the sweep's best. Matches the teacher-sized ceiling architecture's AUC exactly (delta +0.0000) while using 37% of its parameters.

**PR-AUC jumped from the ensemble's 0.161 to the student's 0.878.** This is a real, explainable jump, not an artifact: IF and AE are *unsupervised* (never see the fraud label during training); the student is trained directly against true labels (with `pos_weight` for the 0.4% imbalance) plus the ensemble's soft signal on top — a supervised classifier with distillation as a regularizer, not a pure distillation setup. Worth stating this plainly in any writeup so the comparison isn't read as apples-to-apples.

`tiny_4` (125 params, 0.5KB) already reaches 0.9936 ROC-AUC / 0.8451 PR-AUC — most of the signal comes from the 27 engineered features, not head capacity. Good news for on-device deployment regardless of which arch ships.

Artifacts: `processed/artifacts/student_model.pt` (winner, full-data refit), `processed/artifacts/student_sweep_summary.joblib`, `processed/plots/student/` (5 plots: size-vs-performance, size-vs-latency, loss curves all archs, winner ROC/PR, winner confusion matrix)

---

## File format decisions (settled)

- Teacher-stage artifacts (`isolation_forest.joblib`, `autoencoder.pt`, `ensemble_meta_model.joblib`) stay as-is — both are pickle-based, fine for a Python-only pipeline.
- **SS-7's student model does NOT get pickled for deployment.** SS-8 needs ONNX/TFLite for on-device inference (Python-only formats don't run in a browser or mobile runtime). Path: `torch.onnx.export()` → `.onnx` → `onnx2tf`/`onnx-tf` → `.tflite`.

---

## Git

Both `train_isolation_forest.py` (with `feature_engineering.py` fix) and later `combine_ensemble.py` pushed to `features/ML`:

```
617b9f5..fedc4a1  features/ML -> features/ML
bb6aa49..cb06a45  features/ML -> features/ML
```

---

## Not yet done

- **SS-8:** export `medium_32_16` student (`processed/artifacts/student_model.pt`) to ONNX, then TFLite.
- **SS-9:** benchmark exported model size + inference latency (on-device, not the CPU latency numbers above which were measured in the training venv).
- **SS-10:** define on-device explainability method — Integrated Gradients likely, since the student is a small differentiable MLP (SHAP's fast paths need either a tree model or heavy sampling, neither fits an ONNX/TFLite-deployed NN well).
