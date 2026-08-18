"""
SOAIDEATHON-S40 — monitor_student.py
Production monitoring for the deployed student model — the "is it still
healthy" question, distinct from SS-10's per-prediction explainability
("why did it say that about THIS transaction"). This script tracks
aggregate behavior across many predictions over time:

  1. FEATURE DRIFT — has the incoming transaction data started looking
     different from what the model was trained on? (PSI per feature)
  2. SCORE DRIFT — has the distribution of risk scores the model outputs
     shifted? (PSI on the score distribution itself)
  3. FLAG RATE — what fraction of transactions are getting flagged as
     fraud, and is that fraction changing over time?
  4. LATENCY — is inference still fast, or degrading?

Uses Population Stability Index (PSI), the standard drift metric in fraud/
credit-risk modeling: PSI < 0.1 = no significant shift, 0.1-0.25 = moderate
shift (watch it), > 0.25 = significant shift (investigate / consider
retraining). Computed by binning a REFERENCE distribution (training data)
into deciles, then comparing how a CURRENT batch's proportions fall into
those same bins.

This script is a standalone simulation: it fits a reference from the
training data once, then replays chunks of held-out data as if they were
sequential production batches arriving over time, logging + plotting drift
as it goes. In real deployment, replace `simulate_incoming_batches()` with
your actual batch source (e.g. a rolling window of logged API requests) and
run this on a schedule (cron / Airflow / whatever your infra uses).

Run after export_student.py (uses the ONNX model, since that's what
actually runs in production, not the PyTorch training checkpoint).
"""

import os
import time

import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
ARTIFACTS_DIR = os.path.join(PROCESSED_DIR, "artifacts")
MONITORING_DIR = os.path.join(PROCESSED_DIR, "monitoring")
PLOTS_DIR = os.path.join(PROCESSED_DIR, "plots", "monitoring")

ONNX_PATH = os.path.join(ARTIFACTS_DIR, "student_model.onnx")
REFERENCE_PATH = os.path.join(MONITORING_DIR, "reference_stats.joblib")
LOG_PATH = os.path.join(MONITORING_DIR, "monitoring_log.parquet")

N_BINS = 10                  # decile bins for PSI
PSI_WATCH_THRESHOLD = 0.10   # below this: no action
PSI_ALERT_THRESHOLD = 0.25   # above this: flag for investigation
N_SIMULATED_BATCHES = 20     # how many "time steps" to replay in this demo run
BATCH_SIZE = 5000            # rows per simulated batch


# ---------------------------------------------------------------------------
# Population Stability Index
# ---------------------------------------------------------------------------

def fit_bin_edges(reference_values: np.ndarray, n_bins=N_BINS):
    """Quantile bin edges from the reference (training) distribution.
    Quantile-based rather than equal-width so each reference bin starts with
    ~equal population -- makes PSI meaningful even for skewed features."""
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(reference_values, quantiles))
    if len(edges) < 3:  # degenerate (e.g. a near-constant column) -- pad so binning doesn't crash
        edges = np.array([reference_values.min() - 1e-6, reference_values.max() + 1e-6])
    edges[0], edges[-1] = -np.inf, np.inf  # so any future value, however extreme, still lands in a bin
    return edges


def bin_proportions(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values, bins=edges)
    props = counts / max(len(values), 1)
    return np.clip(props, 1e-6, None)  # avoid log(0) in PSI


def psi(reference_props: np.ndarray, current_props: np.ndarray) -> float:
    return float(np.sum((current_props - reference_props) * np.log(current_props / reference_props)))


# ---------------------------------------------------------------------------
# Reference fitting (run once, persisted)
# ---------------------------------------------------------------------------

def fit_reference(feature_cols: list):
    print("Fitting reference distribution from X_full.parquet (training-time feature + score stats)...")
    X_full = pd.read_parquet(os.path.join(PROCESSED_DIR, "X_full.parquet"))
    meta = pd.read_parquet(os.path.join(PROCESSED_DIR, "meta_full.parquet"))

    import onnxruntime as ort
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    ref_scores = sigmoid(sess.run(None, {input_name: X_full.values.astype(np.float32)})[0].squeeze())

    feature_edges = {col: fit_bin_edges(X_full[col].values) for col in feature_cols}
    score_edges = fit_bin_edges(ref_scores)

    reference = {
        "feature_cols": feature_cols,
        "feature_edges": feature_edges,
        "feature_reference_props": {
            col: bin_proportions(X_full[col].values, feature_edges[col]) for col in feature_cols
        },
        "score_edges": score_edges,
        "score_reference_props": bin_proportions(ref_scores, score_edges),
        "reference_flag_rate": float((ref_scores >= 0.5).mean()),
        "reference_mean_score": float(ref_scores.mean()),
    }
    os.makedirs(MONITORING_DIR, exist_ok=True)
    joblib.dump(reference, REFERENCE_PATH)
    print(f"Saved reference -> {REFERENCE_PATH}")
    return reference


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ---------------------------------------------------------------------------
# Simulated production batches
# ---------------------------------------------------------------------------

def simulate_incoming_batches(feature_cols, n_batches, batch_size):
    """Stands in for a real production data source. Replace this generator
    with however batches actually arrive in your deployment (e.g. read the
    last N logged API requests) -- everything downstream just needs an
    iterator of (X_batch_df, batch_id)."""
    X_full = pd.read_parquet(os.path.join(PROCESSED_DIR, "X_full.parquet"))
    for i in range(n_batches):
        batch = X_full[feature_cols].sample(n=batch_size, random_state=1000 + i)
        yield batch, i


# ---------------------------------------------------------------------------
# Per-batch monitoring
# ---------------------------------------------------------------------------

def monitor_batch(sess, input_name, reference, batch_df, batch_id):
    X_batch = batch_df.values.astype(np.float32)

    t0 = time.perf_counter()
    logits = sess.run(None, {input_name: X_batch})[0].squeeze()
    latency_ms = (time.perf_counter() - t0) * 1000.0 / len(X_batch)  # per-sample

    scores = sigmoid(logits)
    flag_rate = float((scores >= 0.5).mean())

    score_props = bin_proportions(scores, reference["score_edges"])
    score_psi = psi(reference["score_reference_props"], score_props)

    feature_psis = {}
    for col in reference["feature_cols"]:
        props = bin_proportions(batch_df[col].values, reference["feature_edges"][col])
        feature_psis[col] = psi(reference["feature_reference_props"][col], props)
    max_feature = max(feature_psis, key=feature_psis.get)
    max_feature_psi = feature_psis[max_feature]

    alert = score_psi > PSI_ALERT_THRESHOLD or max_feature_psi > PSI_ALERT_THRESHOLD
    watch = not alert and (score_psi > PSI_WATCH_THRESHOLD or max_feature_psi > PSI_WATCH_THRESHOLD)

    return {
        "batch_id": batch_id, "n_samples": len(X_batch), "mean_score": float(scores.mean()),
        "flag_rate": flag_rate, "flag_rate_delta": flag_rate - reference["reference_flag_rate"],
        "score_psi": score_psi, "max_feature_psi_name": max_feature, "max_feature_psi": max_feature_psi,
        "latency_ms_per_sample": latency_ms, "status": "ALERT" if alert else ("WATCH" if watch else "OK"),
    }


def make_plots(log_df: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    ax = axes[0, 0]
    ax.plot(log_df["batch_id"], log_df["score_psi"], "o-", label="score PSI", color="steelblue")
    ax.plot(log_df["batch_id"], log_df["max_feature_psi"], "s-", label="worst feature PSI", color="darkorange")
    ax.axhline(PSI_WATCH_THRESHOLD, color="goldenrod", linestyle="--", linewidth=1, label="watch threshold")
    ax.axhline(PSI_ALERT_THRESHOLD, color="firebrick", linestyle="--", linewidth=1, label="alert threshold")
    ax.set_xlabel("batch (time)"); ax.set_ylabel("PSI"); ax.set_title("Drift over time")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(log_df["batch_id"], log_df["flag_rate"], "o-", color="crimson")
    ax.axhline(log_df["flag_rate"].iloc[0], color="gray", linestyle=":", linewidth=1, label="reference rate (approx)")
    ax.set_xlabel("batch (time)"); ax.set_ylabel("fraction flagged")
    ax.set_title("Fraud flag rate over time"); ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(log_df["batch_id"], log_df["latency_ms_per_sample"], "o-", color="seagreen")
    ax.set_xlabel("batch (time)"); ax.set_ylabel("latency (ms/sample)")
    ax.set_title("Inference latency over time")

    ax = axes[1, 1]
    status_counts = log_df["status"].value_counts()
    colors_map = {"OK": "seagreen", "WATCH": "goldenrod", "ALERT": "firebrick"}
    ax.bar(status_counts.index, status_counts.values,
           color=[colors_map.get(s, "gray") for s in status_counts.index])
    ax.set_title("Batch status counts"); ax.set_ylabel("number of batches")

    fig.suptitle("Student model — production monitoring dashboard", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "monitoring_dashboard.png"), dpi=150)
    plt.close(fig)
    print(f"\nSaved monitoring dashboard -> {os.path.join(PLOTS_DIR, 'monitoring_dashboard.png')}")


def main():
    import onnxruntime as ort

    meta = pd.read_parquet(os.path.join(PROCESSED_DIR, "meta_full.parquet"))
    X_full = pd.read_parquet(os.path.join(PROCESSED_DIR, "X_full.parquet"))
    feature_cols = list(X_full.columns)

    if os.path.exists(REFERENCE_PATH):
        reference = joblib.load(REFERENCE_PATH)
        print(f"Loaded existing reference -> {REFERENCE_PATH}")
    else:
        reference = fit_reference(feature_cols)

    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    print(f"\nSimulating {N_SIMULATED_BATCHES} production batches "
          f"({BATCH_SIZE} rows each, sampled from held-out data)...")
    print(f"Reference: mean_score={reference['reference_mean_score']:.4f}  "
          f"flag_rate={reference['reference_flag_rate']:.4%}")
    print(f"Thresholds: WATCH > {PSI_WATCH_THRESHOLD}, ALERT > {PSI_ALERT_THRESHOLD}\n")

    log_rows = []
    for batch_df, batch_id in simulate_incoming_batches(feature_cols, N_SIMULATED_BATCHES, BATCH_SIZE):
        result = monitor_batch(sess, input_name, reference, batch_df, batch_id)
        log_rows.append(result)
        flag = "  <-- " + result["status"] if result["status"] != "OK" else ""
        print(f"  batch {batch_id:3d}  score_psi={result['score_psi']:.4f}  "
              f"worst_feature={result['max_feature_psi_name']:<20s} psi={result['max_feature_psi']:.4f}  "
              f"flag_rate={result['flag_rate']:.4%}  latency={result['latency_ms_per_sample']:.4f}ms  "
              f"[{result['status']}]{flag}")

    log_df = pd.DataFrame(log_rows)
    os.makedirs(MONITORING_DIR, exist_ok=True)
    log_df.to_parquet(LOG_PATH, index=False)
    print(f"\nSaved monitoring log -> {LOG_PATH}")

    n_alert = (log_df["status"] == "ALERT").sum()
    n_watch = (log_df["status"] == "WATCH").sum()
    print(f"\nSummary: {n_alert} ALERT batches, {n_watch} WATCH batches, "
          f"{len(log_df) - n_alert - n_watch} OK batches out of {len(log_df)}")
    if n_alert:
        print("  -> investigate: check which feature(s) are drifting, consider retraining "
              "the pipeline from feature_engineering.py forward if this persists.")

    make_plots(log_df)


if __name__ == "__main__":
    main()
