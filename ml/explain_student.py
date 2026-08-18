"""
SOAIDEATHON-S40 — explain_student.py  (SS-10)
Defines and implements the on-device explainability method for the deployed
student model: for a single flagged transaction, which features drove the
score, and by how much?

THE ACTUAL DECISION THIS SCRIPT MAKES:
Classic Integrated Gradients needs gradients, normally computed via
backprop through the PyTorch model. But what's actually deployed is the
ONNX export, running through `onnxruntime` (API) or `onnxruntime-web`
(browser) -- neither has autograd. Explaining predictions by loading the
PyTorch checkpoint on the side would mean the explanation and the
prediction come from two different artifacts, which can drift out of sync
and defeats the point of "on-device."

So this implements NUMERICAL Integrated Gradients: same theory (attribute
the score difference between input and a baseline, integrated along the
straight-line path between them), but gradients at each interpolation step
are approximated via central finite differences -- i.e. nothing but forward
passes through the exact ONNX model that's actually serving predictions.
All the finite-difference + interpolation points get batched into ONE
onnxruntime call each, so this is fast despite needing many forward passes
per explanation (see benchmark at the bottom of main()).

A second, much cheaper method (occlusion / feature ablation: zero out one
feature at a time, measure the score change) is also implemented and
benchmarked alongside it, as the actual comparison this ticket asks for --
occlusion is ~15-50x cheaper per explanation but gives a coarser, less
theoretically grounded attribution (it doesn't account for interactions
along a path the way IG's integral does). Both get run so you can see the
tradeoff directly rather than take it on faith.

Run after export_student.py (needs student_model.onnx).
"""

import os
import time

import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
ARTIFACTS_DIR = os.path.join(PROCESSED_DIR, "artifacts")
PLOTS_DIR = os.path.join(PROCESSED_DIR, "plots", "explainability")

ONNX_PATH = os.path.join(ARTIFACTS_DIR, "student_model.onnx")

IG_STEPS = 50           # interpolation steps along the baseline -> input path
FD_EPSILON = 0.05        # finite-difference step size, in standardized-feature units (features are pre-scaled)
N_EXAMPLE_TRANSACTIONS = 5   # how many individual flagged transactions to fully explain + plot
N_GLOBAL_SAMPLE = 500        # how many transactions to average over for the global importance plot


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def load_model_and_data():
    import onnxruntime as ort
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    X_full = pd.read_parquet(os.path.join(PROCESSED_DIR, "X_full.parquet"))
    meta = pd.read_parquet(os.path.join(PROCESSED_DIR, "meta_full.parquet"))
    feature_names = list(X_full.columns)
    return sess, input_name, X_full, meta, feature_names


def forward_batch(sess, input_name, X_batch):
    out = sess.run(None, {input_name: X_batch.astype(np.float32)})[0]
    return out.reshape(-1)  # ONNX graph already applies squeeze(-1) internally, so output shape
                             # varies (batch,) vs (batch,1) depending on batch size -- reshape(-1)
                             # handles both without assuming which one comes back


def compute_baseline(X_full: pd.DataFrame, meta: pd.DataFrame) -> np.ndarray:
    """Baseline = mean feature vector of NORMAL (label==0) transactions -- the
    natural 'what does typical, non-fraud look like' reference point for IG
    to attribute deviations FROM."""
    normal = X_full[meta["label"].values == 0]
    return normal.mean(axis=0).values.astype(np.float32)


# ---------------------------------------------------------------------------
# Method 1: numerical (gradient-free) Integrated Gradients
# ---------------------------------------------------------------------------

def integrated_gradients(sess, input_name, x: np.ndarray, baseline: np.ndarray,
                          steps=IG_STEPS, eps=FD_EPSILON):
    """x, baseline: shape (n_features,). Returns per-feature attributions,
    shape (n_features,), such that attributions.sum() ~= F(x) - F(baseline)
    in logit space (the 'completeness' property IG is supposed to satisfy --
    checked explicitly in main() as a correctness sanity check, the same
    role SS-8's PyTorch-vs-ONNX diff check played for the export)."""
    n_features = len(x)
    alphas = np.linspace(1.0 / steps, 1.0, steps)  # skip alpha=0 (baseline itself, gradient there is uninformative)
    interpolated = baseline[None, :] + alphas[:, None] * (x - baseline)[None, :]  # (steps, n_features)

    # build ALL perturbed points at once: for each interpolation step, +eps and -eps on each feature
    plus = np.repeat(interpolated[:, None, :], n_features, axis=1).reshape(-1, n_features)   # (steps*n_features, n_features)
    minus = plus.copy()
    idx = np.tile(np.arange(n_features), steps)
    plus[np.arange(len(plus)), idx] += eps
    minus[np.arange(len(minus)), idx] -= eps

    all_points = np.concatenate([plus, minus], axis=0)   # (2*steps*n_features, n_features) -- ONE batched forward call
    all_logits = forward_batch(sess, input_name, all_points)
    plus_logits, minus_logits = np.split(all_logits, 2)

    grads = (plus_logits - minus_logits) / (2 * eps)          # (steps*n_features,)
    grads = grads.reshape(steps, n_features)
    avg_grad = grads.mean(axis=0)                              # (n_features,)

    attributions = (x - baseline) * avg_grad
    return attributions


# ---------------------------------------------------------------------------
# Method 2: occlusion / feature ablation (cheap comparison baseline)
# ---------------------------------------------------------------------------

def occlusion_attributions(sess, input_name, x: np.ndarray, baseline: np.ndarray):
    """Set each feature to its baseline value one at a time, measure how much
    the score moves. n_features forward passes total (batched into one
    call) -- no interpolation, no finite-difference gradient, just direct
    ablation. Cheaper than IG but doesn't account for feature interactions
    along a path, only the single-step effect of removing each feature from
    the actual input."""
    n_features = len(x)
    base_logit = forward_batch(sess, input_name, x[None, :])[0]

    occluded = np.tile(x, (n_features, 1))
    occluded[np.arange(n_features), np.arange(n_features)] = baseline
    occluded_logits = forward_batch(sess, input_name, occluded)

    return base_logit - occluded_logits  # positive = removing this feature INCREASES the score -> feature pushed score up


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_example_attribution(feature_names, attributions, score, method_name, out_path, top_n=12):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort(np.abs(attributions))[::-1][:top_n]
    names = [feature_names[i] for i in order]
    vals = attributions[order]

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["firebrick" if v > 0 else "steelblue" for v in vals]
    ax.barh(range(len(vals)), vals[::-1], color=[colors[::-1][i] for i in range(len(vals))])
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(names[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("attribution (logit units, red = pushed toward fraud)")
    ax.set_title(f"{method_name} — risk score {score:.4f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_global_importance(feature_names, mean_abs_attr_ig, mean_abs_attr_occ, out_path, top_n=15):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort(mean_abs_attr_ig)[::-1][:top_n]
    names = [feature_names[i] for i in order]
    ig_vals = mean_abs_attr_ig[order]
    occ_vals = mean_abs_attr_occ[order]

    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(names))
    width = 0.35
    ax.barh(y - width / 2, ig_vals[::-1], width, label="Integrated Gradients", color="steelblue")
    ax.barh(y + width / 2, occ_vals[::-1], width, label="Occlusion", color="darkorange")
    ax.set_yticks(y)
    ax.set_yticklabels(names[::-1])
    ax.set_xlabel("mean |attribution| across sampled transactions")
    ax.set_title(f"Global feature importance (top {top_n}, averaged over {N_GLOBAL_SAMPLE} transactions)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_method_comparison(ig_time_ms, occ_time_ms, ig_completeness_err, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    axes[0].bar(["Integrated\nGradients", "Occlusion"], [ig_time_ms, occ_time_ms],
                color=["steelblue", "darkorange"])
    axes[0].set_ylabel("ms per explanation")
    axes[0].set_title("Explanation latency")
    for i, v in enumerate([ig_time_ms, occ_time_ms]):
        axes[0].text(i, v + max(ig_time_ms, occ_time_ms) * 0.02, f"{v:.2f}ms", ha="center")

    axes[1].hist(ig_completeness_err, bins=30, color="steelblue", alpha=0.8)
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("IG completeness error (attributions.sum() - [F(x)-F(baseline)])")
    axes[1].set_title("IG correctness check across sampled transactions")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    sess, input_name, X_full, meta, feature_names = load_model_and_data()
    baseline = compute_baseline(X_full, meta)
    print(f"Baseline (mean of {(meta['label'] == 0).sum():,} normal transactions) computed, "
          f"{len(feature_names)} features.")

    # ---- explain a handful of individual flagged transactions ----
    scores_all = sigmoid(forward_batch(sess, input_name, X_full.values))
    flagged_idx = np.where(scores_all >= 0.5)[0]
    rng = np.random.default_rng(42)
    example_idx = rng.choice(flagged_idx, size=min(N_EXAMPLE_TRANSACTIONS, len(flagged_idx)), replace=False)

    print(f"\nExplaining {len(example_idx)} example flagged transactions...")
    for i, idx in enumerate(example_idx):
        x = X_full.values[idx].astype(np.float32)
        score = scores_all[idx]

        ig_attr = integrated_gradients(sess, input_name, x, baseline)
        occ_attr = occlusion_attributions(sess, input_name, x, baseline)

        plot_example_attribution(feature_names, ig_attr, score, "Integrated Gradients",
                                  os.path.join(PLOTS_DIR, f"example_{i}_ig.png"))
        plot_example_attribution(feature_names, occ_attr, score, "Occlusion",
                                  os.path.join(PLOTS_DIR, f"example_{i}_occlusion.png"))

        top_feature = feature_names[np.argmax(np.abs(ig_attr))]
        print(f"  transaction #{idx}  score={score:.4f}  top driver (IG): {top_feature} "
              f"(attribution={ig_attr[np.argmax(np.abs(ig_attr))]:+.4f})")

    # ---- global importance + method comparison over a larger sample ----
    print(f"\nComputing global importance + benchmarking both methods over {N_GLOBAL_SAMPLE} transactions...")
    sample_idx = rng.choice(len(X_full), size=N_GLOBAL_SAMPLE, replace=False)

    ig_attrs, occ_attrs, completeness_errs = [], [], []
    t_ig_total, t_occ_total = 0.0, 0.0

    for idx in sample_idx:
        x = X_full.values[idx].astype(np.float32)

        t0 = time.perf_counter()
        ig_attr = integrated_gradients(sess, input_name, x, baseline)
        t_ig_total += time.perf_counter() - t0

        t0 = time.perf_counter()
        occ_attr = occlusion_attributions(sess, input_name, x, baseline)
        t_occ_total += time.perf_counter() - t0

        ig_attrs.append(ig_attr)
        occ_attrs.append(occ_attr)

        # completeness check: sum of attributions should equal F(x) - F(baseline) in logit space
        f_x = forward_batch(sess, input_name, x[None, :])[0]
        f_baseline = forward_batch(sess, input_name, baseline[None, :])[0]
        completeness_errs.append(ig_attr.sum() - (f_x - f_baseline))

    ig_attrs = np.array(ig_attrs)
    occ_attrs = np.array(occ_attrs)
    completeness_errs = np.array(completeness_errs)

    mean_abs_ig = np.abs(ig_attrs).mean(axis=0)
    mean_abs_occ = np.abs(occ_attrs).mean(axis=0)

    ig_ms = (t_ig_total / N_GLOBAL_SAMPLE) * 1000
    occ_ms = (t_occ_total / N_GLOBAL_SAMPLE) * 1000

    print(f"\nMean IG explanation latency:        {ig_ms:.3f} ms  "
          f"({2 * IG_STEPS * len(feature_names)} forward evals/explanation, batched into 1 call)")
    print(f"Mean Occlusion explanation latency: {occ_ms:.3f} ms  "
          f"({len(feature_names)} forward evals/explanation, batched into 1 call)  "
          f"[{ig_ms / occ_ms:.1f}x cheaper than IG]")
    print(f"IG completeness check: mean_abs_error={np.abs(completeness_errs).mean():.4f}  "
          f"(should be small relative to typical logit magnitude -- large values mean the "
          f"finite-difference approximation or step count needs tuning)")

    plot_global_importance(feature_names, mean_abs_ig, mean_abs_occ,
                            os.path.join(PLOTS_DIR, "global_feature_importance.png"))
    plot_method_comparison(ig_ms, occ_ms, completeness_errs,
                            os.path.join(PLOTS_DIR, "method_comparison.png"))

    print(f"\nSaved {len(example_idx) * 2 + 2} plots -> {PLOTS_DIR}")
    print("  example_N_ig.png / example_N_occlusion.png (per-transaction attribution bars)")
    print("  global_feature_importance.png, method_comparison.png")

    summary = {
        "feature_names": feature_names, "baseline": baseline,
        "ig_steps": IG_STEPS, "fd_epsilon": FD_EPSILON,
        "ig_latency_ms": ig_ms, "occlusion_latency_ms": occ_ms,
        "ig_completeness_mean_abs_error": float(np.abs(completeness_errs).mean()),
        "global_mean_abs_attribution_ig": dict(zip(feature_names, mean_abs_ig.tolist())),
        "global_mean_abs_attribution_occlusion": dict(zip(feature_names, mean_abs_occ.tolist())),
    }
    joblib.dump(summary, os.path.join(ARTIFACTS_DIR, "explainability_summary.joblib"))
    print(f"\nSaved -> {os.path.join(ARTIFACTS_DIR, 'explainability_summary.joblib')}")

    print(
        "\nMETHOD RECOMMENDATION: use Integrated Gradients for flagged-transaction explanations "
        "shown to an analyst/user (theoretically grounded, satisfies completeness) -- at "
        f"~{ig_ms:.2f}ms per explanation it's cheap enough to run on every flag, not just on request. "
        "Occlusion is the fallback for extremely constrained environments (e.g. a low-power mobile "
        f"client) where even {ig_ms:.2f}ms per explanation is too much: it's ~{ig_ms/occ_ms:.0f}x faster "
        "at the cost of ignoring feature interactions along the baseline->input path."
    )


if __name__ == "__main__":
    main()