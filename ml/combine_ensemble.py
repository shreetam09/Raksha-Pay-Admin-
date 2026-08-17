"""
SOAIDEATHON-S40 — combine_ensemble.py  (SS-6)
Merges the IsolationForest teacher (if_scores.parquet, SS-4) and the
Autoencoder teacher (ae_scores.parquet, SS-5) into a single ensemble
anomaly score. This is the target the student model (next task, downstream
distillation) will be trained to approximate.

Both raw teacher scores already follow the "higher = more anomalous"
convention (if_anomaly_score = -IF.decision_function, ae_recon_error = MSE),
but they live on very different scales/distributions, so a straight average
of raw values would let whichever teacher happens to have larger numeric
spread dominate. Both are put on a common [0,1] scale first via rank/
percentile normalization -- robust to that scale mismatch and to outliers
in either teacher's raw score (a plain min-max would let one freak IF or AE
value compress the rest of the range).

Run after both train_isolation_forest.py and train_autoencoder.py.
"""

import os

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    confusion_matrix, precision_score, recall_score, f1_score, accuracy_score,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
ARTIFACTS_DIR = os.path.join(PROCESSED_DIR, "artifacts")
PLOTS_DIR = os.path.join(PROCESSED_DIR, "plots", "ensemble")  # separate from plots/autoencoder/

IF_WEIGHT = 0.5  # weight for IsolationForest in ensemble_weighted_avg (AE gets 1 - IF_WEIGHT)
CONTAMINATION = 0.005  # same assumed fraud rate used in SS-4/SS-5, for the confusion-matrix operating point

SCORE_COLS = [
    "if_score_norm", "ae_score_norm", "ensemble_simple_avg",
    "ensemble_weighted_avg", "ensemble_max", "ensemble_logreg_meta",
]
SCORE_LABELS = {
    "if_score_norm": "Isolation Forest",
    "ae_score_norm": "Autoencoder",
    "ensemble_simple_avg": "Ensemble (simple avg)",
    "ensemble_weighted_avg": "Ensemble (weighted avg)",
    "ensemble_max": "Ensemble (max)",
    "ensemble_logreg_meta": "Ensemble (logreg meta)",
}


def percentile_normalize(x: np.ndarray) -> np.ndarray:
    """Rank-based normalization to [0,1]. Robust to the scale/outlier mismatch
    between IsolationForest's decision_function and the Autoencoder's MSE."""
    ranks = rankdata(x, method="average")
    return (ranks - 1) / (len(x) - 1)


def print_auc_table(meta: pd.DataFrame, score_cols: list):
    print("\nROC-AUC / PR-AUC comparison (full dataset):")
    for col in score_cols:
        auc = roc_auc_score(meta["label"], meta[col])
        ap = average_precision_score(meta["label"], meta[col])
        print(f"  {col:22s} ROC-AUC={auc:.4f}  PR-AUC={ap:.4f}")

    print("\nPer-source ROC-AUC breakdown:")
    for source in meta["source"].unique():
        sub = meta[meta["source"] == source]
        if sub["label"].nunique() < 2:
            print(f"  {source}: skipped (only one class present)")
            continue
        row = "  " + f"{source:10s}"
        for col in score_cols:
            row += f"  {col}={roc_auc_score(sub['label'], sub[col]):.4f}"
        print(row)


def make_plots(merged: pd.DataFrame, score_cols: list, contamination: float):
    """Every figure -> processed/plots/ensemble/. One overlay ROC/PR across all
    score columns (both teachers + all ensemble variants), an AUC comparison bar
    chart, a confusion matrix + score distribution for the best-AUC variant, and
    a combined dashboard tying all of it together."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(PLOTS_DIR, exist_ok=True)

    y = merged["label"].values
    aucs, aps = {}, {}
    curves = {}  # col -> (fpr, tpr, prec, rec)

    for col in score_cols:
        scores = merged[col].values
        fpr, tpr, _ = roc_curve(y, scores)
        prec, rec, _ = precision_recall_curve(y, scores)
        aucs[col] = roc_auc_score(y, scores)
        aps[col] = average_precision_score(y, scores)
        curves[col] = (fpr, tpr, prec, rec)

    best_col = max(aucs, key=aucs.get)
    best_scores = merged[best_col].values
    threshold = np.quantile(best_scores, 1 - contamination)
    y_pred = (best_scores >= threshold).astype(int)
    acc = accuracy_score(y, y_pred)
    prec_at_t = precision_score(y, y_pred, zero_division=0)
    rec_at_t = recall_score(y, y_pred, zero_division=0)
    f1_at_t = f1_score(y, y_pred, zero_division=0)
    cm = confusion_matrix(y, y_pred)

    # ---- 1. ROC overlay, all score columns ----
    fig, ax = plt.subplots(figsize=(6, 6))
    for col in score_cols:
        fpr, tpr, _, _ = curves[col]
        ax.plot(fpr, tpr, label=f"{SCORE_LABELS[col]} (AUC={aucs[col]:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="chance")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC — teachers vs ensemble variants (full dataset)")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "roc_overlay.png"), dpi=150); plt.close(fig)

    # ---- 2. PR overlay, all score columns ----
    fig, ax = plt.subplots(figsize=(6, 6))
    for col in score_cols:
        _, _, prec, rec = curves[col]
        ax.plot(rec, prec, label=f"{SCORE_LABELS[col]} (AP={aps[col]:.4f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall — teachers vs ensemble variants (full dataset)")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "pr_overlay.png"), dpi=150); plt.close(fig)

    # ---- 3. AUC / AP comparison bar chart ----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(score_cols))
    width = 0.35
    ax.bar(x - width / 2, [aucs[c] for c in score_cols], width, label="ROC-AUC", color="steelblue")
    ax.bar(x + width / 2, [aps[c] for c in score_cols], width, label="PR-AUC", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels([SCORE_LABELS[c] for c in score_cols], rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1); ax.legend()
    ax.set_title("Full-dataset ROC-AUC / PR-AUC by score")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "auc_comparison.png"), dpi=150); plt.close(fig)

    # ---- 4. per-source AUC, grouped bar (best-AUC variant vs both teachers) ----
    compare_cols = ["if_score_norm", "ae_score_norm", best_col]
    sources = [s for s in merged["source"].unique() if merged.loc[merged["source"] == s, "label"].nunique() >= 2]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(sources))
    width = 0.25
    for i, col in enumerate(compare_cols):
        vals = [roc_auc_score(merged.loc[merged["source"] == s, "label"], merged.loc[merged["source"] == s, col])
                for s in sources]
        ax.bar(x + (i - 1) * width, vals, width, label=SCORE_LABELS[col])
    ax.set_xticks(x); ax.set_xticklabels(sources)
    ax.set_ylim(0, 1); ax.set_ylabel("ROC-AUC"); ax.legend(fontsize=8)
    ax.set_title(f"Per-source ROC-AUC: teachers vs best ensemble ({SCORE_LABELS[best_col]})")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "per_source_auc_comparison.png"), dpi=150); plt.close(fig)

    # ---- 5. score distribution for best variant ----
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(best_scores[y == 0], bins=80, alpha=0.6, label="normal", density=True)
    ax.hist(best_scores[y == 1], bins=80, alpha=0.6, label="fraud", density=True)
    ax.axvline(threshold, color="k", linestyle="--", linewidth=1, label=f"threshold @ {contamination:.1%}")
    ax.set_xlabel(f"{SCORE_LABELS[best_col]} score"); ax.set_ylabel("density")
    ax.set_title(f"Score distribution — best variant ({SCORE_LABELS[best_col]})")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "best_score_distribution.png"), dpi=150); plt.close(fig)

    # ---- 6. confusion matrix for best variant ----
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred normal", "pred fraud"]); ax.set_yticklabels(["true normal", "true fraud"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(f"Confusion matrix @ {contamination:.1%} — {SCORE_LABELS[best_col]}\n"
                 f"acc={acc:.4f}  prec={prec_at_t:.4f}  rec={rec_at_t:.4f}  f1={f1_at_t:.4f}")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "best_confusion_matrix.png"), dpi=150); plt.close(fig)

    # ---- 7. combined dashboard ----
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3)

    ax1 = fig.add_subplot(gs[0, 0])
    for col in score_cols:
        fpr, tpr, _, _ = curves[col]
        ax1.plot(fpr, tpr, label=f"{SCORE_LABELS[col]}", linewidth=1)
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax1.set_title("ROC overlay"); ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR")
    ax1.legend(fontsize=6)

    ax2 = fig.add_subplot(gs[0, 1])
    for col in score_cols:
        _, _, prec, rec = curves[col]
        ax2.plot(rec, prec, label=f"{SCORE_LABELS[col]}", linewidth=1)
    ax2.set_title("PR overlay"); ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    ax2.legend(fontsize=6)

    ax3 = fig.add_subplot(gs[0, 2])
    x = np.arange(len(score_cols))
    ax3.bar(x, [aucs[c] for c in score_cols], color="steelblue")
    ax3.set_xticks(x); ax3.set_xticklabels([SCORE_LABELS[c] for c in score_cols], rotation=40, ha="right", fontsize=6)
    ax3.set_ylim(0, 1); ax3.set_title("ROC-AUC by score")

    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(best_scores[y == 0], bins=80, alpha=0.6, label="normal", density=True)
    ax4.hist(best_scores[y == 1], bins=80, alpha=0.6, label="fraud", density=True)
    ax4.axvline(threshold, color="k", linestyle="--", linewidth=1)
    ax4.set_title(f"Best variant distribution\n({SCORE_LABELS[best_col]})"); ax4.legend(fontsize=8)

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(cm, cmap="Blues")
    ax5.set_xticks([0, 1]); ax5.set_yticks([0, 1])
    ax5.set_xticklabels(["pred norm", "pred fraud"]); ax5.set_yticklabels(["true norm", "true fraud"])
    for i in range(2):
        for j in range(2):
            ax5.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax5.set_title(f"Confusion matrix\nacc={acc:.4f} f1={f1_at_t:.4f}")

    ax6 = fig.add_subplot(gs[1, 2])
    width = 0.25
    xs = np.arange(len(sources))
    for i, col in enumerate(compare_cols):
        vals = [roc_auc_score(merged.loc[merged["source"] == s, "label"], merged.loc[merged["source"] == s, col])
                for s in sources]
        ax6.bar(xs + (i - 1) * width, vals, width, label=SCORE_LABELS[col])
    ax6.set_xticks(xs); ax6.set_xticklabels(sources, fontsize=8)
    ax6.set_ylim(0, 1); ax6.legend(fontsize=6)
    ax6.set_title("Per-source AUC")

    fig.suptitle(f"Ensemble comparison — best: {SCORE_LABELS[best_col]} "
                 f"(ROC-AUC {aucs[best_col]:.4f}, PR-AUC {aps[best_col]:.4f})", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "combined_dashboard.png"), dpi=150); plt.close(fig)

    print(f"\nSaved 7 plots -> {PLOTS_DIR}")
    print("  roc_overlay.png, pr_overlay.png, auc_comparison.png, per_source_auc_comparison.png,")
    print("  best_score_distribution.png, best_confusion_matrix.png, combined_dashboard.png")
    print(f"\nBest variant by full-dataset ROC-AUC: {SCORE_LABELS[best_col]} ({best_col}) -> {aucs[best_col]:.4f}")

    return {
        "best_variant": best_col, "best_roc_auc": float(aucs[best_col]), "best_pr_auc": float(aps[best_col]),
        "best_threshold": float(threshold), "best_accuracy": float(acc), "best_precision": float(prec_at_t),
        "best_recall": float(rec_at_t), "best_f1": float(f1_at_t),
        "all_roc_auc": {c: float(aucs[c]) for c in score_cols},
        "all_pr_auc": {c: float(aps[c]) for c in score_cols},
    }


def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("Loading teacher scores...")
    if_scores = pd.read_parquet(os.path.join(PROCESSED_DIR, "if_scores.parquet"))
    ae_scores = pd.read_parquet(os.path.join(PROCESSED_DIR, "ae_scores.parquet"))
    print(f"  if_scores: {if_scores.shape}")
    print(f"  ae_scores: {ae_scores.shape}")

    merged = if_scores.merge(
        ae_scores[["record_id", "ae_recon_error"]], on="record_id", how="inner"
    )
    n_dropped = len(if_scores) - len(merged)
    if n_dropped:
        print(f"  warning: {n_dropped} rows dropped in merge (record_id mismatch between teachers)")

    if_norm = percentile_normalize(merged["if_anomaly_score"].values)
    ae_norm = percentile_normalize(merged["ae_recon_error"].values)
    merged["if_score_norm"] = if_norm
    merged["ae_score_norm"] = ae_norm

    merged["ensemble_simple_avg"] = (if_norm + ae_norm) / 2.0
    merged["ensemble_weighted_avg"] = IF_WEIGHT * if_norm + (1 - IF_WEIGHT) * ae_norm
    merged["ensemble_max"] = np.maximum(if_norm, ae_norm)

    meta_model = LogisticRegression(class_weight="balanced")
    meta_model.fit(np.column_stack([if_norm, ae_norm]), merged["label"])
    merged["ensemble_logreg_meta"] = meta_model.predict_proba(
        np.column_stack([if_norm, ae_norm])
    )[:, 1]
    joblib.dump(meta_model, os.path.join(ARTIFACTS_DIR, "ensemble_meta_model.joblib"))

    print_auc_table(merged, SCORE_COLS)

    print(
        "\nrecommendation: use whichever ensemble_* column scores highest full-dataset "
        "ROC-AUC above as the distillation target for the student model. "
        "ensemble_logreg_meta usually wins but needs ensemble_meta_model.joblib carried "
        "into inference; ensemble_simple_avg/weighted_avg need nothing extra at inference "
        "beyond both teachers' raw scores."
    )

    out_path = os.path.join(PROCESSED_DIR, "ensemble_scores.parquet")
    merged.to_parquet(out_path, index=False)
    print(f"\nSaved -> {out_path}")
    print(f"Saved meta-model -> {os.path.join(ARTIFACTS_DIR, 'ensemble_meta_model.joblib')}")

    metrics = make_plots(merged, SCORE_COLS, CONTAMINATION)
    joblib.dump(metrics, os.path.join(ARTIFACTS_DIR, "ensemble_metrics.joblib"))


if __name__ == "__main__":
    main()