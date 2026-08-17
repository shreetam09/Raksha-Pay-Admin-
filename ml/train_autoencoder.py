"""
SOAIDEATHON-S40 — train_autoencoder.py  (SS-5)
Trains the Autoencoder teacher on normal-only features (same X_train_normal.parquet
IsolationForest used), then scores the FULL dataset to produce reconstruction-error
anomaly scores for downstream ensemble/distillation.

Mirrors train_isolation_forest.py's convention: higher score = more anomalous.

Run after feature_engineering.py (and can run independently of train_isolation_forest.py —
both teachers consume the same X_train_normal.parquet / X_full.parquet / meta_full.parquet).
"""

import os
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    confusion_matrix, precision_score, recall_score, f1_score, accuracy_score,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
ARTIFACTS_DIR = os.path.join(PROCESSED_DIR, "artifacts")
PLOTS_DIR = os.path.join(PROCESSED_DIR, "plots", "autoencoder")  # all AE figures live here, separate from ensemble/IF plots

LATENT_DIM = 16
HIDDEN_DIMS = (64, 32)
EPOCHS = 100
BATCH_SIZE = 256
LR = 1e-3
VAL_SPLIT = 0.15
PATIENCE = 10
SEED = 42
CONTAMINATION = 0.005  # same assumed fraud rate as SS-4, used only to pick a threshold for the confusion matrix / accuracy plot


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = LATENT_DIM, hidden_dims=HIDDEN_DIMS):
        super().__init__()
        h1, h2 = hidden_dims
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1), nn.BatchNorm1d(h1), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(h1, h2), nn.BatchNorm1d(h2), nn.ReLU(),
            nn.Linear(h2, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2), nn.BatchNorm1d(h2), nn.ReLU(),
            nn.Linear(h2, h1), nn.BatchNorm1d(h1), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(h1, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}  "
              f"(CUDA {torch.version.cuda}, {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    else:
        device = torch.device("cpu")
        print("WARNING: CUDA not available to torch — training on CPU. "
              "If you have an NVIDIA GPU, check `torch.cuda.is_available()` / driver / CUDA build of torch "
              "(you installed the cu126 wheel — confirm THIS venv's "
              "`python -c \"import torch; print(torch.cuda.is_available())\"` is True).")
    return device


def train_model(model, train_loader, val_loader, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
    criterion = nn.MSELoss()

    best_val, best_state, bad_epochs = float("inf"), None, 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        for (batch,) in train_loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device, non_blocking=True)
                val_losses.append(criterion(model(batch), batch).item())

        train_loss, val_loss = float(np.mean(train_losses)), float(np.mean(val_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)
        print(f"  epoch {epoch:3d}/{EPOCHS}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"  early stopping at epoch {epoch} (best val_loss={best_val:.6f})")
                break

    model.load_state_dict(best_state)
    return model, history


def make_plots(history, meta, contamination):
    """Every figure -> processed/plots/autoencoder/. Individual PNGs plus one combined dashboard."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(PLOTS_DIR, exist_ok=True)

    y = meta["label"].values
    scores = meta["ae_recon_error"].values
    fpr, tpr, _ = roc_curve(y, scores)
    roc_auc = roc_auc_score(y, scores)
    prec, rec, _ = precision_recall_curve(y, scores)
    pr_auc = average_precision_score(y, scores)

    # threshold picked so the flagged fraction matches the assumed contamination rate,
    # purely so accuracy/precision/recall/confusion-matrix have a concrete operating point to report at
    threshold = np.quantile(scores, 1 - contamination)
    y_pred = (scores >= threshold).astype(int)
    acc = accuracy_score(y, y_pred)
    prec_at_t = precision_score(y, y_pred, zero_division=0)
    rec_at_t = recall_score(y, y_pred, zero_division=0)
    f1_at_t = f1_score(y, y_pred, zero_division=0)
    cm = confusion_matrix(y, y_pred)

    per_source_auc = {}
    for source in meta["source"].unique():
        sub = meta[meta["source"] == source]
        if sub["label"].nunique() >= 2:
            per_source_auc[source] = roc_auc_score(sub["label"], sub["ae_recon_error"])

    # ---- 1. loss curve ----
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["train_loss"], label="train")
    ax.plot(history["val_loss"], label="val")
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE loss"); ax.set_title("Autoencoder reconstruction loss")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "loss_curve.png"), dpi=150); plt.close(fig)

    # ---- 2. ROC curve ----
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AE (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="chance")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC — Autoencoder (full dataset)")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "roc_curve.png"), dpi=150); plt.close(fig)

    # ---- 3. Precision-Recall curve ----
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(rec, prec, label=f"AE (AP = {pr_auc:.4f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall — Autoencoder (full dataset)")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "pr_curve.png"), dpi=150); plt.close(fig)

    # ---- 4. score distribution (normal vs fraud) ----
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(scores[y == 0], bins=80, alpha=0.6, label="normal", density=True)
    ax.hist(scores[y == 1], bins=80, alpha=0.6, label="fraud", density=True)
    ax.axvline(threshold, color="k", linestyle="--", linewidth=1, label=f"threshold @ {contamination:.1%} contamination")
    ax.set_xlabel("reconstruction error"); ax.set_ylabel("density")
    ax.set_title("AE score distribution: normal vs fraud")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "score_distribution.png"), dpi=150); plt.close(fig)

    # ---- 5. confusion matrix (at threshold) ----
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred normal", "pred fraud"]); ax.set_yticklabels(["true normal", "true fraud"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(f"Confusion matrix @ {contamination:.1%} threshold\n"
                 f"acc={acc:.4f}  prec={prec_at_t:.4f}  rec={rec_at_t:.4f}  f1={f1_at_t:.4f}")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"), dpi=150); plt.close(fig)

    # ---- 6. per-source AUC bar chart ----
    if per_source_auc:
        fig, ax = plt.subplots(figsize=(5, 4))
        sources = list(per_source_auc.keys())
        vals = [per_source_auc[s] for s in sources]
        ax.bar(sources, vals, color="steelblue")
        ax.set_ylim(0, 1); ax.set_ylabel("ROC-AUC")
        ax.set_title("Autoencoder ROC-AUC by source")
        for i, v in enumerate(vals):
            ax.text(i, v + 0.02, f"{v:.4f}", ha="center")
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, "per_source_auc.png"), dpi=150); plt.close(fig)

    # ---- 7. combined dashboard (all of the above in one figure) ----
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(history["train_loss"], label="train"); ax1.plot(history["val_loss"], label="val")
    ax1.set_title("Loss curve"); ax1.set_xlabel("epoch"); ax1.set_ylabel("MSE"); ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}"); ax2.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax2.set_title("ROC curve"); ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR"); ax2.legend()

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(rec, prec, label=f"AP={pr_auc:.4f}")
    ax3.set_title("Precision-Recall"); ax3.set_xlabel("Recall"); ax3.set_ylabel("Precision"); ax3.legend()

    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(scores[y == 0], bins=80, alpha=0.6, label="normal", density=True)
    ax4.hist(scores[y == 1], bins=80, alpha=0.6, label="fraud", density=True)
    ax4.axvline(threshold, color="k", linestyle="--", linewidth=1)
    ax4.set_title("Score distribution"); ax4.legend()

    ax5 = fig.add_subplot(gs[1, 1])
    im = ax5.imshow(cm, cmap="Blues")
    ax5.set_xticks([0, 1]); ax5.set_yticks([0, 1])
    ax5.set_xticklabels(["pred norm", "pred fraud"]); ax5.set_yticklabels(["true norm", "true fraud"])
    for i in range(2):
        for j in range(2):
            ax5.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax5.set_title(f"Confusion matrix\nacc={acc:.4f} f1={f1_at_t:.4f}")

    ax6 = fig.add_subplot(gs[1, 2])
    if per_source_auc:
        sources = list(per_source_auc.keys())
        vals = [per_source_auc[s] for s in sources]
        ax6.bar(sources, vals, color="steelblue")
        ax6.set_ylim(0, 1)
        for i, v in enumerate(vals):
            ax6.text(i, v + 0.02, f"{v:.4f}", ha="center")
    ax6.set_title("Per-source AUC")

    fig.suptitle(f"Autoencoder teacher — full-dataset ROC-AUC {roc_auc:.4f}, PR-AUC {pr_auc:.4f}", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "combined_dashboard.png"), dpi=150); plt.close(fig)

    print(f"\nSaved 7 plots -> {PLOTS_DIR}")
    print(f"  loss_curve.png, roc_curve.png, pr_curve.png, score_distribution.png,")
    print(f"  confusion_matrix.png, per_source_auc.png, combined_dashboard.png")

    return {
        "threshold": float(threshold), "accuracy": float(acc), "precision": float(prec_at_t),
        "recall": float(rec_at_t), "f1": float(f1_at_t), "roc_auc": float(roc_auc), "pr_auc": float(pr_auc),
    }


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    device = get_device()
    use_gpu = device.type == "cuda"

    print("Loading features...")
    X_train = pd.read_parquet(os.path.join(PROCESSED_DIR, "X_train_normal.parquet")).values.astype(np.float32)
    X_full = pd.read_parquet(os.path.join(PROCESSED_DIR, "X_full.parquet")).values.astype(np.float32)
    meta = pd.read_parquet(os.path.join(PROCESSED_DIR, "meta_full.parquet"))
    print(f"  train (normal-only): {X_train.shape}")
    print(f"  full (all labels):   {X_full.shape}")

    n_val = int(len(X_train) * VAL_SPLIT)
    perm = np.random.permutation(len(X_train))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train[train_idx])),
                               batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
                               pin_memory=use_gpu)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_train[val_idx])),
                             batch_size=BATCH_SIZE, shuffle=False,
                             pin_memory=use_gpu)

    print(f"\nTraining Autoencoder (latent_dim={LATENT_DIM}, hidden={HIDDEN_DIMS}) on {device}...")
    t0 = time.time()
    model = Autoencoder(input_dim=X_train.shape[1]).to(device)
    model, history = train_model(model, train_loader, val_loader, device)
    elapsed = time.time() - t0
    print(f"  trained in {elapsed:.1f}s ({device})")

    print("\nScoring full dataset...")
    model.eval()
    with torch.no_grad():
        X_full_t = torch.from_numpy(X_full).to(device, non_blocking=True)
        recon = model(X_full_t)
        # higher = more anomalous, matching IsolationForest's -decision_function convention
        recon_error = torch.mean((X_full_t - recon) ** 2, dim=1).cpu().numpy()

    meta = meta.copy()
    meta["ae_recon_error"] = recon_error

    auc = roc_auc_score(meta["label"], meta["ae_recon_error"])
    ap = average_precision_score(meta["label"], meta["ae_recon_error"])
    print(f"\nEval vs true label (sanity check only -- teacher trained unsupervised):")
    print(f"  ROC-AUC: {auc:.4f}")
    print(f"  PR-AUC (average precision): {ap:.4f}")

    print("\nPer-source AUC breakdown:")
    for source in meta["source"].unique():
        sub = meta[meta["source"] == source]
        if sub["label"].nunique() < 2:
            print(f"  {source}: skipped (only one class present)")
            continue
        sub_auc = roc_auc_score(sub["label"], sub["ae_recon_error"])
        print(f"  {source}: ROC-AUC {sub_auc:.4f}")

    torch.save(
        {"state_dict": model.state_dict(), "input_dim": X_train.shape[1],
         "latent_dim": LATENT_DIM, "hidden_dims": HIDDEN_DIMS},
        os.path.join(ARTIFACTS_DIR, "autoencoder.pt"),
    )
    joblib.dump(history, os.path.join(ARTIFACTS_DIR, "autoencoder_history.joblib"))
    meta.to_parquet(os.path.join(PROCESSED_DIR, "ae_scores.parquet"), index=False)

    print(f"\nSaved model -> {os.path.join(ARTIFACTS_DIR, 'autoencoder.pt')}")
    print(f"Saved scores -> {os.path.join(PROCESSED_DIR, 'ae_scores.parquet')}")

    metrics = make_plots(history, meta, CONTAMINATION)
    metrics["train_seconds"] = elapsed
    metrics["device"] = str(device)
    joblib.dump(metrics, os.path.join(ARTIFACTS_DIR, "autoencoder_metrics.joblib"))


if __name__ == "__main__":
    main()