"""
SOAIDEATHON-S40 — train_student.py  (SS-7)
Distills the ensemble teacher (ensemble_logreg_meta, SS-6) into a small MLP
student, trained end-to-end on raw features so it needs NO teacher artifacts
at inference (no IsolationForest, no Autoencoder, no scaler chain beyond
what's already baked into X_full.parquet).

Distillation target: ensemble_logreg_meta (chosen over the other SS-6
variants because it's already a calibrated probability — the other variants
are percentile ranks, which only mean something relative to a reference
population and don't translate to a single transaction hitting an API in
isolation).

Loss = alpha * BCE(student_logit, true_label) + (1-alpha) * BCE(student_logit, ensemble_soft_target)
  - hard-label term keeps the student anchored to ground truth
  - soft-target term is the actual distillation signal (teaches the student
    the ensemble's *shape* of uncertainty, not just its final 0/1 calls)

Output is a single sigmoid logit: >0.5 = flagged fraud, and the raw
probability doubles as the on-device risk score (single number covers both
asks: "yes/no" and "risk score").

SIZE SEARCH (the actual point of this script): "as small as possible
without harming inference stats" is an empirical question, not something to
guess up front. This trains a sweep of architectures from tiny to
teacher-sized, evaluates all of them identically, and auto-selects the
smallest one whose ROC-AUC AND PR-AUC are within a small tolerance of the
best architecture in the sweep — rather than picking a size a priori.

Run after combine_ensemble.py (needs ensemble_scores.parquet for soft targets).
"""

import os
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    confusion_matrix, precision_score, recall_score, f1_score, accuracy_score,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
ARTIFACTS_DIR = os.path.join(PROCESSED_DIR, "artifacts")
PLOTS_DIR = os.path.join(PROCESSED_DIR, "plots", "student")

# ---- distillation config ----
SOFT_TARGET_COL = "ensemble_logreg_meta"
ALPHA = 0.5          # weight on hard-label BCE; (1-ALPHA) on soft-target BCE
EPOCHS = 60
BATCH_SIZE = 1024
LR = 1e-3
PATIENCE = 8
SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15       # of the remaining train+val pool

# ---- size sweep: name -> hidden layer widths ----
# "large_64_32" matches the AE's capacity, included as a ceiling reference —
# if a tiny arch matches it, that's the strongest possible evidence to go small.
ARCH_SWEEP = {
    "tiny_4":        [4],
    "tiny_8":        [8],
    "small_16":      [16],
    "small_16_8":    [16, 8],
    "medium_32_16":  [32, 16],
    "large_64_32":   [64, 32],
}

# selection tolerance: a smaller arch is chosen over the sweep-best arch only if
# it's within this much absolute AUC/AP of the best — i.e. "no meaningful cost"
AUC_TOLERANCE = 0.002
AP_TOLERANCE = 0.003


class StudentMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.1)]
            prev = h
        layers += [nn.Linear(prev, 1)]  # raw logit, no sigmoid (use BCEWithLogitsLoss)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")
    else:
        device = torch.device("cpu")
        print("WARNING: training on CPU — see prior GPU install notes if this is unexpected.")
    return device


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_kb(model: nn.Module) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1024.0


def measure_cpu_latency(model: nn.Module, input_dim: int, n_runs: int = 500):
    """Single-sample forward latency on CPU — the realistic deployment case for
    an API endpoint or browser (onnxruntime-web), not the GPU used for training."""
    model_cpu = model.to("cpu").eval()
    x = torch.randn(1, input_dim)
    with torch.no_grad():
        for _ in range(20):  # warmup
            model_cpu(x)
        t0 = time.perf_counter()
        for _ in range(n_runs):
            model_cpu(x)
        elapsed = time.perf_counter() - t0
    return (elapsed / n_runs) * 1000.0  # ms per single-sample inference


def train_one_arch(name, hidden_dims, X_train, y_train, soft_train, X_val, y_val, soft_val, device):
    input_dim = X_train.shape[1]
    model = StudentMLP(input_dim, hidden_dims).to(device)

    pos_weight = torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)], device=device)
    hard_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    soft_criterion = nn.BCEWithLogitsLoss()  # soft target already reflects class rarity via the ensemble

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    train_ds = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train), torch.from_numpy(soft_train)
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val), torch.from_numpy(y_val), torch.from_numpy(soft_val)
    )
    use_gpu = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=use_gpu)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=use_gpu)

    best_val, best_state, bad_epochs = float("inf"), None, 0
    history = {"train_loss": [], "val_loss": []}

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        for xb, yb, sb in train_loader:
            xb, yb, sb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True), sb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = ALPHA * hard_criterion(logits, yb) + (1 - ALPHA) * soft_criterion(logits, sb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb, sb in val_loader:
                xb, yb, sb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True), sb.to(device, non_blocking=True)
                logits = model(xb)
                loss = ALPHA * hard_criterion(logits, yb) + (1 - ALPHA) * soft_criterion(logits, sb)
                val_losses.append(loss.item())

        train_loss, val_loss = float(np.mean(train_losses)), float(np.mean(val_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                break

    elapsed = time.time() - t0
    model.load_state_dict(best_state)
    print(f"  [{name}] hidden={hidden_dims}  params={param_count(model):,}  "
          f"trained {len(history['train_loss'])} epochs in {elapsed:.1f}s  best_val_loss={best_val:.5f}")

    return model, history, elapsed


def evaluate(model, X, y, device):
    model.eval()
    with torch.no_grad():
        logits = model.to(device)(torch.from_numpy(X).to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
    auc = roc_auc_score(y, probs)
    ap = average_precision_score(y, probs)
    return probs, auc, ap


def make_plots(results: dict, best_probs, y_test, contamination: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(PLOTS_DIR, exist_ok=True)
    names = list(results.keys())

    # ---- 1. AUC/AP vs param count (the core "how small can we go" plot) ----
    fig, ax1 = plt.subplots(figsize=(8, 5))
    params = [results[n]["params"] for n in names]
    aucs = [results[n]["test_auc"] for n in names]
    aps = [results[n]["test_ap"] for n in names]
    order = np.argsort(params)
    params_s = [params[i] for i in order]
    names_s = [names[i] for i in order]
    aucs_s = [aucs[i] for i in order]
    aps_s = [aps[i] for i in order]

    ax1.plot(params_s, aucs_s, "o-", color="steelblue", label="ROC-AUC")
    ax1.set_xscale("log")
    ax1.set_xlabel("parameter count (log scale)")
    ax1.set_ylabel("ROC-AUC", color="steelblue")
    for x, y_, n in zip(params_s, aucs_s, names_s):
        ax1.annotate(n, (x, y_), fontsize=7, textcoords="offset points", xytext=(0, 8))
    ax2 = ax1.twinx()
    ax2.plot(params_s, aps_s, "s--", color="darkorange", label="PR-AUC")
    ax2.set_ylabel("PR-AUC", color="darkorange")
    ax1.set_title("Student capacity sweep: size vs performance")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "size_vs_performance.png"), dpi=150)
    plt.close(fig)

    # ---- 2. latency vs param count ----
    fig, ax = plt.subplots(figsize=(7, 5))
    lat = [results[n]["latency_ms"] for n in names_s]
    ax.plot(params_s, lat, "o-", color="seagreen")
    ax.set_xscale("log")
    for x, y_, n in zip(params_s, lat, names_s):
        ax.annotate(n, (x, y_), fontsize=7, textcoords="offset points", xytext=(0, 8))
    ax.set_xlabel("parameter count (log scale)"); ax.set_ylabel("CPU latency (ms/sample)")
    ax.set_title("Student capacity sweep: size vs single-sample CPU latency")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "size_vs_latency.png"), dpi=150)
    plt.close(fig)

    # ---- 3. loss curves, all archs overlaid ----
    fig, ax = plt.subplots(figsize=(7, 5))
    for n in names:
        ax.plot(results[n]["history"]["val_loss"], label=n, linewidth=1)
    ax.set_xlabel("epoch"); ax.set_ylabel("val loss (distillation)")
    ax.set_title("Validation loss by architecture"); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "loss_curves_all_archs.png"), dpi=150)
    plt.close(fig)

    # ---- 4. ROC + PR for winner only ----
    fpr, tpr, _ = roc_curve(y_test, best_probs)
    prec, rec, _ = precision_recall_curve(y_test, best_probs)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].plot(fpr, tpr); axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[0].set_title("ROC — selected student"); axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[1].plot(rec, prec)
    axes[1].set_title("Precision-Recall — selected student"); axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "winner_roc_pr.png"), dpi=150)
    plt.close(fig)

    # ---- 5. confusion matrix for winner ----
    threshold = np.quantile(best_probs, 1 - contamination)
    y_pred = (best_probs >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred normal", "pred fraud"]); ax.set_yticklabels(["true normal", "true fraud"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(f"Winner confusion matrix @ {contamination:.1%}\nacc={acc:.4f} f1={f1:.4f}")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "winner_confusion_matrix.png"), dpi=150)
    plt.close(fig)

    print(f"\nSaved 5 plots -> {PLOTS_DIR}")
    print("  size_vs_performance.png, size_vs_latency.png, loss_curves_all_archs.png,")
    print("  winner_roc_pr.png, winner_confusion_matrix.png")


def select_smallest_viable(results: dict) -> str:
    """Pick the smallest-param arch whose test AUC/AP are within tolerance of the
    sweep's best AUC/AP. This is the actual 'small as possible without harming
    stats' decision, made empirically rather than assumed."""
    best_auc = max(r["test_auc"] for r in results.values())
    best_ap = max(r["test_ap"] for r in results.values())

    viable = [
        name for name, r in results.items()
        if r["test_auc"] >= best_auc - AUC_TOLERANCE and r["test_ap"] >= best_ap - AP_TOLERANCE
    ]
    winner = min(viable, key=lambda n: results[n]["params"])
    return winner, best_auc, best_ap, viable


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    device = get_device()

    print("Loading features + ensemble soft targets...")
    X_full = pd.read_parquet(os.path.join(PROCESSED_DIR, "X_full.parquet"))
    meta = pd.read_parquet(os.path.join(PROCESSED_DIR, "meta_full.parquet"))
    ensemble = pd.read_parquet(os.path.join(PROCESSED_DIR, "ensemble_scores.parquet"))[
        ["record_id", SOFT_TARGET_COL]
    ]

    df = meta.merge(ensemble, on="record_id", how="inner")
    assert len(df) == len(X_full), "meta/ensemble merge dropped rows — check record_id alignment upstream"
    X = X_full.values.astype(np.float32)
    y = df["label"].values.astype(np.float32)
    soft = df[SOFT_TARGET_COL].values.astype(np.float32)

    print(f"  X: {X.shape}  fraud rate: {y.mean():.4%}")

    # stratified train/val/test split so tiny fraud class stays represented in all three
    X_trainval, X_test, y_trainval, y_test, soft_trainval, soft_test = train_test_split(
        X, y, soft, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val, soft_train, soft_val = train_test_split(
        X_trainval, y_trainval, soft_trainval, test_size=VAL_SIZE, stratify=y_trainval, random_state=SEED
    )
    print(f"  train: {X_train.shape}  val: {X_val.shape}  test: {X_test.shape}")

    print(f"\nRunning capacity sweep ({len(ARCH_SWEEP)} architectures)...")
    results = {}
    for name, hidden_dims in ARCH_SWEEP.items():
        model, history, elapsed = train_one_arch(
            name, hidden_dims, X_train, y_train, soft_train, X_val, y_val, soft_val, device
        )
        probs, auc, ap = evaluate(model, X_test, y_test, device)
        latency = measure_cpu_latency(model, X.shape[1])
        results[name] = {
            "hidden_dims": hidden_dims, "params": param_count(model), "size_kb": model_size_kb(model),
            "test_auc": auc, "test_ap": ap, "latency_ms": latency,
            "history": history, "train_seconds": elapsed, "state_dict": model.state_dict(),
        }
        print(f"           test_auc={auc:.4f}  test_ap={ap:.4f}  size={results[name]['size_kb']:.1f}KB  "
              f"cpu_latency={latency:.4f}ms")

    print("\n" + "=" * 90)
    print(f"{'arch':<15} {'params':>8} {'size_kb':>9} {'test_auc':>10} {'test_ap':>9} {'latency_ms':>11}")
    for name, r in results.items():
        print(f"{name:<15} {r['params']:>8,} {r['size_kb']:>9.1f} {r['test_auc']:>10.4f} "
              f"{r['test_ap']:>9.4f} {r['latency_ms']:>11.4f}")
    print("=" * 90)

    winner, best_auc, best_ap, viable = select_smallest_viable(results)
    w = results[winner]
    print(f"\nSweep best:  ROC-AUC={best_auc:.4f}  PR-AUC={best_ap:.4f}")
    print(f"Within tolerance (AUC>=-{AUC_TOLERANCE}, AP>=-{AP_TOLERANCE}): {viable}")
    print(f"SELECTED (smallest viable): {winner}  hidden={w['hidden_dims']}  "
          f"params={w['params']:,}  size={w['size_kb']:.1f}KB")
    print(f"  test_auc={w['test_auc']:.4f} (vs best {best_auc:.4f}, delta={best_auc - w['test_auc']:+.4f})")
    print(f"  test_ap={w['test_ap']:.4f} (vs best {best_ap:.4f}, delta={best_ap - w['test_ap']:+.4f})")
    print(f"  cpu_latency={w['latency_ms']:.4f} ms/sample")

    # save winner
    torch.save(
        {"state_dict": w["state_dict"], "input_dim": X.shape[1], "hidden_dims": w["hidden_dims"],
         "arch_name": winner, "soft_target_col": SOFT_TARGET_COL, "alpha": ALPHA},
        os.path.join(ARTIFACTS_DIR, "student_model.pt"),
    )

    # save full sweep comparison (minus state_dicts — those aren't needed except for the winner)
    sweep_summary = {
        name: {k: v for k, v in r.items() if k not in ("state_dict", "history")}
        for name, r in results.items()
    }
    joblib.dump(sweep_summary, os.path.join(ARTIFACTS_DIR, "student_sweep_summary.joblib"))
    joblib.dump(
        {"winner": winner, "best_auc": best_auc, "best_ap": best_ap, "viable": viable},
        os.path.join(ARTIFACTS_DIR, "student_selection.joblib"),
    )

    print(f"\nSaved winning model -> {os.path.join(ARTIFACTS_DIR, 'student_model.pt')}")
    print(f"Saved sweep summary -> {os.path.join(ARTIFACTS_DIR, 'student_sweep_summary.joblib')}")

    winner_model = StudentMLP(X.shape[1], w["hidden_dims"])
    winner_model.load_state_dict(w["state_dict"])
    probs, _, _ = evaluate(winner_model, X_test, y_test, device)
    make_plots(results, probs, y_test, contamination=0.005)


if __name__ == "__main__":
    main()
