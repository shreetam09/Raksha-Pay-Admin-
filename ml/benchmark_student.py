"""
SOAIDEATHON-S40 — benchmark_student.py  (SS-9)
Benchmarks the SS-7/SS-8 student model across all three formats it now
exists in: the original PyTorch checkpoint, the ONNX export, and the TFLite
export. Measures file size and inference latency (single-sample AND across
a range of batch sizes) for each, on CPU -- the realistic deployment
environment for an API endpoint or browser, not the GPU used for training.

TFLite benchmarking is skipped gracefully if student_model.tflite doesn't
exist (e.g. you decided not to chase the onnx2tf dependency chain) -- the
PyTorch/ONNX comparison still runs and is the one that matters most given
the stated API/website deployment target.

Run after export_student.py.
"""

import os
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
ARTIFACTS_DIR = os.path.join(PROCESSED_DIR, "artifacts")
PLOTS_DIR = os.path.join(PROCESSED_DIR, "plots", "benchmark")

PT_PATH = os.path.join(ARTIFACTS_DIR, "student_model.pt")
ONNX_PATH = os.path.join(ARTIFACTS_DIR, "student_model.onnx")
TFLITE_PATH = os.path.join(ARTIFACTS_DIR, "student_model.tflite")

BATCH_SIZES = [1, 8, 32, 128, 512, 2048]
N_WARMUP = 20
N_REPEATS = 200   # timed repeats per (format, batch_size) combo


class StudentMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.1)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def get_test_data(input_dim, n=2048 * 2):
    path = os.path.join(PROCESSED_DIR, "X_full.parquet")
    if os.path.exists(path):
        df = pd.read_parquet(path)
        return df.sample(n=min(n, len(df)), random_state=42).values.astype(np.float32)
    return np.random.randn(n, input_dim).astype(np.float32)


def time_calls(fn, n_repeats):
    """Runs fn() n_repeats times, returns array of per-call elapsed seconds.
    fn should already be warmed up before this is called."""
    times = np.empty(n_repeats)
    for i in range(n_repeats):
        t0 = time.perf_counter()
        fn()
        times[i] = time.perf_counter() - t0
    return times


def benchmark_pytorch(input_dim, hidden_dims, data):
    ckpt = torch.load(PT_PATH, map_location="cpu", weights_only=False)
    model = StudentMLP(input_dim, hidden_dims)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    size_kb = os.path.getsize(PT_PATH) / 1024.0
    results = {}
    with torch.no_grad():
        for bs in BATCH_SIZES:
            batch = torch.from_numpy(data[:bs])
            for _ in range(N_WARMUP):
                model(batch)
            times = time_calls(lambda: model(batch), N_REPEATS)
            results[bs] = times
    return size_kb, results


def benchmark_onnx(data):
    if not os.path.exists(ONNX_PATH):
        print("student_model.onnx not found -- skipping ONNX benchmark.")
        return None, None
    import onnxruntime as ort

    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    size_kb = os.path.getsize(ONNX_PATH) / 1024.0

    results = {}
    for bs in BATCH_SIZES:
        batch = data[:bs]
        for _ in range(N_WARMUP):
            sess.run(None, {input_name: batch})
        times = time_calls(lambda: sess.run(None, {input_name: batch}), N_REPEATS)
        results[bs] = times
    return size_kb, results


def benchmark_tflite(data, input_dim):
    if not os.path.exists(TFLITE_PATH):
        print("student_model.tflite not found -- skipping TFLite benchmark "
              "(fine if you decided ONNX alone covers your deployment target).")
        return None, None
    import tensorflow as tf

    size_kb = os.path.getsize(TFLITE_PATH) / 1024.0
    results = {}
    for bs in BATCH_SIZES:
        interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
        in_details = interpreter.get_input_details()[0]
        out_details = interpreter.get_output_details()[0]
        interpreter.resize_tensor_input(in_details["index"], [bs, input_dim])
        interpreter.allocate_tensors()
        batch = data[:bs]

        def run():
            interpreter.set_tensor(in_details["index"], batch)
            interpreter.invoke()
            interpreter.get_tensor(out_details["index"])

        for _ in range(N_WARMUP):
            run()
        times = time_calls(run, N_REPEATS)
        results[bs] = times
    return size_kb, results


def summarize(times_by_batch):
    """times_by_batch: {batch_size: array of per-call seconds} -> per-batch stats dict."""
    summary = {}
    for bs, times_s in times_by_batch.items():
        times_ms = times_s * 1000.0
        summary[bs] = {
            "mean_ms": float(np.mean(times_ms)),
            "std_ms": float(np.std(times_ms)),
            "p50_ms": float(np.percentile(times_ms, 50)),
            "p95_ms": float(np.percentile(times_ms, 95)),
            "p99_ms": float(np.percentile(times_ms, 99)),
            "per_sample_ms": float(np.mean(times_ms) / bs),
            "throughput_per_sec": float(bs / (np.mean(times_ms) / 1000.0)),
        }
    return summary


def make_plots(all_summaries: dict, all_sizes: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(PLOTS_DIR, exist_ok=True)
    formats = [f for f in all_summaries if all_summaries[f] is not None]
    colors = {"pytorch": "steelblue", "onnx": "darkorange", "tflite": "seagreen"}

    # ---- 1. model size comparison ----
    fig, ax = plt.subplots(figsize=(6, 4))
    sizes = [all_sizes[f] for f in formats]
    ax.bar(formats, sizes, color=[colors.get(f, "gray") for f in formats])
    for i, s in enumerate(sizes):
        ax.text(i, s + max(sizes) * 0.02, f"{s:.1f} KB", ha="center")
    ax.set_ylabel("size (KB)")
    ax.set_title("Model size by export format")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "model_size_comparison.png"), dpi=150)
    plt.close(fig)

    # ---- 2. per-sample latency vs batch size (log-log) ----
    fig, ax = plt.subplots(figsize=(7, 5))
    for f in formats:
        bss = sorted(all_summaries[f].keys())
        per_sample = [all_summaries[f][bs]["per_sample_ms"] for bs in bss]
        ax.plot(bss, per_sample, "o-", label=f, color=colors.get(f, "gray"))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("batch size (log)"); ax.set_ylabel("latency per sample (ms, log)")
    ax.set_title("Per-sample latency vs batch size")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "latency_vs_batch_size.png"), dpi=150)
    plt.close(fig)

    # ---- 3. throughput vs batch size ----
    fig, ax = plt.subplots(figsize=(7, 5))
    for f in formats:
        bss = sorted(all_summaries[f].keys())
        tput = [all_summaries[f][bs]["throughput_per_sec"] for bs in bss]
        ax.plot(bss, tput, "o-", label=f, color=colors.get(f, "gray"))
    ax.set_xscale("log")
    ax.set_xlabel("batch size (log)"); ax.set_ylabel("throughput (samples/sec)")
    ax.set_title("Throughput vs batch size")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "throughput_vs_batch_size.png"), dpi=150)
    plt.close(fig)

    # ---- 4. single-sample latency distribution (the realistic API-call case) ----
    fig, ax = plt.subplots(figsize=(7, 5))
    box_data = [all_summaries[f][1]["mean_ms"] for f in formats]  # placeholder if raw times not kept
    # use p50/p95/p99 as a compact distribution view instead of a full boxplot (raw arrays not retained post-summarize)
    x = np.arange(len(formats))
    width = 0.25
    p50 = [all_summaries[f][1]["p50_ms"] for f in formats]
    p95 = [all_summaries[f][1]["p95_ms"] for f in formats]
    p99 = [all_summaries[f][1]["p99_ms"] for f in formats]
    ax.bar(x - width, p50, width, label="p50")
    ax.bar(x, p95, width, label="p95")
    ax.bar(x + width, p99, width, label="p99")
    ax.set_xticks(x); ax.set_xticklabels(formats)
    ax.set_ylabel("latency (ms)")
    ax.set_title("Single-sample (batch=1) latency percentiles")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "single_sample_latency_percentiles.png"), dpi=150)
    plt.close(fig)

    # ---- 5. combined dashboard ----
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(formats, sizes, color=[colors.get(f, "gray") for f in formats])
    for i, s in enumerate(sizes):
        ax1.text(i, s + max(sizes) * 0.02, f"{s:.1f} KB", ha="center")
    ax1.set_title("Model size"); ax1.set_ylabel("KB")

    ax2 = fig.add_subplot(gs[0, 1])
    for f in formats:
        bss = sorted(all_summaries[f].keys())
        per_sample = [all_summaries[f][bs]["per_sample_ms"] for bs in bss]
        ax2.plot(bss, per_sample, "o-", label=f, color=colors.get(f, "gray"))
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_title("Per-sample latency vs batch size"); ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(gs[1, 0])
    for f in formats:
        bss = sorted(all_summaries[f].keys())
        tput = [all_summaries[f][bs]["throughput_per_sec"] for bs in bss]
        ax3.plot(bss, tput, "o-", label=f, color=colors.get(f, "gray"))
    ax3.set_xscale("log")
    ax3.set_title("Throughput vs batch size"); ax3.legend(fontsize=8)

    ax4 = fig.add_subplot(gs[1, 1])
    x = np.arange(len(formats))
    ax4.bar(x - width, p50, width, label="p50")
    ax4.bar(x, p95, width, label="p95")
    ax4.bar(x + width, p99, width, label="p99")
    ax4.set_xticks(x); ax4.set_xticklabels(formats)
    ax4.set_title("Batch=1 latency percentiles"); ax4.legend(fontsize=8)

    fig.suptitle("Student model benchmark — size, latency, throughput", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "combined_dashboard.png"), dpi=150)
    plt.close(fig)

    print(f"\nSaved 5 plots -> {PLOTS_DIR}")
    print("  model_size_comparison.png, latency_vs_batch_size.png, throughput_vs_batch_size.png,")
    print("  single_sample_latency_percentiles.png, combined_dashboard.png")


def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    ckpt = torch.load(PT_PATH, map_location="cpu", weights_only=False)
    input_dim, hidden_dims = ckpt["input_dim"], ckpt["hidden_dims"]
    print(f"Model: arch={ckpt.get('arch_name', '?')}  hidden={hidden_dims}  input_dim={input_dim}")

    data = get_test_data(input_dim, n=max(BATCH_SIZES) * 2)
    print(f"Benchmark data: {data.shape}\n")

    all_sizes, all_summaries = {}, {}

    print("Benchmarking PyTorch...")
    pt_size, pt_times = benchmark_pytorch(input_dim, hidden_dims, data)
    all_sizes["pytorch"], all_summaries["pytorch"] = pt_size, summarize(pt_times)

    print("Benchmarking ONNX...")
    onnx_size, onnx_times = benchmark_onnx(data)
    if onnx_times is not None:
        all_sizes["onnx"], all_summaries["onnx"] = onnx_size, summarize(onnx_times)

    print("Benchmarking TFLite...")
    tflite_size, tflite_times = benchmark_tflite(data, input_dim)
    if tflite_times is not None:
        all_sizes["tflite"], all_summaries["tflite"] = tflite_size, summarize(tflite_times)

    print("\n" + "=" * 100)
    print(f"{'format':<10} {'size_kb':>9} {'batch':>7} {'per_sample_ms':>15} {'p95_ms':>10} {'throughput/s':>13}")
    for fmt in all_summaries:
        for bs in BATCH_SIZES:
            s = all_summaries[fmt][bs]
            print(f"{fmt:<10} {all_sizes[fmt]:>9.1f} {bs:>7} {s['per_sample_ms']:>15.5f} "
                  f"{s['p95_ms']:>10.4f} {s['throughput_per_sec']:>13.0f}")
    print("=" * 100)

    print("\nSingle-sample (batch=1) summary:")
    for fmt in all_summaries:
        s = all_summaries[fmt][1]
        print(f"  {fmt:<10} mean={s['mean_ms']:.4f}ms  p50={s['p50_ms']:.4f}ms  "
              f"p95={s['p95_ms']:.4f}ms  p99={s['p99_ms']:.4f}ms  size={all_sizes[fmt]:.1f}KB")

    results = {"sizes_kb": all_sizes, "latency_summary": all_summaries, "batch_sizes": BATCH_SIZES}
    joblib.dump(results, os.path.join(ARTIFACTS_DIR, "benchmark_results.joblib"))
    print(f"\nSaved -> {os.path.join(ARTIFACTS_DIR, 'benchmark_results.joblib')}")

    make_plots(all_summaries, all_sizes)


if __name__ == "__main__":
    main()
