"""
SOAIDEATHON-S40 — export_student.py  (SS-8)
Exports the SS-7 winning student model (processed/artifacts/student_model.pt)
to ONNX, then optionally TFLite. Verifies numerical correctness at each step
(PyTorch vs ONNX Runtime vs TFLite all get compared on the same held-out
sample, not just "did the file get written").

ONNX alone already covers both realistic deployment targets discussed for
this project (a backend API via `onnxruntime`, a website frontend via
`onnxruntime-web` — same .onnx file for both, no separate build). TFLite is
included because SS-8 explicitly asks for it, but if `tensorflow`/`onnx2tf`
aren't installed, this script still finishes successfully with a working
.onnx file and just skips the TFLite step with a clear message — it's not a
hard requirement to get a usable exported model out of this run.

Run after train_student.py (needs processed/artifacts/student_model.pt).
"""

import os

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
ARTIFACTS_DIR = os.path.join(PROCESSED_DIR, "artifacts")

ONNX_PATH = os.path.join(ARTIFACTS_DIR, "student_model.onnx")
TFLITE_PATH = os.path.join(ARTIFACTS_DIR, "student_model.tflite")
SAVED_MODEL_DIR = os.path.join(ARTIFACTS_DIR, "student_saved_model")  # intermediate, kept for inspection

ONNX_OPSET = 17
N_VERIFY_SAMPLES = 256  # rows pulled from X_full.parquet to sanity-check exported outputs against PyTorch


class StudentMLP(nn.Module):
    """Must match train_student.py's architecture exactly — loaded state_dict
    won't line up otherwise."""
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


def load_student():
    ckpt = torch.load(os.path.join(ARTIFACTS_DIR, "student_model.pt"), map_location="cpu", weights_only=False)
    model = StudentMLP(ckpt["input_dim"], ckpt["hidden_dims"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Loaded student: arch={ckpt.get('arch_name', '?')}  hidden={ckpt['hidden_dims']}  "
          f"input_dim={ckpt['input_dim']}")
    return model, ckpt


def get_verification_batch(input_dim: int, n: int):
    """Real feature rows (not random noise) for a meaningful numerical
    comparison across export formats — BatchNorm statistics were fit on real
    data, so garbage inputs can hide export bugs that only show up in-distribution."""
    path = os.path.join(PROCESSED_DIR, "X_full.parquet")
    if os.path.exists(path):
        df = pd.read_parquet(path)
        sample = df.sample(n=min(n, len(df)), random_state=42).values.astype(np.float32)
        print(f"Verification batch: {sample.shape[0]} real rows from X_full.parquet")
        return sample
    print("X_full.parquet not found -- falling back to random verification batch "
          "(export will still work, but this is a weaker correctness check).")
    return np.random.randn(n, input_dim).astype(np.float32)


def export_onnx(model: nn.Module, input_dim: int, verify_batch: np.ndarray):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    dummy_input = torch.from_numpy(verify_batch[:1])  # batch size 1 for tracing

    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        input_names=["features"],
        output_names=["logit"],
        dynamic_axes={"features": {0: "batch_size"}, "logit": {0: "batch_size"}},
        opset_version=ONNX_OPSET,
        dynamo=False,  # torch>=2.x defaults to the dynamo-based exporter, which needs onnxscript;
                       # the legacy TorchScript-based exporter handles this simple MLP fine without it
    )
    size_kb = os.path.getsize(ONNX_PATH) / 1024.0
    print(f"Saved ONNX -> {ONNX_PATH} ({size_kb:.1f} KB)")

    # ---- verify: PyTorch vs ONNX Runtime on the same batch ----
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed -- skipping ONNX numerical verification. "
              "Install with `pip install onnxruntime` to confirm the export is correct before trusting it.")
        return size_kb, None

    with torch.no_grad():
        torch_logits = model(torch.from_numpy(verify_batch)).numpy()
        torch_probs = 1 / (1 + np.exp(-torch_logits))

    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    onnx_logits = sess.run(["logit"], {"features": verify_batch})[0]
    onnx_probs = 1 / (1 + np.exp(-onnx_logits))

    max_abs_diff = float(np.max(np.abs(torch_probs - onnx_probs.squeeze())))
    print(f"PyTorch vs ONNX Runtime — max abs diff in predicted probability: {max_abs_diff:.2e} "
          f"({'OK, effectively identical' if max_abs_diff < 1e-4 else 'WARNING: larger than expected, inspect before shipping'})")

    return size_kb, max_abs_diff


def export_tflite(input_dim: int, verify_batch: np.ndarray, torch_model: nn.Module):
    """Best-effort: needs onnx2tf + tensorflow, both heavy optional installs.
    Fails gracefully (prints why, doesn't crash the whole export run) if missing."""
    try:
        import onnx2tf  # noqa: F401
        import tensorflow as tf
    except ImportError as e:
        print(f"\nTFLite export skipped -- missing dependency ({e.name}). "
              f"Install with:\n  pip install onnx2tf tensorflow\n"
              f"then rerun this script -- the .onnx file above is already valid and usable without this step.")
        return None, None

    print("\nConverting ONNX -> TFLite via onnx2tf...")
    os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
    onnx2tf.convert(
        input_onnx_file_path=ONNX_PATH,
        output_folder_path=SAVED_MODEL_DIR,
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=True,
    )

    # onnx2tf drops a handful of *.tflite variants (float32, float16, dynamic-range-quantized)
    # into the output folder -- prefer the plain float32 one for the correctness check below,
    # since quantized variants are EXPECTED to diverge slightly from the float PyTorch model.
    candidates = [f for f in os.listdir(SAVED_MODEL_DIR) if f.endswith(".tflite") and "float32" in f]
    if not candidates:
        candidates = [f for f in os.listdir(SAVED_MODEL_DIR) if f.endswith(".tflite")]
    if not candidates:
        print("onnx2tf ran but produced no .tflite file -- check the output above for errors.")
        return None, None

    src = os.path.join(SAVED_MODEL_DIR, candidates[0])
    import shutil
    shutil.copy(src, TFLITE_PATH)
    size_kb = os.path.getsize(TFLITE_PATH) / 1024.0
    print(f"Saved TFLite -> {TFLITE_PATH} ({size_kb:.1f} KB, from {candidates[0]})")

    # ---- verify: PyTorch vs TFLite on the same batch ----
    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()
    in_details = interpreter.get_input_details()[0]
    out_details = interpreter.get_output_details()[0]

    tflite_probs = []
    with torch.no_grad():
        torch_logits = torch_model(torch.from_numpy(verify_batch)).numpy()
    torch_probs = 1 / (1 + np.exp(-torch_logits))

    for row in verify_batch:
        interpreter.resize_tensor_input(in_details["index"], [1, input_dim])
        interpreter.allocate_tensors()
        interpreter.set_tensor(in_details["index"], row.reshape(1, -1))
        interpreter.invoke()
        out = interpreter.get_tensor(out_details["index"]).squeeze()
        tflite_probs.append(1 / (1 + np.exp(-out)))  # output is a raw logit (same as ONNX/PyTorch) -- always apply sigmoid, no guessing
    tflite_probs = np.array(tflite_probs)

    max_abs_diff = float(np.max(np.abs(torch_probs - tflite_probs)))
    print(f"PyTorch vs TFLite — max abs diff in predicted probability: {max_abs_diff:.2e} "
          f"({'OK' if max_abs_diff < 1e-2 else 'WARNING: larger than expected for a float32 conversion, inspect before shipping'})")

    return size_kb, max_abs_diff


def main():
    model, ckpt = load_student()
    input_dim = ckpt["input_dim"]

    verify_batch = get_verification_batch(input_dim, N_VERIFY_SAMPLES)

    onnx_size_kb, onnx_diff = export_onnx(model, input_dim, verify_batch)
    tflite_size_kb, tflite_diff = export_tflite(input_dim, verify_batch, model)

    summary = {
        "arch_name": ckpt.get("arch_name"), "input_dim": input_dim, "hidden_dims": ckpt["hidden_dims"],
        "onnx_path": ONNX_PATH, "onnx_size_kb": onnx_size_kb, "onnx_max_abs_diff": onnx_diff,
        "tflite_path": TFLITE_PATH if tflite_size_kb else None,
        "tflite_size_kb": tflite_size_kb, "tflite_max_abs_diff": tflite_diff,
    }
    joblib.dump(summary, os.path.join(ARTIFACTS_DIR, "export_summary.joblib"))

    print("\n" + "=" * 60)
    print("EXPORT SUMMARY")
    print("=" * 60)
    print(f"  ONNX:   {ONNX_PATH if onnx_size_kb else 'FAILED'}"
          + (f"  ({onnx_size_kb:.1f} KB)" if onnx_size_kb else ""))
    print(f"  TFLite: {TFLITE_PATH if tflite_size_kb else 'skipped (see message above)'}"
          + (f"  ({tflite_size_kb:.1f} KB)" if tflite_size_kb else ""))
    print(f"\nSaved -> {os.path.join(ARTIFACTS_DIR, 'export_summary.joblib')}")


if __name__ == "__main__":
    main()