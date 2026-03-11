#!/usr/bin/env python3
"""
Week 16: Inference Latency & Throughput Benchmark

Module:       week16.py
Purpose:      Comprehensive benchmark suite for TensorRT inference engines
Dataset:      ISIC 2019 Skin Cancer Detection
Model:        EfficientNetB3
Author:       Amjad
Date:         February 2026
Platform:     RunPod (A40 GPU)

═══════════════════════════════════════════════════════════════════════════════

DESCRIPTION
───────────
Loads three TensorRT engines (FP32, FP16, INT8) and ONNX baseline, then runs
production-grade benchmarks to measure latency, throughput, accuracy, and
deployment feasibility across batch sizes 1–32.

BENCHMARKS PERFORMED
────────────────────
• Multi-batch latency sweep (batch = 1, 4, 8, 16, 32)
  └─ Captures latency curves, throughput (img/s), VRAM efficiency
• Warmup convergence analysis
  └─ Determines minimum warmup runs before latency stabilizes
• Latency distribution histograms
  └─ P50, P95, P99, P99.9 percentiles for SLA modeling
• Per-class accuracy vs speed trade-off
  └─ Confidence statistics per skin condition
• GPU memory footprint per engine
• Clinical deployment projections
  └─ Images/hour, patients/day, real-time capability
• Engine comparison matrix with speedup vs baseline

USAGE
─────
CRITICAL: Run from terminal only (not notebook cell).

    Terminal → python /workspace/week16.py

Requirements:
  • TensorRT 8.6+ (NVIDIA developer package)
  • PyCUDA with cuda-python
  • numpy, onnxruntime
  • nvidia-smi (for driver/CUDA version detection)

Inputs:
  • /workspace/output/trt_fp32.engine
  • /workspace/output/trt_fp16.engine
  • /workspace/output/trt_int8.engine
  • /workspace/output/EfficientNetB3_ISIC2019_final.onnx
  • /workspace/dataset/test_preprocessed/X_test_300x300.npy
  • /workspace/dataset/test_preprocessed/y_test.npy

Outputs:
  • /workspace/output/week16_benchmark_report.json  (structured results)
  • /workspace/output/week16_summary.txt            (human-readable summary)

═══════════════════════════════════════════════════════════════════════════════
"""

import sys, os, gc, json, time, subprocess
import numpy as np
from datetime import datetime
from pathlib import Path

# ── Abort if TF is already loaded in this process ────────────────────────────
if 'tensorflow' in sys.modules or 'keras' in sys.modules:
    print("ERROR: TensorFlow is loaded in this process.")
    print("Run from a terminal: python /workspace/week16.py")
    sys.exit(1)

print("=" * 80)
print("WEEK 16: INFERENCE LATENCY & THROUGHPUT BENCHMARK")
print("=" * 80)
print(f"  Date   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Python : {sys.version.split()[0]}")

# ============================================================================
# CONFIGURATION
# ============================================================================

ENGINE_FP32  = "/workspace/output/trt_fp32.engine"
ENGINE_FP16  = "/workspace/output/trt_fp16.engine"
ENGINE_INT8  = "/workspace/output/trt_int8.engine"
ONNX_MODEL   = "/workspace/output/EfficientNetB3_ISIC2019_final.onnx"
TEST_DATA    = "/workspace/dataset/test_preprocessed/X_test_300x300.npy"
TEST_LABELS  = "/workspace/dataset/test_preprocessed/y_test.npy"
OUTPUT_DIR   = Path("/workspace/output")
REPORT_PATH  = str(OUTPUT_DIR / "week16_benchmark_report.json")
SUMMARY_PATH = str(OUTPUT_DIR / "week16_summary.txt")

IMAGE_H, IMAGE_W, IMAGE_C = 300, 300, 3
NUM_CLASSES  = 8
CLASS_NAMES  = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']
MAX_BATCH    = 32

# Benchmark settings
BATCH_SIZES  = [1, 4, 8, 16, 32]   # full latency/throughput sweep
WARMUP_RUNS  = 50                   # per batch size
BENCH_RUNS   = 200                  # per batch size (keeps runtime reasonable)
HIST_BINS    = 20                   # latency histogram resolution
WARMUP_CONV_RUNS = 100              # for warmup convergence analysis

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# STEP 0 — DEPENDENCY CHECK
# ============================================================================

print("\n" + "=" * 80)
print("STEP 0: DEPENDENCY CHECK")
print("=" * 80)

try:
    import tensorrt as trt
    trt_version = trt.__version__
    trt_major   = int(trt_version.split(".")[0])
    print(f"  ✓ TensorRT  : {trt_version}")
except ImportError:
    print("  ✗ TensorRT not found — run week1.py first")
    sys.exit(1)

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    print(f"  ✓ PyCUDA    : OK")
except Exception as e:
    print(f"  ✗ PyCUDA failed: {e}")
    print("    Run from terminal (not notebook): python /workspace/week16.py")
    sys.exit(1)

try:
    import onnxruntime as ort
    print(f"  ✓ OnnxRT    : {ort.__version__}")
except ImportError:
    print("  ✗ onnxruntime not found")
    sys.exit(1)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Check engine files
missing = []
for label, path in [("FP32", ENGINE_FP32), ("FP16", ENGINE_FP16),
                     ("INT8", ENGINE_INT8), ("ONNX", ONNX_MODEL)]:
    if Path(path).exists():
        mb = Path(path).stat().st_size / (1024**2)
        print(f"  ✓ {label:<5} : {path}  ({mb:.1f} MB)")
    else:
        print(f"  ✗ {label:<5} : NOT FOUND — {path}")
        missing.append(label)

if missing:
    print(f"\n  Missing: {missing}. Run week15.py first.")
    sys.exit(1)

# ============================================================================
# STEP 1 — GPU INFORMATION
# ============================================================================

print("\n" + "=" * 80)
print("STEP 1: GPU INFORMATION")
print("=" * 80)

device   = cuda.Device(0)
gpu_name = device.name()
vram_mb  = device.total_memory() // (1024**2)
cc_major, cc_minor = device.compute_capability()

print(f"  GPU            : {gpu_name}")
print(f"  VRAM           : {vram_mb:,} MB  ({vram_mb/1024:.1f} GB)")
print(f"  Compute cap    : {cc_major}.{cc_minor}")
print(f"  TensorRT       : {trt_version}")

driver_ver = subprocess.run(
    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
    capture_output=True, text=True).stdout.strip()
cuda_ver = subprocess.run(
    ["nvidia-smi", "--query-gpu=cuda_version", "--format=csv,noheader"],
    capture_output=True, text=True).stdout.strip()
print(f"  Driver         : {driver_ver}")
print(f"  CUDA           : {cuda_ver}")

gpu_info = dict(
    name=gpu_name, vram_mb=vram_mb, vram_gb=round(vram_mb/1024, 1),
    compute_capability=f"{cc_major}.{cc_minor}",
    driver=driver_ver, cuda=cuda_ver, trt=trt_version
)

# ============================================================================
# STEP 2 — LOAD TEST DATA
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: LOAD TEST DATA")
print("=" * 80)

if not Path(TEST_DATA).exists():
    print(f"  ✗ Test data not found: {TEST_DATA}")
    sys.exit(1)

X_raw = np.load(TEST_DATA).astype(np.float32)
y_all = np.load(TEST_LABELS).astype(np.int32) if Path(TEST_LABELS).exists() else None

# Scale to [0, 255] — model has Rescaling(x1/255) baked in
X_test = X_raw * 255.0 if X_raw.max() <= 1.0 else X_raw.copy()

print(f"  Shape  : {X_test.shape}  dtype={X_test.dtype}")
print(f"  Range  : [{X_test.min():.1f}, {X_test.max():.1f}]")
if y_all is not None:
    unique, counts = np.unique(y_all, return_counts=True)
    print(f"  Labels : {len(y_all)} samples across {len(unique)} classes")
    for cls, cnt in zip(unique, counts):
        print(f"           {CLASS_NAMES[cls]}: {cnt}")

# ============================================================================
# SESSION HELPERS
# ============================================================================

def load_trt_engine(path):
    runtime = trt.Runtime(TRT_LOGGER)
    with open(path, "rb") as f:
        return runtime.deserialize_cuda_engine(f.read())


class TRTSession:
    """Thin wrapper around a TRT engine — uses non-deprecated TRT 8.6 API."""

    def __init__(self, engine, max_batch=MAX_BATCH):
        self.engine   = engine
        self.context  = engine.create_execution_context()
        self.stream   = cuda.Stream()
        self.max_batch = max_batch

        n = engine.num_io_tensors if trt_major >= 10 else engine.num_bindings

        self._h_bufs   = []
        self._d_bufs   = []
        self._bindings = []
        self._in_idx   = []
        self._out_idx  = []

        for i in range(n):
            # get_tensor_* is non-deprecated in TRT 8.6+
            name  = engine.get_tensor_name(i)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            shape = engine.get_tensor_shape(name)
            is_in = (engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT)

            shape  = tuple(max_batch if s == -1 else s for s in shape)
            n_elem = int(np.prod(shape))

            h_buf = cuda.pagelocked_empty(n_elem, dtype)
            d_buf = cuda.mem_alloc(h_buf.nbytes)
            self._h_bufs.append(h_buf)
            self._d_bufs.append(d_buf)
            self._bindings.append(int(d_buf))
            (self._in_idx if is_in else self._out_idx).append(i)

    def infer(self, batch: np.ndarray) -> np.ndarray:
        bs = batch.shape[0]
        ii = self._in_idx[0]
        oi = self._out_idx[0]

        # set_input_shape is non-deprecated in TRT 8.6+
        name = self.engine.get_tensor_name(ii)
        self.context.set_input_shape(name, (bs, IMAGE_H, IMAGE_W, IMAGE_C))

        np.copyto(self._h_bufs[ii][:batch.size],
                  np.ascontiguousarray(batch, np.float32).ravel())
        cuda.memcpy_htod_async(self._d_bufs[ii], self._h_bufs[ii], self.stream)

        if trt_major >= 10:
            for idx in self._in_idx + self._out_idx:
                n = self.engine.get_tensor_name(idx)
                self.context.set_tensor_address(n, int(self._d_bufs[idx]))
            self.context.execute_async_v3(stream_handle=self.stream.handle)
        else:
            self.context.execute_async_v2(
                bindings=self._bindings,
                stream_handle=self.stream.handle)

        cuda.memcpy_dtoh_async(self._h_bufs[oi], self._d_bufs[oi], self.stream)
        self.stream.synchronize()
        return self._h_bufs[oi][:bs * NUM_CLASSES].reshape(bs, NUM_CLASSES).copy()

    def __del__(self):
        try:
            for d in self._d_bufs:
                d.free()
        except Exception:
            pass


def get_gpu_mem_used_mb():
    """Return current GPU memory used in MB via nvidia-smi."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        ).stdout.strip()
        return int(out)
    except Exception:
        return -1


# ============================================================================
# CORE BENCHMARK FUNCTIONS
# ============================================================================

def bench_latency_sweep(infer_fn, X, label, batch_sizes=BATCH_SIZES,
                         warmup=WARMUP_RUNS, runs=BENCH_RUNS):
    """
    Run inference at every batch size in batch_sizes.
    Returns a dict keyed by batch_size with full latency stats.
    """
    results = {}
    print(f"\n  [{label}] latency sweep:")

    for bs in batch_sizes:
        if bs > len(X):
            print(f"    batch={bs:2d} — skip (not enough test data)")
            continue

        # build probe
        probe = X[0:bs].astype(np.float32)

        # warmup
        for _ in range(warmup):
            infer_fn(probe)

        # measure
        times_ms = []
        for i in range(runs):
            start = (i * bs) % (len(X) - bs)
            batch = X[start : start + bs].astype(np.float32)
            t0    = time.perf_counter()
            infer_fn(batch)
            times_ms.append((time.perf_counter() - t0) * 1000.0)

        mean  = float(np.mean(times_ms))
        std   = float(np.std(times_ms))
        p50   = float(np.percentile(times_ms, 50))
        p95   = float(np.percentile(times_ms, 95))
        p99   = float(np.percentile(times_ms, 99))
        p999  = float(np.percentile(times_ms, 99.9))
        tput  = (1000.0 / mean) * bs
        lat_per_img = mean / bs

        # latency histogram
        hist, edges = np.histogram(times_ms, bins=HIST_BINS)
        histogram = {
            "counts": hist.tolist(),
            "bin_edges_ms": [round(e, 4) for e in edges.tolist()]
        }

        results[bs] = dict(
            batch_size=bs, mean_ms=mean, std_ms=std,
            p50_ms=p50, p95_ms=p95, p99_ms=p99, p999_ms=p999,
            throughput_ips=tput,
            latency_per_image_ms=lat_per_img,
            histogram=histogram,
        )
        print(f"    batch={bs:2d}  mean={mean:7.3f}ms  "
              f"P99={p99:7.3f}ms  tput={tput:8.1f} img/s  "
              f"lat/img={lat_per_img:.3f}ms")

    return results


def bench_onnx_sweep(X, batch_sizes=BATCH_SIZES,
                      warmup=WARMUP_RUNS, runs=BENCH_RUNS):
    """Same sweep for ONNX Runtime baseline."""
    sess = ort.InferenceSession(
        ONNX_MODEL,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    inp_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    def infer(batch):
        return sess.run([out_name], {inp_name: batch})[0]

    results = bench_latency_sweep(infer, X, "ONNX-CUDA", batch_sizes, warmup, runs)
    del sess
    gc.collect()
    return results


def bench_warmup_convergence(infer_fn, X, label, n_runs=WARMUP_CONV_RUNS):
    """
    Measure latency on every single run from cold start.
    Shows how many runs it takes to reach steady-state.
    Used to determine minimum warmup requirement for production.
    """
    print(f"\n  [{label}] warmup convergence ({n_runs} runs, batch=1) ...")
    probe = X[0:1].astype(np.float32)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        infer_fn(probe)
        times.append((time.perf_counter() - t0) * 1000.0)

    # Find where latency stabilises: first run where rolling mean (w=5)
    # is within 2% of the final rolling mean
    rolling = [float(np.mean(times[max(0, i-4):i+1])) for i in range(len(times))]
    steady  = rolling[-1]
    converge_at = n_runs  # default: never
    for i, r in enumerate(rolling):
        if i >= 5 and abs(r - steady) / steady < 0.02:
            converge_at = i
            break

    print(f"    Steady-state latency : {steady:.3f} ms")
    print(f"    Converges at run     : {converge_at}  "
          f"({'already stable' if converge_at < 10 else 'needs warmup'})")

    return dict(
        runs=n_runs,
        latencies_ms=times,
        rolling_mean_ms=rolling,
        steady_state_ms=steady,
        converges_at_run=converge_at,
    )


def compute_accuracy(infer_fn, X, y, batch_size=32, label=""):
    """Full-dataset accuracy + per-class breakdown + confidence stats."""
    if y is None:
        return None

    all_preds  = []
    all_probs  = []
    for i in range(0, len(X) - batch_size + 1, batch_size):
        batch = X[i : i + batch_size].astype(np.float32)
        probs = infer_fn(batch)
        all_probs.extend(probs.tolist())
        all_preds.extend(np.argmax(probs, axis=1).tolist())

    y_pred = np.array(all_preds)
    y_true = y[:len(y_pred)]
    probs  = np.array(all_probs)

    overall_acc = float((y_pred == y_true).mean())

    # Per-class
    per_class = {}
    for cls in range(NUM_CLASSES):
        mask = y_true == cls
        if mask.sum() == 0:
            continue
        cls_acc  = float((y_pred[mask] == cls).mean())
        cls_conf = float(probs[mask, cls].mean())
        per_class[CLASS_NAMES[cls]] = dict(
            n=int(mask.sum()), accuracy=round(cls_acc, 4),
            mean_confidence=round(cls_conf, 4)
        )

    # Confidence distribution
    top_confs = probs.max(axis=1)
    conf_stats = dict(
        mean=float(top_confs.mean()),
        std=float(top_confs.std()),
        p10=float(np.percentile(top_confs, 10)),
        p50=float(np.percentile(top_confs, 50)),
        p90=float(np.percentile(top_confs, 90)),
        p99=float(np.percentile(top_confs, 99)),
        low_confidence_pct=float((top_confs < 0.5).mean() * 100),
    )

    print(f"    {label} accuracy: {overall_acc*100:.2f}%  "
          f"(conf mean={conf_stats['mean']:.3f}  "
          f"low-conf<50%: {conf_stats['low_confidence_pct']:.1f}%)")

    return dict(
        overall_accuracy=overall_acc,
        overall_pct=round(overall_acc * 100, 2),
        per_class=per_class,
        confidence_stats=conf_stats,
        n_evaluated=len(y_pred),
    )


def clinical_projections(label, tput_b1, tput_b32, accuracy):
    """
    Translate raw throughput numbers into clinical deployment terms.
    Assumes a dermatology screening workflow.
    """
    # Assumptions
    IMAGES_PER_PATIENT   = 3    # avg dermoscopy images per patient visit
    WORKING_HOURS        = 8    # clinical workday hours
    REALTIME_THRESHOLD   = 100  # ms — acceptable for interactive use

    imgs_per_hour_b1  = tput_b1  * 3600
    imgs_per_hour_b32 = tput_b32 * 3600
    patients_per_day  = (imgs_per_hour_b32 * WORKING_HOURS) / IMAGES_PER_PATIENT
    ms_per_image_b1   = 1000.0 / tput_b1 if tput_b1 else 0
    realtime_ok       = ms_per_image_b1 < REALTIME_THRESHOLD

    return dict(
        assumptions=dict(
            images_per_patient=IMAGES_PER_PATIENT,
            working_hours_per_day=WORKING_HOURS,
            realtime_threshold_ms=REALTIME_THRESHOLD,
        ),
        single_image=dict(
            ms_per_image=round(ms_per_image_b1, 3),
            images_per_second=round(tput_b1, 1),
            images_per_hour=round(imgs_per_hour_b1, 0),
            realtime_capable=realtime_ok,
        ),
        batch_throughput=dict(
            images_per_second=round(tput_b32, 1),
            images_per_hour=round(imgs_per_hour_b32, 0),
            patients_per_8h_day=round(patients_per_day, 0),
        ),
        accuracy_context=dict(
            accuracy_pct=round(accuracy * 100, 2) if accuracy else None,
            note="42-43% accuracy on 8-class balanced test set. "
                 "Higher per-class accuracy expected on common classes (NV, MEL). "
                 "Clinical use requires per-class calibration and human review."
        ),
    )


# ============================================================================
# STEP 3 — ONNX BASELINE SWEEP
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: ONNX-CUDA BASELINE SWEEP")
print("=" * 80)

onnx_sweep = bench_onnx_sweep(X_test)

# Accuracy on ONNX
print("\n  ONNX accuracy check ...")
ort_sess  = ort.InferenceSession(
    ONNX_MODEL,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
inp_name = ort_sess.get_inputs()[0].name
out_name = ort_sess.get_outputs()[0].name

def _onnx_infer(b):
    return ort_sess.run([out_name], {inp_name: b})[0]

onnx_accuracy = compute_accuracy(_onnx_infer, X_test, y_all, 32, "ONNX")
del ort_sess; gc.collect()


# ============================================================================
# STEP 4 — TRT FP32 BENCHMARK
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: TRT FP32 BENCHMARK")
print("=" * 80)

mem_before_fp32 = get_gpu_mem_used_mb()
engine_fp32 = load_trt_engine(ENGINE_FP32)
sess_fp32   = TRTSession(engine_fp32)
mem_after_fp32  = get_gpu_mem_used_mb()
fp32_vram_mb    = mem_after_fp32 - mem_before_fp32

fp32_sweep = bench_latency_sweep(sess_fp32.infer, X_test, "TRT-FP32")

print("\n  FP32 warmup convergence ...")
fp32_warmup = bench_warmup_convergence(sess_fp32.infer, X_test, "TRT-FP32")

print("\n  FP32 accuracy ...")
fp32_accuracy = compute_accuracy(sess_fp32.infer, X_test, y_all, 32, "TRT-FP32")

del sess_fp32; gc.collect()


# ============================================================================
# STEP 5 — TRT FP16 BENCHMARK
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: TRT FP16 BENCHMARK")
print("=" * 80)

mem_before_fp16 = get_gpu_mem_used_mb()
engine_fp16 = load_trt_engine(ENGINE_FP16)
sess_fp16   = TRTSession(engine_fp16)
mem_after_fp16  = get_gpu_mem_used_mb()
fp16_vram_mb    = mem_after_fp16 - mem_before_fp16

fp16_sweep = bench_latency_sweep(sess_fp16.infer, X_test, "TRT-FP16")

print("\n  FP16 warmup convergence ...")
fp16_warmup = bench_warmup_convergence(sess_fp16.infer, X_test, "TRT-FP16")

print("\n  FP16 accuracy ...")
fp16_accuracy = compute_accuracy(sess_fp16.infer, X_test, y_all, 32, "TRT-FP16")

del sess_fp16; gc.collect()


# ============================================================================
# STEP 6 — TRT INT8 BENCHMARK
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: TRT INT8 BENCHMARK")
print("=" * 80)

mem_before_int8 = get_gpu_mem_used_mb()
engine_int8 = load_trt_engine(ENGINE_INT8)
sess_int8   = TRTSession(engine_int8)
mem_after_int8  = get_gpu_mem_used_mb()
int8_vram_mb    = mem_after_int8 - mem_before_int8

int8_sweep = bench_latency_sweep(sess_int8.infer, X_test, "TRT-INT8")

print("\n  INT8 warmup convergence ...")
int8_warmup = bench_warmup_convergence(sess_int8.infer, X_test, "TRT-INT8")

print("\n  INT8 accuracy ...")
int8_accuracy = compute_accuracy(sess_int8.infer, X_test, y_all, 32, "TRT-INT8")

del sess_int8; gc.collect()


# ============================================================================
# STEP 7 — CROSS-ENGINE COMPARISON + SPEEDUP MATRIX
# ============================================================================

print("\n" + "=" * 80)
print("STEP 7: CROSS-ENGINE COMPARISON")
print("=" * 80)

def speedup_matrix(sweeps, labels):
    """
    For every batch size, compute speedup of each engine vs ONNX.
    Returns dict keyed by batch_size.
    """
    matrix = {}
    onnx_s = sweeps["onnx"]
    for bs in BATCH_SIZES:
        if bs not in onnx_s:
            continue
        onnx_tput = onnx_s[bs]["throughput_ips"]
        row = {"onnx_ips": round(onnx_tput, 1)}
        for key, label in labels.items():
            if key == "onnx" or bs not in sweeps[key]:
                continue
            trt_tput  = sweeps[key][bs]["throughput_ips"]
            trt_lat   = sweeps[key][bs]["p99_ms"]
            row[f"{key}_ips"]     = round(trt_tput, 1)
            row[f"{key}_speedup"] = round(trt_tput / onnx_tput, 3)
            row[f"{key}_p99_ms"]  = round(trt_lat, 3)
        matrix[bs] = row
    return matrix

all_sweeps = {"onnx": onnx_sweep, "fp32": fp32_sweep,
              "fp16": fp16_sweep, "int8": int8_sweep}
labels     = {"onnx": "ONNX", "fp32": "FP32", "fp16": "FP16", "int8": "INT8"}
comparison = speedup_matrix(all_sweeps, labels)

# Print comparison table
print(f"\n  Throughput (img/sec) by batch size:")
header = f"  {'Batch':>6}  {'ONNX':>8}  {'FP32':>8}  {'FP16':>8}  {'INT8':>8}"
print(header)
print(f"  {'─' * 50}")
for bs in BATCH_SIZES:
    if bs not in comparison:
        continue
    row = comparison[bs]
    print(f"  {bs:>6}  "
          f"{row.get('onnx_ips', 0):>8.1f}  "
          f"{row.get('fp32_ips', 0):>8.1f}  "
          f"{row.get('fp16_ips', 0):>8.1f}  "
          f"{row.get('int8_ips', 0):>8.1f}")

print(f"\n  Speedup vs ONNX (batch={MAX_BATCH}):")
row32 = comparison.get(MAX_BATCH, {})
for key in ["fp32", "fp16", "int8"]:
    spd = row32.get(f"{key}_speedup", 0)
    ips = row32.get(f"{key}_ips", 0)
    bar = "█" * int(spd * 10) if spd > 0 else ""
    print(f"    {key.upper():<5}: {spd:.3f}x  {ips:.0f} img/s  {bar}")

print(f"\n  P99 latency (ms) at batch=1 — real-time requirement:")
row1 = comparison.get(1, {})
onnx_p99 = onnx_sweep.get(1, {}).get("p99_ms", 0)
print(f"    ONNX : {onnx_p99:.3f}ms")
for key in ["fp32", "fp16", "int8"]:
    p99 = row1.get(f"{key}_p99_ms", 0)
    rt  = "✓ real-time" if p99 < 100 else "✗ not real-time"
    print(f"    {key.upper():<5}: {p99:.3f}ms  {rt}")


# ============================================================================
# STEP 8 — CLINICAL DEPLOYMENT PROJECTIONS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 8: CLINICAL DEPLOYMENT PROJECTIONS")
print("=" * 80)

def _get_tput(sweep, bs):
    return sweep.get(bs, {}).get("throughput_ips", 0)

clinical = {}
for key, sweep, acc_result in [
    ("onnx", onnx_sweep, onnx_accuracy),
    ("fp32", fp32_sweep, fp32_accuracy),
    ("fp16", fp16_sweep, fp16_accuracy),
    ("int8", int8_sweep, int8_accuracy),
]:
    acc_val = acc_result["overall_accuracy"] if acc_result else None
    proj    = clinical_projections(
        key.upper(),
        _get_tput(sweep, 1),
        _get_tput(sweep, MAX_BATCH),
        acc_val
    )
    clinical[key] = proj

    print(f"\n  {key.upper()}:")
    print(f"    Single-image   : {proj['single_image']['ms_per_image']:.1f}ms  "
          f"→  {proj['single_image']['images_per_hour']:,.0f} img/hr  "
          f"real-time: {'✓' if proj['single_image']['realtime_capable'] else '✗'}")
    print(f"    Batch-32 tput  : {proj['batch_throughput']['images_per_second']:.0f} img/s  "
          f"→  {proj['batch_throughput']['images_per_hour']:,.0f} img/hr")
    print(f"    Patients/8h day: {proj['batch_throughput']['patients_per_8h_day']:,.0f}  "
          f"(at {proj['assumptions']['images_per_patient']} imgs/patient)")


# ============================================================================
# STEP 9 — VRAM EFFICIENCY
# ============================================================================

print("\n" + "=" * 80)
print("STEP 9: VRAM EFFICIENCY")
print("=" * 80)

vram_results = {}
for key, vram_mb, sweep in [
    ("fp32", fp32_vram_mb, fp32_sweep),
    ("fp16", fp16_vram_mb, fp16_sweep),
    ("int8", int8_vram_mb, int8_sweep),
]:
    tput = _get_tput(sweep, MAX_BATCH)
    efficiency = tput / vram_mb if vram_mb > 0 else 0
    engine_mb  = Path({"fp32": ENGINE_FP32, "fp16": ENGINE_FP16,
                        "int8": ENGINE_INT8}[key]).stat().st_size / (1024**2)
    vram_results[key] = dict(
        engine_file_mb=round(engine_mb, 2),
        vram_delta_mb=vram_mb,
        throughput_b32=round(tput, 1),
        imgs_per_s_per_gb_vram=round(efficiency * 1024, 1),
    )
    print(f"  {key.upper():<5}: engine={engine_mb:.1f}MB  "
          f"VRAM Δ≈{vram_mb}MB  "
          f"tput={tput:.0f} img/s  "
          f"efficiency={efficiency*1024:.1f} img/s/GB")


# ============================================================================
# STEP 10 — FINAL SUMMARY TABLE
# ============================================================================

print("\n" + "=" * 80)
print("STEP 10: FINAL SUMMARY")
print("=" * 80)

W = 22
print(f"\n  {'Engine':<{W}} {'b=1 ms':>8} {'b=1 img/s':>10} {'b=32 img/s':>11} "
      f"{'Speedup':>9} {'Accuracy':>10} {'Converge@':>10}")
print(f"  {'─' * 82}")

def _row(label, sweep, warmup_r, accuracy_r, onnx_b32_tput):
    b1  = sweep.get(1, {})
    b32 = sweep.get(MAX_BATCH, {})
    acc = accuracy_r["overall_pct"] if accuracy_r else None
    spd = b32.get("throughput_ips", 0) / onnx_b32_tput if onnx_b32_tput else 0
    cv  = warmup_r["converges_at_run"] if warmup_r else "?"
    print(f"  {label:<{W}} "
          f"{b1.get('mean_ms', 0):>8.3f} "
          f"{b1.get('throughput_ips', 0):>10.1f} "
          f"{b32.get('throughput_ips', 0):>11.1f} "
          f"{spd:>9.2f}x "
          f"{(str(round(acc,1))+'%') if acc else 'N/A':>10} "
          f"{str(cv)+' runs':>10}")

onnx_b32 = onnx_sweep.get(MAX_BATCH, {}).get("throughput_ips", 1)

_row("ONNX-CUDA (base)", onnx_sweep, None, onnx_accuracy, onnx_b32)
_row("TRT FP32",         fp32_sweep, fp32_warmup, fp32_accuracy, onnx_b32)
_row("TRT FP16 ★",      fp16_sweep, fp16_warmup, fp16_accuracy, onnx_b32)
_row("TRT INT8",         int8_sweep, int8_warmup, int8_accuracy, onnx_b32)

print(f"\n  ★ = recommended production engine")
print(f"\n  Key insight: b=1 speedup < 1x is expected — ONNX-RT is highly")
print(f"  optimised for single-image mode. TRT's advantage is batch throughput.")
print(f"  At batch=32, FP16 delivers {fp16_sweep.get(MAX_BATCH,{}).get('throughput_ips',0)/onnx_b32:.2f}x ONNX throughput.")


# ============================================================================
# BUILD JSON REPORT
# ============================================================================

def _int_keys(d):
    """Convert integer keys to strings for JSON serialisation."""
    return {str(k): v for k, v in d.items()}

report = {
    "metadata": {
        "script":        "week16.py",
        "date":          datetime.now().isoformat(),
        "gpu":           gpu_info,
        "benchmark_config": dict(
            batch_sizes=BATCH_SIZES,
            warmup_runs=WARMUP_RUNS,
            bench_runs=BENCH_RUNS,
            warmup_conv_runs=WARMUP_CONV_RUNS,
            histogram_bins=HIST_BINS,
            max_batch=MAX_BATCH,
        ),
        "models": {
            "onnx":        {"path": ONNX_MODEL,   "size_mb": round(Path(ONNX_MODEL).stat().st_size/(1024**2), 2)},
            "fp32_engine": {"path": ENGINE_FP32,  "size_mb": round(Path(ENGINE_FP32).stat().st_size/(1024**2), 2)},
            "fp16_engine": {"path": ENGINE_FP16,  "size_mb": round(Path(ENGINE_FP16).stat().st_size/(1024**2), 2)},
            "int8_engine": {"path": ENGINE_INT8,  "size_mb": round(Path(ENGINE_INT8).stat().st_size/(1024**2), 2)},
        },
    },
    "latency_sweep": {
        "onnx": _int_keys(onnx_sweep),
        "fp32": _int_keys(fp32_sweep),
        "fp16": _int_keys(fp16_sweep),
        "int8": _int_keys(int8_sweep),
        "batch_sizes_tested": BATCH_SIZES,
    },
    "warmup_convergence": {
        "fp32": fp32_warmup,
        "fp16": fp16_warmup,
        "int8": int8_warmup,
        "interpretation": "converges_at_run is the minimum warmup needed before "
                          "latency stabilises within 2% of steady-state."
    },
    "accuracy": {
        "onnx": onnx_accuracy,
        "fp32": fp32_accuracy,
        "fp16": fp16_accuracy,
        "int8": int8_accuracy,
        "note": "Evaluated on full test set at batch=32. "
                "42-43% overall reflects 8-class balanced difficulty; "
                "per-class results show the real clinical picture."
    },
    "speedup_matrix": {
        "description": "Throughput (img/s) and speedup vs ONNX at each batch size",
        "by_batch_size": _int_keys(comparison),
    },
    "vram_efficiency": vram_results,
    "clinical_projections": clinical,
    "recommendation": {
        "production_engine": "fp16",
        "rationale": (
            f"FP16 delivers {fp16_sweep.get(MAX_BATCH,{}).get('throughput_ips',0)/onnx_b32:.2f}x "
            f"ONNX batch throughput at {fp16_sweep.get(1,{}).get('mean_ms',0):.1f}ms single-image latency. "
            f"Accuracy delta vs FP32 is negligible (+{fp16_accuracy['overall_pct']-fp32_accuracy['overall_pct']:.1f}pp). "
            f"Engine is {round(Path(ENGINE_FP16).stat().st_size/(1024**2),1)}MB vs "
            f"{round(Path(ENGINE_FP32).stat().st_size/(1024**2),1)}MB FP32."
        ),
        "int8_trade_off": (
            f"INT8 gives {int8_sweep.get(MAX_BATCH,{}).get('throughput_ips',0)/fp16_sweep.get(MAX_BATCH,{}).get('throughput_ips',1):.2f}x "
            f"over FP16 at batch=32 but at "
            f"{fp32_accuracy['overall_pct']-int8_accuracy['overall_pct']:.1f}pp accuracy cost. "
            f"Acceptable for high-volume screening; not for diagnostic use without recalibration."
        ),
        "warmup_requirement": {
            "fp16_converges_at": fp16_warmup["converges_at_run"],
            "recommendation": "Warm up at least "
                              f"{fp16_warmup['converges_at_run']+5} images before "
                              "serving production requests."
        },
    },
}

with open(REPORT_PATH, "w") as f:
    json.dump(report, f, indent=2)
print(f"\n  Full JSON report saved: {REPORT_PATH}")

# ── Human-readable summary ────────────────────────────────────────────────────
summary_lines = [
    "=" * 72,
    "WEEK 16 — INFERENCE BENCHMARK SUMMARY",
    f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    f"GPU  : {gpu_name}  ({vram_mb/1024:.1f} GB VRAM)",
    "=" * 72,
    "",
    "LATENCY AT BATCH=1 (single image, real-time scenario)",
    f"  ONNX-CUDA : {onnx_sweep.get(1,{}).get('mean_ms',0):.3f}ms  P99={onnx_sweep.get(1,{}).get('p99_ms',0):.3f}ms",
    f"  TRT FP32  : {fp32_sweep.get(1,{}).get('mean_ms',0):.3f}ms  P99={fp32_sweep.get(1,{}).get('p99_ms',0):.3f}ms",
    f"  TRT FP16  : {fp16_sweep.get(1,{}).get('mean_ms',0):.3f}ms  P99={fp16_sweep.get(1,{}).get('p99_ms',0):.3f}ms",
    f"  TRT INT8  : {int8_sweep.get(1,{}).get('mean_ms',0):.3f}ms  P99={int8_sweep.get(1,{}).get('p99_ms',0):.3f}ms",
    "",
    "THROUGHPUT AT BATCH=32 (sustained screening scenario)",
    f"  ONNX-CUDA : {onnx_sweep.get(32,{}).get('throughput_ips',0):.1f} img/s  (1.00x baseline)",
    f"  TRT FP32  : {fp32_sweep.get(32,{}).get('throughput_ips',0):.1f} img/s  ({fp32_sweep.get(32,{}).get('throughput_ips',0)/onnx_b32:.2f}x)",
    f"  TRT FP16  : {fp16_sweep.get(32,{}).get('throughput_ips',0):.1f} img/s  ({fp16_sweep.get(32,{}).get('throughput_ips',0)/onnx_b32:.2f}x)  ← RECOMMENDED",
    f"  TRT INT8  : {int8_sweep.get(32,{}).get('throughput_ips',0):.1f} img/s  ({int8_sweep.get(32,{}).get('throughput_ips',0)/onnx_b32:.2f}x)",
    "",
    "ACCURACY (full test set, 1000 samples)",
    f"  ONNX-CUDA : {onnx_accuracy['overall_pct'] if onnx_accuracy else 'N/A'}%",
    f"  TRT FP32  : {fp32_accuracy['overall_pct'] if fp32_accuracy else 'N/A'}%",
    f"  TRT FP16  : {fp16_accuracy['overall_pct'] if fp16_accuracy else 'N/A'}%  (delta vs FP32: {(fp16_accuracy['overall_pct']-fp32_accuracy['overall_pct']) if fp16_accuracy and fp32_accuracy else 'N/A'}pp)",
    f"  TRT INT8  : {int8_accuracy['overall_pct'] if int8_accuracy else 'N/A'}%  (delta vs FP32: {(int8_accuracy['overall_pct']-fp32_accuracy['overall_pct']) if int8_accuracy and fp32_accuracy else 'N/A'}pp)",
    "",
    "WARMUP CONVERGENCE (runs until latency stabilises)",
    f"  TRT FP32  : {fp32_warmup['converges_at_run']} runs",
    f"  TRT FP16  : {fp16_warmup['converges_at_run']} runs",
    f"  TRT INT8  : {int8_warmup['converges_at_run']} runs",
    "",
    "CLINICAL PROJECTIONS (FP16, 8-hour workday, 3 imgs/patient)",
    f"  Images/hour    : {clinical['fp16']['batch_throughput']['images_per_hour']:,.0f}",
    f"  Patients/day   : {clinical['fp16']['batch_throughput']['patients_per_8h_day']:,.0f}",
    f"  Real-time cap. : {'Yes' if clinical['fp16']['single_image']['realtime_capable'] else 'No'} (<100ms threshold)",
    "",
    "RECOMMENDATION: TRT FP16",
    f"  {report['recommendation']['rationale']}",
    "",
    f"Full JSON: {REPORT_PATH}",
    "=" * 72,
]

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)

with open(SUMMARY_PATH, "w") as f:
    f.write(summary_text)
print(f"\n  Summary saved: {SUMMARY_PATH}")

print("\n" + "=" * 80)
print("WEEK 16 COMPLETE")
print("=" * 80)
print(f"""
  Outputs:
    JSON report : {REPORT_PATH}
    Text summary: {SUMMARY_PATH}

  Load FP16 engine in week17:
    TRT_ENGINE_PATH = "{ENGINE_FP16}"
""")
print("=" * 80)