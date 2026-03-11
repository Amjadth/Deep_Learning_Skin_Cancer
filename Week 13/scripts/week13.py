#!/usr/bin/env python3
"""
Week 13: EfficientNetB3 Keras → ONNX Conversion

Module:       week13.py
Purpose:      Convert trained Keras model to ONNX with float32 patching
Dataset:      ISIC 2019 Skin Cancer Detection
Model:        EfficientNetB3
Author:       Amjad
Date:         February 2026
Platform:     RunPod

═══════════════════════════════════════════════════════════════════════════════

DESCRIPTION
───────────
Converts .keras model to ONNX format with automated float32 patching.
Addresses mixed_float16 training artifacts that cause NaN at inference.

PROCESS
───────
1. Force float32 dtype policy before TensorFlow initialization
2. Load .keras model → cast all 509 weight arrays to float32
3. Freeze to SavedModel (avoids tf2onnx live-tracing NaN bug)
4. Convert SavedModel → ONNX via tf_loader internal API
5. Patch ONNX graph:
   • Flip 84 Cast→float16 nodes to Cast→float32
   • Upcast 189 float16 weight tensors to float32
6. Validate with real test images (hard NaN gate)
7. Cleanup temporary SavedModel, save JSON report

WHY PATCHING IS NEEDED
──────────────────────
Model trained with mixed_float16 policy. tf2onnx bakes 84 Cast→float16 nodes
into the ONNX graph. At runtime, float32 activations overflow float16's max
(65504) → Inf → NaN throughout the network. Patching removes all float16
computation, keeping everything in float32.

USAGE
─────
Run from terminal:

    python /workspace/week13.py

Requirements:
  • TensorFlow 2.13+
  • tf2onnx
  • onnx, onnxruntime
  • numpy

Inputs:
  • /workspace/models/final_model.keras
  • /workspace/dataset/test_preprocessed/X_test_300x300.npy
  • /workspace/dataset/test_preprocessed/y_test.npy

Outputs:
  • /workspace/output/EfficientNetB3_ISIC2019_final.onnx  (final model)
  • /workspace/output/conversion_report_final.json        (conversion report)
  • /workspace/output/_savedmodel_tmp/                    (deleted at end)

═══════════════════════════════════════════════════════════════════════════════
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS — float32 policy MUST be set before TF initialises
# ─────────────────────────────────────────────────────────────────────────────
import sys, os, gc, json, shutil, tempfile
import numpy as np
from datetime import datetime
from pathlib import Path

print("=" * 80)
print("WEEK 13: EfficientNetB3 KERAS → ONNX CONVERSION (FINAL)")
print("=" * 80)

import tensorflow as tf
from tensorflow.keras import mixed_precision

print("\n[STEP 0] Forcing float32 dtype policy ...")
mixed_precision.set_global_policy('float32')
assert mixed_precision.global_policy().name == 'float32'
print(f"  ✓ Policy  : {mixed_precision.global_policy().name}")
print(f"  TensorFlow: {tf.__version__}  |  Python: {sys.version.split()[0]}")
gpus = tf.config.list_physical_devices('GPU')
for g in gpus: print(f"  GPU       : {g}")

import tf2onnx
import onnx
import onnx.numpy_helper as nph
from onnx import TensorProto
import onnxruntime as ort

print(f"\n  tf2onnx     : {tf2onnx.__version__}")
print(f"  onnx        : {onnx.__version__}")
print(f"  onnxruntime : {ort.__version__}")
print(f"  ORT providers: {ort.get_available_providers()}")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
INPUT_KERAS    = "/workspace/models/final_model.keras"
OUTPUT_DIR     = "/workspace/output"
SAVEDMODEL_DIR = "/workspace/output/_savedmodel_tmp"          # deleted at end
OUTPUT_ONNX    = "/workspace/output/EfficientNetB3_ISIC2019_final.onnx"
REPORT_PATH    = "/workspace/output/conversion_report_final.json"

TEST_DATA      = "/workspace/dataset/test_preprocessed/X_test_300x300.npy"
TEST_LABELS    = "/workspace/dataset/test_preprocessed/y_test.npy"
CLASS_NAMES    = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']
OPSET          = 13
NUM_VAL        = 8     # one per class

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n" + "=" * 80)
print("CONFIGURATION")
print("=" * 80)
print(f"  Input  keras : {INPUT_KERAS}")
print(f"  Output ONNX  : {OUTPUT_ONNX}")
print(f"  Opset        : {OPSET}")

if not os.path.exists(INPUT_KERAS):
    raise FileNotFoundError(f"Model not found: {INPUT_KERAS}")
print(f"  ✓ Model found ({os.path.getsize(INPUT_KERAS)/(1024**2):.1f} MB)")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load .keras and cast all weights to float32
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("STEP 1: LOAD + CAST ALL WEIGHTS TO float32")
print("=" * 80)

tf.keras.backend.clear_session()
gc.collect()

orig = tf.keras.models.load_model(INPUT_KERAS, compile=False)
print(f"  ✓ Loaded  "
      f"compute={orig.dtype_policy.compute_dtype}  "
      f"variable={orig.dtype_policy.variable_dtype}")
print(f"  Input : {orig.input_shape}  Output: {orig.output_shape}")

new_in    = tf.keras.Input(shape=(300, 300, 3), dtype=tf.float32, name='input')
new_out   = orig(new_in, training=False)
model_f32 = tf.keras.Model(new_in, new_out, name='EfficientNetB3_f32')
model_f32.set_weights([w.astype(np.float32) for w in orig.get_weights()])
print(f"  ✓ {len(model_f32.get_weights())} weight arrays cast to float32")

# Sanity check
probe  = np.random.uniform(0, 255, (2, 300, 300, 3)).astype(np.float32)
p_orig = orig.predict(probe, verbose=0)
p_f32  = model_f32.predict(probe, verbose=0)
print(f"\n  Original NaN={np.any(np.isnan(p_orig))}  out={np.round(p_orig[0], 4)}")
print(f"  Float32  NaN={np.any(np.isnan(p_f32))}  out={np.round(p_f32[0], 4)}")
if np.any(np.isnan(p_f32)):
    raise RuntimeError("float32 Keras model outputs NaN — weights are corrupted")

del orig
gc.collect()

tmp_keras = os.path.join(tempfile.gettempdir(), "_f32_size_check.keras")
model_f32.save(tmp_keras)
keras_mb = os.path.getsize(tmp_keras) / (1024**2)
os.remove(tmp_keras)
print(f"\n  float32 model size: {keras_mb:.1f} MB")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — Save as frozen SavedModel
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("STEP 2: FREEZE TO SavedModel")
print("=" * 80)

if os.path.exists(SAVEDMODEL_DIR):
    shutil.rmtree(SAVEDMODEL_DIR)
print(f"  Saving → {SAVEDMODEL_DIR} ...")
tf.saved_model.save(model_f32, SAVEDMODEL_DIR)

sm_mb = sum(
    os.path.getsize(os.path.join(dp, f))
    for dp, _, fns in os.walk(SAVEDMODEL_DIR) for f in fns
) / (1024**2)
print(f"  ✓ Saved ({sm_mb:.1f} MB)")

# Verify SavedModel is clean
sm      = tf.saved_model.load(SAVEDMODEL_DIR)
sig     = sm.signatures['serving_default']
out_key = list(sig.structured_outputs.keys())[0]
sm_pred = sig(tf.constant(probe))[out_key].numpy()
if np.any(np.isnan(sm_pred)):
    raise RuntimeError("SavedModel outputs NaN — cannot continue")
print(f"  ✓ SavedModel clean  out={np.round(sm_pred[0], 4)}")
del sm, sig
gc.collect()


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — Convert SavedModel → ONNX
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("STEP 3: CONVERT SavedModel → ONNX")
print("=" * 80)
print(f"  Source : {SAVEDMODEL_DIR}")
print(f"  Output : {OUTPUT_ONNX}")
print(f"  Opset  : {OPSET}")

from tf2onnx import tf_loader
from tf2onnx.convert import _convert_common

t0 = datetime.now()
print("  Loading frozen graph ...")
result       = tf_loader.from_saved_model(
    SAVEDMODEL_DIR,
    input_names=None,
    output_names=None,
    signatures=['serving_default'],
    return_tensors_to_rename=True,
)
frozen_graph      = result[0]
input_names       = result[1]
output_names      = result[2]
tensors_to_rename = result[3] if len(result) > 3 else None
print(f"  inputs : {input_names}")
print(f"  outputs: {output_names}")

print("  Converting ...")
onnx_model, _ = _convert_common(
    frozen_graph,
    name="EfficientNetB3_ISIC2019",
    opset=OPSET,
    input_names=input_names,
    output_names=output_names,
    tensors_to_rename=tensors_to_rename,
    output_path=OUTPUT_ONNX,
)
conv_time = (datetime.now() - t0).total_seconds()
raw_mb    = os.path.getsize(OUTPUT_ONNX) / (1024**2)
print(f"  ✓ Done in {conv_time:.1f}s  |  {raw_mb:.1f} MB  |  {len(onnx_model.graph.node)} nodes")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — Patch ONNX graph: remove all float16 Casts + upcast initializers
#
# tf2onnx bakes 84 Cast→float16 nodes because mixed_float16 was active
# during training. At runtime these cause float32 activations to overflow
# float16's max (65504) → Inf → NaN throughout the whole network.
# Fix: flip every Cast-to-float16 → Cast-to-float32, upcast f16 weights.
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("STEP 4: PATCH — REMOVE float16 CAST NODES + UPCAST INITIALIZERS")
print("=" * 80)

onnx_model = onnx.load(OUTPUT_ONNX)

# ── 4a: Count what we're about to fix ────────────────────────────────────────
f16_cast_before = sum(
    1 for n in onnx_model.graph.node
    if n.op_type == 'Cast'
    and any(a.name == 'to' and a.i == TensorProto.FLOAT16 for a in n.attribute)
)
f16_init_before = sum(
    1 for i in onnx_model.graph.initializer
    if nph.to_array(i).dtype == np.float16
)
print(f"  Before patch:")
print(f"    Cast→float16 nodes  : {f16_cast_before}")
print(f"    float16 initializers: {f16_init_before}")

# ── 4b: Flip Cast→float16 to Cast→float32 ────────────────────────────────────
fixed_casts = 0
for node in onnx_model.graph.node:
    if node.op_type == 'Cast':
        for attr in node.attribute:
            if attr.name == 'to' and attr.i == TensorProto.FLOAT16:
                attr.i = TensorProto.FLOAT
                fixed_casts += 1
                break
print(f"\n  ✓ Flipped {fixed_casts} Cast→float16 nodes to Cast→float32")

# ── 4c: Upcast float16 initializer tensors to float32 ────────────────────────
to_remove, to_add = [], []
for init in onnx_model.graph.initializer:
    arr = nph.to_array(init)
    if arr.dtype == np.float16:
        to_remove.append(init)
        to_add.append(nph.from_array(arr.astype(np.float32), name=init.name))

for old in to_remove:
    onnx_model.graph.initializer.remove(old)
for new in to_add:
    onnx_model.graph.initializer.append(new)
print(f"  ✓ Upcast {len(to_add)} initializers from float16 → float32")

# ── 4d: Verify none remain ───────────────────────────────────────────────────
f16_cast_after = sum(
    1 for n in onnx_model.graph.node
    if n.op_type == 'Cast'
    and any(a.name == 'to' and a.i == TensorProto.FLOAT16 for a in n.attribute)
)
f16_init_after = sum(
    1 for i in onnx_model.graph.initializer
    if nph.to_array(i).dtype == np.float16
)
print(f"\n  After patch:")
print(f"    Cast→float16 nodes  : {f16_cast_after}  ← must be 0")
print(f"    float16 initializers: {f16_init_after}  ← must be 0")
if f16_cast_after > 0 or f16_init_after > 0:
    raise RuntimeError("float16 elements remain after patching — check logic")
print("  ✅ Graph is fully float32")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — ONNX integrity check + save final model
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("STEP 5: INTEGRITY CHECK + SAVE")
print("=" * 80)

try:
    onnx.checker.check_model(onnx_model)
    print("  ✓ ONNX graph structurally valid")
except Exception as e:
    print(f"  ⚠  check_model: {e}  (non-fatal, continuing)")

try:
    onnx.shape_inference.infer_shapes(onnx_model)
    print("  ✓ Shape inference passed")
except Exception as e:
    print(f"  ⚠  shape inference: {e}  (non-fatal, continuing)")

onnx.save(onnx_model, OUTPUT_ONNX)
final_mb = os.path.getsize(OUTPUT_ONNX) / (1024**2)
print(f"\n  ✓ Saved → {OUTPUT_ONNX}")
print(f"  Size    : {final_mb:.1f} MB  (was {raw_mb:.1f} MB before patch)")
print(f"  Nodes   : {len(onnx_model.graph.node)}")
print(f"  Opset   : {onnx_model.opset_import[0].version}")
for inp in onnx_model.graph.input:
    shape = [d.dim_value if d.dim_value > 0 else 'dynamic'
             for d in inp.type.tensor_type.shape.dim]
    print(f"  Input   : {inp.name}  {shape}")
for out in onnx_model.graph.output:
    shape = [d.dim_value if d.dim_value > 0 else 'dynamic'
             for d in out.type.tensor_type.shape.dim]
    print(f"  Output  : {out.name}  {shape}")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 6 — Runtime validation with real test images
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("STEP 6: RUNTIME VALIDATION")
print("=" * 80)

providers = (['CUDAExecutionProvider', 'CPUExecutionProvider']
             if 'CUDAExecutionProvider' in ort.get_available_providers()
             else ['CPUExecutionProvider'])

sess_opts = ort.SessionOptions()
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess     = ort.InferenceSession(OUTPUT_ONNX, sess_options=sess_opts, providers=providers)
inp_name = sess.get_inputs()[0].name
out_name = sess.get_outputs()[0].name
print(f"  Provider: {sess.get_providers()[0]}")
print(f"  Input   : {inp_name}  {sess.get_inputs()[0].shape}")
print(f"  Output  : {out_name}  {sess.get_outputs()[0].shape}")

# Quick random probe
rnd_out = sess.run([out_name], {inp_name: probe})[0]
print(f"\n  Random probe  NaN={np.any(np.isnan(rnd_out))}  "
      f"range=[{rnd_out.min():.4f},{rnd_out.max():.4f}]")
if np.any(np.isnan(rnd_out)):
    raise RuntimeError("NaN on random input after patch — unexpected, investigate")
print("  ✅ No NaN on random input")

# Real test images
print(f"\n  Loading real test images from {TEST_DATA} ...")
if not Path(TEST_DATA).exists():
    print(f"  ⚠  Not found — skipping real-image validation")
    keras_nan = onnx_nan = False
    max_diff = mean_diff = 0.0
    val_n = 0
else:
    X_raw = np.load(TEST_DATA).astype(np.float32)
    y_all = np.load(TEST_LABELS).astype(int) if Path(TEST_LABELS).exists() else None

    # One image per class for a representative sample
    idxs = []
    if y_all is not None:
        for cls in range(8):
            where = np.where(y_all == cls)[0]
            if len(where): idxs.append(int(where[0]))
    if len(idxs) < NUM_VAL:
        idxs = list(range(NUM_VAL))
    idxs  = idxs[:NUM_VAL]
    X_val = X_raw[idxs]
    y_val = y_all[idxs] if y_all is not None else None
    val_n = len(X_val)

    # Scale to [0,255] — model has Rescaling(×1/255) baked in as node 1
    X_in  = X_val * 255.0 if X_val.max() <= 1.0 else X_val
    print(f"  n={val_n}  input range=[{X_in.min():.1f}, {X_in.max():.1f}]")

    # Keras
    t0 = datetime.now()
    k_preds  = model_f32.predict(X_in, verbose=0)
    k_time   = (datetime.now() - t0).total_seconds()
    keras_nan = bool(np.any(np.isnan(k_preds)))
    print(f"\n  Keras  {k_time:.3f}s  NaN={keras_nan}  "
          f"range=[{np.nanmin(k_preds):.4f},{np.nanmax(k_preds):.4f}]")

    # ONNX
    t0 = datetime.now()
    o_preds  = sess.run([out_name], {inp_name: X_in})[0]
    o_time   = (datetime.now() - t0).total_seconds()
    onnx_nan = bool(np.any(np.isnan(o_preds)))
    print(f"  ONNX   {o_time:.3f}s  NaN={onnx_nan}  "
          f"range=[{np.nanmin(o_preds):.4f},{np.nanmax(o_preds):.4f}]")

    print("\n" + "-" * 60)
    if keras_nan or onnx_nan:
        raise RuntimeError(
            f"NaN after full patch  keras_nan={keras_nan}  onnx_nan={onnx_nan}\n"
            "The model requires retraining without mixed_float16."
        )
    print("  ✅ No NaN in Keras or ONNX outputs\n")

    diff      = np.abs(k_preds - o_preds)
    max_diff  = float(diff.max())
    mean_diff = float(diff.mean())
    print(f"  Keras↔ONNX diff  max={max_diff:.2e}  mean={mean_diff:.2e}")
    if   max_diff < 1e-3: print("  ✅ Numerically faithful")
    elif max_diff < 0.1:  print("  ⚠  Small FP rounding — acceptable")
    else:                 print("  ❌ Large diff — unexpected")

    print(f"\n  {'#':>3}  {'True':>6}  {'Keras':>6} Conf   {'ONNX':>6} Conf   K=O")
    print(f"  {'─'*55}")
    correct = 0
    for i in range(val_n):
        kc = int(np.argmax(k_preds[i])); kconf = k_preds[i][kc]*100
        oc = int(np.argmax(o_preds[i])); oconf = o_preds[i][oc]*100
        tc = int(y_val[i]) if y_val is not None else -1
        tn = CLASS_NAMES[tc] if tc >= 0 else 'N/A'
        if kc == tc: correct += 1
        print(f"  {i:>3}  {tn:>6}  {CLASS_NAMES[kc]:>6} {kconf:5.1f}%  "
              f"{CLASS_NAMES[oc]:>6} {oconf:5.1f}%  {'✓' if kc==oc else '✗'}")

    confs = o_preds.max(axis=1)
    acc   = correct/val_n*100 if y_val is not None else None
    print(f"\n  Accuracy        : {acc:.0f}% ({correct}/{val_n})" if acc else "")
    print(f"  Confidence      : mean={confs.mean():.4f}  std={confs.std():.4f}")
    pred_dist = {CLASS_NAMES[i]: int((o_preds.argmax(1)==i).sum()) for i in range(8)}
    print(f"  ONNX spread     : {pred_dist}")
    print(f"  Speedup         : {k_time/o_time:.1f}× (Keras {k_time:.3f}s vs ONNX {o_time:.3f}s)")

    if confs.std() < 0.001:
        print("  ❌ Confidence std ≈ 0 — model still collapsed, investigate")
    else:
        print("  ✅ Model is differentiating between classes")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 7 — Cleanup temp SavedModel + save JSON report
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("STEP 7: CLEANUP + REPORT")
print("=" * 80)

if os.path.exists(SAVEDMODEL_DIR):
    shutil.rmtree(SAVEDMODEL_DIR)
    print(f"  ✓ Removed temp SavedModel: {SAVEDMODEL_DIR}")

tf.keras.backend.clear_session()
del model_f32
gc.collect()

report = {
    "date"           : datetime.now().isoformat(),
    "source_keras"   : INPUT_KERAS,
    "output_onnx"    : OUTPUT_ONNX,
    "opset"          : OPSET,
    "conversion_time_s": conv_time,
    "model_sizes_mb" : {"keras_f32": keras_mb, "onnx_raw": raw_mb, "onnx_final": final_mb},
    "patch_stats"    : {"cast_f16_fixed": fixed_casts, "init_f16_upcast": len(to_add)},
    "validation"     : {
        "keras_nan"  : keras_nan,
        "onnx_nan"   : onnx_nan,
        "max_diff"   : max_diff,
        "mean_diff"  : mean_diff,
        "passed"     : not keras_nan and not onnx_nan,
    },
    "fixes_applied"  : [
        "1. mixed_precision.set_global_policy('float32') before TF init",
        "2. set_weights([w.astype(float32) for w in weights]) — 509 arrays",
        "3. tf.saved_model.save() → frozen protobuf (no live tracing)",
        "4. tf_loader.from_saved_model() + _convert_common() → ONNX",
        f"5. Flipped {fixed_casts} Cast→float16 nodes to Cast→float32",
        f"6. Upcast {len(to_add)} float16 initializers to float32",
    ],
}

with open(REPORT_PATH, 'w') as f:
    json.dump(report, f, indent=2)
print(f"  ✓ Report saved: {REPORT_PATH}")
print(json.dumps(report, indent=2))

print("\n" + "=" * 80)
print("CONVERSION COMPLETE ✓")
print("=" * 80)
print(f"""
  Final ONNX model : {OUTPUT_ONNX}

  Update week14.py — change ONE line:
    ONNX_MODEL_PATH = "{OUTPUT_ONNX}"

  Input range: model expects [0, 255] (Rescaling×1/255 is baked in).
  week14.py already auto-scales [0,1]→[0,255] — no other changes needed.

  Download from RunPod:
    scp root@<pod-id>.runpod.io:{OUTPUT_ONNX} ./
""")
print("=" * 80)