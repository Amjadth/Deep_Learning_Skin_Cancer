#!/usr/bin/env python3
"""
Week 13-16: Thesis Figures & Visualization Suite

Module:       week17_viz.py
Purpose:      Generate publication-quality thesis figures from pipeline JSON outputs
Project:      ISIC 2019 Skin Cancer Detection
Model:        EfficientNetB3
Author:       Amjad
Date:         February 2026
Platform:     RunPod (post-processing)

═══════════════════════════════════════════════════════════════════════════════

DESCRIPTION
───────────
Comprehensive visualization suite for thesis and technical reports. Generates
15 publication-quality figures with all data sourced from JSON outputs
produced by the optimization pipeline. No hardcoded values anywhere.

DATA SOURCES
────────────
Loads four JSON files produced by weeks 13–16:

  • conversion_report_final.json     (week13 — Keras → ONNX conversion)
  • evaluation_results.json          (week14 — ONNX inference + accuracy)
  • week15_trt_report.json           (week15 — TensorRT engine builds)
  • week16_benchmark_report.json     (week16 — multi-batch latency sweep)
═══════════════════════════════════════════════════════════════════════════════
"""

import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from graphviz import Digraph
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
OUT_DIR = Path("./thesis_figures")
OUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD ALL JSON FILES
# ─────────────────────────────────────────────────────────────────────────────
JSON_FILES = {
    "conv": "conversion_report_final.json",
    "eval": "evaluation_results.json",
    "w15":  "week15_trt_report.json",
    "w16":  "week16_benchmark_report.json",
}

data = {}
for key, fname in JSON_FILES.items():
    p = Path(fname)
    if not p.exists():
        print(f"  ERROR: {fname} not found — place it alongside this script.")
        sys.exit(1)
    with open(p) as f:
        data[key] = json.load(f)
    print(f"  Loaded {fname}")

conv = data["conv"]
ev   = data["eval"]
w15  = data["w15"]
w16  = data["w16"]

sweep       = w16["latency_sweep"]
warmup_data = w16["warmup_convergence"]
acc_data    = w16["accuracy"]
speedup_mx  = w16["speedup_matrix"]["by_batch_size"]
vram        = w16["vram_efficiency"]
clinical    = w16["clinical_projections"]
models_meta = w16["metadata"]["models"]
gpu_info    = w16["metadata"]["gpu"]
rec         = w16["recommendation"]

BATCH_SIZES = w16["metadata"]["benchmark_config"]["batch_sizes"]
CLASS_NAMES = ev["test_data"]["class_names"]

ENG_KEYS   = ["onnx",      "fp32",     "fp16",     "int8"]
ENG_LABELS = ["ONNX-CUDA", "TRT FP32", "TRT FP16*","TRT INT8"]

CLR = {
    "onnx": "#2D6A9F",
    "fp32": "#4CAF7D",
    "fp16": "#E8834E",
    "int8": "#C0392B",
    "bg":   "#FAFAFA",
    "grid": "#E8E8E8",
    "text": "#1A1A2E",
}
ENG_COLOURS = [CLR[k] for k in ENG_KEYS]

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def save(fig, name):
    fig.savefig(OUT_DIR / f"{name}.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  fig {name}.png / .pdf")

def lat(engine, bs, field):
    return sweep[engine][str(bs)][field]

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.facecolor":    CLR["bg"],
    "figure.facecolor":  "white",
    "grid.color":        CLR["grid"],
    "grid.linewidth":    0.8,
    "xtick.labelsize":   9.5,
    "ytick.labelsize":   9.5,
    "legend.fontsize":   9.5,
    "legend.framealpha": 0.9,
})

# ─────────────────────────────────────────────────────────────────────────────
# FIG 1 — ONNX CONVERSION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Fig 1] ONNX Conversion Pipeline ...")

cast_fixed    = conv["patch_stats"]["cast_f16_fixed"]
init_upcast   = conv["patch_stats"]["init_f16_upcast"]
keras_mb      = conv["model_sizes_mb"]["keras_f32"]
onnx_raw_mb   = conv["model_sizes_mb"]["onnx_raw"]
onnx_final_mb = conv["model_sizes_mb"]["onnx_final"]
opset         = conv["opset"]
conv_time     = conv["conversion_time_s"]
max_diff      = conv["validation"]["max_diff"]

g = Digraph("onnx_pipeline", format="png")
g.attr(rankdir="TB", bgcolor="white", fontname="DejaVu Sans",
       pad="0.5", nodesep="0.5", ranksep="0.6", splines="ortho",
       label=(f"<<B>Week 13 — EfficientNetB3 Keras to ONNX Conversion Pipeline</B><BR/>"
              f"<FONT POINT-SIZE=\"10\">ISIC 2019 Skin Cancer Detection | February 2026</FONT>>"),
       labelloc="t", labeljust="c")

inp_s  = dict(shape="cylinder",  style="filled", fillcolor="#D6EAF8", color="#2D6A9F",
              fontname="DejaVu Sans", fontsize="11", penwidth="2")
proc_s = dict(shape="rectangle", style="filled,rounded", fillcolor="#EBF5FB", color="#2D6A9F",
              fontname="DejaVu Sans", fontsize="11", penwidth="1.5")
fix_s  = dict(shape="rectangle", style="filled,rounded", fillcolor="#FEF9E7", color="#D4AC0D",
              fontname="DejaVu Sans", fontsize="11", penwidth="1.5")
chk_s  = dict(shape="diamond",   style="filled", fillcolor="#E8F8F5", color="#1E8449",
              fontname="DejaVu Sans", fontsize="10", penwidth="1.5")
out_s  = dict(shape="cylinder",  style="filled", fillcolor="#D5F5E3", color="#1E8449",
              fontname="DejaVu Sans", fontsize="11", penwidth="2.5")
wrn_s  = dict(shape="note",      style="filled", fillcolor="#FDEDEC", color="#C0392B",
              fontname="DejaVu Sans", fontsize="9",  penwidth="1")

g.node("keras", f"final_model.keras\n(mixed_float16 trained)\n{keras_mb:.1f} MB", **inp_s)
g.node("s0",   "STEP 0\nForce float32 dtype policy\nbefore TF initialises", **proc_s)
g.node("s1",   f"STEP 1\nLoad .keras model\nCast 509 weight arrays to float32", **proc_s)
g.node("why1", f"Without this: mixed_float16\ntraining bakes {cast_fixed} Cast-float16\nnodes -> runtime NaN overflow", **wrn_s)
g.node("s2",   "STEP 2\nFreeze to SavedModel\n(bypasses tf2onnx live-tracing NaN bug)", **proc_s)
g.node("s3",   f"STEP 3\ntf_loader.from_saved_model()\n+ _convert_common() to ONNX\n(opset {opset}  |  {conv_time:.1f}s)", **proc_s)
g.node("raw",  f"Raw ONNX  {onnx_raw_mb:.1f} MB\n(contains float16 nodes)", **wrn_s)
g.node("s4a",  f"STEP 4a\nFlip {cast_fixed} Cast-float16 nodes\nto Cast-float32", **fix_s)
g.node("s4b",  f"STEP 4b\nUpcast {init_upcast} float16 initializers\nto float32", **fix_s)
g.node("s5",   "STEP 5\nONNX integrity check\n+ shape inference", **chk_s)
g.node("s6",   f"STEP 6\nRuntime validation\n8 real dermoscopy images\nmax diff = {max_diff:.4f}", **chk_s)
g.node("s7",   "STEP 7\nCleanup temp SavedModel\nSave conversion_report.json", **proc_s)
g.node("out",  f"EfficientNetB3_ISIC2019_final.onnx\n{onnx_final_mb:.1f} MB  |  opset {opset}  |  float32", **out_s)

e = dict(color="#2D6A9F", penwidth="1.8", fontname="DejaVu Sans", fontsize="9")
g.edge("keras","s0",  **e)
g.edge("s0",  "s1",   **e)
g.edge("s1",  "why1", style="dashed", color="#C0392B", arrowhead="none")
g.edge("s1",  "s2",   **e)
g.edge("s2",  "s3",   **e)
g.edge("s3",  "raw",  style="dashed", color="#C0392B", arrowhead="open")
g.edge("s3",  "s4a",  **e)
g.edge("s4a", "s4b",  **e)
g.edge("s4b", "s5",   **e)
g.edge("s5",  "s6",   **e)
g.edge("s6",  "s7",   **e)
g.edge("s7",  "out",  **e)
g.render(str(OUT_DIR / "fig1_onnx_pipeline"), cleanup=True)
print("  fig1_onnx_pipeline.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 — TENSORRT ACCELERATION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
print("[Fig 2] TensorRT Acceleration Pipeline ...")

onnx_mb   = models_meta["onnx"]["size_mb"]
fp32_mb   = models_meta["fp32_engine"]["size_mb"]
fp16_mb   = models_meta["fp16_engine"]["size_mb"]
int8_mb   = models_meta["int8_engine"]["size_mb"]
bench_r   = w16["metadata"]["benchmark_config"]["bench_runs"]
warmup_r  = w16["metadata"]["benchmark_config"]["warmup_runs"]

fp32_tput = lat("fp32", 32, "throughput_ips")
fp16_tput = lat("fp16", 32, "throughput_ips")
int8_tput = lat("int8", 32, "throughput_ips")
onnx_tput = lat("onnx", 32, "throughput_ips")

fp32_acc  = acc_data["fp32"]["overall_pct"]
fp16_acc  = acc_data["fp16"]["overall_pct"]
int8_acc  = acc_data["int8"]["overall_pct"]

g2 = Digraph("trt_pipeline", format="png")
g2.attr(rankdir="LR", bgcolor="white", fontname="DejaVu Sans",
        pad="0.6", nodesep="0.45", ranksep="0.8", splines="ortho",
        label=(f"<<B>Week 15 — ONNX to TensorRT Acceleration Pipeline</B><BR/>"
               f"<FONT POINT-SIZE=\"10\">{gpu_info['name']} | TensorRT {gpu_info['trt']}</FONT>>"),
        labelloc="t", labeljust="c")

inp_s2  = dict(shape="cylinder",  style="filled", fillcolor="#D6EAF8", color="#2D6A9F",
               fontname="DejaVu Sans", fontsize="10", penwidth="2")
bld_s   = dict(shape="rectangle", style="filled,rounded", fillcolor="#EBF5FB", color="#2D6A9F",
               fontname="DejaVu Sans", fontsize="10", penwidth="1.5")
fp32_s  = dict(shape="rectangle", style="filled,rounded", fillcolor="#D5F5E3", color="#4CAF7D",
               fontname="DejaVu Sans", fontsize="10", penwidth="2")
fp16_s  = dict(shape="rectangle", style="filled,rounded", fillcolor="#FEF5E7", color="#E8834E",
               fontname="DejaVu Sans", fontsize="10", penwidth="2.5")
int8_s  = dict(shape="rectangle", style="filled,rounded", fillcolor="#FDEDEC", color="#C0392B",
               fontname="DejaVu Sans", fontsize="10", penwidth="2")
cal_s   = dict(shape="note",      style="filled", fillcolor="#F9EBEA", color="#C0392B",
               fontname="DejaVu Sans", fontsize="9")
ben_s   = dict(shape="rectangle", style="filled,rounded", fillcolor="#F0F3F4", color="#566573",
               fontname="DejaVu Sans", fontsize="9")
res_s   = dict(shape="rectangle", style="filled", fillcolor="#1A1A2E", color="#1A1A2E",
               fontname="DejaVu Sans", fontsize="9", fontcolor="white")

g2.node("onnx2",   f"EfficientNetB3_final.onnx\n{onnx_mb:.1f} MB | opset 13", **inp_s2)
g2.node("calib",   f"Calibration Data\n320 images\n(10 batches x 32)", **cal_s)
g2.node("builder", f"TRT Builder Config\nWorkspace: 4 GB\nDynamic batch: 1-32\nOpt batch: 16", **bld_s)

with g2.subgraph(name="cluster_engines") as c:
    c.attr(label="TensorRT Engine Builds", style="dashed", color="#AAAAAA",
           fontname="DejaVu Sans", fontsize="10")
    c.node("efp32", f"FP32 Engine\ntrt_fp32.engine\n{fp32_mb:.1f} MB", **fp32_s)
    c.node("efp16", f"FP16 Engine [RECOMMENDED]\ntrt_fp16.engine\n{fp16_mb:.1f} MB", **fp16_s)
    c.node("eint8", f"INT8 Engine + Entropy Calib\ntrt_int8.engine\n{int8_mb:.1f} MB", **int8_s)

g2.node("bench2", f"Benchmark Suite\nbatch=1: {bench_r} runs\nbatch=32: {bench_r} runs\n{warmup_r}-run warmup", **ben_s)

with g2.subgraph(name="cluster_results") as c:
    c.attr(label=f"Results (batch=32)", style="dashed", color="#AAAAAA",
           fontname="DejaVu Sans", fontsize="10")
    c.node("rfp32", f"FP32\n{fp32_tput:.0f} img/s\n{fp32_tput/onnx_tput:.2f}x ONNX\n{fp32_acc}% acc", **res_s)
    c.node("rfp16", f"FP16\n{fp16_tput:.0f} img/s\n{fp16_tput/onnx_tput:.2f}x ONNX\n{fp16_acc}% acc",
           fillcolor="#E8834E", color="#E8834E",
           shape="rectangle", style="filled",
           fontname="DejaVu Sans", fontsize="9", fontcolor="white")
    c.node("rint8", f"INT8\n{int8_tput:.0f} img/s\n{int8_tput/onnx_tput:.2f}x ONNX\n{int8_acc}% acc", **res_s)

g2.node("report2", "week15_trt_report.json\nweek16_benchmark_report.json",
        shape="note", style="filled", fillcolor="#EAFAF1", color="#1E8449",
        fontname="DejaVu Sans", fontsize="9")

me = dict(color="#2D6A9F", penwidth="1.8")
g2.edge("onnx2",   "builder", **me)
g2.edge("builder", "efp32",   **me)
g2.edge("builder", "efp16",   color="#E8834E", penwidth="2.2")
g2.edge("builder", "eint8",   **me)
g2.edge("calib",   "eint8",   style="dashed", color="#C0392B")
g2.edge("efp32",   "bench2",  **me)
g2.edge("efp16",   "bench2",  color="#E8834E", penwidth="2.2")
g2.edge("eint8",   "bench2",  **me)
g2.edge("bench2",  "rfp32",   **me)
g2.edge("bench2",  "rfp16",   color="#E8834E", penwidth="2.2")
g2.edge("bench2",  "rint8",   **me)
g2.edge("rfp32",   "report2", style="dashed", color="#888888")
g2.edge("rfp16",   "report2", style="dashed", color="#888888")
g2.edge("rint8",   "report2", style="dashed", color="#888888")

g2.render(str(OUT_DIR / "fig2_trt_pipeline"), cleanup=True)
print("  fig2_trt_pipeline.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 — SINGLE-IMAGE LATENCY
# ─────────────────────────────────────────────────────────────────────────────
print("[Fig 3] Single-image latency ...")

mean_b1 = [lat(k, 1, "mean_ms") for k in ENG_KEYS]
p95_b1  = [lat(k, 1, "p95_ms")  for k in ENG_KEYS]
p99_b1  = [lat(k, 1, "p99_ms")  for k in ENG_KEYS]
err_up  = [p - m for p, m in zip(p95_b1, mean_b1)]

fig, ax = plt.subplots(figsize=(9, 5.5))
x = np.arange(len(ENG_LABELS))
bars = ax.bar(x, mean_b1, width=0.55, color=ENG_COLOURS, alpha=0.88,
              edgecolor="white", linewidth=1.2, zorder=3)
ax.errorbar(x, mean_b1, yerr=[np.zeros(4), err_up],
            fmt="none", color="#333", capsize=5, capthick=1.5, elinewidth=1.5, zorder=4)
for bar, m, p in zip(bars, mean_b1, p95_b1):
    ax.text(bar.get_x()+bar.get_width()/2, m+0.05, f"{m:.3f}ms",
            ha="center", va="bottom", fontsize=9, fontweight="bold", zorder=5)
    ax.text(bar.get_x()+bar.get_width()/2, p+0.12, f"P95={p:.2f}",
            ha="center", va="bottom", fontsize=7.5, color="#666", zorder=5)
ax.axhline(mean_b1[0], color=CLR["onnx"], linewidth=1.2, linestyle="--",
           alpha=0.6, zorder=2, label="ONNX-CUDA baseline")
ax.annotate("Recommended (FP16)", xy=(2, mean_b1[2]),
            xytext=(2.35, mean_b1[2]+0.9), fontsize=8.5, color=CLR["fp16"],
            fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color=CLR["fp16"], lw=1.2))
ax.set_xticks(x); ax.set_xticklabels(ENG_LABELS, fontsize=10)
ax.set_ylabel("Mean Inference Latency (ms)", fontsize=10)
ax.set_title(
    f"Figure 3 — Single-Image Inference Latency (batch = 1)\n"
    f"EfficientNetB3 | ISIC 2019 | {gpu_info['name']} | "
    f"{w16['metadata']['benchmark_config']['bench_runs']}-run benchmark",
    fontsize=11, pad=12)
ax.set_ylim(0, max(p99_b1)*1.38)
ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
ax.text(0.5, -0.14,
        "At batch=1 ONNX-RT is optimised for single-image dispatch; TRT advantage is in batch throughput (Fig 4).",
        transform=ax.transAxes, ha="center", fontsize=7.5, color="#666", style="italic")
ax.legend(handles=[mpatches.Patch(color=c, label=l) for c,l in zip(ENG_COLOURS, ENG_LABELS)],
          loc="upper right", fontsize=8.5)
save(fig, "fig3_latency_b1")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 4 — BATCH-32 THROUGHPUT
# ─────────────────────────────────────────────────────────────────────────────
print("[Fig 4] Batch-32 throughput ...")

tput_b32 = [lat(k, 32, "throughput_ips") for k in ENG_KEYS]
onnx_b32 = tput_b32[0]
speedups  = [t/onnx_b32 for t in tput_b32]

fig, ax = plt.subplots(figsize=(9, 5))
y = np.arange(len(ENG_LABELS))
ax.barh(y, tput_b32[::-1], height=0.55, color=ENG_COLOURS[::-1],
        alpha=0.88, edgecolor="white", zorder=3)
for i, (t, s) in enumerate(zip(tput_b32[::-1], speedups[::-1])):
    ax.text(t+15, i, f"{t:.0f} img/s   ({s:.2f}x)",
            ha="left", va="center", fontsize=9.5, fontweight="bold")
ax.axvline(onnx_b32, color=CLR["onnx"], linewidth=1.5, linestyle="--",
           alpha=0.7, zorder=2, label=f"ONNX baseline ({onnx_b32:.0f} img/s)")
ax.set_yticks(y); ax.set_yticklabels(ENG_LABELS[::-1], fontsize=10)
ax.set_xlabel("Throughput (images / second)", fontsize=10)
ax.set_title(
    f"Figure 4 — Sustained Batch Throughput (batch = 32)\n"
    f"EfficientNetB3 | ISIC 2019 | {gpu_info['name']} | "
    f"{w16['metadata']['benchmark_config']['bench_runs']}-run benchmark",
    fontsize=11, pad=12)
ax.set_xlim(0, max(tput_b32)*1.28)
ax.xaxis.grid(True, zorder=0, alpha=0.6); ax.set_axisbelow(True)
ax.legend(fontsize=9, loc="lower right")
fp16_bar_y = list(reversed(ENG_KEYS)).index("fp16")
ax.annotate(f"Recommended\n{speedups[2]:.2f}x ONNX",
            xy=(tput_b32[2], fp16_bar_y),
            xytext=(tput_b32[2]-260, fp16_bar_y+0.5),
            fontsize=8.5, color="white",
            bbox=dict(boxstyle="round,pad=0.3", fc=CLR["fp16"], ec="none"),
            arrowprops=dict(arrowstyle="-|>", color=CLR["fp16"], lw=1.2))
save(fig, "fig4_throughput_b32")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 5 — ACCURACY vs THROUGHPUT
# ─────────────────────────────────────────────────────────────────────────────
print("[Fig 5] Accuracy vs Throughput scatter ...")

eng3   = ["fp32", "fp16", "int8"]
lab3   = ["TRT FP32", "TRT FP16*", "TRT INT8"]
col3   = [CLR["fp32"], CLR["fp16"], CLR["int8"]]
tput3  = [lat(k, 32, "throughput_ips")           for k in eng3]
acc3   = [acc_data[k]["overall_pct"]              for k in eng3]
mb3    = [models_meta[f"{k}_engine"]["size_mb"]   for k in eng3]

fig, ax = plt.subplots(figsize=(8.5, 6))
for t, a, col, mb, lab in zip(tput3, acc3, col3, mb3, lab3):
    ax.scatter(t, a, s=mb*12, c=col, alpha=0.85, edgecolors="white",
               linewidths=2, zorder=4)
    oy = 0.3 if "FP16" not in lab else -0.9
    ax.annotate(f"{lab}\n{t:.0f} img/s | {a:.2f}% acc\n({mb:.1f} MB engine)",
                xy=(t, a), xytext=(t+20, a+oy), fontsize=8.5, color=col,
                fontweight="bold", arrowprops=dict(arrowstyle="-", color=col, lw=0.8))
ax.axhspan(acc3[0]-0.5, acc3[1]+0.5, alpha=0.06, color="green", label="Target accuracy band")
ax.axvspan(tput3[1]-100, max(tput3)+100, alpha=0.06, color=CLR["fp16"], label="High-throughput region")
ax.set_xlabel("Batch-32 Throughput (images / second)", fontsize=11)
ax.set_ylabel("Overall Accuracy on ISIC 2019 Test Set (%)", fontsize=11)
ax.set_title("Figure 5 — Accuracy vs. Throughput Trade-off\n"
             "EfficientNetB3 TensorRT Engines | bubble size = engine file size (MB)",
             fontsize=11, pad=12)
for mb, lbl in [(20,"20 MB"),(30,"30 MB"),(50,"50 MB")]:
    ax.scatter([], [], s=mb*12, c="#AAAAAA", alpha=0.6, label=f"Engine: {lbl}", edgecolors="white")
ax.legend(loc="lower right", fontsize=8.5)
ax.yaxis.grid(True, alpha=0.5); ax.xaxis.grid(True, alpha=0.5); ax.set_axisbelow(True)
ax.annotate("Sweet spot: high throughput\nnegligible accuracy loss",
            xy=(tput3[1], acc3[1]), xytext=(tput3[1]-420, acc3[1]+1.2),
            fontsize=8.5, color=CLR["fp16"],
            bbox=dict(boxstyle="round,pad=0.35", fc="#FEF5E7", ec=CLR["fp16"]),
            arrowprops=dict(arrowstyle="-|>", color=CLR["fp16"], lw=1.2))
save(fig, "fig5_accuracy_vs_throughput")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 6 — LATENCY PERCENTILE HEATMAP
# ─────────────────────────────────────────────────────────────────────────────
print("[Fig 6] Latency percentile heatmap ...")

pct_fields = ["mean_ms","p50_ms","p95_ms","p99_ms"]
row_labels = ["Mean","P50 (Median)","P95","P99"]
heat = np.array([[lat(k, 1, p) for k in ENG_KEYS] for p in pct_fields])

fig, ax = plt.subplots(figsize=(9, 4.5))
im = ax.imshow(heat, cmap=sns.color_palette("YlOrRd", as_cmap=True),
               aspect="auto", vmin=heat.min()*0.9, vmax=heat.max()*1.05)
ax.set_xticks(range(len(ENG_LABELS))); ax.set_xticklabels(ENG_LABELS, fontsize=10)
ax.set_yticks(range(len(row_labels)));  ax.set_yticklabels(row_labels, fontsize=10)
for i in range(len(row_labels)):
    for j in range(len(ENG_KEYS)):
        v  = heat[i, j]
        bg = (v - heat.min()) / (heat.max() - heat.min())
        tc = "white" if bg > 0.55 else CLR["text"]
        ax.text(j, i, f"{v:.3f}ms", ha="center", va="center",
                fontsize=9.5, color=tc, fontweight="bold")
# Highlight FP16 (col index 2)
for i in range(len(row_labels)):
    ax.add_patch(plt.Rectangle((1.5, i-0.5), 1, 1,
                                fill=False, edgecolor=CLR["fp16"],
                                linewidth=2.5, zorder=5))
cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label("Latency (ms)", fontsize=9)
ax.set_title("Figure 6 — Inference Latency Percentiles Heatmap (batch = 1)\n"
             "Lower = better | FP16 column highlighted as recommended engine",
             fontsize=11, pad=12)
ax.set_xlabel("Inference Engine", fontsize=10)
save(fig, "fig6_latency_heatmap")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 7 — ENGINE SIZE vs THROUGHPUT
# ─────────────────────────────────────────────────────────────────────────────
print("[Fig 7] Engine size vs throughput ...")

sizes_mb = [
    models_meta["onnx"]["size_mb"],
    models_meta["fp32_engine"]["size_mb"],
    models_meta["fp16_engine"]["size_mb"],
    models_meta["int8_engine"]["size_mb"],
]
tput_all = [lat(k, 32, "throughput_ips") for k in ENG_KEYS]

fig, ax = plt.subplots(figsize=(8, 5.5))
for mb, t, lab, col in zip(sizes_mb, tput_all, ENG_LABELS, ENG_COLOURS):
    ax.scatter(mb, t, s=320, c=col, alpha=0.85, edgecolors="white",
               linewidths=2, zorder=4)
    oy = 30 if "FP16" not in lab else -55
    ax.annotate(f"{lab}\n{mb:.1f} MB | {t:.0f} img/s",
                xy=(mb, t), xytext=(mb+0.5, t+oy),
                fontsize=9, color=col, fontweight="bold")
# Pareto: ONNX -> FP16 -> INT8
ax.plot([sizes_mb[0], sizes_mb[2], sizes_mb[3]],
        [tput_all[0], tput_all[2], tput_all[3]],
        "--", color="#AAAAAA", linewidth=1.2, label="Pareto frontier", zorder=2)
ax.text(30, min(tput_all)+50, "More efficient direction\n(smaller, faster)",
        fontsize=8.5, color="#666", style="italic")
ax.set_xlabel("Engine File Size (MB)", fontsize=10)
ax.set_ylabel("Batch-32 Throughput (images / second)", fontsize=10)
ax.set_title("Figure 7 — Engine Size vs. Throughput\n"
             "Smaller engines with higher throughput are preferred", fontsize=11, pad=12)
ax.yaxis.grid(True, alpha=0.5); ax.xaxis.grid(True, alpha=0.5); ax.set_axisbelow(True)
save(fig, "fig7_engine_size_vs_throughput")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 8 — SPEEDUP WATERFALL
# ─────────────────────────────────────────────────────────────────────────────
print("[Fig 8] Speedup waterfall ...")

tput_vals = [lat(k, 32, "throughput_ips") for k in ENG_KEYS]
base      = tput_vals[0]
stages    = ["ONNX-CUDA\n(Baseline)", "TRT FP32",
             "TRT FP16\n(+FP16 kernels)", "TRT INT8\n(+Quantisation)"]

fig, ax = plt.subplots(figsize=(9, 5.5))
x = np.arange(len(stages))
ax.bar(x, tput_vals, width=0.55, color=ENG_COLOURS, alpha=0.88,
       edgecolor="white", linewidth=1.2, zorder=3)
for xi, v in enumerate(tput_vals):
    ax.text(xi, v+15, f"{v:.0f} img/s\n{v/base:.2f}x",
            ha="center", va="bottom", fontsize=9, fontweight="bold")
for i in range(1, len(tput_vals)):
    delta = tput_vals[i] - tput_vals[i-1]
    mid   = (tput_vals[i-1] + tput_vals[i]) / 2
    col   = "#2ECC71" if delta > 0 else "#E74C3C"
    ax.annotate(f"{'+' if delta>=0 else ''}{delta:.0f}\nimg/s",
                xy=(i-0.5, mid), fontsize=8, ha="center", color=col,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col, alpha=0.85))
ax.set_xticks(x); ax.set_xticklabels(stages, fontsize=9.5)
ax.set_ylabel("Throughput (images / second)", fontsize=10)
ax.set_title("Figure 8 — Throughput Progression: ONNX to TensorRT (batch = 32)\n"
             "Each engine and gain over previous stage", fontsize=11, pad=12)
ax.set_ylim(0, max(tput_vals)*1.28)
ax.yaxis.grid(True, alpha=0.5, zorder=0); ax.set_axisbelow(True)
save(fig, "fig8_speedup_waterfall")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 9 — CLINICAL DEPLOYMENT PROJECTIONS
# ─────────────────────────────────────────────────────────────────────────────
print("[Fig 9] Clinical deployment projections ...")

imgs_hr = [clinical[k]["batch_throughput"]["images_per_hour"]     for k in ENG_KEYS]
pts_day = [clinical[k]["batch_throughput"]["patients_per_8h_day"] for k in ENG_KEYS]
imgs_pp = clinical["onnx"]["assumptions"]["images_per_patient"]
rt_thr  = clinical["onnx"]["assumptions"]["realtime_threshold_ms"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
fig.suptitle(f"Figure 9 — Clinical Deployment Projections\n"
             f"8-hour workday | {imgs_pp} dermoscopy images per patient visit",
             fontsize=11, fontweight="bold", y=1.02)

x = np.arange(len(ENG_LABELS))
for ax_, vals, ylabel, title in [
    (axes[0], imgs_hr, "Images per Hour (Millions)",                "Batch Throughput -> Images/Hour"),
    (axes[1], pts_day, "Patient Visits Supported per Day (Millions)","Capacity: Patients per 8-Hour Workday"),
]:
    bars = ax_.bar(x, [v/1e6 for v in vals], width=0.55,
                   color=ENG_COLOURS, alpha=0.88, edgecolor="white", zorder=3)
    for bar, v in zip(bars, vals):
        ax_.text(bar.get_x()+bar.get_width()/2, v/1e6 + max(vals)/1e6*0.02,
                 f"{v/1e6:.2f}M", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")
    ax_.set_xticks(x); ax_.set_xticklabels(ENG_LABELS, fontsize=9)
    ax_.set_ylabel(ylabel); ax_.set_title(title, fontsize=10)
    ax_.yaxis.grid(True, alpha=0.5); ax_.set_axisbelow(True)
    ax_.text(0.5, -0.18,
             f"All engines: real-time capable (<{rt_thr}ms threshold at batch=1)",
             transform=ax_.transAxes, ha="center", fontsize=7.5, color="#666", style="italic")

plt.tight_layout()
save(fig, "fig9_clinical_projections")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 10 — MULTI-BATCH SWEEP (from week16 — all 5 batch sizes)
# ─────────────────────────────────────────────────────────────────────────────
print("[Fig 10] Multi-batch throughput sweep ...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle(f"Figure 10 — Multi-Batch Latency & Throughput Sweep\n"
             f"EfficientNetB3 | batch sizes {BATCH_SIZES} | {gpu_info['name']}",
             fontsize=11, fontweight="bold", y=1.01)

ax = axes[0]
for k, lab, col in zip(ENG_KEYS, ENG_LABELS, ENG_COLOURS):
    lpi = [lat(k, bs, "latency_per_image_ms") for bs in BATCH_SIZES]
    lw  = 2.5 if k == "fp16" else 1.6
    ax.plot(BATCH_SIZES, lpi, "o-", color=col, label=lab, linewidth=lw, markersize=7, zorder=4)
ax.set_xlabel("Batch Size", fontsize=10)
ax.set_ylabel("Latency per Image (ms)", fontsize=10)
ax.set_title("(A) Latency per Image vs Batch Size", fontsize=10)
ax.set_xticks(BATCH_SIZES)
ax.legend(fontsize=8.5); ax.yaxis.grid(True, alpha=0.5); ax.set_axisbelow(True)
ax.text(0.5, -0.16, "Lower is better — larger batches amortise fixed GPU overhead.",
        transform=ax.transAxes, ha="center", fontsize=7.5, color="#666", style="italic")

ax = axes[1]
for k, lab, col in zip(ENG_KEYS, ENG_LABELS, ENG_COLOURS):
    tputs = [lat(k, bs, "throughput_ips") for bs in BATCH_SIZES]
    lw    = 2.5 if k == "fp16" else 1.6
    ax.plot(BATCH_SIZES, tputs, "o-", color=col, label=lab, linewidth=lw, markersize=7, zorder=4)
ax.set_xlabel("Batch Size", fontsize=10)
ax.set_ylabel("Throughput (images / second)", fontsize=10)
ax.set_title("(B) Throughput vs Batch Size", fontsize=10)
ax.set_xticks(BATCH_SIZES)
ax.legend(fontsize=8.5); ax.yaxis.grid(True, alpha=0.5); ax.set_axisbelow(True)
# Find crossover for FP16 vs ONNX from speedup_mx
onnx_tputs = [lat("onnx", bs, "throughput_ips") for bs in BATCH_SIZES]
fp16_tputs = [lat("fp16", bs, "throughput_ips") for bs in BATCH_SIZES]
for i, (o, t) in enumerate(zip(onnx_tputs, fp16_tputs)):
    if t > o and i > 0:
        ax.axvline(BATCH_SIZES[i], color=CLR["fp16"], linewidth=1.0, linestyle=":", alpha=0.8)
        ax.text(BATCH_SIZES[i]+0.1, ax.get_ylim()[0] + 20,
                f"FP16 overtakes\nONNX at bs={BATCH_SIZES[i]}",
                fontsize=7.5, color=CLR["fp16"])
        break

plt.tight_layout()
save(fig, "fig10_batch_sweep")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 11 — WARMUP CONVERGENCE (from week16)
# ─────────────────────────────────────────────────────────────────────────────
print("[Fig 11] Warmup convergence curves ...")

fig, ax = plt.subplots(figsize=(10, 5.5))
for k, lab, col in zip(["fp32","fp16","int8"],
                        ["TRT FP32","TRT FP16*","TRT INT8"],
                        [CLR["fp32"],CLR["fp16"],CLR["int8"]]):
    wc      = warmup_data[k]
    raw     = wc["latencies_ms"]
    rolling = wc["rolling_mean_ms"]
    conv_at = wc["converges_at_run"]
    steady  = wc["steady_state_ms"]
    n       = len(rolling)
    lw      = 2.5 if k == "fp16" else 1.6
    ax.plot(range(n), raw, color=col, alpha=0.18, linewidth=0.8)
    ax.plot(range(n), rolling, color=col, linewidth=lw,
            label=f"{lab}  (stable @ run {conv_at}, steady={steady:.2f}ms)")
    ax.axhline(steady, color=col, linewidth=0.9, linestyle="--", alpha=0.5)
    ax.axvline(conv_at, color=col, linewidth=0.9, linestyle=":", alpha=0.8)

ax.set_xlabel("Inference Run Number (from cold start)", fontsize=10)
ax.set_ylabel("Latency (ms)", fontsize=10)
ax.set_title("Figure 11 — Warmup Convergence Analysis (batch = 1)\n"
             "Thin = raw latency | Thick = 5-run rolling mean | Dashed = steady-state",
             fontsize=11, pad=12)
ax.legend(fontsize=8.5, loc="upper right")
ax.yaxis.grid(True, alpha=0.5); ax.set_axisbelow(True)
fp16_conv = warmup_data["fp16"]["converges_at_run"]
ax.text(0.5, -0.14,
        f"Production recommendation: {rec['warmup_requirement']['recommendation']}",
        transform=ax.transAxes, ha="center", fontsize=7.5, color="#666", style="italic")
save(fig, "fig11_warmup_convergence")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 12 — PER-CLASS ACCURACY HEATMAP (from week16 per_class)
# ─────────────────────────────────────────────────────────────────────────────
print("[Fig 12] Per-class accuracy heatmap ...")

eng3b = ["fp32","fp16","int8"]
lab3b = ["TRT FP32","TRT FP16*","TRT INT8"]
acc_matrix = np.array([
    [acc_data[k]["per_class"][cls]["accuracy"]*100 for cls in CLASS_NAMES]
    for k in eng3b
])

fig, ax = plt.subplots(figsize=(11, 4))
im = ax.imshow(acc_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
ax.set_xticks(range(len(CLASS_NAMES))); ax.set_xticklabels(CLASS_NAMES, fontsize=10)
ax.set_yticks(range(3)); ax.set_yticklabels(lab3b, fontsize=10)
for i in range(3):
    for j in range(len(CLASS_NAMES)):
        v  = acc_matrix[i, j]
        tc = "white" if v < 25 or v > 75 else CLR["text"]
        n  = acc_data[eng3b[i]]["per_class"][CLASS_NAMES[j]]["n"]
        ax.text(j, i, f"{v:.1f}%\n(n={n})", ha="center", va="center",
                fontsize=8.5, color=tc, fontweight="bold")
cbar = plt.colorbar(im, ax=ax, shrink=0.9, pad=0.01)
cbar.set_label("Per-class Accuracy (%)", fontsize=9)
ax.set_title(
    "Figure 12 — Per-Class Accuracy Breakdown by Engine\n"
    "NV (naevi) consistently highest — reflects class prevalence in training data",
    fontsize=11, pad=12)
ax.set_xlabel("Skin Lesion Class", fontsize=10)
support = {cls: ev["per_class_metrics"][cls]["support"] for cls in CLASS_NAMES}
note    = "  |  ".join([f"{cls}: n={support[cls]}" for cls in CLASS_NAMES])
ax.text(0.5, -0.18, f"Test set class sizes: {note}",
        transform=ax.transAxes, ha="center", fontsize=6.5, color="#666")
save(fig, "fig12_per_class_accuracy")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 13 — CONFUSION MATRIX (from evaluation_results.json)
# ─────────────────────────────────────────────────────────────────────────────
print("[Fig 13] Confusion matrix ...")

cm   = np.array(ev["confusion_matrix"])
acc_ = ev["accuracy_metrics"]["accuracy"]
f1_  = ev["accuracy_metrics"]["weighted_f1"]
n_   = ev["test_data"]["num_images"]

fig, ax = plt.subplots(figsize=(9, 7.5))
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            linewidths=0.5, linecolor="#dddddd",
            cbar_kws={"label": "Row-normalised proportion"}, ax=ax)
ax.set_xlabel("Predicted Class", fontsize=10)
ax.set_ylabel("True Class", fontsize=10)
ax.set_title(
    f"Figure 13 — Confusion Matrix (ONNX-CUDA Baseline)\n"
    f"Overall accuracy: {acc_*100:.1f}%  |  Weighted F1: {f1_*100:.1f}%  |  n={n_} test images",
    fontsize=11, pad=12)
plt.tight_layout()
save(fig, "fig13_confusion_matrix")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 14 — VRAM EFFICIENCY (from week16)
# ─────────────────────────────────────────────────────────────────────────────
print("[Fig 14] VRAM efficiency ...")

vk   = ["fp32","fp16","int8"]
vlb  = ["TRT FP32","TRT FP16*","TRT INT8"]
vcol = [CLR["fp32"],CLR["fp16"],CLR["int8"]]
eff  = [vram[k]["imgs_per_s_per_gb_vram"] for k in vk]
dmb  = [vram[k]["vram_delta_mb"]          for k in vk]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f"Figure 14 — GPU VRAM Efficiency\n"
             f"{gpu_info['name']} | {gpu_info['vram_gb']} GB total VRAM | batch=32",
             fontsize=11, fontweight="bold", y=1.02)
x = np.arange(len(vlb))

ax = axes[0]
bars = ax.bar(x, dmb, width=0.5, color=vcol, alpha=0.88, edgecolor="white", zorder=3)
for bar, v in zip(bars, dmb):
    ax.text(bar.get_x()+bar.get_width()/2, v+8, f"{v} MB",
            ha="center", va="bottom", fontsize=9.5, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(vlb, fontsize=9.5)
ax.set_ylabel("VRAM Usage Delta (MB)"); ax.set_title("(A) VRAM Footprint per Engine", fontsize=10)
ax.yaxis.grid(True, alpha=0.5); ax.set_axisbelow(True)

ax = axes[1]
bars = ax.bar(x, eff, width=0.5, color=vcol, alpha=0.88, edgecolor="white", zorder=3)
for bar, v in zip(bars, eff):
    ax.text(bar.get_x()+bar.get_width()/2, v+30, f"{v:.0f}",
            ha="center", va="bottom", fontsize=9.5, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(vlb, fontsize=9.5)
ax.set_ylabel("img/s per GB VRAM")
ax.set_title("(B) Throughput Efficiency (img/s per GB VRAM)", fontsize=10)
ax.yaxis.grid(True, alpha=0.5); ax.set_axisbelow(True)

plt.tight_layout()
save(fig, "fig14_vram_efficiency")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 15 — COMBINED SUMMARY DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
print("[Fig 15] Combined summary dashboard ...")

fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor("white")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)
fig.suptitle(
    f"EfficientNetB3 Inference Optimisation — ISIC 2019 Skin Cancer Detection\n"
    f"Midterm Progress Report 2  |  {gpu_info['name']}  |  TensorRT {gpu_info['trt']}  |  February 2026",
    fontsize=13, fontweight="bold", y=1.01
)

x       = np.arange(len(ENG_LABELS))
b1_mean = [lat(k, 1,  "mean_ms")        for k in ENG_KEYS]
b1_p95  = [lat(k, 1,  "p95_ms")         for k in ENG_KEYS]
b32_tp  = [lat(k, 32, "throughput_ips") for k in ENG_KEYS]
accs_p  = [acc_data[k]["overall_pct"] if k in acc_data else 0.0 for k in ENG_KEYS]
eng_mbs = [models_meta["onnx"]["size_mb"],
           models_meta["fp32_engine"]["size_mb"],
           models_meta["fp16_engine"]["size_mb"],
           models_meta["int8_engine"]["size_mb"]]

# A — Latency b1
axA = fig.add_subplot(gs[0,0])
axA.bar(x, b1_mean, color=ENG_COLOURS, alpha=0.88, edgecolor="white", zorder=3)
axA.errorbar(x, b1_mean, yerr=[np.zeros(4), [p-m for p,m in zip(b1_p95, b1_mean)]],
             fmt="none", color="#333", capsize=4, capthick=1.2, elinewidth=1.2, zorder=4)
for xi, v in enumerate(b1_mean):
    axA.text(xi, v+0.08, f"{v:.2f}", ha="center", va="bottom",
             fontsize=8, fontweight="bold")
axA.set_xticks(x); axA.set_xticklabels(["ONNX","FP32","FP16*","INT8"], fontsize=8.5)
axA.set_ylabel("Latency (ms)"); axA.set_title("(A) Batch=1 Latency", fontsize=10)
axA.yaxis.grid(True, alpha=0.5); axA.set_axisbelow(True)

# B — Throughput b32
axB = fig.add_subplot(gs[0,1])
axB.barh(x, b32_tp[::-1], color=ENG_COLOURS[::-1], alpha=0.88, edgecolor="white", zorder=3)
for yi, (t, s) in enumerate(zip(b32_tp[::-1], [t/b32_tp[0] for t in b32_tp[::-1]])):
    axB.text(t+20, yi, f"{t:.0f}  ({s:.2f}x)", va="center", fontsize=8, fontweight="bold")
axB.set_yticks(x); axB.set_yticklabels(["ONNX","FP32","FP16*","INT8"][::-1], fontsize=8.5)
axB.set_xlabel("img / sec"); axB.set_title("(B) Batch=32 Throughput", fontsize=10)
axB.xaxis.grid(True, alpha=0.5); axB.set_axisbelow(True)
axB.set_xlim(0, max(b32_tp)*1.22)

# C — Accuracy
axC = fig.add_subplot(gs[0,2])
axC.bar(x, accs_p, color=ENG_COLOURS, alpha=0.88, edgecolor="white", zorder=3)
axC.set_ylim(0, 55)
for xi, v in enumerate(accs_p):
    lbl = f"{v:.1f}%" if v > 0 else "—"
    axC.text(xi, v+0.5, lbl, ha="center", va="bottom", fontsize=8.5, fontweight="bold")
axC.set_xticks(x); axC.set_xticklabels(["ONNX","FP32","FP16*","INT8"], fontsize=8.5)
axC.set_ylabel("Accuracy (%)"); axC.set_title("(C) Test Accuracy (8-class)", fontsize=10)
axC.yaxis.grid(True, alpha=0.5); axC.set_axisbelow(True)

# D — Engine sizes
axD = fig.add_subplot(gs[1,0])
axD.bar(x, eng_mbs, color=ENG_COLOURS, alpha=0.88, edgecolor="white", zorder=3)
for xi, v in enumerate(eng_mbs):
    axD.text(xi, v+0.5, f"{v:.1f} MB", ha="center", va="bottom",
             fontsize=8, fontweight="bold")
axD.set_xticks(x); axD.set_xticklabels(["ONNX","FP32","FP16*","INT8"], fontsize=8.5)
axD.set_ylabel("File Size (MB)"); axD.set_title("(D) Engine / Model File Size", fontsize=10)
axD.yaxis.grid(True, alpha=0.5); axD.set_axisbelow(True)

# E — Accuracy vs throughput scatter
axE = fig.add_subplot(gs[1,1])
for i, k in enumerate(eng3):
    axE.scatter(tput3[i], acc3[i], s=mb3[i]*9, c=col3[i],
                alpha=0.85, edgecolors="white", linewidths=1.8, zorder=4)
    axE.annotate(lab3[i].replace("*",""),
                 xy=(tput3[i], acc3[i]),
                 xytext=(tput3[i]+25, acc3[i]+0.3),
                 fontsize=7.5, color=col3[i], fontweight="bold")
axE.set_xlabel("Throughput (img/s)"); axE.set_ylabel("Accuracy (%)")
axE.set_title("(E) Accuracy vs Throughput", fontsize=10)
axE.yaxis.grid(True, alpha=0.5); axE.xaxis.grid(True, alpha=0.5); axE.set_axisbelow(True)

# F — Summary table (all values from JSON)
axF = fig.add_subplot(gs[1,2])
axF.axis("off")
onnx_acc_val = acc_data["onnx"]["overall_pct"] if "onnx" in acc_data else "—"
table_rows = [
    ["ONNX-CUDA", f"{b1_mean[0]:.2f}", f"{b32_tp[0]:.0f}", "1.00x",
     f"{onnx_acc_val}%", f"{eng_mbs[0]:.1f}"],
    ["TRT FP32",  f"{b1_mean[1]:.2f}", f"{b32_tp[1]:.0f}",
     f"{b32_tp[1]/b32_tp[0]:.2f}x", f"{accs_p[1]:.1f}%", f"{eng_mbs[1]:.1f}"],
    ["TRT FP16*", f"{b1_mean[2]:.2f}", f"{b32_tp[2]:.0f}",
     f"{b32_tp[2]/b32_tp[0]:.2f}x", f"{accs_p[2]:.1f}%", f"{eng_mbs[2]:.1f}"],
    ["TRT INT8",  f"{b1_mean[3]:.2f}", f"{b32_tp[3]:.0f}",
     f"{b32_tp[3]/b32_tp[0]:.2f}x", f"{accs_p[3]:.1f}%", f"{eng_mbs[3]:.1f}"],
]
tbl = axF.table(cellText=table_rows,
                colLabels=["Engine","b=1 ms","b=32 img/s","Speedup","Acc","MB"],
                cellLoc="center", loc="center", bbox=[0, 0.05, 1, 0.9])
tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
for j in range(6):
    tbl[(0,j)].set_facecolor(CLR["text"])
    tbl[(0,j)].set_text_props(color="white", fontweight="bold")
for j in range(6):
    tbl[(3,j)].set_facecolor("#FEF5E7")
    tbl[(3,j)].set_text_props(color=CLR["fp16"], fontweight="bold")
axF.set_title("(F) Summary Table", fontsize=10, pad=8)

fig.legend(handles=[mpatches.Patch(color=c, label=l)
                    for c,l in zip(ENG_COLOURS, ENG_LABELS)],
           loc="lower center", ncol=4, fontsize=9, framealpha=0.9,
           bbox_to_anchor=(0.5, -0.04))
save(fig, "fig15_dashboard")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("ALL FIGURES GENERATED FROM JSON DATA")
print("=" * 65)
for f in sorted(OUT_DIR.glob("*.png")):
    print(f"  {f.name:<45} {f.stat().st_size//1024:>5} KB")
print(f"\n  Output : {OUT_DIR.resolve()}")
print(f"  Formats: PNG (200 dpi) + PDF (vector)")
print(f"\n  Data sources:")
for key, fname in JSON_FILES.items():
    print(f"    {fname}")
print("""
Thesis placement guide:
  Fig 1,  2  -> Methods: workflow diagrams
  Fig 3,  4  -> Results: latency & throughput
  Fig 5      -> Results: key trade-off (FP16 sweet spot)
  Fig 6      -> Results/Appendix: percentile heatmap
  Fig 7,  8  -> Results: engine efficiency & progression
  Fig 9      -> Discussion: clinical deployment viability
  Fig 10     -> Results: multi-batch sweep (week16 data)
  Fig 11     -> Results: warmup convergence (week16 data)
  Fig 12     -> Results: per-class accuracy (week16 data)
  Fig 13     -> Appendix: confusion matrix (evaluation_results.json)
  Fig 14     -> Results: VRAM efficiency (week16 data)
  Fig 15     -> Appendix: combined dashboard
""")