"""
===============================================
WEEK 2: COMPREHENSIVE THESIS VISUALIZATIONS
===============================================

This script creates detailed, publication-ready visualizations from the 
Week 2 preprocessed ISIC 2019 dataset for thesis documentation.

IMPORTANT:
- This version does NOT require X_full.npy / y_full.npy.
- It uses:
    - full_metadata.csv          (Week 2 output)
    - custom_dataset_statistics.json (Week 2 output, if present)
    - Original ISIC images on disk (same location Week 2 used)

Visualizations created (publication-ready, 300 DPI):
  1. Dataset Overview Dashboard
  2. Sample Images Gallery per Class (recomputed on a small sample)
  3. Color Distribution Analysis (sample-based)
  4. Image Quality Metrics (brightness / contrast / sharpness)
  5. Pixel Value Distribution per Class (sample-based)
  6. Statistical Summary Report (text figure)
  7. Preprocessing Methodology Flowchart
  8. Class Comparison & Imbalance Analysis

Usage:
  python week2_thesis_visualizations.py

Dependencies:
  - numpy, pandas, matplotlib, seaborn, opencv-python
  - tqdm

Author: Thesis Documentation
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION & PATHS
# ============================================

# Try to mimic Week 2's base directory logic
BASE_DIR = Path(os.getcwd())
if Path("/workspace").exists():
    BASE_DIR = Path("/workspace")
elif Path("/notebooks").exists():
    BASE_DIR = Path("/notebooks")

# Outputs directory (Week 2 used /workspace/outputs or similar)
OUTPUT_DIR = (BASE_DIR / "outputs").resolve()

# Visualization directory requested by you:
VIZ_DIR = OUTPUT_DIR / "viz" / "week2"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# Week 2 artifacts we rely on
METADATA_PATH = OUTPUT_DIR / "full_metadata.csv"
STATS_PATH = OUTPUT_DIR / "custom_dataset_statistics.json"

# Original dataset directory (same as Week 2)
# Week 2 used STORAGE_BASE / 'data' / 'isic2019'
INPUT_DIR = (BASE_DIR / "data").resolve()

# Class names
CLASS_NAMES = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"]
CLASS_FULL_NAMES = {
    "AK": "Actinic Keratosis",
    "BCC": "Basal Cell Carcinoma",
    "BKL": "Benign Keratosis-like Lesion",
    "DF": "Dermatofibroma",
    "MEL": "Melanoma",
    "NV": "Nevus",
    "SCC": "Squamous Cell Carcinoma",
    "VASC": "Vascular Lesion",
}

# Color palette per class
COLORS = {
    "AK": "#FF6B6B",    # Red
    "BCC": "#4ECDC4",   # Teal
    "BKL": "#45B7D1",   # Blue
    "DF": "#FFA07A",    # Light Salmon
    "MEL": "#000000",   # Black
    "NV": "#FFD93D",    # Yellow
    "SCC": "#6BCB77",   # Green
    "VASC": "#A78BFA",  # Purple
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

TARGET_SIZE = (600, 600)  # Week 2 resolution

print("📊 Week 2 Thesis Visualization Generator")
print("=" * 70)
print(f"Base directory:   {BASE_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Viz directory:    {VIZ_DIR}")
print(f"Input dataset:    {INPUT_DIR}")
print("=" * 70)

# ============================================
# SANITY CHECKS
# ============================================

if not METADATA_PATH.exists():
    print(f"❌ ERROR: full_metadata.csv not found at {METADATA_PATH}")
    print("   You need to copy full_metadata.csv from the RunPod volume or rerun Week 2.")
    sys.exit(1)

# Load metadata
try:
    metadata_df = pd.read_csv(METADATA_PATH)
    if "class" not in metadata_df.columns or "image_name" not in metadata_df.columns:
        raise ValueError("full_metadata.csv must contain 'class' and 'image_name' columns.")
except Exception as e:
    print(f"❌ ERROR loading metadata: {e}")
    sys.exit(1)

# Load statistics if available
stats_dict = None
if STATS_PATH.exists():
    try:
        with open(STATS_PATH, "r") as f:
            stats_dict = json.load(f)
        print("✓ Loaded custom_dataset_statistics.json")
    except Exception as e:
        print(f"⚠ Could not load custom_dataset_statistics.json: {e}")

total_images = len(metadata_df)
print(f"\n✅ Metadata loaded: {total_images:,} images\n")

# ============================================
# WEEK 2 PREPROCESSING HELPERS (LIGHT VERSION)
# ============================================

def _shades_of_gray_color_constancy(img_rgb, power=6, gamma=None):
    try:
        img = img_rgb.astype(np.float32)
        if gamma is not None:
            img = np.power(img, gamma)
        eps = 1e-6
        mean_per_channel = np.power(
            np.mean(np.power(img, power), axis=(0, 1)) + eps, 1.0 / power
        )
        norm_factor = np.sqrt(np.sum(np.power(mean_per_channel, 2.0))) + eps
        img = img / (mean_per_channel / norm_factor)
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)
    except Exception:
        return img_rgb


def _advanced_medical_preprocessing(img_rgb):
    try:
        img_denoised = cv2.bilateralFilter(img_rgb, 9, 75, 75)
        kernel_edge = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        img_enhanced = cv2.filter2D(img_denoised, -1, kernel_edge)
        img_enhanced = np.clip(img_enhanced, 0, 255)
        gamma = 1.2
        img_gamma = np.power(img_enhanced / 255.0, gamma) * 255.0
        img_gamma = np.clip(img_gamma, 0, 255).astype(np.uint8)
        return img_gamma
    except Exception:
        return img_rgb


def _lesion_enhancement(img_rgb):
    try:
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        img_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        return img_enhanced
    except Exception:
        return img_rgb


def _hair_removal_dullrazor(img_rgb):
    try:
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, thresh = cv2.threshold(
            blackhat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_clean)
        inpainted = cv2.inpaint(img_bgr, thresh, 5, cv2.INPAINT_TELEA)
        return cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
    except Exception:
        return img_rgb


def preprocess_image_medical_fast(
    image_path, target_size=TARGET_SIZE, normalize=True
):
    """
    Simplified Week 2 pipeline for on-the-fly visualization:
      - Read
      - Color constancy
      - Advanced preprocessing
      - Lesion enhancement + CLAHE
      - Hair removal
      - Resize with reflect padding (600x600)
      - ImageNet normalization (if normalize=True)
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = _shades_of_gray_color_constancy(img, power=6, gamma=None)
        img = _advanced_medical_preprocessing(img)
        img = _lesion_enhancement(img)
        img = _hair_removal_dullrazor(img)

        # Resize with aspect ratio preserved + reflect padding
        h, w = img.shape[:2]
        aspect = w / h
        target_aspect = target_size[1] / target_size[0]

        if aspect > target_aspect:
            new_w = target_size[1]
            new_h = int(new_w / aspect)
        else:
            new_h = target_size[0]
            new_w = int(new_h * aspect)

        interp = cv2.INTER_AREA if (new_w < w or new_h < h) else cv2.INTER_LINEAR
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

        pad_top = (target_size[0] - new_h) // 2
        pad_bottom = target_size[0] - new_h - pad_top
        pad_left = (target_size[1] - new_w) // 2
        pad_right = target_size[1] - new_w - pad_left

        canvas = cv2.copyMakeBorder(
            img_resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_REFLECT,
        )

        canvas = canvas.astype(np.float32) / 255.0

        # ImageNet normalization
        canvas = (canvas - IMAGENET_MEAN) / IMAGENET_STD

        return canvas
    except Exception:
        return None


def denormalize_imagenet(img):
    img = img.astype(np.float32)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img, 0.0, 1.0)
    return img


# ============================================
# IMAGE PATH RESOLUTION (similar to Week 2)
# ============================================

def resolve_image_path(base: Path, image_name: str, label: str) -> Path:
    """
    Resolve image path based on Week 2 logic:
      - class-based dirs: base / label / *.jpg
      - standard dirs: base / ISIC_2019_Training_Input / *.jpg
      - fallback: recursive search
    """
    image_id_clean = image_name
    if image_id_clean.lower().endswith((".jpg", ".jpeg", ".png")):
        image_id_clean = image_id_clean.rsplit(".", 1)[0]

    extensions = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]

    candidates = [
        base / label / image_id_clean,
        base / "ISIC_2019_Training_Input" / image_id_clean,
        base / "images" / image_id_clean,
        base / "train" / image_id_clean,
        base / image_id_clean,
    ]

    for c in candidates:
        for ext in extensions:
            p = c.with_suffix(ext)
            if p.exists():
                return p

    # fallback recursive search
    for ext in extensions:
        pattern = f"{image_id_clean}*{ext}"
        matches = list(base.rglob(pattern))
        if matches:
            return matches[0]

    # last resort: non-existent but deterministic path
    return base / "ISIC_2019_Training_Input" / f"{image_id_clean}.jpg"


# ============================================
# SAMPLE IMAGES PER CLASS (FOR ALL VISUALIZATIONS)
# ============================================

print("🎨 Sampling images per class for visualizations...")

SAMPLES_FOR_GALLERY = 4      # max gallery samples per class
SAMPLES_FOR_METRICS = 50     # for quality metrics & histograms
PER_CLASS_SAMPLES = {}

for cls in CLASS_NAMES:
    df_cls = metadata_df[metadata_df["class"] == cls]
    if df_cls.empty:
        PER_CLASS_SAMPLES[cls] = []
        print(f"  ⚠ No images found for class {cls} in metadata.")
        continue

    # Use up to SAMPLES_FOR_METRICS for each class
    n_samples = min(len(df_cls), SAMPLES_FOR_METRICS)
    sampled_rows = df_cls.sample(n_samples, random_state=42)

    cls_imgs = []
    for _, row in tqdm(
        sampled_rows.iterrows(),
        total=n_samples,
        desc=f"  {cls} (sampling & preprocessing)",
        leave=False,
    ):
        img_path = resolve_image_path(INPUT_DIR, row["image_name"], row["class"])
        img_pre = preprocess_image_medical_fast(img_path, target_size=TARGET_SIZE)
        if img_pre is not None:
            cls_imgs.append(img_pre)

    PER_CLASS_SAMPLES[cls] = cls_imgs
    print(
        f"  ✓ {cls}: prepared {len(cls_imgs)} preprocessed sample images "
        f"(from {len(df_cls)} total in metadata)"
    )

print("\n✅ Sampling & preprocessing complete.\n")

# ============================================
# UTILITY FUNCTIONS
# ============================================

def get_class_color_by_name(cls_name: str):
    return COLORS.get(cls_name, "#999999")


def save_figure(fig, filename, dpi=300, transparent=False):
    path = VIZ_DIR / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight", transparent=transparent)
    plt.close(fig)
    print(f"  ✓ Saved: {path}")
    return path


def calculate_image_metrics(img):
    """Brightness, contrast, sharpness on a single preprocessed image."""
    img_display = denormalize_imagenet(img)
    brightness = float(np.mean(img_display))
    contrast = float(np.std(img_display))
    gray = cv2.cvtColor((img_display * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return {
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": lap_var,
    }

# ============================================
# VIS 1: DATASET OVERVIEW DASHBOARD
# ============================================

print("1️⃣  Dataset Overview Dashboard...")

fig = plt.figure(figsize=(16, 12))
gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
fig.suptitle(
    "ISIC 2019 Skin Lesion Dataset - Overview Dashboard (Week 2 Preprocessed)",
    fontsize=20,
    fontweight="bold",
    y=0.98,
)

# Compute counts from metadata
class_counts = (
    metadata_df["class"].value_counts().reindex(CLASS_NAMES).fillna(0).astype(int)
)
class_names_ordered = CLASS_NAMES
counts_arr = class_counts.values

# (A) Bar chart
ax1 = fig.add_subplot(gs[0, :2])
colors_bar = [get_class_color_by_name(c) for c in class_names_ordered]
bars = ax1.bar(class_names_ordered, counts_arr, color=colors_bar,
               edgecolor="black", linewidth=1.5, alpha=0.85)
ax1.set_ylabel("Number of Images", fontsize=11, fontweight="bold")
ax1.set_title("(A) Class Distribution", fontsize=12, fontweight="bold", loc="left")
ax1.grid(axis="y", alpha=0.3, linestyle="--")

for bar in bars:
    h = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2.0,
        h,
        f"{int(h):,}",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )

# (B) Pie chart
ax2 = fig.add_subplot(gs[0, 2])
percentages = counts_arr / total_images * 100.0
ax2.pie(
    counts_arr,
    labels=class_names_ordered,
    autopct="%1.1f%%",
    colors=colors_bar,
    startangle=90,
    textprops={"fontsize": 9, "fontweight": "bold"},
)
ax2.set_title("(B) Class Distribution (%)", fontsize=12, fontweight="bold", loc="left")

# (C) Stats table
ax3 = fig.add_subplot(gs[1, :])
ax3.axis("off")
table_data = [["Class", "Count", "Percentage", "Full Name"]]
for cls in CLASS_NAMES:
    cnt = class_counts[cls]
    pct = cnt / total_images * 100.0 if total_images > 0 else 0.0
    table_data.append(
        [cls, f"{cnt:,}", f"{pct:.2f}%", CLASS_FULL_NAMES.get(cls, cls)]
    )
table_data.append(["TOTAL", f"{total_images:,}", "100.00%", "All Classes"])

tbl = ax3.table(
    cellText=table_data,
    cellLoc="center",
    loc="center",
    colWidths=[0.1, 0.15, 0.15, 0.6],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 2.5)

# header
for j in range(4):
    tbl[(0, j)].set_facecolor("#4ECDC4")
    tbl[(0, j)].set_text_props(weight="bold", color="white")

# per-class rows
for i, cls in enumerate(CLASS_NAMES):
    for j in range(4):
        tbl[(i + 1, j)].set_facecolor(get_class_color_by_name(cls))
        tbl[(i + 1, j)].set_alpha(0.3)

# total row
row_total = len(CLASS_NAMES) + 1
for j in range(4):
    tbl[(row_total, j)].set_facecolor("#333333")
    tbl[(row_total, j)].set_text_props(weight="bold", color="white")

ax3.set_title(
    "(C) Dataset Statistics Table", fontsize=12, fontweight="bold", loc="left", pad=20
)

# (D) Text block
ax4 = fig.add_subplot(gs[2, :])
ax4.axis("off")

imbalance_ratio = (
    class_counts.max() / class_counts.min() if class_counts.min() > 0 else np.nan
)
most_cls = class_counts.idxmax()
least_cls = class_counts.idxmin()

info_text = f"""
Dataset Information:
  • Total Images: {total_images:,}
  • Number of Classes: {len(CLASS_NAMES)}
  • Preprocessing: Week 2 medical pipeline (color constancy, CLAHE, hair removal, etc.)

Class Imbalance:
  • Class Imbalance Ratio (max : min): {imbalance_ratio:.2f}:1
  • Most Common Class:  {most_cls} - {CLASS_FULL_NAMES[most_cls]} ({class_counts[most_cls]:,} images)
  • Least Common Class: {least_cls} - {CLASS_FULL_NAMES[least_cls]} ({class_counts[least_cls]:,} images)

If custom_dataset_statistics.json is available, it contains:
  • Channel-wise mean and std after preprocessing
  • Preprocessing steps applied to all images
  • Parallelism / worker configuration used during Week 2
"""

ax4.text(
    0.02,
    0.98,
    info_text,
    transform=ax4.transAxes,
    fontsize=10,
    verticalalignment="top",
    family="monospace",
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85, pad=0.8),
)

save_figure(fig, "01_dataset_overview_dashboard.png", dpi=300)

# ============================================
# VIS 2: SAMPLE IMAGES GALLERY PER CLASS
# ============================================

print("2️⃣  Sample Images Gallery per Class...")

fig, axes = plt.subplots(4, 2, figsize=(14, 16))
fig.suptitle(
    "ISIC 2019 - Sample Preprocessed Images by Class (Week 2 Pipeline)",
    fontsize=16,
    fontweight="bold",
    y=0.995,
)

for idx, cls in enumerate(CLASS_NAMES):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]

    samples = PER_CLASS_SAMPLES.get(cls, [])
    if not samples:
        ax.text(
            0.5,
            0.5,
            f"No samples\nfor {cls}",
            ha="center",
            va="center",
            fontsize=11,
        )
        ax.axis("off")
        continue

    # Show up to SAMPLES_FOR_GALLERY images in a montage (here: just first)
    img = samples[0]
    img_disp = denormalize_imagenet(img)
    ax.imshow(img_disp)
    ax.set_title(
        f"{cls}: {CLASS_FULL_NAMES[cls]}\n({class_counts[cls]:,} images)",
        fontsize=11,
        fontweight="bold",
        pad=8,
    )
    ax.axis("off")

plt.tight_layout()
save_figure(fig, "02_sample_images_gallery.png", dpi=300)

# ============================================
# VIS 3: COLOR DISTRIBUTION ANALYSIS (SAMPLE-BASED)
# ============================================

print("3️⃣  Color Distribution Analysis (sample-based)...")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle(
    "Color Channel Distribution by Class (Sampled, Week 2 Preprocessed)",
    fontsize=16,
    fontweight="bold",
)

for idx, cls in enumerate(CLASS_NAMES):
    row = idx // 4
    col = idx % 4
    ax = axes[row, col]

    samples = PER_CLASS_SAMPLES.get(cls, [])
    if not samples:
        ax.text(0.5, 0.5, f"No {cls} samples", ha="center", va="center")
        ax.axis("off")
        continue

    arr = np.stack(samples, axis=0)  # (N, 600, 600, 3)
    den = np.array([denormalize_imagenet(im) for im in arr])  # [0,1]
    mean_colors = np.mean(den, axis=(1, 2))  # (N,3)

    ax.scatter(
        mean_colors[:, 0],
        mean_colors[:, 1],
        c=mean_colors,
        s=15,
        alpha=0.7,
        edgecolor="none",
    )
    ax.set_xlabel("Red Channel Mean", fontsize=9)
    ax.set_ylabel("Green Channel Mean", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(
        f"{cls}\n({len(samples)} sampled)",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")

plt.tight_layout()
save_figure(fig, "03_color_distribution_analysis.png", dpi=300)

# ============================================
# VIS 4: IMAGE QUALITY METRICS PER CLASS
# ============================================

print("4️⃣  Image Quality Metrics by Class (brightness / contrast / sharpness)...")

metrics_rows = []
for cls in CLASS_NAMES:
    samples = PER_CLASS_SAMPLES.get(cls, [])
    for img in samples:
        m = calculate_image_metrics(img)
        m["class"] = cls
        metrics_rows.append(m)

metrics_df = pd.DataFrame(metrics_rows)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Image Quality Metrics by Class (Sampled)", fontsize=16, fontweight="bold")

# map class to x index
cls_to_x = {cls: i for i, cls in enumerate(CLASS_NAMES)}

# Helper for boxplot-per-class
def boxplot_metric(ax, metric_name, title):
    data = []
    positions = []
    colors_bp = []
    for cls in CLASS_NAMES:
        vals = metrics_df[metrics_df["class"] == cls][metric_name].values
        if len(vals) == 0:
            continue
        data.append(vals)
        positions.append(cls_to_x[cls])
        colors_bp.append(get_class_color_by_name(cls))
    bps = ax.boxplot(
        data,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
    )
    for bp, c in zip(bps["boxes"], colors_bp):
        bp.set_facecolor(c)
        bp.set_alpha(0.7)

    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45)
    ax.set_ylabel(metric_name.capitalize(), fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)


boxplot_metric(axes[0], "brightness", "(A) Image Brightness")
boxplot_metric(axes[1], "contrast", "(B) Image Contrast")
boxplot_metric(axes[2], "sharpness", "(C) Image Sharpness")

plt.tight_layout()
save_figure(fig, "04_image_quality_metrics.png", dpi=300)

# ============================================
# VIS 5: PIXEL VALUE DISTRIBUTION PER CLASS
# ============================================

print("5️⃣  Pixel Value Distribution per Class...")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle(
    "Pixel Value Distribution per Class (Sampled, R/G/B, Week 2 Preprocessed)",
    fontsize=16,
    fontweight="bold",
)

for idx, cls in enumerate(CLASS_NAMES):
    row = idx // 4
    col = idx % 4
    ax = axes[row, col]

    samples = PER_CLASS_SAMPLES.get(cls, [])
    if not samples:
        ax.text(0.5, 0.5, f"No {cls} samples", ha="center", va="center")
        ax.axis("off")
        continue

    arr = np.stack(samples, axis=0)
    den = np.array([denormalize_imagenet(im) for im in arr])  # [0,1]
    pixels = den.reshape(-1, 3)

    ax.hist(
        pixels[:, 0],
        bins=40,
        alpha=0.5,
        label="Red",
        color="red",
        edgecolor="black",
    )
    ax.hist(
        pixels[:, 1],
        bins=40,
        alpha=0.5,
        label="Green",
        color="green",
        edgecolor="black",
    )
    ax.hist(
        pixels[:, 2],
        bins=40,
        alpha=0.5,
        label="Blue",
        color="blue",
        edgecolor="black",
    )

    ax.set_xlabel("Pixel Value", fontsize=9)
    ax.set_ylabel("Frequency", fontsize=9)
    ax.set_title(cls, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
save_figure(fig, "05_pixel_value_distribution.png", dpi=300)

# ============================================
# VIS 6: STATISTICAL SUMMARY REPORT (TEXT FIGURE)
# ============================================

print("6️⃣  Statistical Summary Report...")

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111)
ax.axis("off")

stats_text = f"""
ISIC 2019 Skin Lesion Dataset - Statistical Summary (Week 2 Preprocessed)

Dataset Composition:
  • Total images:          {total_images:,}
  • Number of classes:     {len(CLASS_NAMES)}

Class Distribution:
"""

for cls in CLASS_NAMES:
    cnt = class_counts[cls]
    pct = cnt / total_images * 100 if total_images > 0 else 0
    bar_len = int(pct / 2)
    bar = "█" * bar_len
    stats_text += (
        f"  • {cls:4s} ({CLASS_FULL_NAMES[cls]:30s}): "
        f"{cnt:6,} ({pct:5.2f}%) {bar}\n"
    )

if len(metrics_df) > 0:
    stats_text += f"""

Image Quality Metrics (sampled):
  • Avg brightness: {metrics_df['brightness'].mean():.4f} ± {metrics_df['brightness'].std():.4f}
  • Avg contrast:   {metrics_df['contrast'].mean():.4f} ± {metrics_df['contrast'].std():.4f}
  • Avg sharpness:  {metrics_df['sharpness'].mean():.2f} ± {metrics_df['sharpness'].std():.2f}
"""

if stats_dict is not None and "mean_rgb" in stats_dict and "std_rgb" in stats_dict:
    mean_rgb = stats_dict["mean_rgb"]
    std_rgb = stats_dict["std_rgb"]
    stats_text += f"""
Global Pixel Statistics (from custom_dataset_statistics.json):
  • Mean (R,G,B): {mean_rgb}
  • Std  (R,G,B): {std_rgb}
"""

stats_text += f"""

Preprocessing Summary (Week 2):
  ✓ Shades-of-Gray color constancy
  ✓ Advanced medical preprocessing (bilateral, edge enhancement, gamma)
  ✓ Lesion enhancement via CLAHE (LAB color space)
  ✓ Hair/marker removal (DullRazor-style)
  ✓ Aspect ratio preservation with reflection padding to 600×600
  ✓ ImageNet normalization (μ={IMAGENET_MEAN.tolist()}, σ={IMAGENET_STD.tolist()})
"""

ax.text(
    0.02,
    0.98,
    stats_text,
    transform=ax.transAxes,
    fontsize=9.5,
    verticalalignment="top",
    family="monospace",
    bbox=dict(boxstyle="round", facecolor="#fdf7d9", alpha=0.95, pad=1.0),
)

save_figure(fig, "06_statistical_summary_report.png", dpi=300)

# ============================================
# VIS 7: PREPROCESSING METHODOLOGY FLOWCHART
# ============================================

print("7️⃣  Preprocessing Methodology Flowchart...")

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis("off")

input_color = "#FFE5B4"
process_color = "#B4D7FF"
output_color = "#B4FFB4"
arrow_color = "#333333"

def draw_box(ax, x, y, w, h, text, color):
    rect = mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.2",
        edgecolor=arrow_color,
        facecolor=color,
        linewidth=2,
    )
    ax.add_patch(rect)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        wrap=True,
    )

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", lw=2, color=arrow_color),
    )

ax.text(
    5,
    11.5,
    "ISIC 2019 Medical Image Preprocessing Pipeline (Week 2)",
    ha="center",
    fontsize=16,
    fontweight="bold",
)

# Boxes (top to bottom)
draw_box(
    ax,
    5,
    10.5,
    4,
    0.8,
    "Input: Raw ISIC 2019 Images\n(RGB, variable resolution)",
    input_color,
)
draw_arrow(ax, 5, 10.1, 5, 9.7)

draw_box(
    ax,
    5,
    9.2,
    4.2,
    0.9,
    "Shades-of-Gray Color Constancy\n(Illumination normalization)",
    process_color,
)
draw_arrow(ax, 5, 8.8, 5, 8.4)

draw_box(
    ax,
    5,
    7.9,
    4.4,
    0.9,
    "Advanced Medical Preprocessing\n(Bilateral filter, edge enhancement,\ngamma correction)",
    process_color,
)
draw_arrow(ax, 5, 7.5, 5, 7.1)

draw_box(
    ax,
    5,
    6.6,
    4.2,
    0.9,
    "Lesion Enhancement via CLAHE\n(LAB color space)",
    process_color,
)
draw_arrow(ax, 5, 6.2, 5, 5.8)

draw_box(
    ax,
    5,
    5.3,
    4.2,
    0.9,
    "Hair / Marker Removal\n(DullRazor-style inpainting)",
    process_color,
)
draw_arrow(ax, 5, 4.9, 5, 4.5)

draw_box(
    ax,
    5,
    4.0,
    4.2,
    0.9,
    "Aspect Ratio Preservation\n(Reflection padding to 600×600)",
    process_color,
)
draw_arrow(ax, 5, 3.6, 5, 3.2)

draw_box(
    ax,
    5,
    2.7,
    4.2,
    0.9,
    "ImageNet Normalization\n(channel-wise mean/std)",
    process_color,
)
draw_arrow(ax, 5, 2.3, 5, 1.9)

draw_box(
    ax,
    5,
    1.4,
    4,
    0.8,
    "Output: Preprocessed Images\n(600×600×3, float32)",
    output_color,
)

save_figure(fig, "07_preprocessing_methodology_flowchart.png", dpi=300)

# ============================================
# VIS 8: CLASS COMPARISON & IMBALANCE PANEL
# ============================================

print("8️⃣  Class Comparison & Imbalance Analysis Panel...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Class Comparison & Imbalance Analysis", fontsize=16, fontweight="bold")

# (A) Average color per class
axA = axes[0, 0]
avg_colors = []
for cls in CLASS_NAMES:
    samples = PER_CLASS_SAMPLES.get(cls, [])
    if not samples:
        avg_colors.append([0.0, 0.0, 0.0])
        continue
    arr = np.stack(samples, axis=0)
    den = np.array([denormalize_imagenet(im) for im in arr])
    avg_colors.append(np.mean(den, axis=(0, 1, 2)))
avg_colors = np.array(avg_colors)

axA.imshow(avg_colors.reshape(1, len(CLASS_NAMES), 3))
axA.set_xticks(range(len(CLASS_NAMES)))
axA.set_xticklabels(CLASS_NAMES, rotation=45)
axA.set_yticks([])
axA.set_title("(A) Average Color per Class (Sampled)", fontsize=12, fontweight="bold")

# (B) RGB channel bars
axB = axes[0, 1]
x = np.arange(len(CLASS_NAMES))
width = 0.25
axB.bar(x - width, avg_colors[:, 0], width, label="Red", color="red", alpha=0.7)
axB.bar(x, avg_colors[:, 1], width, label="Green", color="green", alpha=0.7)
axB.bar(x + width, avg_colors[:, 2], width, label="Blue", color="blue", alpha=0.7)
axB.set_xticks(x)
axB.set_xticklabels(CLASS_NAMES, rotation=45)
axB.set_ylabel("Average Channel Value", fontsize=11, fontweight="bold")
axB.set_title("(B) RGB Channel Means (Sampled)", fontsize=12, fontweight="bold")
axB.grid(axis="y", alpha=0.3)
axB.legend()

# (C) Horizontal bar plot of class sizes
axC = axes[1, 0]
bars = axC.barh(
    CLASS_NAMES,
    [class_counts[cls] for cls in CLASS_NAMES],
    color=[get_class_color_by_name(cls) for cls in CLASS_NAMES],
    alpha=0.85,
)
axC.set_xlabel("Number of Images", fontsize=11, fontweight="bold")
axC.set_title("(C) Class Sizes from Metadata", fontsize=12, fontweight="bold")
axC.grid(axis="x", alpha=0.3)
for bar, cls in zip(bars, CLASS_NAMES):
    width_val = bar.get_width()
    axC.text(
        width_val + max(class_counts) * 0.01,
        bar.get_y() + bar.get_height() / 2.0,
        f"{class_counts[cls]:,}",
        va="center",
        fontsize=9,
        fontweight="bold",
    )

# (D) Imbalance text
axD = axes[1, 1]
axD.axis("off")
imbalance_text = f"""
Class Imbalance Analysis

  • Max:Min Ratio:     {imbalance_ratio:.2f}:1
  • Largest Class:     {most_cls} - {CLASS_FULL_NAMES[most_cls]}
                        {class_counts[most_cls]:,} images ({class_counts[most_cls]/total_images*100:.2f}%)
  • Smallest Class:    {least_cls} - {CLASS_FULL_NAMES[least_cls]}
                        {class_counts[least_cls]:,} images ({class_counts[least_cls]/total_images*100:.2f}%)

Recommendations for Training:
  • Use class weights in the loss function
  • Monitor per-class metrics (e.g., balanced accuracy, F1 per class)
  • Apply targeted augmentation to minority classes
"""
axD.text(
    0.02,
    0.98,
    imbalance_text,
    transform=axD.transAxes,
    fontsize=10,
    verticalalignment="top",
    family="monospace",
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85),
)

plt.tight_layout()
save_figure(fig, "08_class_comparison_analysis.png", dpi=300)

# ============================================
# SUMMARY
# ============================================

print("\n" + "=" * 70)
print("✅ THESIS VISUALIZATIONS COMPLETE (Week 2, no X_full.npy/y_full.npy needed)")
print("=" * 70)
print(f"\nVisualizations saved in: {VIZ_DIR}")
print("Files generated:")
print("  1. 01_dataset_overview_dashboard.png")
print("  2. 02_sample_images_gallery.png")
print("  3. 03_color_distribution_analysis.png")
print("  4. 04_image_quality_metrics.png")
print("  5. 05_pixel_value_distribution.png")
print("  6. 06_statistical_summary_report.png")
print("  7. 07_preprocessing_methodology_flowchart.png")
print("  8. 08_class_comparison_analysis.png")
print("\nAll figures are 300 DPI and suitable for direct inclusion in your thesis.")
print("=" * 70)