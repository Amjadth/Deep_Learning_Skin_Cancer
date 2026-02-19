#!/usr/bin/env python3
"""
Organize outputs into structured directories and optionally delete train/val/test datasets.

Usage:
  - Dry run (default):   python organize_outputs.py
  - Apply changes:       python organize_outputs.py --apply
  - Custom outputs dir:  python organize_outputs.py --outputs /workspace/outputs --apply
"""

import argparse
import os
from pathlib import Path
import shutil
from typing import Dict, List, Tuple


REQUESTED_DIRS = [
    "cache",
    "checkpoints",
    "configs",
    "data",
    "logs",
    "models",
    "plots",  # kept for compatibility, but we'll prefer visualizations
    "results",
    "visualizations",
]

# Files to delete (train/val/test datasets)
DATASETS_TO_DELETE = [
    "X_train.npy",
    "y_train.npy",
    "X_val.npy",
    "y_val.npy",
    "X_test.npy",
    "y_test.npy",
]


def build_plan(outputs_dir: Path) -> Tuple[Dict[Path, Path], List[Path]]:
    """
    Build a move plan {src: dst} and a delete list for datasets.
    """
    moves: Dict[Path, Path] = {}
    deletes: List[Path] = []

    # Ensure outputs dir exists
    if not outputs_dir.exists():
        return moves, deletes

    # Directory targets
    dir_cache = outputs_dir / "cache"
    dir_checkpoints = outputs_dir / "checkpoints"
    dir_configs = outputs_dir / "configs"
    dir_data = outputs_dir / "data"
    dir_logs = outputs_dir / "logs"
    dir_models = outputs_dir / "models"
    dir_plots = outputs_dir / "plots"
    dir_results = outputs_dir / "results"
    dir_visualizations = outputs_dir / "visualizations"

    # Map known files
    mapping = {
        # results and metadata
        "split_info.json": dir_results,
        "split_summary.csv": dir_results,
        "full_metadata.csv": dir_results,
        # logs
        "cleanup_report.json": dir_logs,
        "QUICK_REFERENCE.txt": dir_logs,
        # checkpoints and cache
        "week4_checkpoint.json": dir_checkpoints,
        "week4_split_indices.npz": dir_cache,
        # data (keep core big arrays)
        "X_full.npy": dir_data,
        "y_full.npy": dir_data,
        "X_augmented_medical.npy": dir_data,
        "y_augmented_medical.npy": dir_data,
        # top-level plots should live under visualizations
        "split_distribution.png": dir_visualizations,
    }

    # First, plan deletions
    for fname in DATASETS_TO_DELETE:
        fpath = outputs_dir / fname
        if fpath.exists():
            deletes.append(fpath)

    # Then, explicit known moves
    for fname, target_dir in mapping.items():
        src = outputs_dir / fname
        if src.exists():
            moves[src] = target_dir / fname

    # Also move any other top-level PNGs into visualizations (but don't move existing subdir files)
    for item in outputs_dir.iterdir():
        if item.is_file() and item.suffix.lower() == ".png" and item.name not in mapping:
            moves[item] = dir_visualizations / item.name

    # Do not move anything already inside requested dirs
    requested_dirs_set = {outputs_dir / d for d in REQUESTED_DIRS}
    safe_moves = {}
    for src, dst in moves.items():
        # Skip if already in a requested subdir
        if any(str(src).startswith(str(d) + os.sep) for d in requested_dirs_set):
            continue
        safe_moves[src] = dst

    return safe_moves, deletes


def ensure_dirs(outputs_dir: Path, apply: bool) -> None:
    for d in REQUESTED_DIRS:
        target = outputs_dir / d
        if apply:
            target.mkdir(parents=True, exist_ok=True)


def perform_plan(outputs_dir: Path, moves: Dict[Path, Path], deletes: List[Path], apply: bool) -> None:
    print(f"Outputs directory: {outputs_dir}")
    print(f"Apply mode: {'YES' if apply else 'NO (dry-run)'}")
    print()

    # Ensure directories
    ensure_dirs(outputs_dir, apply)

    # Report moves
    if moves:
        print("Planned file moves:")
        for src, dst in moves.items():
            print(f"  - {src.name} -> {dst.relative_to(outputs_dir)}")
        if apply:
            for src, dst in moves.items():
                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.move(str(src), str(dst))
                except Exception as e:
                    print(f"    ! Failed to move {src.name}: {e}")
        print()
    else:
        print("No files to move.")
        print()

    # Report deletions
    if deletes:
        print("Planned deletions (train/val/test datasets):")
        for f in deletes:
            print(f"  - {f.name}")
        if apply:
            for f in deletes:
                try:
                    f.unlink(missing_ok=True)
                except Exception as e:
                    print(f"    ! Failed to delete {f.name}: {e}")
        print()
    else:
        print("No dataset files to delete.")
        print()

    print("Done." if apply else "Dry-run complete. Re-run with --apply to execute.")


def main():
    parser = argparse.ArgumentParser(description="Organize outputs into folders and delete train/val/test datasets.")
    # __file__ can be undefined in notebooks; prefer /workspace/outputs if present, else ./outputs
    default_outputs = Path("/workspace/outputs") if Path("/workspace/outputs").exists() else (Path.cwd() / "outputs")
    parser.add_argument("--outputs", type=Path, default=default_outputs,
                        help="Path to outputs directory (default: ./outputs)")
    parser.add_argument("--apply", action="store_true", help="Apply changes (otherwise dry-run)")
    args = parser.parse_args()

    outputs_dir = args.outputs.resolve()
    if not outputs_dir.exists():
        print(f"ERROR: outputs directory does not exist: {outputs_dir}")
        return

    moves, deletes = build_plan(outputs_dir)
    perform_plan(outputs_dir, moves, deletes, apply=args.apply)


if __name__ == "__main__":
    main()


