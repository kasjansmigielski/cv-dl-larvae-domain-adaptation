#!/usr/bin/env python3
"""
STEP 3.1 - Prepare the crop set for StyleGAN2-ADA.

Input:  data/gan_crops/{s1,s2}/*.png   (from step 01_extract_dish_crops.py)
Output: data/stylegan/<session>/*.png   (after quality control, --max-images limit)

StyleGAN uses the SAME crops as CycleGAN. This script:
  1. Loads the crops for a session (s1, s2) or BOTH at once (combined).
  2. Applies the same quality control (QC) as CycleGAN - rejects empty/uniform
     crops based on the standard deviation of brightness (std), independent of the
     domain brightness level (S1 dark, S2 bright - the essence of the domain shift).
  3. Shuffles (seed=42) and copies to data/stylegan/<session>/ with --max-images.

'combined' mode (RECOMMENDED): merges S1+S2 crops into one set so StyleGAN learns
to generate larvae in BOTH illumination styles (universal-model augmentation). When
merging, it draws half of the limit from each session (balanced domains).

After this step the data goes to dataset_tool.py, which converts the PNG folder to
the zip format required by StyleGAN2-ADA (handled by train_stylegan.slurm).

Usage:
    # RECOMMENDED - combined (S1+S2 together):
    python src/dl_pipeline/gan/stylegan/05_prepare_stylegan_dataset.py --session combined --max-images 4000
    # Or a single session:
    python src/dl_pipeline/gan/stylegan/05_prepare_stylegan_dataset.py --session s1 --max-images 3000
"""

import argparse
import random
import shutil
from pathlib import Path
from typing import List

import cv2


def is_valid_crop(path: Path, min_std: float = 4.0,
                  min_mean: float = 2.0, max_mean: float = 253.0) -> bool:
    """
    Crop quality validation - identical logic to 02_split_cyclegan_dataset.py.

    The main indicator is the STANDARD DEVIATION (content variance), NOT mean
    brightness, because S1/S2 domains have fundamentally different brightness. An
    empty/uniform frame has std ~0.
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    mean, std = float(img.mean()), float(img.std())
    return (std >= min_std) and (min_mean <= mean <= max_mean)


def collect_valid(src_dir: Path) -> List[Path]:
    """Collect valid crops from a directory (after quality validation)."""
    all_pngs = sorted(src_dir.glob("*.png"))
    valid = [p for p in all_pngs if is_valid_crop(p)]
    rejected = len(all_pngs) - len(valid)
    print(f"  {src_dir.name}: {len(all_pngs)} crops, {len(valid)} OK, {rejected} rejected")
    return valid


def main():
    parser = argparse.ArgumentParser(
        description="STEP 3.1 - prepare the crop set for StyleGAN2-ADA"
    )
    parser.add_argument("--crops-dir", type=str, default="data/gan_crops",
                        help="Folder with crops from step 01 (default: data/gan_crops)")
    parser.add_argument("--output", type=str, default="data/stylegan",
                        help="Output folder (default: data/stylegan)")
    parser.add_argument("--session", type=str, default="combined",
                        choices=["s1", "s2", "combined"],
                        help="Set: s1, s2 or combined (S1+S2 together; default combined)")
    parser.add_argument("--max-images", type=int, default=4000,
                        help="Max number of images (default 4000; for combined split evenly per session)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    project_root = Path(__file__).resolve().parents[4]
    crops_dir = project_root / args.crops_dir
    out = project_root / args.output / args.session

    print(f"{'='*70}")
    print(f"STYLEGAN DATASET PREPARATION")
    print(f"  Set:        {args.session}")
    print(f"  Input:      {crops_dir}")
    print(f"  Output:     {out}")
    print(f"  Max images: {args.max_images}")
    print(f"{'='*70}")

    # combined mode: balanced draw of half the limit from each session.
    sessions = ["s1", "s2"] if args.session == "combined" else [args.session]
    per_session_limit = (args.max_images // len(sessions)) if args.session == "combined" \
        else args.max_images

    selected = []
    for sess in sessions:
        src = crops_dir / sess
        if not src.exists():
            print(f"\nERROR: missing {src} - run 01_extract_dish_crops.py first")
            raise SystemExit(1)
        print(f"\nSession {sess.upper()}:")
        valid = collect_valid(src)
        if not valid:
            print(f"  WARNING: no valid crops in {sess} after QC.")
            continue
        random.shuffle(valid)
        chosen = valid[:per_session_limit]
        selected.extend(chosen)
        print(f"  -> selected {len(chosen)} crops")

    if not selected:
        print("ERROR: no valid crops after QC.")
        raise SystemExit(1)

    out.mkdir(parents=True, exist_ok=True)
    for f in selected:
        # Session prefix in the name prevents collisions when merging (combined).
        dst_name = f"{f.parent.name}_{f.name}"
        shutil.copy2(f, out / dst_name)

    print(f"\n{'='*70}")
    print(f"DONE. Copied {len(selected)} crops -> {out}")
    print(f"  Next step: dataset_tool.py + train_stylegan.slurm")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
