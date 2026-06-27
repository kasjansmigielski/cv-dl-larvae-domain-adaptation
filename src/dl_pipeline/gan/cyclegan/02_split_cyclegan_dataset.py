#!/usr/bin/env python3
"""
STEP 1.2 + 1.3 - Filter crops and split into CycleGAN sets.

Input:  data/gan_crops/s1/*.png, data/gan_crops/s2/*.png  (from step 01)
Output: data/cyclegan/{trainA,trainB,testA,testB}/

CycleGAN convention (pytorch-CycleGAN-and-pix2pix):
    trainA = domain A = Session I  (S1, top illumination)
    trainB = domain B = Session II (S2, bottom illumination)
    testA / testB = test sets (for translation and evaluation)

CycleGAN is UNPAIRED - there is no 1:1 correspondence between trainA and trainB.
Labels are NOT needed.

Step 1.3 (quality control) is performed here automatically:
    - reject crops that are too dark/bright (empty/corrupt)
    - reject crops with too low variance (uniform background, no content)

Usage:
    python src/dl_pipeline/gan/cyclegan/02_split_cyclegan_dataset.py \
        --train-per-domain 4000 --test-per-domain 500
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
    Quality validation: rejects empty/uniform/blown-out crops.

    NOTE: the main indicator is the STANDARD DEVIATION (content variance), NOT mean
    brightness. Domains S1 (top illumination, mean ~8) and S2 (bottom, mean ~108)
    have fundamentally different brightness - this is the essence of the domain
    shift we study. The 'mean' threshold is therefore very loose (only extreme
    black/white = a decoding error). An empty/uniform frame has std near 0; a real
    dish with a larva has std > ~15.
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    mean, std = float(img.mean()), float(img.std())
    return (std >= min_std) and (min_mean <= mean <= max_mean)


def filter_and_collect(src_dir: Path) -> List[Path]:
    """Collect valid crops from a directory (after quality validation)."""
    all_pngs = sorted(src_dir.glob("*.png"))
    valid = [p for p in all_pngs if is_valid_crop(p)]
    rejected = len(all_pngs) - len(valid)
    print(f"  {src_dir.name}: {len(all_pngs)} crops, "
          f"{len(valid)} OK, {rejected} rejected")
    return valid


def copy_subset(files: List[Path], dst_dir: Path) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, dst_dir / f.name)
    return len(files)


def main():
    parser = argparse.ArgumentParser(
        description="STEP 1.2/1.3 - split crops into CycleGAN sets"
    )
    parser.add_argument("--crops-dir", type=str, default="data/gan_crops")
    parser.add_argument("--output", type=str, default="data/cyclegan")
    parser.add_argument("--train-per-domain", type=int, default=4000)
    parser.add_argument("--test-per-domain", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    project_root = Path(__file__).resolve().parents[4]
    crops_dir = project_root / args.crops_dir
    out_dir = project_root / args.output

    print(f"{'='*70}")
    print(f"CYCLEGAN DATASET SPLIT")
    print(f"  Input:  {crops_dir}")
    print(f"  Output: {out_dir}")
    print(f"  Train/domain: {args.train_per_domain}, Test/domain: {args.test_per_domain}")
    print(f"{'='*70}")

    mapping = {"s1": ("trainA", "testA"), "s2": ("trainB", "testB")}
    for session, (train_name, test_name) in mapping.items():
        src = crops_dir / session
        if not src.exists():
            print(f"\nERROR: missing {src} - run 01_extract_dish_crops.py first")
            continue

        print(f"\nSession {session.upper()}:")
        valid = filter_and_collect(src)
        random.shuffle(valid)

        n_train = min(args.train_per_domain, len(valid))
        n_test = min(args.test_per_domain, len(valid) - n_train)
        train_files = valid[:n_train]
        test_files = valid[n_train:n_train + n_test]

        n1 = copy_subset(train_files, out_dir / train_name)
        n2 = copy_subset(test_files, out_dir / test_name)
        print(f"  -> {train_name}: {n1}, {test_name}: {n2}")

    print(f"\n{'='*70}\nDONE. Structure:")
    for sub in ["trainA", "trainB", "testA", "testB"]:
        d = out_dir / sub
        n = len(list(d.glob("*.png"))) if d.exists() else 0
        print(f"  {out_dir.name}/{sub}/  ({n} images)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
