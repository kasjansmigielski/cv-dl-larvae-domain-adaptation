#!/usr/bin/env python3
"""
STEP 4.3 - Build 3 YOLO dataset variants to evaluate StyleGAN augmentation.

Combines the REAL S1 dataset with pseudo-labeled StyleGAN synthetics into three
variants:

    A_real_only   : train = real S1            | val/test = real S1   (baseline)
    B_augmented   : train = real S1 + N synth  | val/test = real S1   (does it help?)
    C_synth_only  : train = N synth            | val/test = real S1   (are they enough?)

KEY: the val and test sets are ALWAYS real (real S1) - otherwise we would evaluate
the model "on the GAN's memory". Only the TRAINING set differs between variants.

Assumes a standard YOLO layout for the real dataset:
    <real>/{train,valid|val,test}/{images,labels}/    OR    <real>/images/{train,...}
The script auto-detects the layout (tries both conventions).

Pseudo-labeled synthetics (from step 07) have the layout:
    <synth>/images/*.png + <synth>/labels/*.txt

Usage:
    python src/dl_pipeline/gan/stylegan/08_build_augmented_dataset.py \
        --real  data/session1_only \
        --synth data/stylegan_synth/combined_labeled \
        --out   data/stylegan_variants \
        --n-synth 1000
"""

import argparse
import random
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

IMG_EXTS = (".png", ".jpg", ".jpeg")


def find_split_dir(real_root: Path, split: str) -> Optional[Tuple[Path, Path]]:
    """
    Detect images/labels directories for a split in the real YOLO dataset.
    Supports conventions:
        <root>/<split>/images, <root>/<split>/labels
        <root>/images/<split>, <root>/labels/<split>
    'valid' and 'val' are interchangeable.
    Returns (images_dir, labels_dir) or None.
    """
    split_aliases = [split]
    if split == "val":
        split_aliases = ["val", "valid"]

    for s in split_aliases:
        # Convention 1: <root>/<split>/{images,labels}
        img1 = real_root / s / "images"
        lbl1 = real_root / s / "labels"
        if img1.exists():
            return img1, lbl1
        # Convention 2: <root>/{images,labels}/<split>
        img2 = real_root / "images" / s
        lbl2 = real_root / "labels" / s
        if img2.exists():
            return img2, lbl2
    return None


def list_images(d: Path) -> List[Path]:
    out = []
    for ext in IMG_EXTS:
        out += sorted(d.glob(f"*{ext}"))
    return out


def remap_label_to_larva(lbl_text: str, keep_src_cls: Optional[int]) -> str:
    """
    Unify labels to a SINGLE 'larva' class (id=0).

    The real combined dataset has 2 classes (dish=0, larva=1), while synthetics have
    1 class (larva=0). To merge them consistently, we keep ONLY larva objects and
    remap their id to 0. 'dish' (the whole dish frame) is uninformative for larva
    detection - skipped.

    keep_src_cls:
        - int  -> keep only lines with this class_id (real: larva=1); output gets id 0
        - None -> do not filter by class, just force id 0 (synthetics are larva=0 anyway)
    """
    out_lines = []
    for line in lbl_text.strip().splitlines():
        parts = line.split()
        if not parts:
            continue
        src_cls = int(float(parts[0]))
        if keep_src_cls is not None and src_cls != keep_src_cls:
            continue  # skip e.g. dish
        out_lines.append("0 " + " ".join(parts[1:]))  # remap class_id -> 0 (larva)
    return ("\n".join(out_lines) + "\n") if out_lines else ""


def copy_pair(img: Path, src_lbl_dir: Path, dst_img: Path, dst_lbl: Path,
              prefix: str = "", keep_src_cls: Optional[int] = None):
    """
    Copy an image + label, unifying the class to 'larva' (id=0).
    keep_src_cls: which class in the source is the larva (real=1); None for synthetics (already 0).
    """
    name = (prefix + img.name) if prefix else img.name
    shutil.copy2(img, dst_img / name)
    lbl = src_lbl_dir / img.with_suffix(".txt").name
    dst_lbl_path = dst_lbl / (prefix + img.with_suffix(".txt").name if prefix else img.with_suffix(".txt").name)
    if lbl.exists():
        dst_lbl_path.write_text(remap_label_to_larva(lbl.read_text(), keep_src_cls))
    else:
        # no label = background image (empty label is allowed in YOLO)
        dst_lbl_path.write_text("")


def make_split_dirs(variant_root: Path, split: str) -> Tuple[Path, Path]:
    img = variant_root / split / "images"
    lbl = variant_root / split / "labels"
    img.mkdir(parents=True, exist_ok=True)
    lbl.mkdir(parents=True, exist_ok=True)
    return img, lbl


def write_yaml(variant_root: Path):
    yaml_text = (
        "names:\n- larva\n"
        "nc: 1\n"
        f"path: {variant_root}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
    )
    (variant_root / "data.yaml").write_text(yaml_text)


def copy_real_split(real_root: Path, split: str, variant_root: Path, keep_src_cls: int):
    """Copy a real split (e.g. val/test) into the variant, unifying the class to larva (id=0)."""
    found = find_split_dir(real_root, split)
    if found is None:
        print(f"  WARNING: real split '{split}' not found in {real_root}")
        return 0
    src_img, src_lbl = found
    dst_img, dst_lbl = make_split_dirs(variant_root, split)
    imgs = list_images(src_img)
    for im in imgs:
        copy_pair(im, src_lbl, dst_img, dst_lbl, keep_src_cls=keep_src_cls)
    return len(imgs)


def main():
    ap = argparse.ArgumentParser(description="STEP 4.3 - build 3 augmentation dataset variants")
    ap.add_argument("--real", required=True, help="Real S1 dataset (YOLO layout)")
    ap.add_argument("--synth", required=True, help="Pseudo-labeled synthetics (images/ + labels/) from step 07")
    ap.add_argument("--out", required=True, help="Output folder for the variants")
    ap.add_argument("--n-synth", type=int, default=1000, help="How many synthetics to use in B and C (default 1000)")
    ap.add_argument("--real-larva-cls", type=int, default=1,
                    help="larva class_id in the REAL dataset (combined: dish=0, larva=1 -> 1)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    real = Path(args.real)
    synth = Path(args.synth)
    out = Path(args.out)

    # --- Real training data ---
    real_train = find_split_dir(real, "train")
    if real_train is None:
        print(f"ERROR: real split 'train' not found in {real}")
        raise SystemExit(1)
    real_train_img, real_train_lbl = real_train
    real_train_imgs = list_images(real_train_img)

    # --- Synthetics (pseudo-labeled) ---
    synth_img_dir = synth / "images"
    synth_lbl_dir = synth / "labels"
    if not synth_img_dir.exists():
        print(f"ERROR: missing {synth_img_dir} - run 07_pseudo_label.py first")
        raise SystemExit(1)
    synth_imgs = list_images(synth_img_dir)
    random.shuffle(synth_imgs)
    synth_use = synth_imgs[: args.n_synth]

    print(f"{'='*70}")
    print(f"BUILDING STYLEGAN AUGMENTATION DATASET VARIANTS")
    print(f"  Real train:  {len(real_train_imgs)} images")
    print(f"  Synthetics:  {len(synth_imgs)} available, using {len(synth_use)}")
    print(f"  Output:      {out}")
    print(f"{'='*70}")

    variants = ["A_real_only", "B_augmented", "C_synth_only"]
    for v in variants:
        vroot = out / v
        print(f"\n[{v}]")

        # TRAIN. Real: filter to larva (keep_src_cls). Synthetics: already larva=0 (None).
        dst_img, dst_lbl = make_split_dirs(vroot, "train")
        if v == "A_real_only":
            for im in real_train_imgs:
                copy_pair(im, real_train_lbl, dst_img, dst_lbl, keep_src_cls=args.real_larva_cls)
            print(f"  train: {len(real_train_imgs)} real")
        elif v == "B_augmented":
            for im in real_train_imgs:
                copy_pair(im, real_train_lbl, dst_img, dst_lbl, keep_src_cls=args.real_larva_cls)
            for im in synth_use:
                copy_pair(im, synth_lbl_dir, dst_img, dst_lbl, prefix="synth_", keep_src_cls=None)
            print(f"  train: {len(real_train_imgs)} real + {len(synth_use)} synth")
        elif v == "C_synth_only":
            for im in synth_use:
                copy_pair(im, synth_lbl_dir, dst_img, dst_lbl, prefix="synth_", keep_src_cls=None)
            print(f"  train: {len(synth_use)} synth")

        # VAL + TEST - always real (also unified to larva=0)
        n_val = copy_real_split(real, "val", vroot, keep_src_cls=args.real_larva_cls)
        n_test = copy_real_split(real, "test", vroot, keep_src_cls=args.real_larva_cls)
        print(f"  val: {n_val} real | test: {n_test} real")

        write_yaml(vroot)
        print(f"  data.yaml -> {vroot / 'data.yaml'}")

    print(f"\n{'='*70}")
    print("DONE. Train each variant:")
    for v in variants:
        print(f"  sbatch train_yolo_variant.slurm {v}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
