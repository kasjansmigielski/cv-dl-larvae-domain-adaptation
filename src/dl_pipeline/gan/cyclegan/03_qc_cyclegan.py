#!/usr/bin/env python3
"""
STEP 2.5 (QC) - Quality control of the CycleGAN translation.

Input: outputs of test.py from the CycleGAN repo, in the format:
    results/S2_to_S1/images/<name>_real.png   # original S2
    results/S2_to_S1/images/<name>_fake.png   # translated S2->S1 (G_B)

What it computes:
  1. CENTROID DRIFT - runs _real and _fake through a YOLO model and compares the
     larva position. Small drift (<3 px) means CycleGAN did not move the larva, so
     the labels still match and YOLO can be run without re-annotation.
  2. DETECTION RATE - % of images on which YOLO detects a larva, for _real vs _fake.
     An increase on _fake means the translation helps the model (signal that the
     domain gap is being closed).
  3. VISUAL GRID - a real|fake grid (PNG).
  4. Prepares folders for FID (real/ and fake/ separately) - FID is computed with a
     separate command.

Usage (on a GPU):
    python src/dl_pipeline/gan/cyclegan/03_qc_cyclegan.py \
        --results-dir <cyclegan_repo>/results/S2_to_S1/images \
        --yolo-model models/yolo_baseline_combined_best.pt \
        --out qc_S2_to_S1
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics missing. pip install ultralytics")
    raise SystemExit(1)


def larva_centroid(model, img_path: Path, conf: float = 0.25) -> Optional[Tuple[float, float, float]]:
    """Return (cx, cy, conf) of the best larva detection, or None if absent."""
    results = model.predict(str(img_path), imgsz=256, conf=conf, verbose=False)
    best = None
    for r in results:
        if r.boxes is None:
            continue
        for i, cls_id in enumerate(r.boxes.cls.cpu().numpy()):
            if model.names[int(cls_id)] != "larva":
                continue
            c = float(r.boxes.conf[i].cpu().numpy())
            x1, y1, x2, y2 = r.boxes.xyxy[i].cpu().numpy()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if best is None or c > best[2]:
                best = (float(cx), float(cy), c)
    return best


def make_grid(pairs, out_path: Path, n: int = 16):
    """Save a real|fake grid (n pairs) as a single PNG."""
    n = min(n, len(pairs))
    if n == 0:
        return
    cell = 256
    cols = 2  # real | fake
    rows = n
    grid = np.full((rows * cell, cols * cell, 3), 255, dtype=np.uint8)
    for i, (real_p, fake_p) in enumerate(pairs[:n]):
        for j, p in enumerate([real_p, fake_p]):
            img = cv2.imread(str(p))
            if img is None:
                continue
            img = cv2.resize(img, (cell, cell))
            grid[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell] = img
    cv2.imwrite(str(out_path), grid)


def main():
    ap = argparse.ArgumentParser(description="STEP 2.5 QC - CycleGAN translation quality")
    ap.add_argument("--results-dir", required=True, help="images/ folder with *_real / *_fake pairs")
    ap.add_argument("--yolo-model", required=True, help="YOLO model (.pt)")
    ap.add_argument("--out", required=True, help="QC output folder")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--drift-threshold", type=float, default=3.0, help="Centroid drift threshold [px]")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fid_real = out_dir / "fid_real"
    fid_fake = out_dir / "fid_fake"
    fid_real.mkdir(exist_ok=True)
    fid_fake.mkdir(exist_ok=True)

    # Pair _real with _fake by common prefix
    reals = sorted(results_dir.glob("*_real.png"))
    pairs = []
    for r in reals:
        f = r.with_name(r.name.replace("_real.png", "_fake.png"))
        if f.exists():
            pairs.append((r, f))
    print(f"Found {len(pairs)} real/fake pairs in {results_dir}")
    if not pairs:
        raise SystemExit("ERROR: no *_real/*_fake pairs - check --results-dir")

    model = YOLO(args.yolo_model)

    drifts = []
    det_real = det_fake = 0
    for idx, (real_p, fake_p) in enumerate(pairs):
        shutil.copy2(real_p, fid_real / real_p.name)
        shutil.copy2(fake_p, fid_fake / fake_p.name)

        cr = larva_centroid(model, real_p, args.conf)
        cf = larva_centroid(model, fake_p, args.conf)
        if cr is not None:
            det_real += 1
        if cf is not None:
            det_fake += 1
        if cr is not None and cf is not None:
            drifts.append(float(np.hypot(cr[0] - cf[0], cr[1] - cf[1])))

    n = len(pairs)
    drifts = np.array(drifts) if drifts else np.array([np.nan])
    summary = {
        "n_pairs": n,
        "detection_rate_real_pct": round(100 * det_real / n, 1),
        "detection_rate_fake_pct": round(100 * det_fake / n, 1),
        "centroid_drift_px": {
            "n_both_detected": int(np.sum(~np.isnan(drifts))),
            "mean": round(float(np.nanmean(drifts)), 2),
            "median": round(float(np.nanmedian(drifts)), 2),
            "p95": round(float(np.nanpercentile(drifts, 95)), 2),
            "pct_below_threshold": round(100 * float(np.mean(drifts < args.drift_threshold)), 1),
            "threshold_px": args.drift_threshold,
        },
    }

    make_grid(pairs, out_dir / "grid_real_vs_fake.png", n=16)
    with open(out_dir / "qc_summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)

    print("\n" + "=" * 60)
    print("QC SUMMARY")
    print(f"  real/fake pairs:               {n}")
    print(f"  Detection rate REAL (S2):      {summary['detection_rate_real_pct']}%")
    print(f"  Detection rate FAKE (S2->S1):  {summary['detection_rate_fake_pct']}%")
    print(f"  Centroid drift (median):       {summary['centroid_drift_px']['median']} px")
    print(f"  Drift < {args.drift_threshold}px:                   "
          f"{summary['centroid_drift_px']['pct_below_threshold']}% of pairs")
    print(f"  Grid:    {out_dir / 'grid_real_vs_fake.png'}")
    print(f"  FID folders: {fid_real} , {fid_fake}")
    print("=" * 60)
    print("\nCompute FID separately (compare fake to REAL S1 images):")
    print(f"  python -m pytorch_fid {fid_fake} <folder with real S1 crops (trainA/testA)>")


if __name__ == "__main__":
    main()
