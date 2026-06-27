#!/usr/bin/env python3
"""
STEP 4.2 - Pseudo-label synthetic images with a YOLO-seg combined model.

PROBLEM: StyleGAN generates WHOLE synthetic 256x256 dishes without labels (bbox/mask).
SOLUTION: the yolo_baseline_combined model (yolov8s-SEG, within-session mAP ~96%) is
a reliable pseudo-annotator for images in the training style (such as those StyleGAN
generates). Larva detections with confidence >= --conf are accepted as pseudo-labels.

NOTE: the model is SEGMENTATION-based, so we save MASKS in YOLO-seg format:
    <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>     (coordinates normalized 0..1)

We label ONLY the 'larva' class (--class-name). The dish in a 256x256 crop fills the
whole frame - uninformative, skipped.

Images WITHOUT a confident larva detection (conf >= threshold) are REJECTED.

Output:
    <out>/images/synth_NNNNNN.png    # accepted synthetics
    <out>/labels/synth_NNNNNN.txt    # YOLO-seg pseudo-masks (class_id = 0 -> larva remap)
    <out>/qc_preview.png             # grid of 16 synthetics with overlaid masks (sanity check)
    <out>/summary.json               # acceptance statistics

Usage (GPU):
    python src/dl_pipeline/gan/stylegan/07_pseudo_label.py \
        --images data/stylegan_synth/combined \
        --yolo   models/yolo_baseline_combined_best.pt \
        --out    data/stylegan_synth/combined_labeled \
        --conf 0.7 --class-name larva
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics missing. pip install ultralytics")
    raise SystemExit(1)


def mask_to_polygon(mask: np.ndarray, w: int, h: int, eps_frac: float = 0.01):
    """
    Convert a binary mask to a normalized polygon (YOLO-seg format).
    Takes the largest contour, simplifies it (approxPolyDP), normalizes to [0,1].
    Returns a list [x1,y1,x2,y2,...] or None if the contour is too small.
    """
    m = (mask > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 8:  # contour too small = noise
        return None
    eps = eps_frac * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
    if len(approx) < 3:
        return None
    poly = []
    for x, y in approx:
        poly.extend([float(x) / w, float(y) / h])
    return poly


def main():
    ap = argparse.ArgumentParser(description="STEP 4.2 - pseudo-label synthetics (YOLO-seg)")
    ap.add_argument("--images", required=True, help="Folder with synthetics (*.png)")
    ap.add_argument("--yolo", required=True, help="YOLO-seg combined model (.pt)")
    ap.add_argument("--out", required=True, help="Output folder (images/ + labels/)")
    ap.add_argument("--conf", type=float, default=0.7, help="Confidence threshold for acceptance (default 0.7)")
    ap.add_argument("--class-name", type=str, default="larva", help="Class to label (default larva)")
    ap.add_argument("--class-id", type=int, default=0,
                    help="Class ID in pseudo-labels (default 0 - larva as the only class)")
    ap.add_argument("--imgsz", type=int, default=256, help="YOLO input size (256 crops)")
    args = ap.parse_args()

    src = Path(args.images)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    model = YOLO(args.yolo)
    target_cls = None
    for cid, cname in model.names.items():
        if cname == args.class_name:
            target_cls = int(cid)
    if target_cls is None:
        print(f"ERROR: class '{args.class_name}' not in the model. Available: {model.names}")
        raise SystemExit(1)

    imgs = sorted(src.glob("*.png")) + sorted(src.glob("*.jpg"))
    print(f"Images to pseudo-label: {len(imgs)}")
    if not imgs:
        raise SystemExit("ERROR: no images in --images")

    accepted = rejected = 0
    n_instances = 0
    preview_cells = []

    for img_path in imgs:
        results = model.predict(str(img_path), imgsz=args.imgsz, conf=args.conf, verbose=False)
        r = results[0]

        polys = []
        if r.masks is not None and r.boxes is not None:
            cls_arr = r.boxes.cls.cpu().numpy()
            conf_arr = r.boxes.conf.cpu().numpy()
            mask_data = r.masks.data.cpu().numpy()  # (N, mh, mw)
            mh, mw = mask_data.shape[1], mask_data.shape[2]
            for i in range(len(cls_arr)):
                if int(cls_arr[i]) != target_cls or conf_arr[i] < args.conf:
                    continue
                poly = mask_to_polygon(mask_data[i], mw, mh)
                if poly is not None:
                    polys.append(poly)

        if not polys:
            rejected += 1
            continue

        # Accept: copy the image + save YOLO-seg pseudo-masks
        shutil.copy2(img_path, out / "images" / img_path.name)
        label_lines = []
        for poly in polys:
            coords = " ".join(f"{v:.6f}" for v in poly)
            label_lines.append(f"{args.class_id} {coords}")
        (out / "labels" / img_path.with_suffix(".txt").name).write_text("\n".join(label_lines) + "\n")
        accepted += 1
        n_instances += len(polys)

        # QC preview (first 16 accepted)
        if len(preview_cells) < 16:
            vis = cv2.imread(str(img_path))
            for poly in polys:
                pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
                pts[:, 0] *= vis.shape[1]
                pts[:, 1] *= vis.shape[0]
                cv2.polylines(vis, [pts.astype(np.int32)], True, (0, 255, 0), 2)
            preview_cells.append(cv2.resize(vis, (256, 256)))

    # 4x4 QC grid
    if preview_cells:
        cell = 256
        grid = np.full((4 * cell, 4 * cell, 3), 255, dtype=np.uint8)
        for i, c in enumerate(preview_cells[:16]):
            row, col = i // 4, i % 4
            grid[row * cell:(row + 1) * cell, col * cell:(col + 1) * cell] = c
        cv2.imwrite(str(out / "qc_preview.png"), grid)

    total = len(imgs)
    summary = {
        "total_input": total,
        "accepted": accepted,
        "rejected": rejected,
        "acceptance_rate_pct": round(100 * accepted / total, 1) if total else 0.0,
        "n_larva_instances": n_instances,
        "avg_instances_per_image": round(n_instances / accepted, 2) if accepted else 0.0,
        "conf_threshold": args.conf,
        "class_name": args.class_name,
        "yolo_model": args.yolo,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 60)
    print("PSEUDO-LABELING - SUMMARY")
    print(f"  Input:             {total}")
    print(f"  Accepted:          {accepted} ({summary['acceptance_rate_pct']}%)")
    print(f"  Rejected:          {rejected}")
    print(f"  Larva instances:   {n_instances} (avg {summary['avg_instances_per_image']}/image)")
    print(f"  QC grid:           {out / 'qc_preview.png'}  <- INSPECT ~50 samples!")
    print(f"  Summary:           {out / 'summary.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
