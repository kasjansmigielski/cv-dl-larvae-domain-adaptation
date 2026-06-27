#!/usr/bin/env python3
"""
STEP 1.1 - Extract dish crops (256x256 PNG frames) from raw recordings.

Goal: prepare image sets for training CycleGAN (S1 vs S2) and StyleGAN2-ADA.
Unlike 00_preprocess_dishes.py (which writes 640x640 *videos* for kinematic
analysis), this script saves single frames as 256x256 PNGs - the format expected
by pytorch-CycleGAN-and-pix2pix and stylegan2-ada-pytorch.

Pipeline for one recording (1920x1080, 6 dishes):
    Dish detection (YOLO) on the first frames
      -> 6 ROIs (square, sorted)
      -> sample every N frames
      -> per-dish crop + resize 256x256
      -> save PNG: crops/<session>/<experiment>_dish<k>_f<frame>.png

Labels are NOT needed - CycleGAN is unsupervised (unpaired).

Usage:
    # All recordings from both sessions (S1 + S2):
    python src/dl_pipeline/gan/cyclegan/01_extract_dish_crops.py --all

    # A single recording (test):
    python src/dl_pipeline/gan/cyclegan/01_extract_dish_crops.py \
        --video data/raw_videos/s1/control.mp4 --session s1 --experiment control

    # Different sampling step / frame limit (quick test):
    python src/dl_pipeline/gan/cyclegan/01_extract_dish_crops.py --all --frame-step 50 --max-frames 500
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: install ultralytics: pip install ultralytics")
    sys.exit(1)


# ============================================================================
# CONSTANTS
# ============================================================================

OUTPUT_SIZE = (256, 256)       # target crop size (CycleGAN/StyleGAN)
PADDING_FACTOR = 1.10          # 10% margin around the dish (consistent with 00_preprocess)
DEFAULT_FRAME_STEP = 25        # sampling step in frames (25 fps => ~1 s)

# File-name -> experiment-name mapping (consistent with 00_preprocess_dishes.py)
VIDEO_MAP = {
    "control": "control", "pbs": "pbs",
    "coli_2x10_8": "coli_2x10_8",
    "coli_5x10_7": "coli_5x10_7",
    "coli_5x10_8": "coli_5x10_8",
}


# ============================================================================
# DISH DETECTION (simplified version from 00_preprocess_dishes.py)
# ============================================================================

def detect_dish_rois(
    video_path: str,
    model_path: str,
    n_dishes: int = 6,
    conf: float = 0.9,
    n_frames: int = 10,
) -> List[Tuple[int, int, int, int]]:
    """
    Detect dish positions from the first frames (averaged over several frames).
    Returns a list of square ROIs (x1, y1, x2, y2) with a PADDING_FACTOR margin.
    """
    from scipy.cluster.hierarchy import fcluster, linkage

    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: cannot open {video_path}")
        sys.exit(1)

    all_boxes = []
    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, imgsz=640, conf=conf, verbose=False)
        for r in results:
            if r.boxes is None:
                continue
            for i, cls_id in enumerate(r.boxes.cls.cpu().numpy()):
                if model.names[int(cls_id)] == "dish":
                    all_boxes.append(r.boxes.xyxy[i].cpu().numpy())
    cap.release()

    if len(all_boxes) == 0:
        print("ERROR: no dish detections!")
        sys.exit(1)
    if len(all_boxes) < n_dishes:
        print(f"WARNING: detected only {len(all_boxes)} boxes (expected >={n_dishes})")

    all_boxes = np.array(all_boxes)
    centroids = np.column_stack([
        (all_boxes[:, 0] + all_boxes[:, 2]) / 2,
        (all_boxes[:, 1] + all_boxes[:, 3]) / 2,
    ])
    n_clusters = min(n_dishes, len(all_boxes))
    labels = fcluster(linkage(centroids, method="ward"), t=n_clusters, criterion="maxclust")

    dish_rois = []
    for cluster_id in range(1, n_clusters + 1):
        cb = all_boxes[labels == cluster_id]
        dish_rois.append((
            np.median(cb[:, 0]), np.median(cb[:, 1]),
            np.median(cb[:, 2]), np.median(cb[:, 3]),
        ))
    dish_rois = np.array(dish_rois)

    # Sort top-left -> bottom-right (2 rows x 3 cols)
    dc = np.column_stack([
        (dish_rois[:, 0] + dish_rois[:, 2]) / 2,
        (dish_rois[:, 1] + dish_rois[:, 3]) / 2,
    ])
    y_median = np.median(dc[:, 1])
    top = np.where(dc[:, 1] < y_median)[0]
    bot = np.where(dc[:, 1] >= y_median)[0]
    top = top[np.argsort(dc[top, 0])]
    bot = bot[np.argsort(dc[bot, 0])]
    ordered = []
    for t, b in zip(top, bot):
        ordered.extend([t, b])
    ordered.extend([t for t in top if t not in ordered])
    ordered.extend([b for b in bot if b not in ordered])

    sorted_rois = []
    for idx in ordered:
        x1, y1, x2, y2 = dish_rois[idx]
        side = max(x2 - x1, y2 - y1) * PADDING_FACTOR
        cx, cy, half = (x1 + x2) / 2, (y1 + y2) / 2, side / 2
        sorted_rois.append((
            max(0, int(cx - half)), max(0, int(cy - half)),
            int(cx + half), int(cy + half),
        ))
    return sorted_rois


# ============================================================================
# PNG FRAME EXTRACTION
# ============================================================================

def extract_crops_from_video(
    video_path: str,
    dish_model_path: str,
    output_dir: Path,
    session: str,
    experiment: str,
    n_dishes: int = 6,
    frame_step: int = DEFAULT_FRAME_STEP,
    max_frames: Optional[int] = None,
    skip_start: int = 250,
    target_crops: Optional[int] = None,
) -> int:
    """
    Extract 256x256 PNG frames for each dish from one recording.
    Returns the number of saved crops.

    If target_crops is given, frame_step is chosen AUTOMATICALLY so that the
    whole recording (regardless of length) yields ~target_crops crops. This
    balances counts between S1 (32 min) and S2 (15 min) recordings.
    """
    video_path = Path(video_path)
    print(f"\n  Detecting dishes in {video_path.name}...")
    rois = detect_dish_rois(str(video_path), dish_model_path, n_dishes=n_dishes)
    print(f"  Found {len(rois)} dishes")

    out_session = output_dir / session
    out_session.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total = min(total, max_frames + skip_start)

    # Auto step: target_crops images = (usable_frames / step) * n_dishes
    if target_crops:
        usable = max(1, total - skip_start)
        frames_needed = max(1, target_crops // max(1, len(rois)))
        frame_step = max(1, usable // frames_needed)
        print(f"  Auto frame-step={frame_step} (target ~{target_crops} crops from {usable} frames)")

    saved = 0
    frame_idx = 0
    while frame_idx < total:
        ret, frame = cap.read()
        if not ret:
            break
        # Skip the beginning (larvae being placed) and sample every frame_step
        if frame_idx >= skip_start and (frame_idx - skip_start) % frame_step == 0:
            h_frame, w_frame = frame.shape[:2]
            for dish_id, (x1, y1, x2, y2) in enumerate(rois, start=1):
                cx1, cy1 = max(0, x1), max(0, y1)
                cx2, cy2 = min(w_frame, x2), min(h_frame, y2)
                cropped = frame[cy1:cy2, cx1:cx2]
                if cropped.size == 0:
                    continue
                resized = cv2.resize(cropped, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)
                out_file = out_session / f"{experiment}_dish{dish_id}_f{frame_idx:06d}.png"
                cv2.imwrite(str(out_file), resized)
                saved += 1
        frame_idx += 1
    cap.release()

    print(f"  Saved {saved} crops -> {out_session}")
    return saved


def main():
    parser = argparse.ArgumentParser(
        description="STEP 1.1 - extract dish crops (256x256 PNG) for CycleGAN/StyleGAN"
    )
    parser.add_argument("--dish-model", type=str,
                        default="models/yolo_baseline_combined_best.pt",
                        help="YOLO model for dish detection (class 'dish')")
    parser.add_argument("--output", type=str, default="data/gan_crops",
                        help="Output folder (default: data/gan_crops)")
    parser.add_argument("--all", action="store_true",
                        help="Process all recordings from both sessions")
    parser.add_argument("--video", type=str, default=None,
                        help="Single recording")
    parser.add_argument("--session", type=str, default="s1", help="Session (s1/s2)")
    parser.add_argument("--experiment", type=str, default="control",
                        help="Experiment name (with --video)")
    parser.add_argument("--n-dishes", type=int, default=6)
    parser.add_argument("--frame-step", type=int, default=DEFAULT_FRAME_STEP,
                        help=f"Sampling step in frames (default: {DEFAULT_FRAME_STEP})")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Frame limit per recording (for testing)")
    parser.add_argument("--skip-start", type=int, default=250,
                        help="Frames to skip at the start (default 250 = 10 s)")
    parser.add_argument("--target-crops", type=int, default=None,
                        help="Target number of crops PER RECORDING (auto-tunes frame-step, "
                             "balances S1/S2 of different length). Recommended: 600.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[4]
    output_dir = project_root / args.output

    print(f"{'='*70}")
    print(f"DISH CROP EXTRACTION (CycleGAN/StyleGAN)")
    print(f"  Dish model:   {args.dish_model}")
    print(f"  Output:       {output_dir}")
    print(f"  Size:         {OUTPUT_SIZE[0]}x{OUTPUT_SIZE[1]} PNG")
    print(f"  Frame step:   every {args.frame_step} frames")
    print(f"{'='*70}")

    total_saved = 0
    if args.all:
        for session_key in ["s1", "s2"]:
            vdir = project_root / "data" / "raw_videos" / session_key
            if not vdir.exists():
                print(f"\nSkipping session {session_key}: missing {vdir}")
                continue
            video_files = sorted(vdir.glob("*.mp4")) + sorted(vdir.glob("*.mov"))
            print(f"\n{'='*70}\nSESSION {session_key.upper()} - {len(video_files)} recordings\n{'='*70}")
            for vf in video_files:
                stem = vf.stem.lower().replace(" ", "_")
                experiment = next(
                    (exp for key, exp in VIDEO_MAP.items() if key.lower() in stem), None
                )
                if experiment is None:
                    print(f"\n  Skipping: {vf.name} (unrecognized group)")
                    continue
                total_saved += extract_crops_from_video(
                    str(vf), args.dish_model, output_dir, session_key, experiment,
                    args.n_dishes, args.frame_step, args.max_frames, args.skip_start,
                    args.target_crops,
                )
    elif args.video:
        total_saved += extract_crops_from_video(
            args.video, args.dish_model, output_dir, args.session, args.experiment,
            args.n_dishes, args.frame_step, args.max_frames, args.skip_start,
            args.target_crops,
        )
    else:
        print("Usage:\n  python src/dl_pipeline/gan/cyclegan/01_extract_dish_crops.py --all")
        return

    print(f"\n{'='*70}")
    print(f"DONE. Saved {total_saved} crops in total.")
    for session_key in ["s1", "s2"]:
        sdir = output_dir / session_key
        if sdir.exists():
            n = len(list(sdir.glob("*.png")))
            print(f"  {session_key}: {n} crops in {sdir}")
    print(f"{'='*70}")

    summary = {
        "output_dir": str(output_dir),
        "output_size": list(OUTPUT_SIZE),
        "frame_step": args.frame_step,
        "skip_start": args.skip_start,
        "total_saved": total_saved,
    }
    with open(output_dir / "extraction_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
