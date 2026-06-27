#!/usr/bin/env python3
"""
Preprocessing: automatic dish cropping from raw recordings.

Uses a YOLO model to detect dish positions on the first frames, then crops
each dish and saves it as a separate 640x640 video.

Pipeline:
  Raw video (1920x1080, 6 dishes)
    -> dish detection (first frames)
    -> per-dish crop
    -> resize 640x640
    -> 6 separate .mp4 files

Usage:
    # All recordings from both sessions:
    python src/dl_pipeline/yolo/00_preprocess_dishes.py --all

    # A single recording:
    python src/dl_pipeline/yolo/00_preprocess_dishes.py --video data/raw_videos/s1/control.mp4

    # With a custom dish-detection model:
    python src/dl_pipeline/yolo/00_preprocess_dishes.py --all --dish-model models/yolo_baseline_combined_best.pt
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


DISH_DIAMETER_MM = 98.0
OUTPUT_SIZE = (640, 640)
PADDING_FACTOR = 1.10  # 10% margin around the dish

EXPERIMENTS = ["control", "pbs", "coli_5x10_7", "coli_2x10_8", "coli_5x10_8"]

S1_VIDEO_MAP = {
    "control": "control", "pbs": "pbs",
    "coli_2x10_8": "coli_2x10_8",
    "coli_5x10_7": "coli_5x10_7",
    "coli_5x10_8": "coli_5x10_8",
}
S2_VIDEO_MAP = dict(S1_VIDEO_MAP)


def detect_dish_rois(
    video_path: str,
    model_path: str,
    n_dishes: int = 6,
    conf: float = 0.9,
    n_frames: int = 10,
) -> List[Tuple[int, int, int, int]]:
    """
    Detect dish positions from the first frames of a recording.

    Averages detections over several frames for stability. Returns a list of
    ROIs as (x1, y1, x2, y2), sorted top-left -> bottom-right (2 rows x 3 cols).
    """
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"ERROR: cannot open {video_path}")
        sys.exit(1)

    all_boxes = []
    for frame_idx in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, imgsz=640, conf=conf, verbose=False)
        for r in results:
            if r.boxes is None:
                continue
            for i, cls_id in enumerate(r.boxes.cls.cpu().numpy()):
                if model.names[int(cls_id)] == "dish":
                    box = r.boxes.xyxy[i].cpu().numpy()
                    all_boxes.append(box)

    cap.release()

    if len(all_boxes) < n_dishes:
        print(f"WARNING: detected only {len(all_boxes)} dish boxes (expected >={n_dishes})")
        if len(all_boxes) == 0:
            print("ERROR: no dish detections!")
            sys.exit(1)

    all_boxes = np.array(all_boxes)

    from scipy.cluster.hierarchy import fcluster, linkage

    centroids = np.column_stack([
        (all_boxes[:, 0] + all_boxes[:, 2]) / 2,
        (all_boxes[:, 1] + all_boxes[:, 3]) / 2,
    ])

    Z = linkage(centroids, method="ward")
    n_clusters = min(n_dishes, len(all_boxes))
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")

    dish_rois = []
    for cluster_id in range(1, n_clusters + 1):
        mask = labels == cluster_id
        cluster_boxes = all_boxes[mask]

        x1 = np.median(cluster_boxes[:, 0])
        y1 = np.median(cluster_boxes[:, 1])
        x2 = np.median(cluster_boxes[:, 2])
        y2 = np.median(cluster_boxes[:, 3])

        dish_rois.append((x1, y1, x2, y2))

    dish_rois = np.array(dish_rois)
    dish_centroids = np.column_stack([
        (dish_rois[:, 0] + dish_rois[:, 2]) / 2,
        (dish_rois[:, 1] + dish_rois[:, 3]) / 2,
    ])

    y_median = np.median(dish_centroids[:, 1])
    top_row_idx = np.where(dish_centroids[:, 1] < y_median)[0]
    bot_row_idx = np.where(dish_centroids[:, 1] >= y_median)[0]

    top_sorted = top_row_idx[np.argsort(dish_centroids[top_row_idx, 0])]
    bot_sorted = bot_row_idx[np.argsort(dish_centroids[bot_row_idx, 0])]

    ordered = []
    for t, b in zip(top_sorted, bot_sorted):
        ordered.append(t)
        ordered.append(b)
    remaining_top = [t for t in top_sorted if t not in ordered]
    remaining_bot = [b for b in bot_sorted if b not in ordered]
    ordered.extend(remaining_top)
    ordered.extend(remaining_bot)

    sorted_rois = []
    for idx in ordered:
        x1, y1, x2, y2 = dish_rois[idx]

        w = x2 - x1
        h = y2 - y1
        side = max(w, h) * PADDING_FACTOR

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        half = side / 2

        rx1 = max(0, int(cx - half))
        ry1 = max(0, int(cy - half))
        rx2 = int(cx + half)
        ry2 = int(cy + half)

        sorted_rois.append((rx1, ry1, rx2, ry2))

    return sorted_rois


def crop_and_save_dish(
    video_path: str,
    output_path: str,
    roi: Tuple[int, int, int, int],
    output_size: Tuple[int, int] = OUTPUT_SIZE,
    max_frames: Optional[int] = None,
) -> dict:
    """Crop a single dish from the recording and save it as a 640x640 mp4."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if max_frames:
        total = min(total, max_frames)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, output_size)

    x1, y1, x2, y2 = roi
    crop_w = x2 - x1
    crop_h = y2 - y1

    i = -1
    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        h_frame, w_frame = frame.shape[:2]
        cx1 = max(0, x1)
        cy1 = max(0, y1)
        cx2 = min(w_frame, x2)
        cy2 = min(h_frame, y2)

        cropped = frame[cy1:cy2, cx1:cx2]
        resized = cv2.resize(cropped, output_size, interpolation=cv2.INTER_AREA)
        out.write(resized)

    cap.release()
    out.release()

    pixels_per_mm = output_size[0] / (DISH_DIAMETER_MM * PADDING_FACTOR)
    scale_mm_per_px = 1.0 / pixels_per_mm

    return {
        "n_frames": i + 1,
        "fps": fps,
        "roi": list(roi),
        "crop_size_px": [crop_w, crop_h],
        "output_size": list(output_size),
        "pixels_per_mm": round(pixels_per_mm, 4),
        "scale_mm_per_px": round(scale_mm_per_px, 4),
    }


def preprocess_video(
    video_path: str,
    dish_model_path: str,
    output_dir: str,
    session: str,
    experiment: str,
    n_dishes: int = 6,
    max_frames: Optional[int] = None,
):
    """Full preprocessing of one recording: dish detection -> crop -> save."""
    video_path = Path(video_path)
    output_path = Path(output_dir) / session / experiment
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n  Detecting dishes in {video_path.name}...")
    rois = detect_dish_rois(str(video_path), dish_model_path, n_dishes=n_dishes)
    print(f"  Found {len(rois)} dishes")

    metadata = {
        "source_video": str(video_path),
        "session": session,
        "experiment": experiment,
        "n_dishes": len(rois),
        "dishes": {},
    }

    for dish_id, roi in enumerate(rois, start=1):
        out_file = output_path / f"dish_{dish_id}.mp4"
        print(f"    Dish {dish_id}: ROI={roi} -> {out_file.name}...", end=" ", flush=True)

        info = crop_and_save_dish(
            str(video_path),
            str(out_file),
            roi,
            max_frames=max_frames,
        )
        print(f"({info['n_frames']} frames, scale={info['scale_mm_per_px']:.4f} mm/px)")

        metadata["dishes"][str(dish_id)] = {
            "output_file": str(out_file),
            **info,
        }

    meta_path = output_path / "preprocessing_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata: {meta_path}")

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Preprocessing: automatic dish cropping from raw recordings"
    )
    parser.add_argument("--dish-model", type=str,
                        default="models/yolo_baseline_combined_best.pt",
                        help="YOLO model for dish detection (must contain a 'dish' class)")
    parser.add_argument("--output", type=str, default="data/preprocessed_videos",
                        help="Output folder (default: data/preprocessed_videos)")
    parser.add_argument("--all", action="store_true",
                        help="Process all recordings from both sessions")
    parser.add_argument("--video", type=str, default=None,
                        help="Single recording to process")
    parser.add_argument("--session", type=str, default="s1",
                        help="Session (s1 or s2)")
    parser.add_argument("--experiment", type=str, default="control",
                        help="Experiment name")
    parser.add_argument("--n-dishes", type=int, default=6,
                        help="Expected number of dishes (default: 6)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Frame limit (for testing, e.g. 1000)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[3]
    output_dir = project_root / args.output

    print(f"{'='*70}")
    print(f"DISH PREPROCESSING")
    print(f"  Dish model:   {args.dish_model}")
    print(f"  Output:       {output_dir}")
    print(f"  Output size:  {OUTPUT_SIZE[0]}x{OUTPUT_SIZE[1]}")
    print(f"  Padding:      {PADDING_FACTOR:.0%}")
    print(f"{'='*70}")

    if args.all:
        for session_key, vmap in [("s1", S1_VIDEO_MAP), ("s2", S2_VIDEO_MAP)]:
            vdir = project_root / "data" / "raw_videos" / session_key
            if not vdir.exists():
                print(f"\nSkipping session {session_key}: missing {vdir}")
                continue

            video_files = sorted(vdir.glob("*.mp4")) + sorted(vdir.glob("*.mov"))
            print(f"\n{'='*70}")
            print(f"SESSION {session_key.upper()} - {len(video_files)} recordings")
            print(f"{'='*70}")

            for vf in video_files:
                stem = vf.stem.lower().replace(" ", "_")
                experiment = None
                for key, exp_name in vmap.items():
                    if key.lower() in stem:
                        experiment = exp_name
                        break
                if experiment is None:
                    print(f"\n  Skipping: {vf.name} (unrecognized group)")
                    continue

                preprocess_video(
                    str(vf), args.dish_model, str(output_dir),
                    session_key, experiment, args.n_dishes, args.max_frames,
                )

    elif args.video:
        preprocess_video(
            args.video, args.dish_model, str(output_dir),
            args.session, args.experiment, args.n_dishes, args.max_frames,
        )
    else:
        print("Usage:")
        print(f"  python {__file__} --all")
        print(f"  python {__file__} --video data/raw_videos/s1/control.mp4")


if __name__ == "__main__":
    main()
