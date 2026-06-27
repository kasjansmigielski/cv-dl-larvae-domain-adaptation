#!/usr/bin/env python3
"""
YOLO pipeline with a tracker (ByteTrack) - solves the multi-detection problem.

Instead of 'highest confidence per dish per frame' (unstable), we use the YOLO
tracker, which maintains object identity across frames and eliminates detection
jumping.

Usage:
    # Single recording (quick test, ~5 min):
    python src/dl_pipeline/yolo/01_track_and_analyze.py --video data/raw_videos/s1/control.mp4 \
        --session s1 --experiment control

    # All recordings from both sessions:
    python src/dl_pipeline/yolo/01_track_and_analyze.py --all

    # With a custom model:
    python src/dl_pipeline/yolo/01_track_and_analyze.py --all --model models/yolo_baseline_combined_best.pt
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: install ultralytics: pip install ultralytics")
    sys.exit(1)

# ============================================================================
# CONSTANTS
# ============================================================================

DISH_DIAMETER_MM = 98.0
FPS = 25


def get_device() -> str:
    """Detect the best available backend (mps > cuda > cpu)."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


MIN_MOVEMENT_THRESHOLD_MM = 0.0  # Consistent with OpenCV - no MIN distance filter
MAX_MOVEMENT_THRESHOLD_MM = 5.0
ARTIFACT_PROXIMITY_FRAMES = 5  # +/-5 frames around an artifact are ignored (as in OpenCV)
NOISE_THRESHOLD_PX = 1.5
SKIP_START_FRAMES = 250  # 10 s
TRAJECTORY_JUMP_THRESHOLD_PX = 10.0
MIN_DETECTION_RATE = 95.0
MIN_ANGLE_THRESHOLD_MM = 0.5  # MIN filter only for rose diagrams (angles)

LARVA_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"
]

EXPERIMENTS = ["control", "pbs", "coli_5x10_7", "coli_2x10_8", "coli_5x10_8"]
SESSIONS = ["s1", "s2"]

S1_VIDEO_MAP = {
    "control": "control", "pbs": "pbs",
    "coli_2x10_8": "coli_2x10_8",
    "coli_5x10_7": "coli_5x10_7",
    "coli_5x10_8": "coli_5x10_8",
}
S2_VIDEO_MAP = dict(S1_VIDEO_MAP)


# ============================================================================
# DISH CALIBRATION (from the first frames, without the tracker)
# ============================================================================

def calibrate_dishes(
    model: YOLO,
    video_path: str,
    n_frames: int = 100,
    dish_class_name: str = "dish",
) -> Tuple[float, np.ndarray, float]:
    """
    Determine dish positions and the mm/px scale from the first frames.
    Returns (scale_mm_per_px, ordered_centers, avg_diameter_px).
    """
    from scipy.cluster.hierarchy import fcluster, linkage

    device = get_device()
    results = model.predict(
        source=video_path, imgsz=640, conf=0.7,
        stream=True, verbose=False, device=device,
    )

    all_dish_centroids = []
    for i, r in enumerate(results):
        if i >= n_frames:
            break
        if r.boxes is None:
            continue
        classes = r.boxes.cls.cpu().numpy()
        boxes = r.boxes.xyxy.cpu().numpy()
        for j, cls_id in enumerate(classes):
            if model.names[int(cls_id)] == dish_class_name:
                box = boxes[j]
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                w = box[2] - box[0]
                h = box[3] - box[1]
                all_dish_centroids.append((cx, cy, w, h))

    if len(all_dish_centroids) < 6:
        print(f"    WARNING: only {len(all_dish_centroids)} dish detections")
        if len(all_dish_centroids) == 0:
            print("    ERROR: no dish detections! Falling back to scale 0.2 mm/px")
            # Return 3 values (consistent with the normal return) to avoid an unpacking error.
            # avg_diameter_px derived from the default scale: 98mm / 0.2 = 490 px.
            return 0.2, np.array([]), DISH_DIAMETER_MM / 0.2

    all_dish_centroids = np.array(all_dish_centroids)
    positions = all_dish_centroids[:, :2]
    Z = linkage(positions, method="ward")
    n_clusters = min(6, len(all_dish_centroids))
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")

    dish_centers = []
    dish_diameters = []
    for cluster_id in range(1, n_clusters + 1):
        mask = labels == cluster_id
        cluster_pts = all_dish_centroids[mask]
        center_x = np.median(cluster_pts[:, 0])
        center_y = np.median(cluster_pts[:, 1])
        avg_w = np.median(cluster_pts[:, 2])
        avg_h = np.median(cluster_pts[:, 3])
        dish_centers.append((center_x, center_y))
        dish_diameters.append(max(avg_w, avg_h))

    dish_centers = np.array(dish_centers)
    dish_diameters = np.array(dish_diameters)

    y_median = np.median(dish_centers[:, 1])
    top_row_idx = np.where(dish_centers[:, 1] < y_median)[0]
    bot_row_idx = np.where(dish_centers[:, 1] >= y_median)[0]

    top_sorted = top_row_idx[np.argsort(dish_centers[top_row_idx, 0])]
    bot_sorted = bot_row_idx[np.argsort(dish_centers[bot_row_idx, 0])]

    dish_order = []
    for t, b in zip(top_sorted, bot_sorted):
        dish_order.append(t)
        dish_order.append(b)
    remaining_top = [t for t in top_sorted if t not in dish_order]
    remaining_bot = [b for b in bot_sorted if b not in dish_order]
    dish_order.extend(remaining_top)
    dish_order.extend(remaining_bot)

    ordered_centers = dish_centers[dish_order]
    ordered_diameters = dish_diameters[dish_order]

    avg_diameter_px = np.mean(ordered_diameters)
    scale_mm_per_px = DISH_DIAMETER_MM / avg_diameter_px
    print(f"    Calibration: {avg_diameter_px:.1f} px -> {scale_mm_per_px:.4f} mm/px")

    return scale_mm_per_px, ordered_centers, float(avg_diameter_px)


# ============================================================================
# TRACKING + DETECTION COLLECTION
# ============================================================================

def track_video(
    model: YOLO,
    video_path: str,
    conf: float = 0.4,
    iou: float = 0.7,
    tracker: str = "bytetrack.yaml",
    larva_class_name: str = "larva",
) -> pd.DataFrame:
    """
    Run YOLO with a tracker over the full recording.
    Returns a DataFrame: frame, track_id, cx, cy, confidence, w, h, mask_cx, mask_cy.
    """
    print(f"    Tracking: {video_path}")

    device = get_device()
    print(f"    Device: {device}", flush=True)
    results = model.track(
        source=video_path,
        imgsz=640, conf=conf, iou=iou,
        tracker=tracker,
        stream=True, verbose=False,
        persist=True,
        device=device,
    )

    all_detections = []
    frame_id = -1
    for frame_id, r in enumerate(results):
        if frame_id % 5000 == 0:
            n_det = len(all_detections)
            print(f"      Frame {frame_id}... ({n_det} detections)", flush=True)

        if r.boxes is None or r.boxes.id is None:
            continue

        classes = r.boxes.cls.cpu().numpy()
        ids = r.boxes.id.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        boxes = r.boxes.xyxy.cpu().numpy()

        has_masks = r.masks is not None and len(r.masks) > 0
        masks_xy = r.masks.xy if has_masks else None

        for j, (cls_id, tid) in enumerate(zip(classes, ids)):
            if model.names[int(cls_id)] != larva_class_name:
                continue

            box = boxes[j]
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            w = box[2] - box[0]
            h = box[3] - box[1]

            mask_cx, mask_cy = float(cx), float(cy)
            if has_masks and j < len(masks_xy):
                polygon = masks_xy[j]
                if len(polygon) >= 3:
                    polygon_int = polygon.astype(np.int32)
                    M = cv2.moments(polygon_int)
                    if M["m00"] > 0:
                        mask_cx = float(M["m10"] / M["m00"])
                        mask_cy = float(M["m01"] / M["m00"])

            all_detections.append({
                "frame": frame_id,
                "track_id": tid,
                "confidence": float(confs[j]),
                "cx": float(cx),
                "cy": float(cy),
                "w": float(w),
                "h": float(h),
                "mask_cx": mask_cx,
                "mask_cy": mask_cy,
            })

    df = pd.DataFrame(all_detections)
    n_tracks = df["track_id"].nunique() if not df.empty else 0
    print(f"    Tracking finished: {len(df)} detections, {n_tracks} tracks, {frame_id+1} frames")
    return df


# ============================================================================
# TRACK -> DISH ASSIGNMENT
# ============================================================================

def assign_tracks_to_dishes(
    tracks_df: pd.DataFrame,
    ordered_centers: np.ndarray,
    n_dishes: int = 6,
) -> Dict[int, int]:
    """
    Assign each track_id to a dish based on the median position.
    Returns a dict: {track_id: dish_id (1-indexed)}.
    """
    track_to_dish = {}
    for tid in tracks_df["track_id"].unique():
        track_data = tracks_df[tracks_df["track_id"] == tid]
        median_cx = track_data["cx"].median()
        median_cy = track_data["cy"].median()

        distances = np.sqrt(
            (ordered_centers[:, 0] - median_cx) ** 2 +
            (ordered_centers[:, 1] - median_cy) ** 2
        )
        nearest_dish = np.argmin(distances) + 1
        track_to_dish[tid] = nearest_dish

    return track_to_dish


def merge_tracks_per_dish(
    tracks_df: pd.DataFrame,
    track_to_dish: Dict[int, int],
    ordered_centers: np.ndarray,
    avg_dish_diameter_px: float,
    n_dishes: int = 6,
    spatial_margin: float = 1.1,
) -> Dict[int, pd.DataFrame]:
    """
    Merge all tracks assigned to a given dish into a single DataFrame.
    Filters out detections outside the dish boundary (spatial containment).
    When several tracks have a detection in the same frame, the one with the
    higher confidence is kept.
    Returns a dict: {dish_id: merged_dataframe}.
    """
    half_dish_px = avg_dish_diameter_px / 2 * spatial_margin

    dish_tracks = {}
    for dish_id in range(1, n_dishes + 1):
        tids = [tid for tid, did in track_to_dish.items() if did == dish_id]
        if not tids:
            print(f"      Dish {dish_id} -> NO TRACKS!")
            continue

        dcx, dcy = ordered_centers[dish_id - 1]
        dish_data = tracks_df[tracks_df["track_id"].isin(tids)].copy()
        total_before = len(dish_data)

        dist_from_center = np.sqrt(
            (dish_data["cx"] - dcx) ** 2 + (dish_data["cy"] - dcy) ** 2
        )
        dish_data = dish_data[dist_from_center < half_dish_px].copy()
        dropped = total_before - len(dish_data)

        merged = dish_data.loc[
            dish_data.groupby("frame")["confidence"].idxmax()
        ].sort_values("frame").reset_index(drop=True)

        n_tids = len(tids)
        print(f"      Dish {dish_id} -> {n_tids} tracks, "
              f"{len(merged)} frames (dropped {dropped} outside dish)")

        dish_tracks[dish_id] = merged

    return dish_tracks


# ============================================================================
# TRAJECTORY CONSTRUCTION
# ============================================================================

def build_trajectory_from_track(
    track_data: pd.DataFrame,
    scale: float,
    total_frames: int,
    dish_center_px: np.ndarray,
    skip_start: int = SKIP_START_FRAMES,
) -> pd.DataFrame:
    """
    Build a trajectory from a single track's data.
    Normalizes to a dish-centric coordinate system.
    """
    track_data = track_data[track_data["frame"] >= skip_start].copy()
    track_data = track_data.sort_values("frame").reset_index(drop=True)

    frames = np.arange(total_frames)
    x_mm = np.full(total_frames, np.nan)
    y_mm = np.full(total_frames, np.nan)

    for _, row in track_data.iterrows():
        f = int(row["frame"])
        if f < total_frames:
            x_mm[f] = (row["cx"] - dish_center_px[0]) * scale
            y_mm[f] = (row["cy"] - dish_center_px[1]) * scale

    valid_mask = ~np.isnan(x_mm)
    if valid_mask.sum() > 1:
        x_mm = np.interp(frames, frames[valid_mask], x_mm[valid_mask])
        y_mm = np.interp(frames, frames[valid_mask], y_mm[valid_mask])

    detected = valid_mask.astype(int)

    return pd.DataFrame({
        "frame": frames,
        "x_mm": x_mm,
        "y_mm": y_mm,
        "detected": detected,
    })


# ============================================================================
# KINEMATIC METRICS
# ============================================================================

def compute_kinematics(
    traj: pd.DataFrame,
    scale: float,
    max_threshold: float = MAX_MOVEMENT_THRESHOLD_MM,
) -> dict:
    """Compute kinematic metrics (consistent with the OpenCV pipeline: MAX + artifact proximity)."""
    x = traj["x_mm"].values
    y = traj["y_mm"].values

    dx = np.diff(x)
    dy = np.diff(y)
    distances = np.sqrt(dx ** 2 + dy ** 2)

    # Filtering identical to OpenCV: artifact proximity (+/-5 frames around jumps)
    artifact_mask = distances > max_threshold
    expanded_mask = np.zeros(len(distances), dtype=bool)
    for idx in np.where(artifact_mask)[0]:
        lo = max(0, idx - ARTIFACT_PROXIMITY_FRAMES)
        hi = min(len(distances), idx + ARTIFACT_PROXIMITY_FRAMES + 1)
        expanded_mask[lo:hi] = True

    valid_distances = distances.copy()
    valid_distances[expanded_mask] = 0.0

    total_distance = np.sum(valid_distances)
    duration_s = len(x) / FPS
    avg_speed = total_distance / duration_s if duration_s > 0 else 0.0

    instant_speeds = valid_distances * FPS

    detection_rate = traj["detected"].sum() / len(traj) * 100

    # Angles - MIN=0.5mm filter (only for rose diagrams, as in OpenCV)
    angles = np.degrees(np.arctan2(-dx, dy)) % 360
    angle_mask = (
        (distances >= MIN_ANGLE_THRESHOLD_MM)
        & (distances <= max_threshold)
        & (~expanded_mask)
    )
    filtered_angles = angles[angle_mask]
    filtered_distances = distances[angle_mask]

    speed_timeseries = {}
    frames_per_5min = 5 * 60 * FPS
    n_intervals = int(np.ceil(len(instant_speeds) / frames_per_5min))
    for i in range(n_intervals):
        start = i * frames_per_5min
        end = min((i + 1) * frames_per_5min, len(instant_speeds))
        interval_speeds = instant_speeds[start:end]
        time_label = f"{i * 5}-{(i + 1) * 5}"
        speed_timeseries[time_label] = float(np.mean(interval_speeds))

    is_valid = detection_rate >= MIN_DETECTION_RATE
    pos_std = np.sqrt(np.std(x) ** 2 + np.std(y) ** 2)
    is_stationary = pos_std < 2.0

    return {
        "total_distance_mm": float(total_distance),
        "avg_speed_mm_s": float(avg_speed),
        "detection_rate": float(detection_rate),
        "is_valid": is_valid,
        "is_stationary": is_stationary,
        "position_std_mm": float(pos_std),
        "n_frames": int(len(x)),
        "instant_speeds": instant_speeds,
        "angles": angles,
        "filtered_angles": filtered_angles,
        "filtered_distances": filtered_distances,
        "n_filtered_movements": int(len(filtered_angles)),
        "speed_timeseries": speed_timeseries,
        "x_mm": x,
        "y_mm": y,
    }


# ============================================================================
# PLOTS
# ============================================================================

def plot_all(
    metrics: Dict[int, dict],
    output_path: Path,
    title_suffix: str = "",
):
    """Generate the full set of plots (distance, speed, trajectories, heatmaps, rose)."""

    dish_ids = sorted(metrics.keys())

    # Distance
    fig, ax = plt.subplots(figsize=(10, 6))
    distances = [metrics[d]["total_distance_mm"] for d in dish_ids]
    colors = [LARVA_COLORS[i % len(LARVA_COLORS)] for i in range(len(dish_ids))]
    labels = [f"L{d}" for d in dish_ids]
    bars = ax.bar(labels, distances, color=colors, edgecolor="black", linewidth=0.5)
    for bar, dist, d in zip(bars, distances, dish_ids):
        v = "v" if metrics[d]["is_valid"] else "x"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(distances) * 0.02,
                f"{dist:.0f}\n{v}", ha="center", va="bottom", fontsize=8)
    avg_dist = np.mean(distances)
    ax.axhline(y=avg_dist, color="red", linestyle="--", linewidth=1, label=f"Mean: {avg_dist:.0f} mm")
    ax.set_ylabel("Distance [mm]")
    ax.set_title(f"Distance - YOLO+tracker {title_suffix}", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "distances_plot_tracker.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Trajectories
    n_dishes = len(dish_ids)
    ncols = min(3, n_dishes)
    nrows = (n_dishes + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
    if n_dishes == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten()
    half = DISH_DIAMETER_MM / 2
    for i, dish_id in enumerate(dish_ids):
        ax = axes_flat[i]
        m = metrics[dish_id]
        x, y = m["x_mm"], m["y_mm"]
        ax.plot(x, y, color=LARVA_COLORS[i % len(LARVA_COLORS)], linewidth=0.3, alpha=0.7)
        ax.plot(x[0], y[0], "go", markersize=6, label="Start")
        ax.plot(x[-1], y[-1], "ro", markersize=6, label="End")
        v = "valid" if m["is_valid"] else f"inv ({m['detection_rate']:.0f}%)"
        ax.set_title(f"L{dish_id} ({m['total_distance_mm']:.0f} mm) [{v}]", fontsize=10)
        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Y [mm]")
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xlim(-half - 5, half + 5)
        ax.set_ylim(half + 5, -half - 5)
        ax.axhline(0, color="gray", linewidth=0.5, alpha=0.3)
        ax.axvline(0, color="gray", linewidth=0.5, alpha=0.3)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
    for i in range(len(dish_ids), len(axes_flat)):
        axes_flat[i].set_visible(False)
    fig.suptitle(f"Trajectories - YOLO+tracker {title_suffix}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path / "trajectories_combined_tracker.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Heatmaps
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
    if n_dishes == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten()
    for i, dish_id in enumerate(dish_ids):
        ax = axes_flat[i]
        m = metrics[dish_id]
        x, y = m["x_mm"], m["y_mm"]
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)
        heatmap = heatmap.T
        coverage = np.count_nonzero(heatmap) / heatmap.size * 100
        im = ax.imshow(heatmap, origin="lower", cmap="hot",
                       extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect="auto")
        ax.set_title(f"L{dish_id} (coverage {coverage:.1f}%)", fontsize=10)
        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Y [mm]")
        plt.colorbar(im, ax=ax, shrink=0.8)
    for i in range(len(dish_ids), len(axes_flat)):
        axes_flat[i].set_visible(False)
    fig.suptitle(f"Heatmaps - YOLO+tracker {title_suffix}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path / "combined_heatmaps_tracker.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Rose diagrams
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows),
                             subplot_kw={"projection": "polar"})
    if n_dishes == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten()
    for i, dish_id in enumerate(dish_ids):
        ax = axes_flat[i]
        m = metrics[dish_id]
        angles_deg = m["filtered_angles"]
        n_obs = len(angles_deg)
        if n_obs == 0:
            ax.set_title(f"L{dish_id}\n(no data)", fontsize=10)
            continue
        n_bins = 24
        bins = np.linspace(0, 360, n_bins + 1)
        counts, _ = np.histogram(angles_deg, bins=bins)
        theta = np.deg2rad(bins[:-1] + 7.5)
        width = np.deg2rad(15)
        ax.bar(theta, counts, width=width, color=LARVA_COLORS[i % len(LARVA_COLORS)],
               edgecolor="black", linewidth=0.5, alpha=0.8)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_title(f"L{dish_id} (n={n_obs})", fontsize=10, pad=15)
    for i in range(len(dish_ids), len(axes_flat)):
        axes_flat[i].set_visible(False)
    fig.suptitle(f"Rose Diagrams - YOLO+tracker {title_suffix}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path / "rose_diagrams_tracker.png", dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def process_video(
    video_path: str,
    model_path: str,
    output_dir: str,
    session: str,
    experiment: str,
    conf: float = 0.4,
    tracker: str = "bytetrack.yaml",
    dish_model_path: Optional[str] = None,
):
    """Full pipeline: tracking -> assignment -> kinematics -> plots."""
    video_path_obj = Path(video_path)
    output_path = Path(output_dir) / session / experiment
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n  {'='*50}")
    print(f"  {experiment} ({session}): {video_path_obj.name}")
    print(f"  {'='*50}")

    model = YOLO(model_path)

    # Resolve class names
    larva_class = "larva"
    dish_class = "dish"
    for cls_id, name in model.names.items():
        if "dish" in name.lower() or "petri" in name.lower():
            dish_class = name

    # Dish calibration - any model with a 'dish' class can be used
    dm = dish_model_path or model_path
    dish_model = YOLO(dm) if dm != model_path else model
    print(f"  Dish calibration (model: {Path(dm).name})...")
    scale, ordered_centers, avg_diam_px = calibrate_dishes(dish_model, video_path, dish_class_name="dish")

    if len(ordered_centers) < 6:
        print("    ERROR: could not calibrate 6 dishes!")
        return None

    # Tracking (or loading from cache)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"    Frames: {total_frames}, FPS: {FPS}")

    det_csv = output_path / "raw_tracked_detections.csv"
    if det_csv.exists():
        print(f"    Loading cache: {det_csv}")
        tracks_df = pd.read_csv(det_csv)
        n_tracks = tracks_df["track_id"].nunique() if not tracks_df.empty else 0
        print(f"    Cache: {len(tracks_df)} detections, {n_tracks} tracks")
    else:
        tracks_df = track_video(model, video_path, conf=conf, tracker=tracker, larva_class_name=larva_class)
        tracks_df.to_csv(det_csv, index=False)
        print(f"    Raw detections saved: {det_csv}")

    if tracks_df.empty:
        print("    ERROR: no detections!")
        return None

    # Track->dish assignment and fragment merging
    print(f"    Assigning tracks to dishes (merging fragments):")
    track_to_dish = assign_tracks_to_dishes(tracks_df, ordered_centers)
    dish_merged = merge_tracks_per_dish(
        tracks_df, track_to_dish, ordered_centers, avg_diam_px,
    )

    # Build per-dish trajectories
    print(f"    Building trajectories (dish-centric, skip_start={SKIP_START_FRAMES/FPS:.0f}s):")
    all_metrics = {}
    dish_to_track = {}
    for dish_id in sorted(dish_merged.keys()):
        merged_data = dish_merged[dish_id]
        dish_center = ordered_centers[dish_id - 1]

        traj = build_trajectory_from_track(merged_data, scale, total_frames, dish_center)

        m = compute_kinematics(traj, scale)
        all_metrics[dish_id] = m

        n_tracks = merged_data["track_id"].nunique()
        main_track = int(merged_data["track_id"].mode().iloc[0])
        dish_to_track[dish_id] = main_track

        det_pct = m["detection_rate"]
        v = "valid" if m["is_valid"] else "INVALID"
        s = "stationary" if m["is_stationary"] else "active"
        print(f"      L{dish_id} ({n_tracks} tracks): {m['total_distance_mm']:.0f} mm, "
              f"{m['avg_speed_mm_s']:.2f} mm/s, det={det_pct:.1f}% [{v}] [{s}]")

        traj.to_csv(output_path / f"larva_positions_{dish_id}_tracker.csv", index=False)

    # Plots
    title_suffix = f"({experiment}, {session})"
    plot_all(all_metrics, output_path, title_suffix)

    # Summary CSV
    rows = []
    for dish_id in sorted(all_metrics.keys()):
        m = all_metrics[dish_id]
        rows.append({
            "session": session,
            "experiment": experiment,
            "larva_id": dish_id,
            "track_id": dish_to_track[dish_id],
            "total_distance_mm": round(m["total_distance_mm"], 2),
            "avg_speed_mm_s": round(m["avg_speed_mm_s"], 4),
            "detection_rate_pct": round(m["detection_rate"], 2),
            "is_valid": m["is_valid"],
            "is_stationary": m["is_stationary"],
            "position_std_mm": round(m["position_std_mm"], 2),
            "n_frames": m["n_frames"],
        })

    summary_df = pd.DataFrame(rows)
    summary_path = output_path / "summary_tracker.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n    Summary: {summary_path}")
    print(summary_df[["larva_id", "track_id", "total_distance_mm", "avg_speed_mm_s",
                       "detection_rate_pct", "is_valid", "is_stationary"]].to_string(index=False))

    # Metadata
    meta = {
        "model": model_path,
        "dish_model": dm,
        "tracker": tracker,
        "conf": float(conf),
        "scale_mm_per_px": float(round(scale, 4)),
        "total_frames": int(total_frames),
        "dish_to_track": {str(k): int(v) for k, v in dish_to_track.items()},
        "n_unique_tracks": int(tracks_df["track_id"].nunique()),
    }
    with open(output_path / "tracking_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    return summary_df


def main():
    parser = argparse.ArgumentParser(
        description="YOLO pipeline with the ByteTrack tracker"
    )
    parser.add_argument("--model", type=str,
                        default="models/yolo_baseline_combined_best.pt",
                        help="YOLO model for larva detection")
    parser.add_argument("--dish-model", type=str, default=None,
                        help="YOLO model for dish detection (default: same as --model)")
    parser.add_argument("--output", type=str, default="results/yolo_tracker",
                        help="Output folder")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="YOLO confidence threshold")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml",
                        help="Tracker config (bytetrack.yaml or botsort.yaml)")
    parser.add_argument("--all", action="store_true",
                        help="Process all recordings from both sessions")
    parser.add_argument("--video", type=str, default=None,
                        help="Single recording")
    parser.add_argument("--session", type=str, default="s1")
    parser.add_argument("--experiment", type=str, default="control")
    parser.add_argument("--videos-dir", type=str, default=None,
                        help="Directory with recordings (contains s1/ and s2/). "
                             "When given, overrides auto-detection of project_root/data/raw_videos.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[3]
    output_dir = project_root / args.output
    # Recordings directory: explicit --videos-dir, or default project_root/data/raw_videos.
    videos_base = Path(args.videos_dir) if args.videos_dir else (project_root / "data" / "raw_videos")

    print(f"{'=' * 70}")
    print(f"YOLO + TRACKER PIPELINE")
    print(f"  Model:        {args.model}")
    print(f"  Dish model:   {args.dish_model or '(same model)'}")
    print(f"  Tracker:      {args.tracker}")
    print(f"  Confidence:   {args.conf}")
    print(f"  Output:       {output_dir}")
    print(f"{'=' * 70}")

    all_summaries = []

    if args.all:
        for session_key, vmap in [("s1", S1_VIDEO_MAP), ("s2", S2_VIDEO_MAP)]:
            vdir = videos_base / session_key
            if not vdir.exists():
                print(f"\nSkipping session {session_key}: missing {vdir}")
                continue

            video_files = sorted(vdir.glob("*.mp4")) + sorted(vdir.glob("*.mov"))
            for vf in video_files:
                stem = vf.stem.lower().replace(" ", "_")
                experiment = None
                for key, exp_name in vmap.items():
                    if key.lower() in stem:
                        experiment = exp_name
                        break
                if experiment is None:
                    continue

                summary = process_video(
                    str(vf), args.model, str(output_dir),
                    session_key, experiment, args.conf, args.tracker,
                    args.dish_model,
                )
                if summary is not None:
                    all_summaries.append(summary)

    elif args.video:
        summary = process_video(
            args.video, args.model, str(output_dir),
            args.session, args.experiment, args.conf, args.tracker,
            args.dish_model,
        )
        if summary is not None:
            all_summaries.append(summary)
    else:
        print("Usage:")
        print(f"  python {__file__} --video data/raw_videos/s1/control.mp4 --session s1 --experiment control")
        print(f"  python {__file__} --all")
        return

    if all_summaries:
        combined = pd.concat(all_summaries, ignore_index=True)
        combined_path = output_dir / "combined_summary_tracker.csv"
        combined.to_csv(combined_path, index=False)
        print(f"\n{'=' * 70}")
        print(f"COMBINED SUMMARY: {combined_path}")
        print(f"{'=' * 70}")

        for session in SESSIONS:
            session_data = combined[combined["session"] == session]
            if session_data.empty:
                continue
            print(f"\n--- Session {session.upper()} ---")
            group_means = session_data.groupby("experiment").agg({
                "total_distance_mm": "mean",
                "avg_speed_mm_s": "mean",
                "detection_rate_pct": "mean",
            }).round(2)
            print(group_means.to_string())


if __name__ == "__main__":
    main()
