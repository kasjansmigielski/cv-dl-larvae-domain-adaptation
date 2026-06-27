#!/usr/bin/env python3
"""Train a YOLOv8s-seg baseline model.

Trains an instance-segmentation model (larva + dish classes) from the
pretrained COCO weights. Used for the within-session and combined baselines
(S1-only, S2-only, S1+S2) described in the manuscript.

Usage:
    python src/dl_pipeline/yolo/train_baseline.py \
        --data data/session1_only/data.yaml \
        --name yolo_s1_only \
        --project runs \
        --epochs 100 --imgsz 640 --batch 8
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Train a YOLOv8s-seg baseline.")
    p.add_argument("--data", required=True, help="Path to the YOLO data.yaml file.")
    p.add_argument("--name", required=True, help="Run name (output subfolder).")
    p.add_argument("--project", default="runs", help="Project directory for outputs.")
    p.add_argument("--weights", default="yolov8s-seg.pt", help="Initial weights.")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--device", default=0, help="CUDA device index or 'cpu'.")
    return p.parse_args()


def main():
    args = parse_args()
    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    model = YOLO(args.weights)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        save=True,
        plots=True,
        project=args.project,
        name=args.name,
    )
    print(f"Training finished. Results: {results.save_dir}")


if __name__ == "__main__":
    main()
