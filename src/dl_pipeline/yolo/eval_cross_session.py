#!/usr/bin/env python3
"""Cross-session evaluation of YOLOv8s-seg models (domain-shift quantification).

Evaluates each session-specific model on the *other* session's data:
    - S1-trained model  ->  S2 data
    - S2-trained model  ->  S1 data

This quantifies the cross-session domain shift reported in the manuscript
(severe larva-mask mAP drop under unseen illumination).

Usage:
    python src/dl_pipeline/yolo/eval_cross_session.py \
        --s1-model runs/yolo_s1_only/weights/best.pt \
        --s2-model runs/yolo_s2_only/weights/best.pt \
        --s1-data  data/session1_only/data.yaml \
        --s2-data  data/session2_only/data.yaml \
        --project  runs/eval
"""

import argparse

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Cross-session YOLO evaluation.")
    p.add_argument("--s1-model", required=True, help="Weights of the S1-trained model.")
    p.add_argument("--s2-model", required=True, help="Weights of the S2-trained model.")
    p.add_argument("--s1-data", required=True, help="S1 data.yaml.")
    p.add_argument("--s2-data", required=True, help="S2 data.yaml.")
    p.add_argument("--project", default="runs/eval", help="Output project directory.")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=0, help="CUDA device index or 'cpu'.")
    return p.parse_args()


def main():
    args = parse_args()

    # S1 model -> S2 data
    model_s1 = YOLO(args.s1_model)
    model_s1.val(
        data=args.s2_data,
        imgsz=args.imgsz, batch=args.batch, device=args.device,
        project=args.project, name="s1_model_on_s2_data",
    )

    # S2 model -> S1 data
    model_s2 = YOLO(args.s2_model)
    model_s2.val(
        data=args.s1_data,
        imgsz=args.imgsz, batch=args.batch, device=args.device,
        project=args.project, name="s2_model_on_s1_data",
    )


if __name__ == "__main__":
    main()
