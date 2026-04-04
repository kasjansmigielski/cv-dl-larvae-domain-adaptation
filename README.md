# Classical CV vs. Deep Learning for Robust Invertebrate Larval Tracking Under Varying Illumination Conditions

> **Status:** WIP — code and results will be added as experiments are completed.

## Overview

This repository contains the source code and experimental pipeline accompanying the manuscript submitted to *Computers in Biology and Medicine*.

We present a comparative study of classical computer vision (CV) and deep learning (DL) approaches for automated tracking of invertebrate larvae (*Zophobas morio*) across sessions with different illumination conditions. The key contributions are:

- **End-to-end classical CV pipeline** — adaptive preprocessing (CLAHE), background subtraction, contour-based detection
- **Deep learning baseline** — YOLOv8-based detection with within-session and cross-session evaluation
- **Generative data augmentation** — GAN-synthesized training data to improve detector robustness
- **Domain adaptation** — CycleGAN-based illumination transfer (Session I - Session II) to bridge the domain gap without manual re-annotation
- **Behavioral analysis** — trajectory-derived metrics (velocity, distance, movement patterns) demonstrating biological interpretability

## Repository Structure

```
├── configs/              # Experiment configurations (CV params, YOLO config)
├── data/                 # Data files and download instructions 
├── notebooks/            # Jupyter notebooks for exploration and visualization
├── results/              # Generated tables (CSV) and figures (PNG)
├── src/
│   ├── cv_pipeline/      # Classical CV: detection, tracking, preprocessing
│   ├── dl_pipeline/      # Deep learning: YOLO, GAN, CycleGAN
│   ├── analysis/         # Behavioral metrics, heatmaps, statistical tests
│   └── visualization/    # Figure and table generation for the manuscript
├── environment.yml       # Conda environment specification
└── requirements.txt      # Python dependencies (pip)
```

## Requirements

- Python ≥ 3.11
- Key dependencies: OpenCV, NumPy, SciPy, scikit-image, matplotlib, pandas
- DL experiments additionally require: PyTorch, ultralytics (YOLOv8)

## Installation

```bash
# Option A: conda (recommended)
git clone https://github.com/<user>/cv-dl-larvae-domain-adaptation.git
cd cv-dl-larvae-domain-adaptation
conda env create -f environment.yml
conda activate cv-dl-larvae

# Option B: pip
git clone https://github.com/<user>/cv-dl-larvae-domain-adaptation.git
cd cv-dl-larvae-domain-adaptation
pip install -r requirements.txt
```

## Data

Video recordings of *Galleria mellonella* larvae were captured under two illumination conditions:


| Session    | Illumination | Frames  | Resolution |
| ---------- | ------------ | ------- | ---------- |
| Session I  | Top-down     | ~36 000 | 1920×1080  |
| Session II | Bottom-up    | ~36 000 | 1920×1080  |


Due to file size constraints, raw video data is not included in this repository.

## Results

Key findings (classical CV pipeline):


| Metric                        | Session I            | Session II |
| ----------------------------- | -------------------- | ---------- |
| Detection rate                | 99.05%               | 99.71%     |
| Missing frame rate            | 0.95%                | 0.29%      |
| Cross-session relative change | −0.66% (improvement) | —          |


*DL comparison results and domain adaptation experiments will be added upon completion.*



