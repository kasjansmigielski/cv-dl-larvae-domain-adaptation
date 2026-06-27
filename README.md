# CV–DL Larvae Domain Adaptation

Code accompanying the publication on automated tracking and behavioral analysis of
*Galleria mellonella* larvae, combining classical computer vision, deep learning
(YOLOv8 instance segmentation + ByteTrack), and generative domain adaptation
(CycleGAN, StyleGAN2-ADA) to bridge the gap between imaging sessions.

## Repository structure

```
.
├── data/                       # Raw video download scripts (S3)
├── notebooks/
│   └── cv_pipeline/            # Classical CV pipeline (OpenCV) as notebooks
│       ├── 01_area_selection.ipynb
│       ├── 02_dish_isolation.ipynb
│       ├── 03_larva_tracker.ipynb
│       ├── 04_kinematics.ipynb
│       ├── 05_plot_generator.ipynb
│       └── 06_angle_processor.ipynb
├── src/
│   └── dl_pipeline/
│       ├── yolo/               # YOLOv8-seg training, tracking, evaluation
│       └── gan/
│           ├── cyclegan/       # Unpaired domain translation
│           ├── stylegan/       # Synthetic data + pseudo-labeling
│           └── eval/           # A/B/C augmentation evaluation (SLURM)
├── results/                    # (git-ignored) generated outputs
├── environment.yml             # Conda environment (recommended)
├── requirements.txt            # pip alternative
└── .env.example                # Template for S3 credentials
```

## Installation

### Option A — Conda (recommended)

```bash
conda env create -f environment.yml
conda activate larvixon
```

### Option B — pip / venv

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> For GPU-accelerated PyTorch, install the matching build, e.g.:
> `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`

## Configuration

The data-download scripts read S3 credentials from environment variables:

```bash
cp .env.example .env
# then edit .env and fill in your S3 endpoint, keys, bucket and prefix
```

## Pipelines

1. **Classical CV** (`notebooks/cv_pipeline/`) — dish isolation, ROI/scale
   calibration, larva tracking, kinematics and behavioral plots. Run the
   notebooks in numerical order.
2. **YOLO** (`src/dl_pipeline/yolo/`) — preprocess dishes, train a YOLOv8s-seg
   baseline, track with ByteTrack and analyze, evaluate across sessions.
3. **GAN** (`src/dl_pipeline/gan/`) — CycleGAN for illumination domain adaptation
   and StyleGAN2-ADA for synthetic augmentation, with A/B/C evaluation. Training
   scripts are provided as SLURM jobs for HPC clusters.

## Citation

If you use this code, please cite the accompanying publication (see `CITATION`/paper).

## License

Released under the MIT License — see `LICENSE`.
