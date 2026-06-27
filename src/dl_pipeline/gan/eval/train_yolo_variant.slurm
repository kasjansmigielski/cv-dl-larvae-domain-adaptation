#!/bin/bash
# ============================================================================
# STEP 4.4 - Train YOLO-seg for a StyleGAN augmentation variant (SLURM)
# ============================================================================
# NOTE: example SLURM script. Adjust PROJECT_HOME, the SLURM account,
# partition/GRES, the conda env and the email to your own cluster.
#
# Submit (variant as argument: A_real_only | B_augmented | C_synth_only):
#   sbatch train_yolo_variant.slurm A_real_only
#   sbatch train_yolo_variant.slurm B_augmented
#   sbatch train_yolo_variant.slurm C_synth_only
# Logs: <PROJECT_HOME>/logs/yolo_variant_<jobid>.out
# ----------------------------------------------------------------------------
# The logs/ folder must exist: mkdir -p <PROJECT_HOME>/logs

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<your-email>
#SBATCH --job-name=yolo_variant
#SBATCH --partition=<gpu-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/project/logs/yolo_variant_%j.out
#SBATCH --error=/home/%u/project/logs/yolo_variant_%j.err
#SBATCH --account=<your-slurm-account>

set -euo pipefail

VARIANT="${1:?Provide variant: A_real_only | B_augmented | C_synth_only}"

PROJECT_HOME="/home/${USER}/project"
DATA_YAML="${PROJECT_HOME}/data/stylegan_variants/${VARIANT}/data.yaml"
RUNS_DIR="${PROJECT_HOME}/runs"
RUN_NAME="stylegan_${VARIANT}"

command -v module >/dev/null 2>&1 && module load CUDA/11.8 || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate larvixon

echo "=========================================="
echo "Job:       ${SLURM_JOB_ID:-local}"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader || echo none)"
echo "Variant:   ${VARIANT}"
echo "data.yaml: ${DATA_YAML}"
echo "=========================================="

if [ ! -f "${DATA_YAML}" ]; then
    echo "ERROR: missing ${DATA_YAML} - run 08_build_augmented_dataset.py first"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=0

# Training identical to the baseline (train_baseline.py): yolov8s-seg, 100 epochs, imgsz 640.
yolo segment train \
    model=yolov8s-seg.pt \
    data="${DATA_YAML}" \
    epochs=100 \
    imgsz=640 \
    batch=8 \
    patience=20 \
    device=0 \
    plots=True \
    project="${RUNS_DIR}" \
    name="${RUN_NAME}"

echo "=========================================="
echo "Training finished. Weights: ${RUNS_DIR}/${RUN_NAME}/weights/best.pt"
echo ""
echo "Evaluate on the real S1 test set:"
echo "  yolo segment val model=${RUNS_DIR}/${RUN_NAME}/weights/best.pt \\"
echo "      data=${DATA_YAML} split=test name=eval_${VARIANT}"
echo "=========================================="
