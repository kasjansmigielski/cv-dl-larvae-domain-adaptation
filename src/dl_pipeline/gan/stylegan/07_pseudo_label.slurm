#!/bin/bash
# ============================================================================
# STEP 4.2 - Pseudo-label synthetics with a YOLO-seg model (SLURM, batch on GPU)
# ============================================================================
# NOTE: example SLURM script. Adjust PROJECT_HOME, the SLURM account,
# partition/GRES, the conda env and the email to your own cluster.
#
# Submit (optional arg: path to YOLO weights):
#   sbatch 07_pseudo_label.slurm
#   sbatch 07_pseudo_label.slurm <PROJECT_HOME>/models/other_model.pt
# Logs: <PROJECT_HOME>/logs/pseudo_label_<jobid>.out
# ----------------------------------------------------------------------------
# The logs/ folder must exist: mkdir -p <PROJECT_HOME>/logs

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<your-email>
#SBATCH --job-name=pseudo_label
#SBATCH --partition=<gpu-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/%u/project/logs/pseudo_label_%j.out
#SBATCH --error=/home/%u/project/logs/pseudo_label_%j.err
#SBATCH --account=<your-slurm-account>

set -eo pipefail

PROJECT_HOME="/home/${USER}/project"
SCRIPT="${PROJECT_HOME}/src/dl_pipeline/gan/stylegan/07_pseudo_label.py"
IMAGES="${PROJECT_HOME}/data/stylegan_synth/combined"
# YOLO weights: arg $1 or the default path (adjust if different):
YOLO="${1:-${PROJECT_HOME}/runs/yolo_baseline_combined/weights/best.pt}"
OUT="${PROJECT_HOME}/data/stylegan_synth/combined_labeled"
CONF="0.7"

command -v module >/dev/null 2>&1 && module load CUDA/11.8 || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate larvixon
export CUDA_VISIBLE_DEVICES=0

echo "=========================================="
echo "Job:       ${SLURM_JOB_ID:-local}"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader || echo none)"
echo "Images:    ${IMAGES}"
echo "YOLO:      ${YOLO}"
echo "Out:       ${OUT}   (conf=${CONF})"
echo "=========================================="

if [ ! -f "${YOLO}" ]; then
    echo "ERROR: YOLO weights ${YOLO} not found." >&2
    exit 1
fi

python -u "${SCRIPT}" \
    --images "${IMAGES}" \
    --yolo   "${YOLO}" \
    --out    "${OUT}" \
    --conf "${CONF}" --class-name larva

echo "=========================================="
echo "DONE. Pseudo-labeled: ${OUT}"
echo "  Inspect: ${OUT}/qc_preview.png + ${OUT}/summary.json"
echo "  Next step: 08_build_augmented_dataset.py"
echo "=========================================="
