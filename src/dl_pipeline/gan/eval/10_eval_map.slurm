#!/bin/bash
# ============================================================================
# STEP 4.5 (auxiliary) - Evaluate mAP (yolo val) for variants A/B/C
# ============================================================================
# NOTE: example SLURM script. Adjust PROJECT_HOME, the SLURM account,
# partition/GRES, the conda env and the email to your own cluster.
#
# AUXILIARY metric (main = detection rate from 09_eval_detection_rate.slurm).
# mAP on a static real combined test set is near its ceiling (~95%) - reported
# for table completeness. All 3 variants are evaluated in ONE job.
#
# Submit (no arguments - runs A + B + C in sequence):
#   sbatch 10_eval_map.slurm
# Logs:    <PROJECT_HOME>/logs/eval_map_<jobid>.out
# Results: <PROJECT_HOME>/runs/segment/eval_<variant>/ (results.csv + PR curves)
# ----------------------------------------------------------------------------
# The logs/ folder must exist: mkdir -p <PROJECT_HOME>/logs

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<your-email>
#SBATCH --job-name=eval_map
#SBATCH --partition=<gpu-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/%u/project/logs/eval_map_%j.out
#SBATCH --error=/home/%u/project/logs/eval_map_%j.err
#SBATCH --account=<your-slurm-account>

set -eo pipefail

PROJECT_HOME="/home/${USER}/project"
RUNS_DIR="${PROJECT_HOME}/runs"
VARIANTS="A_real_only B_augmented C_synth_only"

command -v module >/dev/null 2>&1 && module load CUDA/11.8 || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate larvixon
export CUDA_VISIBLE_DEVICES=0

echo "=========================================="
echo "Job:       ${SLURM_JOB_ID:-local}"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader || echo none)"
echo "Variants:  ${VARIANTS}"
echo "=========================================="

cd "${PROJECT_HOME}"

for VARIANT in ${VARIANTS}; do
    MODEL="${RUNS_DIR}/stylegan_${VARIANT}/weights/best.pt"
    DATA_YAML="${PROJECT_HOME}/data/stylegan_variants/${VARIANT}/data.yaml"

    echo ""
    echo "------------------------------------------"
    echo ">>> Variant: ${VARIANT}"
    echo "    Model:     ${MODEL}"
    echo "    data.yaml: ${DATA_YAML}"
    echo "------------------------------------------"

    if [ ! -f "${MODEL}" ]; then
        echo "ERROR: missing weights ${MODEL} - train the variant first (train_yolo_variant.slurm)." >&2
        exit 1
    fi
    if [ ! -f "${DATA_YAML}" ]; then
        echo "ERROR: missing ${DATA_YAML} - run 08_build_augmented_dataset.py first." >&2
        exit 1
    fi

    # Evaluate on the real combined test set (split=test). Output: runs/segment/eval_<variant>/
    yolo segment val \
        model="${MODEL}" \
        data="${DATA_YAML}" \
        split=test \
        project="${RUNS_DIR}/segment" \
        name="eval_${VARIANT}" \
        plots=True
done

echo ""
echo "=========================================="
echo "DONE. mAP results (auxiliary):"
for VARIANT in ${VARIANTS}; do
    echo "  ${RUNS_DIR}/segment/eval_${VARIANT}/results.csv  (+ BoxPR_curve.png, MaskPR_curve.png)"
done
echo "=========================================="
