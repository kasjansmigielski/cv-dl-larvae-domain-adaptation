#!/bin/bash
# ============================================================================
# STEP 4.5 - Evaluate detection rate (YOLO+ByteTrack tracking) for variant A/B/C
# ============================================================================
# NOTE: example SLURM script. Adjust PROJECT_HOME, the SLURM account,
# partition/GRES, the conda env and the email to your own cluster.
#
# MAIN experiment metric: detection rate on FULL recordings.
# Each variant -> its own output folder (so they don't overwrite each other).
#
# REQUIRES: recordings in <PROJECT_HOME>/data/raw_videos/{s1,s2}/*.mp4
#
# Submit (variant as argument: A_real_only | B_augmented | C_synth_only):
#   sbatch 09_eval_detection_rate.slurm A_real_only
#   sbatch 09_eval_detection_rate.slurm B_augmented
#   sbatch 09_eval_detection_rate.slurm C_synth_only
# Logs: <PROJECT_HOME>/logs/eval_detrate_<jobid>.out
# ----------------------------------------------------------------------------
# The logs/ folder must exist: mkdir -p <PROJECT_HOME>/logs

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<your-email>
#SBATCH --job-name=eval_dr
#SBATCH --partition=<gpu-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/home/%u/project/logs/eval_detrate_%j.out
#SBATCH --error=/home/%u/project/logs/eval_detrate_%j.err
#SBATCH --account=<your-slurm-account>

set -eo pipefail

VARIANT="${1:?Provide variant: A_real_only | B_augmented | C_synth_only}"

PROJECT_HOME="/home/${USER}/project"
SCRIPT="${PROJECT_HOME}/src/dl_pipeline/yolo/01_track_and_analyze.py"
MODEL="${PROJECT_HOME}/runs/stylegan_${VARIANT}/weights/best.pt"
# CRITICAL: variant models A/B/C have ONLY the 'larva' class (remap in step 08) - they do NOT
# know 'dish'. Dish calibration needs a 'dish' class, so we use the ORIGINAL combined model for it.
# Larvae are detected by the variant model (--model), dishes by --dish-model.
DISH_MODEL="${PROJECT_HOME}/runs/yolo_baseline_combined/weights/best.pt"
# Separate output folder per variant (otherwise combined_summary_tracker.csv overwrites).
OUTPUT="results/yolo_tracker_${VARIANT}"
VIDEOS_DIR="${PROJECT_HOME}/data/raw_videos"

command -v module >/dev/null 2>&1 && module load CUDA/11.8 || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate larvixon
export CUDA_VISIBLE_DEVICES=0

echo "=========================================="
echo "Job:       ${SLURM_JOB_ID:-local}"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader || echo none)"
echo "Variant:   ${VARIANT}"
echo "Model:     ${MODEL}"
echo "Output:    ${OUTPUT}"
echo "=========================================="

if [ ! -f "${MODEL}" ]; then
    echo "ERROR: missing weights ${MODEL}. Train the variant first (train_yolo_variant.slurm)." >&2
    exit 1
fi
if [ ! -f "${DISH_MODEL}" ]; then
    echo "ERROR: missing dish-calibration model ${DISH_MODEL} (needed because variants lack 'dish')." >&2
    exit 1
fi
if [ ! -d "${VIDEOS_DIR}/s1" ] && [ ! -d "${VIDEOS_DIR}/s2" ]; then
    echo "ERROR: no recordings in ${VIDEOS_DIR}/{s1,s2}. Download them before evaluation." >&2
    exit 1
fi

cd "${PROJECT_HOME}"
python -u "${SCRIPT}" --all \
    --model "${MODEL}" \
    --dish-model "${DISH_MODEL}" \
    --output "${OUTPUT}" \
    --videos-dir "${VIDEOS_DIR}"

echo "=========================================="
echo "DONE. Results: ${PROJECT_HOME}/${OUTPUT}/combined_summary_tracker.csv"
echo "  Detection rate per experiment - compare A vs B vs C."
echo "=========================================="
