#!/bin/bash
# ============================================================================
# STEP 4.1 - Generate StyleGAN2-ADA synthetics (SLURM, batch on GPU)
# ============================================================================
# NOTE: example SLURM script. Adjust PROJECT_HOME, the env path, the SLURM
# account, partition/GRES and the email to your own cluster.
#
# Submit (with optional args: NUM TRUNC):
#   sbatch 06_generate_synthetic.slurm              # default 1500 / 0.7
#   sbatch 06_generate_synthetic.slurm 2000 0.7     # 2000 images, trunc 0.7
# Logs: <PROJECT_HOME>/logs/gen_synth_<jobid>.out
# ----------------------------------------------------------------------------
# The logs/ folder must exist: mkdir -p <PROJECT_HOME>/logs

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<your-email>
#SBATCH --job-name=gen_synth
#SBATCH --partition=<gpu-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/%u/project/logs/gen_synth_%j.out
#SBATCH --error=/home/%u/project/logs/gen_synth_%j.err
#SBATCH --account=<your-slurm-account>

set -eo pipefail   # no -u: conda activate.d scripts use unset variables

# --- Parameters (from args or defaults) -------------------------------------
NUM="${1:-1500}"    # number of images to generate
TRUNC="${2:-0.7}"   # truncation psi (quality vs diversity)

# --- Paths (adjust to your cluster) -----------------------------------------
PROJECT_HOME="/home/${USER}/project"
SCRIPT="${PROJECT_HOME}/src/dl_pipeline/gan/stylegan/06_generate_synthetic.py"
NET="${PROJECT_HOME}/stylegan3/best_models/best_model.pkl"
REPO_DIR="${PROJECT_HOME}/stylegan3"
OUTDIR="${PROJECT_HOME}/data/stylegan_synth/combined"

# --- Environment ------------------------------------------------------------
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${HOME}/conda_envs/stylegan"
export CUDA_VISIBLE_DEVICES=0
# CUDA runtime libs (cu12) from torch - so G(...) on GPU finds libcudart.so.12.
TORCH_LIB="$(python -c 'import os,torch;print(os.path.join(os.path.dirname(torch.__file__),"lib"))' 2>/dev/null || true)"
export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH:-}"

echo "=========================================="
echo "Job:       ${SLURM_JOB_ID:-local}"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader || echo none)"
echo "Model:     ${NET}"
echo "Num:       ${NUM}   Trunc: ${TRUNC}"
echo "Outdir:    ${OUTDIR}"
echo "=========================================="

if [ ! -f "${NET}" ]; then
    echo "ERROR: model ${NET} not found. Check the path to the best snapshot." >&2
    exit 1
fi

python -u "${SCRIPT}" \
    --network "${NET}" \
    --num "${NUM}" \
    --trunc "${TRUNC}" \
    --outdir "${OUTDIR}" \
    --repo "${REPO_DIR}"

echo "=========================================="
echo "DONE. Synthetics: ${OUTDIR}"
echo "  Preview grid: ${OUTDIR}_grid.png  (inspect before pseudo-labeling!)"
echo "  Next step: 07_pseudo_label.py"
echo "=========================================="
