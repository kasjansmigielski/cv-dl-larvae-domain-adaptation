#!/bin/bash
# ============================================================================
# STEP 3.5 - Train StyleGAN2-ADA on an HPC cluster (stylegan3 repo, SLURM)
# ============================================================================
# NOTE: example SLURM script. Adjust PROJECT_HOME, the env path, the SLURM
# account, partition/GRES and the email to your own cluster.
#
# Submit:  sbatch train_stylegan.slurm
# Watch:   squeue -u $USER
# Logs:    tail -f <PROJECT_HOME>/stylegan3/logs/stylegan_<jobid>.out
# Cancel:  scancel <jobid>
# ----------------------------------------------------------------------------
# NOTE: we use the stylegan3 repo (torch 2.x on H100/sm_90) in '--cfg stylegan2'
#   mode - i.e. still STYLEGAN2-ADA. The old repo (torch 1.8) does NOT run on H100.
# ----------------------------------------------------------------------------
# The logs/ folder must exist before submitting:
#   mkdir -p <PROJECT_HOME>/stylegan3/logs

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<your-email>
#SBATCH --job-name=stylegan
#SBATCH --partition=<gpu-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=72:00:00
#SBATCH --output=/home/%u/project/stylegan3/logs/stylegan_%j.out
#SBATCH --error=/home/%u/project/stylegan3/logs/stylegan_%j.err
#SBATCH --account=<your-slurm-account>

# NOTE: do NOT use 'set -u' globally - conda activate.d/deactivate.d scripts
# (gcc_linux-64, cuda-nvcc) reference unset variables and would fail the job.
set -eo pipefail

# --- Paths (adjust to your cluster) -----------------------------------------
PROJECT_HOME="/home/${USER}/project"
REPO_DIR="${PROJECT_HOME}/stylegan3"
DATASET="${DATASET:-combined}"                               # combined | s1 | s2
SRC_IMAGES="${PROJECT_HOME}/data/stylegan/${DATASET}"        # PNG folder
DATA_ZIP="${PROJECT_HOME}/data/stylegan/${DATASET}_256.zip"  # zip for StyleGAN
OUTDIR="${REPO_DIR}/runs/stylegan_${DATASET}"
# RESUME: empty = train from scratch (RECOMMENDED for a clean FID curve).
RESUME=""

# --- Environment ------------------------------------------------------------
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${HOME}/conda_envs/stylegan"

# CRITICAL: op-compilation cache and TMPDIR NOT on /tmp (too small -> Errno 28).
export TORCH_EXTENSIONS_DIR="${HOME}/.torch_extensions"
export TMPDIR="${HOME}/.tmp_build"
mkdir -p "${TMPDIR}"

# CRITICAL - nvcc and CUDA libraries from the CONDA env (consistent toolkit with
# cusparse/cublas headers), NOT from system modules.
# Conda installs the toolkit in ONE of two layouts:
#   a) $CONDA_PREFIX/targets/x86_64-linux/{include,lib}  ("toolkit" layout)
#   b) $CONDA_PREFIX/{include,lib}                        ("flat" layout)
# CUDA_HOME MUST point to the one that actually contains cuda_runtime_api.h.
if [ -f "${CONDA_PREFIX}/targets/x86_64-linux/include/cuda_runtime_api.h" ]; then
    CUDA_TARGET="${CONDA_PREFIX}/targets/x86_64-linux"
    export CUDA_HOME="${CUDA_TARGET}"
else
    CUDA_TARGET="${CONDA_PREFIX}"        # flat layout: headers in $CONDA_PREFIX/include
    export CUDA_HOME="${CONDA_PREFIX}"
fi
export PATH="${CONDA_PREFIX}/bin:${PATH}"
# Add both include dirs to op compiler flags (just in case, regardless of layout).
export CFLAGS="-I${CONDA_PREFIX}/include -I${CONDA_PREFIX}/targets/x86_64-linux/include ${CFLAGS:-}"
export CXXFLAGS="-I${CONDA_PREFIX}/include -I${CONDA_PREFIX}/targets/x86_64-linux/include ${CXXFLAGS:-}"

# CUDA runtime libs (cu12) must be on LD_LIBRARY_PATH so the op finds libcudart.so.12.
TORCH_LIB="$(python -c 'import os,torch;print(os.path.join(os.path.dirname(torch.__file__),"lib"))')"
NVIDIA_LIBS="$(python -c 'import os,nvidia;base=os.path.dirname(nvidia.__file__);print(":".join(os.path.join(base,d,"lib") for d in os.listdir(base) if os.path.isdir(os.path.join(base,d,"lib"))))' 2>/dev/null || true)"
export LD_LIBRARY_PATH="${CUDA_TARGET}/lib:${CONDA_PREFIX}/lib:${TORCH_LIB}:${NVIDIA_LIBS}:${LD_LIBRARY_PATH:-}"

# Compile the op only for H100 (sm_90) - faster and unambiguous.
export TORCH_CUDA_ARCH_LIST="9.0"

echo "=========================================="
echo "Job:       ${SLURM_JOB_ID:-local}"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader || echo none)"
echo "nvcc:      $(nvcc --version 2>/dev/null | grep release || echo 'nvcc MISSING! Run setup_stylegan_env.sh')"
echo "CUDA_HOME: ${CUDA_HOME}"
echo "Repo:      ${REPO_DIR}  (stylegan3, --cfg stylegan2 mode)"
echo "Dataset:   ${DATA_ZIP}"
echo "Outdir:    ${OUTDIR}"
echo "=========================================="

cd "${REPO_DIR}"
export CUDA_VISIBLE_DEVICES=0

# Clear any stale, incorrectly compiled op cache from previous attempts.
rm -rf "${TORCH_EXTENSIONS_DIR}" ~/.cache/torch_extensions ~/.torch_extensions 2>/dev/null || true
mkdir -p "${TORCH_EXTENSIONS_DIR}"

# --- DISK GUARD (useful under a tight quota) --------------------------------
# A snapshot .pkl is ~357 MB; at --kimg 5000 and --snap 10 this accumulates to
# ~44 GB. We keep only the last few snapshots + every Nth as a milestone. FID is
# read from metric-fid50k_full.jsonl, so no metric history is lost.
KEEP_LAST=3            # how many latest snapshots to always keep
KEEP_EVERY_KIMG=500    # additionally keep snapshots at round kimg (milestones)
disk_guard() {
    while true; do
        for exp in "${OUTDIR}"/*/; do
            [ -d "${exp}" ] || continue
            mapfile -t snaps < <(ls -1 "${exp}"network-snapshot-*.pkl 2>/dev/null | sort)
            n=${#snaps[@]}
            (( n <= KEEP_LAST )) && continue
            for ((i=0; i<n-KEEP_LAST; i++)); do
                f="${snaps[i]}"
                kimg=$(basename "${f}" | sed -E 's/network-snapshot-0*([0-9]+)\.pkl/\1/')
                if (( kimg % KEEP_EVERY_KIMG == 0 )); then
                    continue   # keep milestone
                fi
                rm -f "${f}" 2>/dev/null || true
            done
        done
        sleep 120   # check every 2 minutes
    done
}
disk_guard &
GUARD_PID=$!
trap 'kill ${GUARD_PID} 2>/dev/null || true' EXIT

# --- Convert PNG folder -> zip (if not done yet) ----------------------------
# stylegan3 dataset_tool syntax: --resolution=WxH (different from stylegan2-ada!)
if [ ! -f "${DATA_ZIP}" ]; then
    echo "Converting PNG folder -> zip (dataset_tool.py)..."
    python dataset_tool.py \
        --source="${SRC_IMAGES}" \
        --dest="${DATA_ZIP}" \
        --resolution=256x256
else
    echo "Dataset zip already exists: ${DATA_ZIP}"
fi

# --- Train StyleGAN2-ADA (via the stylegan3 repo) ---------------------------
# --cfg stylegan2  : StyleGAN2 architecture (NOT stylegan3-t/r) = classic StyleGAN2-ADA
# --aug ada        : adaptive discriminator augmentation (KEY for <5k images)
# --target 0.6     : target D overfitting level (ADA control)
# --gpus 1         : 1x H100
# --batch 32       : global batch
# --gamma 6.6      : R1 regularization (tune by FID)
# --mirror 1       : horizontal flip (2x effective set)
# --snap 50        : snapshot every 50 ticks (fewer .pkl files under a tight quota)
# --metrics ...    : FID as the main quality metric
# --kimg 5000      : training budget (thousands of images)
RESUME_ARG=""
if [ -n "${RESUME}" ]; then
    RESUME_ARG="--resume=${RESUME}"
    echo "Resuming from: ${RESUME}"
fi

python -u train.py \
    --outdir="${OUTDIR}" \
    --data="${DATA_ZIP}" \
    --cfg=stylegan2 \
    --gpus=1 \
    --batch=32 \
    --gamma=6.6 \
    --mirror=1 \
    --aug=ada \
    --target=0.6 \
    --snap=50 \
    --metrics=fid50k_full \
    --kimg=5000 \
    ${RESUME_ARG}

echo "=========================================="
echo "Training finished. Snapshots: ${OUTDIR}/<EXP>/network-snapshot-*.pkl"
echo "Pick the .pkl with the LOWEST FID from metric-fid50k_full.jsonl"
echo "=========================================="
