#!/bin/bash
# ============================================================================
# STEP 3.3 - StyleGAN environment setup on an HPC cluster (clean env for H100)
# ============================================================================
# NOTE: example HPC setup script. Adjust PROJECT_HOME, ENV_PREFIX and cache
# locations to your own cluster. Run ONCE on a login node:
#   bash setup_stylegan_env.sh
#
# Why this setup differs from the default:
#   - The GPU is an NVIDIA H100 (sm_90) -> requires torch 2.x + CUDA 12.x
#     (old torch 1.8 fails).
#   - StyleGAN custom ops (bias_act/upfirdn2d) require a FULL CUDA toolkit with
#     headers (cusparse.h, cublas.h, ...). We install a consistent CUDA 12.4
#     (mature, fewer conflicts than 12.6, fully compatible with torch cu121 + sm_90).
#
# What it does:
#   1. Sets cache locations off /tmp (which is too small on compute nodes).
#   2. (Optionally) removes an old, inconsistent 'stylegan' env.
#   3. Creates a conda env (--prefix), Python 3.9.
#   4. Installs torch 2.4.1+cu121 (pip) + CUDA dev libraries (conda, with headers).
#   5. Installs stylegan3 dependencies + clones the repo.
#   6. Smoke test (nvcc 12.4 + cusparse/cublas headers).
# ============================================================================

set -eo pipefail   # NOTE: no -u (conda activate.d scripts break on unbound vars)

# --- Locations (adjust to your cluster) -------------------------------------
PROJECT_HOME="/home/${USER}/project"
ENV_PREFIX="${HOME}/conda_envs/stylegan"
REPO_DIR="${PROJECT_HOME}/stylegan3"
# CRITICAL: cache NOT on /tmp! On a compute node /tmp is too small and fails with
# [Errno 28] No space left on device when unpacking ~1.1 GB of CUDA packages.
export CONDA_PKGS_DIRS="${HOME}/.conda_pkgs"
export PIP_CACHE_DIR="${HOME}/.pip_cache"
export TMPDIR="${HOME}/.tmp_build"
CUDA_VER="12.4"                              # mature, compatible with torch cu121 + sm_90

mkdir -p "${PROJECT_HOME}" "${CONDA_PKGS_DIRS}" "${PIP_CACHE_DIR}" "${TMPDIR}"

echo "=========================================="
echo "StyleGAN setup (torch 2.x + CUDA ${CUDA_VER}, for H100)"
echo "  Env:        ${ENV_PREFIX}"
echo "  Repo:       ${REPO_DIR}"
echo "  pkgs cache: ${CONDA_PKGS_DIRS} (NOT /tmp)"
echo "  TMPDIR:     ${TMPDIR} (NOT /tmp)"
echo "=========================================="

source "$(conda info --base)/etc/profile.d/conda.sh"

# --- 1. Remove an old, inconsistent env -------------------------------------
# NOTE: run ONLY when rebuilding the env from scratch. If the env already works, skip.
echo "[1/6] Cleaning old env + cache..."
conda deactivate 2>/dev/null || true
conda env remove --prefix "${ENV_PREFIX}" -y 2>/dev/null || true
conda env remove -n "stylegan" -y 2>/dev/null || true
conda clean -a -y 2>/dev/null || true

# --- 2. stylegan3 repo ------------------------------------------------------
if [ ! -d "${REPO_DIR}" ]; then
    echo "[2/6] Cloning stylegan3..."
    git clone https://github.com/NVlabs/stylegan3.git "${REPO_DIR}"
else
    echo "[2/6] Repo already exists - skipping."
fi

# --- 3. conda env (--prefix) ------------------------------------------------
echo "[3/6] Creating conda env '${ENV_PREFIX}' (Python 3.9)..."
mkdir -p "$(dirname "${ENV_PREFIX}")"
conda create -y --prefix "${ENV_PREFIX}" python=3.9
conda activate "${ENV_PREFIX}"

# --- 4. torch (pip) + CUDA dev libraries (conda) ----------------------------
echo "[4/6] Installing torch 2.4.1+cu121..."
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

# Lightweight: instead of the full cuda-toolkit (~4 GB) - only nvcc + dev headers
# needed to compile StyleGAN ops (cudart, cusparse, cublas, cusolver, curand, cufft, nvjitlink).
# All in a single ${CUDA_VER} from the nvidia channel -> consistent, no conflicts.
echo "[4/6] Installing nvcc + CUDA dev libraries (${CUDA_VER})..."
conda install -y -c "nvidia/label/cuda-${CUDA_VER}.0" \
    cuda-nvcc cuda-cudart-dev cuda-cccl \
    libcusparse-dev libcublas-dev libcusolver-dev \
    libcurand-dev libcufft-dev libnvjitlink-dev

# --- 5. stylegan3 dependencies ----------------------------------------------
echo "[5/6] Installing stylegan3 dependencies..."
pip install ninja scipy click requests tqdm pyspng imageio imageio-ffmpeg \
    psutil matplotlib "pillow>=9" "numpy<2"

# --- 6. Smoke test ----------------------------------------------------------
echo "[6/6] Smoke test..."
echo "  nvcc:    $(nvcc --version 2>/dev/null | grep release || echo 'nvcc MISSING!')"
# Conda installs CUDA headers in one of two layouts:
#   a) $CONDA_PREFIX/targets/x86_64-linux/include  ("toolkit" layout)
#   b) $CONDA_PREFIX/include                        ("flat" layout)
# Check BOTH - otherwise we get a false MISSING despite a correct install.
CUDA_INC_T="${CONDA_PREFIX}/targets/x86_64-linux/include"
CUDA_INC_F="${CONDA_PREFIX}/include"
for h in cuda_runtime_api.h cusparse.h cublas_v2.h cusolverDn.h; do
    if [ -f "${CUDA_INC_T}/${h}" ]; then
        echo "  header ${h}: OK (targets/)"
    elif [ -f "${CUDA_INC_F}/${h}" ]; then
        echo "  header ${h}: OK (include/)"
    else
        echo "  header ${h}: MISSING!"
    fi
done
python -c "import torch; print('  torch:', torch.__version__, '| cuda:', torch.version.cuda)"

echo "=========================================="
echo "DONE. Activate: conda activate ${ENV_PREFIX}"
echo "Logs folder + start training:"
echo "  mkdir -p ${REPO_DIR}/logs"
echo "  sbatch train_stylegan.slurm"
echo "=========================================="
