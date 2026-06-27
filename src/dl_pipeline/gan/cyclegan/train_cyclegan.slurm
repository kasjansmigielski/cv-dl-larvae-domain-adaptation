#!/bin/bash
# ============================================================================
# STEP 2.4 - Train CycleGAN on an HPC cluster (SLURM)
# ============================================================================
# NOTE: This is an example SLURM script. Adjust PROJECT_HOME, the SLURM account,
# the partition/GRES, and the conda environment name to your own cluster.
#
# Submit:  sbatch train_cyclegan.slurm
# Watch:   squeue -u $USER
# Logs:    tail -f <PROJECT_HOME>/pytorch-CycleGAN-and-pix2pix/logs/cyclegan_<jobid>.out
# Cancel:  scancel <jobid>
# ----------------------------------------------------------------------------
# The logs/ folder must exist before submitting:
#   mkdir -p <PROJECT_HOME>/pytorch-CycleGAN-and-pix2pix/logs

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<your-email>
#SBATCH --job-name=cyclegan
#SBATCH --partition=<gpu-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/%u/project/pytorch-CycleGAN-and-pix2pix/logs/cyclegan_%j.out
#SBATCH --error=/home/%u/project/pytorch-CycleGAN-and-pix2pix/logs/cyclegan_%j.err
#SBATCH --account=<your-slurm-account>

set -euo pipefail

# --- Cluster paths (adjust to your $HOME) -----------------------------------
PROJECT_HOME="/home/${USER}/project"
CYCLEGAN_REPO="${PROJECT_HOME}/pytorch-CycleGAN-and-pix2pix"
DATAROOT="${PROJECT_HOME}/data/cyclegan"
RUN_NAME="larvae_s1_to_s2"

# --- Environment ------------------------------------------------------------
# NOTE: on a compute node 'module' may be unavailable - guard with || true.
command -v module >/dev/null 2>&1 && module load CUDA/11.8 || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate larvixon

echo "=========================================="
echo "Job:       ${SLURM_JOB_ID:-local}"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader || echo none)"
echo "Dataroot:  ${DATAROOT}"
echo "Repo:      ${CYCLEGAN_REPO}"
echo "=========================================="

cd "${CYCLEGAN_REPO}"

# SLURM allocates 1 GPU via --gres; restrict visibility to it.
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline   # no --use_wandb => visdom/wandb disabled

# --- CycleGAN training ------------------------------------------------------
# netG resnet_9blocks + PatchGAN (n_layers_D=3) = standard for 256x256.
# n_epochs 100 (constant LR) + n_epochs_decay 100 (linear decay) = 200 epochs.
# python -u = unbuffered output -> progress visible in the log immediately.
python -u train.py \
    --dataroot "${DATAROOT}" \
    --name "${RUN_NAME}" \
    --model cycle_gan \
    --netG resnet_9blocks \
    --netD n_layers --n_layers_D 3 \
    --lambda_A 10.0 --lambda_B 10.0 --lambda_identity 0.5 \
    --pool_size 50 \
    --no_dropout \
    --load_size 286 --crop_size 256 \
    --n_epochs 100 --n_epochs_decay 100 \
    --batch_size 4 \
    --save_epoch_freq 10

echo "=========================================="
echo "Training finished. Checkpoints: ${CYCLEGAN_REPO}/checkpoints/${RUN_NAME}/"
echo "=========================================="
