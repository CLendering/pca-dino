#!/bin/bash
#SBATCH --partition=gpu_h100     # GPU partition
#SBATCH --gres=gpu:1             # request 1 GPU
#SBATCH --job-name=full-shot     # job name
#SBATCH --ntasks=1               # number of tasks
#SBATCH --cpus-per-task=4        # number of CPU cores
#SBATCH --mem=128G               # memory per node
#SBATCH --time=24:00:00          # walltime
#SBATCH --output=logs/out_%j.txt # standard output
#SBATCH --error=logs/err_%j.txt  # standard error

eval "$($CONDA_EXE shell.bash hook)"
conda activate pcadino

# -----------------------------------------------------------------
# --- 1. !! EDIT YOUR DATASET PATHS HERE !! ---
# -----------------------------------------------------------------
# Set the absolute path to your MVTec AD LOCO dataset directory
MVTEC_PATH="datasets/mvtec-loco-ad"
# -----------------------------------------------------------------

echo "--- Running Full-Shot (all train images) for MVTec-LOCO ---"
conda run -n pcadino python -u main.py \
    --dataset_name mvtec_loco \
    --dataset_path "$MVTEC_PATH" \
    --image_res 672 \
    --layers="-12,-13,-14,-15,-16,-17,-18" \
    --model_ckpt "facebook/dinov2-with-registers-giant" \
    --pca_ev 0.99 \
    --agg_method "mean" \
    --outdir "LOCO_results" \
    --use_logical_branch \

echo "--- All experiments complete ---"