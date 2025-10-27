#!/bin/bash
#SBATCH --partition=gpu_h100      # GPU partition
#SBATCH --gres=gpu:1             # request 1 GPU
#SBATCH --job-name=grouped_gmm   # job name
#SBATCH --ntasks=1               # number of tasks
#SBATCH --cpus-per-task=4        # number of CPU cores
#SBATCH --mem=168G               # memory per node
#SBATCH --time=2:00:00          # walltime
#SBATCH --output=logs/out_%j.txt # standard output
#SBATCH --error=logs/err_%j.txt  # standard error

module purge
module load 2023
module load 2024
module load  Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.4.0
source ~/venvs/anomalib-env/bin/activate

# python -u main.py --dataset_name mvtec_ad2 --dataset_path ../anomalib/datasets/MVTec_AD_2 --use_specular_filter --patch_size 448 --use_clahe
python -u main.py --dataset_name mvtec_ad --dataset_path ../anomalib/datasets/MVTec --use_specular_filter