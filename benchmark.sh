#!/bin/bash
#SBATCH --partition=gpu_h100     # GPU partition
#SBATCH --gres=gpu:1             # request 1 GPU
#SBATCH --job-name=few-shot      # job name
#SBATCH --ntasks=1               # number of tasks
#SBATCH --cpus-per-task=4        # number of CPU cores
#SBATCH --mem=128G               # memory per node
#SBATCH --time=8:00:00          # walltime
#SBATCH --output=logs/out_%j.txt # standard output
#SBATCH --error=logs/err_%j.txt  # standard error

conda activate pcadino


#python -u main.py --dataset_name mvtec_ad --dataset_path datasets/mvtec-ad --use_specular_filter --image_res 336 --k_shot 8 --aug_count 30
conda run -n pcadino python -u main.py --dataset_name visa --dataset_path datasets/visa --use_specular_filter --image_res 336 --k_shot 1 --use_clahe --aug_count 200
