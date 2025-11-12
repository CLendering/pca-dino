#!/bin/bash
#SBATCH --partition=gpu_h100     # GPU partition
#SBATCH --gres=gpu:1             # request 1 GPU
#SBATCH --job-name=few-shot      # job name
#SBATCH --ntasks=1               # number of tasks
#SBATCH --cpus-per-task=4        # number of CPU cores
#SBATCH --mem=128G               # memory per node
#SBATCH --time=24:00:00          # walltime
#SBATCH --output=logs/out_%j.txt # standard output
#SBATCH --error=logs/err_%j.txt  # standard error

eval "$($CONDA_EXE shell.bash hook)"
conda activate pcadino

# conda run -n pcadino python -u visualize_tsne.py \
#     --dataset_name mvtec_ad \
#     --dataset_path datasets/mvtec-ad/ \
#     --category hazelnut \
#     --model_ckpt facebook/dinov3-vit7b16-pretrain-lvd1689m \
#     --layers="-12,-13,-14,-15,-16,-17,-18" \
#     --image_res 448 \
#     --aug_count 30 \
#     --aug_list "rotate" \
#     --pca_ev 0.85 \

conda run -n pcadino python -u main.py --dataset_name visa --dataset_path ../AnomalyDINO/VisA_pytorch/1cls/  --image_res 448 --k_shot 1 --layers="-17,-18,-19,-20,-21,-22,-23" --model_ckpt "facebook/dinov3-vit7b16-pretrain-lvd1689m" --aug_count 35 --pca_ev 0.99 --save_intro_overlays --outdir "visuals/overlays_k1"