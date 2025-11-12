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



#python -u main.py --dataset_name mvtec_ad --dataset_path datasets/mvtec-ad --use_specular_filter --image_res 672 --k_shot 8 --aug_count 30
# conda run -n pcadino python -u  main.py --dataset_name visa --dataset_path ../AnomalyDINO/VisA_pytorch/1cls/ --image_res 672 --k_shot 4 --aug_count 30
# conda run -n pcadino python -u main.py --dataset_name mvtec_ad --dataset_path datasets/mvtec-ad --image_res 672 --k_shot 8 --layers="-12,-13,-14,-15,-17,-18" --model_ckpt "facebook/dinov3-vit7b16-pretrain-lvd1689m" --aug_count 30 --pca_ev 0.99 --agg_method "mean"
conda run -n pcadino python -u main.py --dataset_name mvtec_ad --dataset_path datasets/mvtec-ad --image_res 448 --layers="-12,-13,-14,-15,-16,-17,-18" --model_ckpt "facebook/dinov3-vit7b16-pretrain-lvd1689m" --pca_ev 0.99 --agg_method "mean"
# conda run -n pcadino python -u main.py --dataset_name mvtec_ad --dataset_path datasets/mvtec-ad --image_res 672 --k_shot 4 --layers="-12,-13,-14,-15,-17,-18" --model_ckpt "facebook/dinov3-vit7b16-pretrain-lvd1689m" --aug_count 30 --pca_ev 0.99 --agg_method "mean"
# conda run -n pcadino python -u main.py --dataset_name visa --dataset_path ../AnomalyDINO/VisA_pytorch/1cls/ --image_res 672 --k_shot 8 --layers="-12,-13,-14,-15,-17,-18" --model_ckpt "facebook/dinov3-vit7b16-pretrain-lvd1689m" --aug_count 30 --pca_ev 0.99 --agg_method "mean"
conda run -n pcadino python -u main.py --dataset_name visa --dataset_path ../AnomalyDINO/VisA_pytorch/1cls/ --image_res 448 --layers="-12,-13,-14,-15,-16,-17,-18" --model_ckpt "facebook/dinov3-vit7b16-pretrain-lvd1689m" --pca_ev 0.99 --agg_method "mean"
# conda run -n pcadino python -u main.py --dataset_name visa --dataset_path ../AnomalyDINO/VisA_pytorch/1cls/ --image_res 672 --k_shot 4 --layers="-12,-13,-14,-15,-17,-18" --model_ckpt "facebook/dinov3-vit7b16-pretrain-lvd1689m" --aug_count 30 --pca_ev 0.99 --agg_method "mean"