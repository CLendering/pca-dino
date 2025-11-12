## Install
```
# TODO: Add install instructions (conda/venv, packages, CUDA, torch, etc.)
```

## Data
- MVTec-AD: download from https://www.mvtec.com/company/research/datasets/mvtec-ad and place under your data directory (e.g., data/mvtec).
- VisA: download the archive and extract to your data directory (e.g., data/visa).
    - Example:
        ```
        # download VisA (may require changing URL or using a browser)
        wget https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar
        tar -xvf VisA_20220922.tar -C data/
        ```

**VisA preprocessing**
- Update source and target paths inside prepare_visa_public.py to match your local layout.
- Run preprocessing:
```
python prepare_visa_public.py
```

**Scripts and paths**
- Update source paths for both MvTec and VisA in ./scripts/*.sh to point to your data directories.
- Verify any other path variables (checkpoints, outputs, logs) in the scripts.

**Run**
- Submit experiments via the provided SLURM scripts:
```
sbatch scripts/benchmark_few_shot.sh      # few-shot
sbatch scripts/benchmark_batched0shot.sh  # batched, zero-shot
sbatch scripts/benchmark_full_shot.sh     # full-shot
```
- For quick local testing, run the equivalent script or main entrypoint directly.
