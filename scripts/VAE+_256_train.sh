#!/bin/sh

# A100 GPU queue, there is also gpua40 and gpua10
#BSUB -q gpua100

# job name
#BSUB -J job_name_goes_here

# 4 cpus, 1 machine, 1 gpu, 24 hours (the max)
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00

# at least 32 GB RAM
#BSUB -R "rusage[mem=32GB]"

# stdout/stderr files for debugging (%J is substituted for job ID)
#BSUB -o logs/my_run_%J.out
#BSUB -e logs/my_run_%J.err

# your training script here, e.g.
# activate environment ...
source .venv/bin/activate
python src/train.py --config configs/VAE+_256.yaml