#!/bin/bash

AVAILABLE_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sort -n -k2 | head -n1 | cut -d, -f1)
export CUDA_VISIBLE_DEVICES=$AVAILABLE_GPU

export LOCAL_RANK=-1
ulimit -s unlimited
ulimit -n 10000

python train.py --dataset=DATASET --msa_dataset=MSA --model_hparams=MODEL_HPARAMS --backend=nccl --run_hparams=RUN_HPARAMS --run_dir=RUNDIR --n_nodes=NUM_NODES --n_trials=NUM_TRIALS --train=DATASET/TRAIN.in --validation=DATASET/VALIDATION.in --test=DATASET/TEST.in --converter_path=CONVERTER


