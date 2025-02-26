#!/bin/bash
#SBATCH --mem=250G
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH -p sched_mit_keating
#SBATCH -o RUNDIR/tune-output_runRUNNO_NODERANK.out
#SBATCH -e RUNDIR/tune-error_runRUNNO_NODERANK.out

MASTER_PORT=1234
ulimit -s unlimited
ulimit -n 10000
export TF_ENABLE_ONEDNN_OPTS=0
export LOCAL_RANK=-1
export LD_LIBRARY_PATH=/home/fosterb/.conda/envs/esmfold/lib
CONDA_ROOT=/home/software/anaconda3/2021.11
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate esmfold

python finetune.py --dataset=DATASET --orig_model=ORIG --model_hparams=MODEL_HPARAMS --backend=nccl --run_hparams=RUN_HPARAMS --run_dir=RUNDIR --n_nodes=NUM_NODES --n_trials=NUM_TRIALS --train=DATASET/TRAIN.in --validation=DATASET/VALIDATION.in --test=DATASET/TEST.in --converter_path=CONVERTER


