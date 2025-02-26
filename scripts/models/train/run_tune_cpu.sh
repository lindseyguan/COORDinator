#!/bin/bash
#SBATCH --mincpu=24
#SBATCH --time=96:00:00
#SBATCH -o RUNDIR/train-output_runRUNNO_NODERANK.out
#SBATCH -e RUNDIR/train-error_runRUNNO_NODERANK.out

CONDA_ROOT=/state/partition1/llgrid/pkg/anaconda/anaconda3-2022b/
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate terminator_fine_tune
export LOCAL_RANK=-1
ulimit -s unlimited
ulimit -n 10000

python finetune.py --dataset=DATASET --orig_model=ORIG --model_hparams=MODEL_HPARAMS --backend=nccl --run_hparams=RUN_HPARAMS --run_dir=RUNDIR --n_nodes=NUM_NODES --n_trials=NUM_TRIALS --train=DATASET/TRAIN.in --validation=DATASET/VALIDATION.in --test=DATASET/TEST.in --converter_path=CONVERTER --dev=cpu


