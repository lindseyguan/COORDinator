#!/bin/bash
#SBATCH --mincpu=20
#SBATCH --gres=gpu:volta:1
#SBATCH --time=96:00:00
#SBATCH -o RUNDIR/train-output_runRUNNO_NODERANK.out
#SBATCH -e RUNDIR/train-error_runRUNNO_NODERANK.out

CONDA_ROOT=/state/partition1/llgrid/pkg/anaconda/anaconda3-2022b/
source ${CONDA_ROOT}/etc/profile.d/conda.sh

MASTER_PORT=1234
ulimit -s unlimited
ulimit -n 10000
echo NODERANK
module load cuda/11.1
module load nccl/2.8.3-cuda11.1
conda activate terminator_fine_tune
export NCCL_DEBUG=INFO
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_LAUNCH_BLOCKING=1

python -m torch.distributed.run --nnodes NUM_NODES --nproc_per_node NUM_GPUS_PER_NODE --node_rank NODERANK --master_addr MASTER_ADDR --master_port "$MASTER_PORT" --rdzv_id 2 --rdzv_backend c10d --rdzv_endpoint MASTER_ADDR:"$MASTER_PORT" train.py --dataset=DATASET --msa_dataset=MSA --model_hparams=MODEL_HPARAMS --backend=nccl --run_hparams=RUN_HPARAMS --run_dir=RUNDIR --n_nodes=NUM_NODES --n_trials=NUM_TRIALS --train=DATASET/TRAIN.in --validation=DATASET/VALIDATION.in --test=DATASET/TEST.in


