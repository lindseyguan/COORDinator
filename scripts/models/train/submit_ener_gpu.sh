#!/bin/bash

# collect args
DATASET=$(readlink -f $1)
DATANAME=${1##*/}
ORIG=$2
MODEL_HPARAMS=$(readlink -f $3)
RUN_HPARAMS=$(readlink -f $4)
RUNDIR=$(readlink -f $5)
RUNNAME=${5##*/}
OUTPUTDIR=$(readlink -f $6)
LOGDIR=$(readlink -f $7)
HOURS=$8
TRAIN=$9
VALIDATION=${10}
TEST=${11}
echo "$DATANAME $ORIG $RUNNAME $OUTPUTDIR $LOGDIR"
NUM_GPUS_PER_NODE="$NUM_GPUS_PER_NODE"
NUM_NODES="$NUM_NODES"
NUM_TRIALS="$NUM_TRIALS"
MPN_TYPE=${12}
NODE_RANK=${13}
MASTER_ADDR=${14}
CONVERTER=${15}
echo "$NUM_GPUS_PER_NODE $NUM_NODES $MPN_TYPE $NODE_RANK $MASTER_ADDR"

# compute what directory this file is in
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
# DIR is the directory this file is in, e.g. postprocessing
DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"

cd $DIR

# create the folder to store the submission script
if [[ ! -d bash_files ]];
then
  mkdir bash_files
fi

# create the run dir and output dir
if [[ ! -d $RUNDIR ]];
then
  mkdir $RUNDIR
fi
if [[ ! -d $OUTPUTDIR ]];
then
  mkdir $OUTPUTDIR
fi

sed \
  -e "s|DATASET|${DATASET}|g" \
  -e "s|ORIG|${ORIG}|g" \
  -e "s|DATANAME|${DATANAME}|g" \
  -e "s|RUNNO|${RUNNO}|g" \
  -e "s|MODEL_HPARAMS|${MODEL_HPARAMS}|g" \
  -e "s|RUN_HPARAMS|${RUN_HPARAMS}|g" \
  -e "s|RUNDIR|${RUNDIR}|g" \
  -e "s|OUTPUTDIR|${OUTPUTDIR}|g" \
  -e "s|RUNNAME|${RUNNAME}|g" \
  -e "s|HOURS|${HOURS}|g" \
  -e "s|TRAIN|${TRAIN}|g" \
  -e "s|VALIDATION|${VALIDATION}|g" \
  -e "s|TEST|${TEST}|g" \
  -e "s|NODERANK|${NODE_RANK}|g" \
  -e "s|NUM_GPUS_PER_NODE|${NUM_GPUS_PER_NODE}|g" \
  -e "s|NUM_NODES|${NUM_NODES}|g" \
  -e "s|MASTER_ADDR|${MASTER_ADDR}|g" \
  -e "s|NUM_TRIALS|${NUM_TRIALS}|g" \
  -e "s|CONVERTER|${CONVERTER}|g" \
  <run_ener_gpu.sh \
  >bash_files/run_${DATANAME}_${RUNNAME}_ener_gpu.sh
jid0=$(sbatch --parsable bash_files/run_${DATANAME}_${RUNNAME}_ener_gpu.sh)
echo $jid0