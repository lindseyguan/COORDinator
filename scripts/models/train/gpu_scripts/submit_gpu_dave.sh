#!/bin/bash
IFS=","
OUTDIR="/home/fosterb/coordinator_runs/coord_mpnn_multi_node_self_sub_45"
mkdir -p ${OUTDIR}
export NUM_GPUS_PER_NODE=1
export NUM_NODES=1
export NUM_TRIALS=1
for args in \
	dgat,0,localhost; \
	do set -- $args
	. /home/fosterb/TERMinator-public_mirror/scripts/models/train/submit_train_gpu_dave.sh /mnt/shared/fosterb/multichain_data/multichain_features /mnt/shared/fosterb/multicahin_data/multichain_features /home/fosterb/TERMinator-public_mirror/hparams/model/coordinator_supercloud.json /home/fosterb/TERMinator-public_mirror/hparams/run/default_supercloud.json ${OUTDIR} ${OUTDIR} ${OUTDIR} 240 scratch scratch scratch $1 $2 $3 /home/fosterb/TERMinator-public_mirror/models/converter_model_coord.pt
done


	#. /home/fosterb/TERMinator-public_mirror/scripts/models/train/submit_train_gpu_dave.sh /mnt/shared/fosterb/ingraham_data/ingraham_data_features /mnt/shared/fosterb/ingraham_data/ingraham_data_features /home/fosterb/TERMinator-public_mirror/hparams/model/coordinator.json /home/fosterb/TERMinator-public_mirror/hparams/run/default.json ${OUTDIR} ${OUTDIR} ${OUTDIR} 240 scratch scratch scratch $1 $2 $3 /home/fosterb/TERMinator-public_mirror/models/converter_model_coord.pt
