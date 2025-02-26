#!/bin/bash
IFS=","
. /etc/profile.d/modules.sh
OUTDIR="/home/fosterb/coordinator_runs/coord_mpnn_multi_node_self_sub_center_node_rand"
mkdir -p ${OUTDIR}
export NUM_GPUS_PER_NODE=1
export NUM_NODES=1
export NUM_TRIALS=1
for args in \
	dgat,0,localhost; \
	do set -- $args
	. /home/fosterb/TERMinator-public_mirror/scripts/models/train/submit_train_gpu_engaging.sh /nobackup1c/users/fosterb/multichain_data/multichain_features /nobackup1c/users/fosterb/ingraham_data/ingraham_features /home/fosterb/TERMinator-public_mirror/hparams/model/coordinator.json /home/fosterb/TERMinator-public_mirror/hparams/run/default.json ${OUTDIR} ${OUTDIR} ${OUTDIR} 240 train validation test $1 $2 $3 /home/fosterb/TERMinator-public_mirror/converter_model_coord.pt 
done

	
	#     /home/gridsan/fbirnbaum/TERMinator-public_mirror/scripts/models/train/submit_train_gpu.sh /data1/groups/keatinglab/fosterb/ingraham_data/ingraham_data_features /data1/groups/keatinglab/fosterb/ingraham_data/ingraham_features_msa_2 /home/gridsan/fbirnbaum/TERMinator-public_mirror/hparams/model/coordinator.json /home/gridsan/fbirnbaum/TERMinator-public_mirror/hparams/run/default.json ${OUTDIR} ${OUTDIR} ${OUTDIR} 240 train validation test $1 $2 $3 /home/gridsan/fbirnbaum/TERMinator/analysis/converter_model_coord.pt