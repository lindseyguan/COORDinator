#!/bin/bash
IFS=","
. /etc/profile.d/modules.sh
OUTDIR="/home/fosterb/coordinator_runs/coord_mpnn_multi_node_self_sub_ft_last_covid_mut_split"
mkdir -p ${OUTDIR}
export NUM_GPUS_PER_NODE=1
export NUM_NODES=1
export NUM_TRIALS=1
for args in \
	dgat,0,localhost; \
	do set -- $args
	. /home/fosterb/TERMinator-public_mirror/scripts/models/train/submit_tune_gpu_engaging.sh /nobackup1c/users/fosterb/covid_data/covid_features /home/fosterb/coordinator_runs/coord_mpnn_multi_node_self_sub/net_best_checkpoint.pt /home/fosterb/TERMinator-public_mirror/hparams/model/coordinator_fine_tuning.json /home/fosterb/TERMinator-public_mirror/hparams/run/fine_tuning.json ${OUTDIR} ${OUTDIR} ${OUTDIR} 240 scratch scratch scratch $1 $2 $3 /home/fosterb/TERMinator-public_mirror/converter_model_coord.pt 
done