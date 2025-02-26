#!/bin/bash
IFS=","
. /etc/profile.d/modules.sh
OUTDIR="/data1/groups/keatinglab/fosterb/running_COORDinator_experiments/ingraham_true_base_msa_weighted_nlcpl_cutoff_0.9_1000_V2"
mkdir -p ${OUTDIR}
export NUM_GPUS_PER_NODE=1
export NUM_NODES=3
export NUM_TRIALS=1
for args in \
	dgat,0,localhost; \
	do set -- $args
	. /home/gridsan/fbirnbaum/TERMinator-public_mirror/scripts/models/train/submit_train_ddp.sh /data1/groups/keatinglab/fosterb/ingraham_data/ingraham_data_features /data1/groups/keatinglab/fosterb/ingraham_data/ingraham_features_msa_2 /home/gridsan/fbirnbaum/TERMinator-public_mirror/hparams/model/coordinator.json /home/gridsan/fbirnbaum/TERMinator-public_mirror/hparams/run/default.json ${OUTDIR} ${OUTDIR} ${OUTDIR} 240 train validation test $1 $2 $3
done
