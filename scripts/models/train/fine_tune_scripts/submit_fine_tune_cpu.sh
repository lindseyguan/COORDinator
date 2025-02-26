#!/bin/bash
IFS=","
. /etc/profile.d/modules.sh
OUTDIR="/data1/groups/keatinglab/fosterb/COORDinator_finetune_evc/ingraham_true_base_corr_loss_masked_V1"
mkdir -p ${OUTDIR}
export NUM_GPUS_PER_NODE=1
export NUM_NODES=1
export NUM_TRIALS=1
for args in \
	dgat,0,localhost; \
	do set -- $args
	. /home/gridsan/fbirnbaum/TERMinator-public_mirror/scripts/models/train/submit_train.sh /data1/groups/keatinglab/fosterb/rocklin_data_2022/rocklin_2022_features_evcouplings /data1/groups/keatinglab/fosterb/rocklin_data_2022/rocklin_2022_features_evcouplings /home/gridsan/fbirnbaum/TERMinator-public_mirror/hparams/model/coordinator_fine_tuning.json /home/gridsan/fbirnbaum/TERMinator-public_mirror/hparams/run/fine_tuning.json ${OUTDIR} ${OUTDIR} ${OUTDIR} 240 train validation test $1 $2 $3
done
