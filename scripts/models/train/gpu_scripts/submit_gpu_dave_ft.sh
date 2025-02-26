#!/bin/bash
IFS=","
OUTDIR="/home/fosterb/coordinator_runs/coord_mpnn_multi_node_self_sub_covid_ft_last_corr_only_site_split_R1"
mkdir -p ${OUTDIR}
export NUM_GPUS_PER_NODE=1
export NUM_NODES=1
export NUM_TRIALS=1
for args in \
	dgat,0,localhost; \
	do set -- $args
	. /home/fosterb/TERMinator-public_mirror/scripts/models/train/submit_tune_gpu_dave.sh /mnt/shared/fosterb/covid_data/covid /home/fosterb/coordinator_runs/coord_mpnn_multi_node_self_sub/net_best_checkpoint.pt /home/fosterb/TERMinator-public_mirror/hparams/model/coordinator_fine_tuning.json /home/fosterb/TERMinator-public_mirror/hparams/run/fine_tuning.json ${OUTDIR} ${OUTDIR} ${OUTDIR} 240 scratch scratch scratch $1 $2 $3 /home/fosterb/TERMinator-public_mirror/models/converter_model_coord.pt
done


	#. /home/gridsan/fbirnbaum/TERMinator-public_mirror/scripts/models/train/submit_train_gpu.sh /data1/groups/keatinglab/fosterb/ingraham_data/ingraham_data_features /data1/groups/keatinglab/fosterb/ingraham_data/ingraham_features_msa_2 /home/gridsan/fbirnbaum/TERMinator-public_mirror/hparams/model/coordinator.json /home/gridsan/fbirnbaum/TERMinator-public_mirror/hparams/run/default.json ${OUTDIR} ${OUTDIR} ${OUTDIR} 240 scratch scratch scratch $1 $2 $3
