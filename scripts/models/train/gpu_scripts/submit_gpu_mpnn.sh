#!/bin/bash
IFS=","
. /etc/profile.d/modules.sh
OUTDIR="/data1/groups/keatinglab/fosterb/running_COORDinator_experiments/resseq_mpnn_data_mpnnn_feat_decoder_loss_smoothed_nlcpl_re1_tf_esm_drop_0.2_olr"
mkdir -p ${OUTDIR}
export NUM_GPUS_PER_NODE=1
export NUM_NODES=1
export NUM_TRIALS=1
for args in \
	dgat,0,localhost; \
	do set -- $args
	. /home/gridsan/fbirnbaum/TERMinator-public_mirror/scripts/models/train/submit_train_gpu_mpnn.sh /data1/groups/keatinglab/ProteinMPNN/data/pdb_2021aug02 /data1/groups/keatinglab/fosterb/ingraham_data/ingraham_features_msa_2 /home/gridsan/fbirnbaum/TERMinator-public_mirror/hparams/model/coordinator.json /home/gridsan/fbirnbaum/TERMinator-public_mirror/hparams/run/default.json ${OUTDIR} ${OUTDIR} ${OUTDIR} 240 train validation test $1 $2 $3 /home/gridsan/fbirnbaum/TERMinator/analysis/converter_model_coord.pt
done

	
	# /home/gridsan/fbirnbaum/TERMinator-public_mirror/scripts/models/train/submit_train_gpu.sh /data1/groups/keatinglab/fosterb/data/multichain_features /data1/groups/keatinglab/fosterb/data/multichain_features /home/gridsan/fbirnbaum/TERMinator-public_mirror/hparams/model/coordinator.json /home/gridsan/fbirnbaum/TERMinator-public_mirror/hparams/run/default.json ${OUTDIR} ${OUTDIR} ${OUTDIR} 240 train validation test $1 $2 $3 /home/gridsan/fbirnbaum/TERMinator/analysis/converter_model_coord.pt