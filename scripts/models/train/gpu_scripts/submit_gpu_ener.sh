#!/bin/bash
IFS=","
. /etc/profile.d/modules.sh
OUTDIR="/data1/groups/keatinglab/fosterb/running_COORDinator_experiments/resseq_mpnn_data_mpnnn_feat_decoder_loss_smoothed_nlcpl_re1_tf_esm_drop_0.2_ft_rocklin_mse_single_fix"
mkdir -p ${OUTDIR}
export NUM_GPUS_PER_NODE=1
export NUM_NODES=1
export NUM_TRIALS=1
for args in \
	dgat,0,localhost; \
	do set -- $args
	. /home/gridsan/fbirnbaum/TERMinator-public_mirror/scripts/models/train/submit_ener_gpu.sh /data1/groups/keatinglab/fosterb/rocklin_data_2022/rocklin_features /data1/groups/keatinglab/fosterb/running_COORDinator_experiments/resseq_mpnn_data_mpnnn_feat_decoder_loss_smoothed_nlcpl_re1_tf_esm_drop_0.2/net_best_checkpoint.pt /home/gridsan/fbirnbaum/TERMinator-public_mirror/hparams/model/coordinator_fine_tuning_2.json /home/gridsan/fbirnbaum/TERMinator-public_mirror/hparams/run/fine_tuning_2.json ${OUTDIR} ${OUTDIR} ${OUTDIR} 240 scratch scratch scratch $1 $2 $3 /home/gridsan/fbirnbaum/TERMinator/analysis/converter_model_coord.pt
done
	
	
	#. /home/gridsan/fbirnbaum/TERMinator-public_mirror/scripts/models/train/submit_ener_gpu.sh /data1/groups/keatinglab/fosterb/ingraham_data/ingraham_data_features /data1/groups/keatinglab/fosterb/ingraham_data/ingraham_data_features /home/gridsan/fbirnbaum/TERMinator-public_mirror/hparams/model/coordinator.json /home/gridsan/fbirnbaum/TERMinator-public_mirror/hparams/run/default.json ${OUTDIR} ${OUTDIR} ${OUTDIR} 240 scratch scratch scratch $1 $2 $3
	
	#. /home/gridsan/fbirnbaum/TERMinator-public_mirror/scripts/models/train/submit_ener_gpu.sh /data1/groups/keatinglab/fosterb/data/multichain_features /data1/groups/keatinglab/fosterb/data/multichain_features /home/gridsan/fbirnbaum/TERMinator-public_mirror/hparams/model/coordinator.json /home/gridsan/fbirnbaum/TERMinator-public_mirror/hparams/run/default.json ${OUTDIR} ${OUTDIR} ${OUTDIR} 240 scratch scratch scratch $1 $2 $3
	
	#. /home/gridsan/fbirnbaum/TERMinator-public_mirror/scripts/models/train/submit_ener_gpu.sh /data1/groups/keatinglab/fosterb/bcl2_data/bcl2_features_with_sortcery /data1/groups/keatinglab/fosterb/running_COORDinator_experiments/multichain_side_chain_half_interface_esm_rep_nlcpl/net_best_checkpoint.pt /home/gridsan/fbirnbaum/TERMinator-public_mirror/hparams/model/coordinator_fine_tuning.json /home/gridsan/fbirnbaum/TERMinator-public_mirror/hparams/run/fine_tuning.json ${OUTDIR} ${OUTDIR} ${OUTDIR} 240 scratch scratch scratch $1 $2 $3 /home/gridsan/fbirnbaum/TERMinator/analysis/
