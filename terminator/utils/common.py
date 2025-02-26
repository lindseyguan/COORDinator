import torch.nn as nn
import torch.nn.functional as F

## Create converter
class Converter(nn.Module):
    def __init__(self, h_in, h_out):
        super(Converter, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(h_in, h_out))
        self.layers.append(nn.Linear(h_out, h_out))
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x


# zero is used as padding
AA_to_int = {
    'A': 1,
    'ALA': 1,
    'C': 2,
    'CYS': 2,
    'D': 3,
    'ASP': 3,
    'E': 4,
    'GLU': 4,
    'F': 5,
    'PHE': 5,
    'G': 6,
    'GLY': 6,
    'H': 7,
    'HIS': 7,
    'I': 8,
    'ILE': 8,
    'K': 9,
    'LYS': 9,
    'L': 10,
    'LEU': 10,
    'M': 11,
    'MET': 11,
    'N': 12,
    'ASN': 12,
    'P': 13,
    'PRO': 13,
    'Q': 14,
    'GLN': 14,
    'R': 15,
    'ARG': 15,
    'S': 16,
    'SER': 16,
    'T': 17,
    'THR': 17,
    'V': 18,
    'VAL': 18,
    'W': 19,
    'TRP': 19,
    'Y': 20,
    'TYR': 20,
    '-': 21,
    'X': 22
}

esm_list = [ 5, 23, 13,  9, 18,  6, 21, 12, 15,  4, 20, 17, 14, 16, 10,  8, 11,
          7, 22, 19, 30, 24] # Alphabetical
esm_encodings = {}
for i, e in enumerate(esm_list):
    esm_encodings[i] = e
esm_decodings = {}
for i, e in enumerate(esm_list):
    esm_decodings[e] = i

AA_to_int = {key: val - 1 for key, val in AA_to_int.items()}

int_to_AA = {y: x for x, y in AA_to_int.items() if len(x) == 1}

int_to_3lt_AA = {y: x for x, y in AA_to_int.items() if len(x) == 3}

def seq_to_ints(sequence):
    """
    Given a string of one-letter encoded AAs, return its corresponding integer encoding
    """
    return [AA_to_int[residue] for residue in sequence]


def ints_to_seq(int_list):
    return [int_to_AA[i] if i in int_to_AA.keys() else 'X' for i in int_list]

def aa_three_to_one(residue):
    return int_to_AA[AA_to_int[residue]]

def esm_convert(seq):
    return [esm_encodings[s] for s in seq]

def esm_deconvert(seq):
    return [esm_decodings[s] for s in seq]

def ints_to_seq_torch(seq):
    return "".join(ints_to_seq(seq.cpu().numpy()))

def esm_ints_to_seq_torch(seq):
    return "".join(ints_to_seq(esm_deconvert(seq)))

def ints_to_seq_normal(seq):
    return "".join(ints_to_seq(seq))

def aa_to_similar(residue):
    if residue == 'MSE':  # convert MSE (seleno-met) to MET
        residue = 'MET'
    elif residue == 'FME':  # convert MSE (n-formylmethionine) to MET
        residue = 'MET'
    elif residue == 'HIC': # convert HIC (4-methyl-histidine) to HIS
        residue = 'HIS'
    elif residue == 'SEP':  # convert SEP (phospho-ser) to SER
        residue = 'SER'
    elif residue == 'SAC':  # convert SAC (n-acetyl-ser) to SER
        residue = 'SER'
    elif residue == 'OAS':  # convert OAS (o-acetyl-ser) to SER
        residue = 'SER'
    elif residue == 'TPO':  # convert TPO (phospho-thr) to THR
        residue = 'THR'
    elif residue == 'IYT':  # convert IYT (n-alpha-acetyl-3,5-diiodotyrosyl-d-threonine) to THR
        residue = 'THR'
    elif residue == 'PTR':  # convert PTR (phospho-tyr) to TYR
        residue = 'TYR'
    elif residue == 'TYS':  # convert TYS (o-sulfo-l-tyr) to TYR
        residue = 'TYR'
    elif residue == 'CSO':  # convert CSO (hydroxy-cys) to CYS
        residue = 'CYS'
    elif residue == 'SEC':  # convert SEC (seleno-cys) to CYS
        residue = 'CYS'
    elif residue == 'CSS':  # convert CSS (s-mercaptocysteine) to CYS
        residue = 'CYS'
    elif residue == 'CAS':  # convert CAS (s-(dimethylarsenic)cysteine) to CYS
        residue = 'CYS'
    elif residue == 'CAF':  # convert CAF (s-dimethylarsinoyl-cysteine) to CYS
        residue = 'CYS'
    elif residue == 'OCS':  # convert OCS (cysteine sulfonic acid) to CYS
        residue = 'CYS'
    elif residue == 'CSD':  # convert CSD (3-sulfinoalanine) to CYS
        residue = 'CYS'
    elif residue == 'CME':  # convert CME (s,s-(2-hydroxyethyl)thiocysteine) to CYS
        residue = 'CYS'
    elif residue == 'YCM':  # convert YCM (s-(2-amino-2-oxoethyl)-l-cysteine) to CYS
        residue = 'CYS'
    elif residue == 'SAH':  # convert SAH (s-adenosyl-l-homocysteine) to CYS
        residue = 'CYS'
    elif residue == 'HYP':  # convert HYP (4-hydroxyproline) to PRO
        residue = 'PRO'
    elif residue == 'M3L':  # convert M3L (n-trimethyllysine) to LYS
        residue = 'LYS'
    elif residue == 'LLP':  # convert LLP (n'-pyridoxyl-lysine-5'-monophosphate) to LYS
        residue = 'LYS'
    elif residue == 'KPI':  # convert KPI ((2s)-2-amino-6-[(1-hydroxy-1-oxo-propan-2-ylidene)amino]hexanoic acid) to LYS
        residue = 'LYS'
    elif residue == 'KPX':  # convert KPX (lysine nz-corboxylic acid) to LYS
        residue = 'LYS'
    elif residue == 'MLY':  # convert MLY (n-dimethyl-lysine) to LYS
        residue = 'LYS'
    elif residue == 'KCX':  # convert KCX (lysine nz-carboxylic acid) to LYS
        residue = 'LYS'
    elif residue == 'PCA':  # convert PCA (pyroglutamic acid) to GLN
        residue = 'GLN'
    elif residue == 'DGL':  # convert DGL (d-glutamic acid) to GLU
        residue = 'GLU'
    elif residue == 'BHD':  # convert BHD (beta-hydroxyaspartic acid) to ASP
        residue = 'ASP'
    elif residue == 'IAS':  # convert IAS (beta-l-aspartic acid) to ASP
        residue = 'ASP'
    elif residue == 'ABA':  # convert ABA (alpha-aminobutyric acid) to ALA
        residue = 'ALA'
    elif residue == '0A9':  # convert 0A9 (methyl l-phenylalaninate) to PHE
        residue = 'PHE'
    elif residue == 'KYN':  # convert KYN (l-kynurenine) to TRP
        residue = 'TRP'
    elif residue == '0Q4':
        residue = '0Q4'
    elif residue == 'FOL':
        residue = 'FOL' 
    
    return residue

def afmm_int_to_seq(seq, res_types_dict):
    string = ""
    for s in seq:
        string += res_types_dict[s]
    return string

def calc_seq_id(seq1, seq2):
    identity = 0
    for s1, s2 in zip(seq1, seq2):
        if s1 == s2:
            identity += 1
    return identity / len(seq1)



def backwards_compat(model_hparams, run_hparams):
    if "cov_features" not in model_hparams.keys():
        model_hparams["cov_features"] = False
    if "term_use_mpnn" not in model_hparams.keys():
        model_hparams["term_use_mpnn"] = False
    if "matches" not in model_hparams.keys():
        model_hparams["matches"] = "resnet"
    if "struct2seq_linear" not in model_hparams.keys():
        model_hparams['struct2seq_linear'] = False
    if "energies_gvp" not in model_hparams.keys():
        model_hparams['energies_gvp'] = False
    if "num_sing_stats" not in model_hparams.keys():
        model_hparams['num_sing_stats'] = 0
    if "num_pair_stats" not in model_hparams.keys():
        model_hparams['num_pair_stats'] = 0
    if "contact_idx" not in model_hparams.keys():
        model_hparams['contact_idx'] = False
    if "fe_dropout" not in model_hparams.keys():
        model_hparams['fe_dropout'] = 0.1
    if "fe_max_len" not in model_hparams.keys():
        model_hparams['fe_max_len'] = 1000
    if "cie_dropout" not in model_hparams.keys():
        model_hparams['cie_dropout'] = 0.1
    if "energy_merge_fn" not in model_hparams.keys():
        model_hparams['energy_merge_fn'] = 'default'
    if "edge_merge_fn" not in model_hparams.keys():
        model_hparams["edge_merge_fn"] = 'default'
    if "featurizer" not in model_hparams.keys():
        model_hparams["featurizer"] = 'multichain'
    if "nonlinear_features" not in model_hparams.keys():
        model_hparams["nonlinear_features"] = False
    if "esm_feats" not in model_hparams.keys():
        model_hparams["esm_feats"] = False
    if "rla_feats" not in model_hparams.keys():
        model_hparams["rla_feats"] = False
    if "esm_embed_layer" not in model_hparams.keys():
        model_hparams["esm_embed_layer"] = 30
    if "esm_embed_dim" not in model_hparams.keys():
        model_hparams["esm_embed_dim"] = 640
    if "esm_rep_feat_ins" not in model_hparams.keys():
        model_hparams["esm_rep_feat_ins"] = [640]
    if "esm_rep_feat_outs" not in model_hparams.keys():
        model_hparams["esm_rep_feat_outs"] = [32]
    if "esm_attn_feat_ins" not in model_hparams.keys():
        model_hparams["esm_attn_feat_ins"] = [600, 100]
    if "esm_attn_feat_outs" not in model_hparams.keys():
        model_hparams["esm_attn_feat_outs"] = [100, 20]
    if "esm_model" not in model_hparams.keys():
        model_hparams["esm_model"] = '150'
    if "sc_info" not in run_hparams.keys():
        run_hparams["sc_info"] = 'full'
    if "use_esm_attns" not in run_hparams.keys():
        run_hparams["use_esm_attns"] = False
    if "use_reps" not in run_hparams.keys():
        run_hparams["use_reps"] = False
    if "connect_chains" not in run_hparams.keys():
        run_hparams["connect_chains"] = False
    if "use_sc" not in run_hparams.keys():
        run_hparams["use_sc"] = False
    if "old" not in model_hparams.keys():
        model_hparams['old'] = True
    if "mask_interface" not in run_hparams.keys():
        run_hparams["mask_interface"] = False
    if "num_recycles" not in run_hparams.keys():
        run_hparams["num_recycles"] = 0
    if "use_edges_for_nodes" not in model_hparams.keys():
        model_hparams["use_edges_for_nodes"] = False
    if "mask_neighbors" not in run_hparams.keys():
        run_hparams["mask_neighbors"] = False
    if "convert_to_esm" not in model_hparams.keys():
        model_hparams["convert_to_esm"] = False
    if "one_hot" not in model_hparams.keys():
        model_hparams["one_hot"] = False
    if "keep_seq_recycle" not in run_hparams.keys():
        run_hparams["keep_seq_recycle"] = True
    if "nodes_to_probs" not in model_hparams.keys():
        model_hparams["nodes_to_probs"] = True
    if "sc_screen" not in run_hparams.keys():
        run_hparams["sc_screen"] = False
    if "sc_screen_range" not in run_hparams.keys():
        run_hparams["sc_screen_range"] = []
    if "half_interface" not in run_hparams.keys():
        run_hparams["half_interface"] = True
    if "inter_cutoff" not in run_hparams.keys():
        run_hparams["inter_cutoff"] = 8
    if "msa_id_cutoff" not in run_hparams.keys():
        run_hparams["msa_id_cutoff"] = 0.5
    if "edges_to_seq" not in model_hparams.keys():
        model_hparams["edges_to_seq"] = False
    if "post_esm_mask" not in run_hparams.keys():
        run_hparams["post_esm_mask"] = False
    if "sc_mask_rate" not in run_hparams.keys():
        run_hparams['sc_mask_rate'] = 0.0
    if "sc_mask_schedule" not in run_hparams.keys():
        run_hparams["sc_mask_schedule"] = False
    if "replace_muts" not in run_hparams.keys():
        run_hparams["replace_muts"] = False
    if "mpnn_dihedrals" not in model_hparams.keys():
        model_hparams["mpnn_dihedrals"] = False
    if "suffix" not in run_hparams.keys():
        run_hparams["suffix"] = None
    if "dataset_name" not in run_hparams.keys():
        run_hparams["dataset_name"] = 'bcl2'
    if "ener_finetune_mlp" not in model_hparams.keys():
        model_hparams["ener_finetune_mlp"] = False
    if "mpnn_decoder" not in model_hparams.keys():
        model_hparams["mpnn_decoder"] = False
    if "num_decoder_layers" not in model_hparams.keys():
        model_hparams["num_decoder_layers"] = 3
    if "side_chain_graph" not in model_hparams.keys():
        model_hparams["side_chain_graph"] = False
    if "sc_hidden_dim" not in model_hparams.keys():
        model_hparams["sc_hidden_dim"] = 128
    if "sc_dropout" not in model_hparams.keys():
        model_hparams["sc_dropout"] = 0.1
    if "teacher_forcing" not in model_hparams.keys():
        model_hparams["teacher_forcing"] = True
    if "data_source" not in run_hparams.keys():
        run_hparams["data_source"] = 'bcl2'
    if "esm_dropout" not in model_hparams.keys():
        model_hparams["esm_dropout"] = 0.0
    if "keep_sc_mask_recycle" not in run_hparams.keys():
        run_hparams["keep_sc_mask_recycle"] = False
    if "all_batch" not in run_hparams.keys():
        run_hparams["all_batch"] = False
    if "bias" not in model_hparams.keys():
        model_hparams["bias"] = True
    if "only_loss_recycle" not in run_hparams.keys():
        run_hparams["only_loss_recycle"] = False
    if "zero_node_features" not in model_hparams.keys():
        model_hparams["zero_node_features"] = False
    if "zero_edge_features" not in model_hparams.keys():
        model_hparams["zero_edge_features"] = False
    if "recycle_teacher_forcing" not in run_hparams.keys():
        run_hparams["recycle_teacher_forcing"] = False
    if "keep_sc_mask_loss" not in run_hparams.keys():
        run_hparams["keep_sc_mask_loss"] = True
    if "predict_confidence" not in model_hparams.keys():
        model_hparams["predict_confidence"] = False
    if "predict_blosum" not in model_hparams.keys():
        model_hparams["predict_blosum"] = False
    if "pifold_decoder" not in model_hparams.keys():
        model_hparams["pifold_decoder"] = False
    if "recycle_confidence" not in run_hparams.keys():
        run_hparams["recycle_confidence"] = False
    if "random_type" not in model_hparams.keys():
        model_hparams["random_type"] = ''
    if "blosum_diff" not in model_hparams.keys():
        run_hparams["blosum_diff"] = False
    if "restrict_pifold_output" not in model_hparams.keys():
        model_hparams["restrict_pifold_output"] = False
    if "confidence_vector" not in model_hparams.keys():
        model_hparams["confidence_vector"] = False
    if "confidence_vector_dim" not in model_hparams.keys():
        model_hparams["confidence_vector_dim"] = 1
    if "confidence_matrix_type" not in run_hparams.keys():
        run_hparams["confidence_matrix_type"] = 'blosum'
    if "interface_pep" not in run_hparams.keys():
        run_hparams["interface_pep"] = True
    if "use_sc_mask" not in run_hparams.keys():
        run_hparams["use_sc_mask"] = True
    if "node_self_sub" not in model_hparams.keys():
        model_hparams["node_self_sub"] = False
    if "aux_loss" not in run_hparams.keys():
        run_hparams["aux_loss"] = None
    if "aux_dataset" not in run_hparams.keys():
        run_hparams["aux_dataset"] = None
    if "aux_suffix" not in run_hparams.keys():
        run_hparams["aux_suffix"] = None
    if "aux_dataset_name" not in run_hparams.keys():
        run_hparams["aux_dataset_name"] = None
    if "name_cluster" not in run_hparams.keys():
        run_hparams["name_cluster"] = False
    if "aux_name_cluster" not in run_hparams.keys():
        run_hparams["aux_name_cluster"] = False
    if "per_protein_loss" not in run_hparams.keys():
        run_hparams["per_protein_loss"] = False
    if "chain_handle" not in model_hparams.keys():
        model_hparams["chain_handle"] = ''
    if "aux_dev" not in run_hparams.keys():
        run_hparams["aux_dev"] = 'cuda:0'
    if "aux_grad" not in run_hparams.keys():
        run_hparams["aux_grad"] = True
    if "nodes_output_dim" not in model_hparams.keys():
        model_hparams["nodes_output_dim"] = 22
    if "struct_predict" not in model_hparams.keys():
        model_hparams["struct_predict"] = False
    if "data_dev" not in run_hparams.keys():
        run_hparams["data_dev"] = 'cpu'
    if "weight_inter_types" not in run_hparams.keys():
        run_hparams["weight_inter_types"] = None
    if "use_transfer_model" not in model_hparams.keys():
        model_hparams["use_transfer_model"] = False
    if "random_interface" not in run_hparams.keys():
        run_hparams["random_interface"] = False
    if "post_esm_module" not in model_hparams.keys():
        model_hparams["post_esm_module"] = False
    if "esm_once" not in model_hparams.keys():
        model_hparams["esm_once"] = False
    if "use_pepmlm" not in model_hparams.keys():
        model_hparams["use_pepmlm"] = False
    if "esm_module_only" not in model_hparams.keys():
        model_hparams["esm_module_only"] = False
    if "esm_module" not in model_hparams.keys():
        model_hparams["esm_module"] = False
    if "transfer_model" not in model_hparams.keys():
        model_hparams["transfer_model"] = None
    if "no_mask" not in run_hparams.keys():
        run_hparams["no_mask"] = False
    if "transfer_hidden_dim" not in model_hparams.keys():
        model_hparams["transfer_hidden_dim"] = 484
    if "use_pretrained_out" not in run_hparams.keys():
        run_hparams["use_pretrained_out"] = True
    if "unfreeze_esm" not in run_hparams.keys():
        run_hparams["unfreeze_esm"] = False
    if "from_pepmlm" not in model_hparams.keys():
        model_hparams["from_pepmlm"] = False
    if "multimer_structure_module" not in model_hparams.keys():
        model_hparams["multimer_structure_module"] = False
    if "mult_etabs" not in run_hparams.keys():
        run_hparams["mult_etabs"] = True
    model_hparams["mult_etabs"] = run_hparams["mult_etabs"]
    if "train_transfer_only" not in run_hparams.keys():
        run_hparams['train_transfer_only'] = False
    if "single_etab_dense" not in model_hparams.keys():
        model_hparams['single_etab_dense'] = False
    if "nrg_noise" not in run_hparams.keys():
        run_hparams["nrg_noise"] = 0
    if "transfer_use_light_attn" not in model_hparams.keys():
        model_hparams['transfer_use_light_attn'] = True
    if "transfer_num_linear" not in model_hparams.keys():
        model_hparams["transfer_num_linear"] = 2
    if "transfer_use_out" not in model_hparams.keys():
        model_hparams["transfer_use_out"] = True
    if "transfer_use_esm" not in model_hparams.keys():
        model_hparams["transfer_use_esm"] = False
    if "use_struct_predict" not in run_hparams.keys():
        run_hparams["use_struct_predict"] = False
    if "fix_multi_rate" not in run_hparams.keys():
        run_hparams["fix_multi_rate"] = False
    if "struct_dropout" not in model_hparams.keys():
        model_hparams["struct_dropout"] = 0   
    if "struct_node_dropout" not in model_hparams.keys():
        model_hparams["struct_node_dropout"] = 0   
    if "struct_edge_dropout" not in model_hparams.keys():
        model_hparams["struct_edge_dropout"] = 0   
    if "load_etab_dir" not in model_hparams.keys():
        model_hparams["load_etab_dir"] = ''
    if "ft_dropout" not in run_hparams.keys():
        run_hparams["ft_dropout"] = 0
    if "distill_checkpoint" not in model_hparams.keys():
        model_hparams["distill_checkpoint"] = ''
    if "distill_model_hparams" not in model_hparams.keys():
        model_hparams["distill_model_hparams"] = ''
    if "distill_run_hparams" not in model_hparams.keys():
        model_hparams["distill_run_hparams"] = ''
    if "struct_predict_pairs" not in model_hparams.keys():
        model_hparams["struct_predict_pairs"] = True
    if "struct_predict" not in model_hparams.keys():
        model_hparams["struct_predict"] = False
    if "loss_weight_schedule" not in run_hparams.keys():
        run_hparams["loss_weight_schedule"] = {}
    if "struct_predict_seq" not in model_hparams.keys():
        model_hparams["struct_predict_seq"] = True
    if "max_loop_tokens" not in run_hparams.keys():
        run_hparams["max_loop_tokens"] = 20000
    if "alphabetize_data" not in run_hparams.keys():
        run_hparams["alphabetize_data"] = False
    if "center_node" not in model_hparams.keys():
        model_hparams["center_node"] = False
    if "random_graph" not in model_hparams.keys():
        model_hparams["random_graph"] = False
    if "center_node_ablation" not in model_hparams.keys():
        model_hparams["center_node_ablation"] = False
    if "center_node_only" not in model_hparams.keys():
        model_hparams["center_node_only"] = False
    if "node_finetune" not in run_hparams.keys():
        run_hparams["node_finetune"] = False
        
    run_hparams['confidence_vector'] = model_hparams['confidence_vector']
    model_hparams['ft_dropout'] = run_hparams['ft_dropout']
    
    return model_hparams, run_hparams