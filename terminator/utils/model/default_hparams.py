""" Default hyperparameter set for TERMinator

Parameters
==========
    matches : str, default='transformer'
        How to processes singleton statistics.

        Options
            'resnet'
                process using a convolutional ResNet
            'transformer'
                process using MatchAttention
            'ablate'
                perform no processing

    term_hidden_dim : int, default=32
        Hidden dimensionality for TERM Information Condenser
        (e.g. net1, self.bot, or CondenseMSA variant)
    
    flex_hidden_dim : int, default=1
        Hidden dimensionality for Flex Information Condenser

    msa_type : str, default=''
        Type of msa data to use

    condense_options : str, default=''
        Type of condense to undirected info to use

    energies_hidden_dim : int, default=32
        Hidden dimensionality for GNN Potts Model Encoder
        (e.g. net2, self.top, or PairEnergies variant)

    gradient_checkpointing : bool, default=True
        Enable gradient checkpointing at most
        memory-intensive steps (currently, MatchAttention)

    cov_features : str, default='all_raw'
        What features to include for covariance matrix computation. See
        :code:`terminator.models.layers.condense.EdgeFeatures` for more
        information.

        Options
            'shared_learned':
                use those produced by the :code:`ResidueFeatures` module.
            'all_raw':
                concatenate a one-hot encoding of residue identity and the fed-in additional features.
            'all_learned':
                'all_raw', but fed through a dense layer.
            'aa_learned':
                use an embedding matrix for residue identity
            'aa_counts':
                use a one-hot encoding of residue identity
            'cnn':
                use a 2D convolutional neural net on fed-in matches
                (WARNING: not tested due to extreme memory consumption)

    cov_compress : str, default='ffn'
        The method the covariance matrix is compressed into a vector.

        Options
            'ffn'
                Use a 2-layer dense network
            'project'
                Use a linear transformation
            'ablate'
                Use a 0 vector

    num_pair_stats : int, default=28
        [DEPRECIATED] Number of precomputed pairwise match statistics fed into TERMinator

    num_sing_stats : int, default=0
        [DEPRECIATED] Number of precomputed singleton match statistics fed into TERMinator

    resnet_blocks : int, default=4
        Number of ResNet blocks to use if :code:`matches='resnet'`

    term_layers : int, default=4
        Number of TERM MPNN layers to use.

    flex_layers : int, default=4
        Number of flex layers to use

    term_heads : int, default=4
        Number of heads to use in TERMAttention if :code:`term_use_mpnn=False`

    conv_filter : int, default=3
        Length of convolutional filter if code:`matches='resnet'`

    matches_layers : int, default=4
        Number of Transformer layers to use in MatchesCondensor if :code:`matches='transformer'`

    matches_num_heads : int, default=4
        Number of heads to use in MatchAttention if :code:`matches='transformer'`

    k_neighbors : int, default=30
        What `k` is for kNN computation

    k_cutoff : int, default=None
        When outputting a kNN potts model, take the top :code:`k_cutoff` edges and output the truncated etab

    contact_idx : bool, default=True
        Whether or not to include contact indices in computation

    cie_dropout : float, default=0.1
        Dropout rate for sinusoidal encoding of contact index

    cie_scaling : int, default=500
        Multiplicative factor by which to scale contact indices

    cie_offset : int, default=0
        Additive factor by which to offset contact indices

    transformer_dropout : float, default=0.1
        Dropout rate for Transformers used in the TERM Information Condensor

    term_use_mpnn : bool, default=True
        If set to :code:`True`, use a feedforward network to compute TERM graph messages.
        Otherwise, update TERM graph representations using an Attention-based mechanism.

    energies_protein_features : str, default='full'
        Feature set for coordinates fed into the GNN Potts Model Encoder

    energies_augment_eps : float, default=0
        Scaling factor for Gaussian noise added to coordinates before featurization

    energies_encoder_layers : int, default=6
        Number of {node_update, edge_update} layers to include in the GNN Potts Model Encoder

    energies_dropout : float, default=0.1
        Dropout rate in the GNN Potts Model Encoder

    energies_use_mpnn : bool, default=False
        If set to :code:`True`, use a feedforward network to compute kNN graph messages.
        Otherwise, update kNN graph representations using an Attention-based mechanism.

    energies_output_dim : int, default=400
        Output dimension of GNN Potts Model Encoder

    energies_geometric : bool, default=False
        Use Torch Geometric version of GNN Potts Model Encoder instead

    energies_gvp : bool, default=False
        Use GVP version of GNN Potts Model Encoder instead

    energies_full_graph : bool, default=True
        [DEPRECIATED] Update both node and edge representations in the GNN Potts Model Encoder.
        GNN Potts Model Encoder always updates node and edge representations now,
        making this option do nothing.

    res_embed_linear : bool, default=False
        Replace the singleton matches residue embedding layer with a linear layer.

    matches_linear : bool, default=False
        Remove the Matches Condensor

    term_mpnn_linear : bool, default=False
        Remove the TERM MPNN

    struct2seq_linear : bool, default=False
        Linearize the GNN Potts Model Encoder

    use_terms : bool, default=True
        Whether or not to use the TERM Information Condensor / net1

    use_flex : bool, default=True
        Whether or not to use the Flex Information Condensor

    term_matches_cutoff : int or None, default=None
        Use the top :code:`term_matches_cutoff` TERM matches for featurization.
        If :code:`None`, apply no cutoff.

    test_term_matches_cutoff : int, optional
        Apply a different :code:`term_matches_cutoff` for validation/evaluation

    use_coords : bool, default=True
        Whether or not to use coordinate-based features in the GNN Potts Model Encoder

    train_batch_size : int or None, default=16
        Batch size for training

    shuffle : bool, default=True
        Whether to do a complete shuffle of the data

    sort_data : bool, default=False
        Create deterministic batches by sorting the data according to the
        specified length metric and creating batches from the sorted data.
        Incompatible with :code:`shuffle=True` and :code:`semi_shuffle=True`.

    shuffle : bool, default=True
        Shuffle the data completely before creating batches.
        Incompatible with :code:`sort_data=True` and :code:`semi_shuffle=True`.

    semi_shuffle : bool, default=False
        Sort the data according to the specified length metric,
        then partition the data into :code:`semi_shuffle_cluster_size`-sized partitions.
        Within each partition perform a complete shuffle. The upside is that
        batching with similar lengths reduces padding making for more efficient computation,
        but the downside is that it does a less complete shuffle.

    regularization : float, default=0
        Amount of L2 regularization to apply to the internal Adam optimizer

    max_term_res : int or None, default=55000
        When :code:`train_batch_size=None, max_term_res>0, max_seq_tokens=None`,
        batch by fitting as many datapoints as possible with the total number of
        TERM residues included below `max_term_res`.
        Calibrated using :code:`nn.DataParallel` on two V100 GPUs.

    max_seq_tokens : int or None, default=None
        When :code:`train_batch_size=None, max_term_res=None, max_seq_tokens>0`,
        batch by fitting as many datapoints as possible with the total number of
        sequence residues included below `max_seq_tokens`.

    term_dropout : str or None, default=None
        Let `t` be the number of TERM matches in the given datapoint.
        Select a random int `n` from 1 to `t`, and take a random subset `n`
        of the given TERM matches to keep. If :code:`term_dropout='keep_first'`,
        keep the first match and choose `n-1` from the rest.
        If :code:`term_dropout='all'`, choose `n` matches from all matches.

    num_features : int, default=9
        The number of non-sequence TERM-based features included per TERM residue.

    loss_config : dict of str (loss component) -> float (scaling factor)
        Dictionary that describes how to construct a loss function. An example dictionary follows:
        .. code-block :
            {
                'nlcpl': 1,
                'etab_norm_penalty': 0.01
            }

    finetune : bool
        Whether or not to train the model in finetuning mode (i.e. freezing all weights but the output layer)

    noise : float, default=0
        Std of noise to add to all backbone atoms

    patience : int, default=20
        Number of epochs to wait during early stopping

    pdb_dataset : str, default=None
        Path to directory containing .pdb files

    flex_folder : str, default=None
        Path to directory containing flex files
    
    num_ensembles : int, default=1
        Number of conformational ensembles input to model

    replicate : int, default=1
        Replicate number for setting random seeds
"""
DEFAULT_MODEL_HPARAMS = {
    'model': 'multichain',
    'matches': 'transformer',
    'term_hidden_dim': 32,
    'flex_hidden_dim': 1,
    'condense_options': '',
    'energies_hidden_dim': 32,
    'gradient_checkpointing': True,
    'cov_features': 'all_raw',
    'cov_compress': 'ffn',  #
    'num_pair_stats': 28,  #
    'num_sing_stats': 0,  #
    'resnet_blocks': 4,  #
    'term_layers': 4,  #
    'flex_layers': 4, #
    'term_heads': 4,  #
    'conv_filter': 3,  #
    'matches_layers': 4,  #
    'matches_num_heads': 4,  #
    'k_neighbors': 30,  #
    'k_cutoff': None,  #
    'contact_idx': True,  #
    'cie_dropout': 0.1,  #
    'cie_scaling': 500,  #
    'cie_offset': 0,  #
    'transformer_dropout': 0.1,  #
    'term_use_mpnn': True,  #
    'energies_protein_features': 'full',  #
    'energies_augment_eps': 0,  #
    'energies_encoder_layers': 6,  #
    'energies_dropout': 0.1,  #
    'esm_dropout': 0.0,
    'energies_use_mpnn': False,  #
    'energies_output_dim': 20 * 20,  #
    'nodes_output_dim': 22, #
    'energies_gvp': False,  #
    'energies_graphformer': False, #
    'energies_geometric': False,  #
    'energies_full_graph': True,  #
    'res_embed_linear': False,  #
    'matches_linear': False,  #
    'term_mpnn_linear': False,  #
    'struct2seq_linear': False,
    'use_terms': True,  #
    'use_flex': False,
    'use_coords': True,
    'num_features':
    len(['sin_phi', 'sin_psi', 'sin_omega', 'cos_phi', 'cos_psi', 'cos_omega', 'env', 'rmsd', 'term_len']),  #
    'chain_handle': '',
    'test_code': '',
    'edge_update': '',
    'edge_update_k': 0,
    'energies_graph_type': 'undirected',
    'loss_graph_type': 'undirected',
    'activation_layers': 'relu',
    'voxel_max': 15,
    'voxel_size': 16,
    'voxel_width': 2,
    'interaction_mlp_layers': 3,
    'interaction_mlp_in': [],
    'interaction_mlp_out': [],
    'mlp_dropout': 0,
    'skip_attention': False,
    'kernel_width': 0,
    'kernel_sigma': 1,
    'masking': '',
    'neighborhood_multiplier': 2,
    'graphformer_num_heads': 4,
    'graphformer_num_layers': 3,
    'graphformer_mlp_multiplier': 1,
    'graphformer_edge_type': 'basic',
    'graphformer_dropout': 0,
    'num_positional_embeddings': 16,
    'edge_merge_fn': 'default',
    'energy_merge_fn': 'default',
    'featurizer': 'multichain',
    'side_chain': False,
    'nonlinear_features': False,
    'esm_feats': False,
    'rla_feats': False,
    'esm_embed_layer': 30,
    'esm_embed_dim':  640,
    'esm_rep_feat_ins': [640],
    'esm_rep_feat_outs': [32],
    'esm_attn_feat_ins': [600, 100],
    'esm_attn_feat_outs': [100, 20],
    'esm_model': 150,
    'convert_to_esm': False,
    'old': False,
    'one_hot': False,
    'nodes_to_probs': False,
    'edges_to_seq': False,
    'mpnn_dihedrals': False,
    'ener_finetune_mlp': False,
    'mpnn_decoder': False,
    'pifold_decoder': False,
    'num_decoder_layers': 3,
    'side_chain_graph': False,
    'post_esm_module': False,
    'sc_hidden_dim': 128,
    'sc_dropout': 0.1,
    'teacher_forcing': True,
    'bias': True,
    'zero_node_features': False,
    'zero_edge_features': False,
    'predict_confidence': False,
    'predict_blosum': False,
    'restrict_pifold_output': False,
    'random_type': '',
    'confidence_vector_dim': 1,
    'confidence_vector': False,
    'node_self_sub': False,
    'struct_predict': False,
    'use_transfer_model': False,
    'transfer_model': None,
    'esm_once': False,
    'use_pepmlm': False,
    'esm_module': False,
    'esm_module_only': False,
    'transfer_hidden_dim': 484,
    'from_pepmlm': False,
    'multimer_structure_module': False,
    'mult_etabs': True,
    'single_etab_dense': False,
    'transfer_use_light_attn': True,
    'transfer_num_linear': 2,
    'transfer_use_out': True,
    'transfer_use_esm': False,
    'struct_dropout': 0,
    'struct_node_dropout': 0,
    'struct_edge_dropout': 0,
    'load_etab_dir': '',
    'distill_checkpoint': '',
    'distill_model_hparams': '',
    'distill_run_hparams': '',
    'struct_predict_pairs': True,
    'struct_predict_seq': True,
    'struct_predict': False,
    'center_node': False,
    'center_node_ablation': False,
    'center_node_only': False,
    'random_graph': False
}

DEFAULT_TRAIN_HPARAMS = {
    'term_matches_cutoff': None,
    # 'test_term_matches_cutoff': None,
    # ^ is an optional hparam if you want to use a different TERM matches cutoff during validation/testing vs training
    'train_batch_size': 16,
    'shuffle': True,
    'sort_data': True,
    'semi_shuffle': False,
    'regularization': 0,
    'max_term_res': 55000,
    'max_seq_tokens': None,
    'min_seq_tokens': 30,
    'term_dropout': None,
    'loss_config': {
        'nlcpl': 1
    },
    'finetune': False,
    'lr_multiplier': 1,
    'finetune_lr': 1e-6,
    'bond_length_noise_level': 0,
    'bond_angle_noise_level': 0,
    'patience': 20,
    'pdb_dataset': None,
    'flex_folder': None,
    'num_ensembles': 1,
    'msa_type': '',
    'msa_id_cutoff': 0.5,
    'msa_depth_lim': 60,
    'undirected_edge_scale': 1,
    'flex_type': '',
    'noise_level': 0.0,
    'replicate': 1,
    'noise_lim': 2,
    'use_sc': False,
    'sc_mask_rate': 0.15,
    'base_sc_mask': 0.05,
    'sc_mask': [],
    'chain_mask': False,
    'sc_mask_schedule': False,
    'sc_info': 'full',
    'sc_noise': 0,
    'mask_neighbors': False,
    'mask_interface': False,
    'half_interface': True,
    'interface_pep': True,
    'inter_cutoff': 16,
    'warmup': 4000,
    'post_esm_mask': False,
    'use_esm_attns': False,
    'use_reps': False,
    'connect_chains': True,
    'from_wds': True,
    'num_recycles': 0,
    'keep_seq_recycle': True,
    'keep_sc_mask_recycle': False,
    'sc_screen': False,
    'sc_screen_range': [],
    'replace_muts': False,
    'reload_data_every_n_epochs': 2,
    'suffix': None,
    'dataset_name': 'multichain',
    'data_source': 'multichain',
    'all_batch': False,
    'only_loss_recycle': False,
    'recycle_teacher_forcing': False,
    'recycle_confidence': False,
    'keep_sc_mask_loss': True,
    'blosum_diff': False,
    'confidence_matrix_type': 'blosum',
    'use_sc_mask': True,
    'aux_loss': None,
    'aux_dataset': None,
    'aux_suffix': None,
    'aux_dataset_name': None,
    'name_cluster': False,
    'aux_name_cluster': False,
    'per_protein_loss': False,
    'aux_dev': 'cuda:0',
    'aux_grad': False,
    'data_dev': 'cpu',
    'weight_inter_types': None,
    'random_interface': False,
    'no_mask': False,
    'use_pretrained_out': True,
    'unfreeze_esm': False,
    'train_transfer_only': False,
    'nrg_noise': 0,
    'use_struct_predict': False,
    'fix_multi_rate': False,
    'ft_dropout': 0,
    'loss_weight_schedule': {},
    'max_loop_tokens': 20000,
    'alphabetize_data': False,
    'node_finetune': False,
    }
