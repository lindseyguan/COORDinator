""" GNN Potts Model Encoder modules

This file contains the GNN Potts Model Encoder, as well as an ablated version of
itself. """
from __future__ import print_function
import numpy as np
import copy
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from terminator.models.layers.graph_features import MultiChainProteinFeatures, ProteinMPNNFeatures, reSSeqMPNNFeatures, MultiLayerLinear
from terminator.models.layers.s2s_modules import (EdgeMPNNLayer, EdgeTransformerLayer, NodeMPNNLayer,
                                                  NodeTransformerLayer, DecLayer)
from terminator.models.layers.utils import (cat_edge_endpoints, cat_neighbors_nodes, gather_edges, gather_nodes, insert_vectors_2D,
                                            merge_duplicate_pairE, merge_duplicate_edges, get_merge_dups_mask, gelu, _esm_featurize, MultiLayerLinear)
from terminator.utils.model.decoder_utils import _S_to_seq

# pylint: disable=no-member, not-callable
# from torch_scatter import scatter_mean


# def merge_dups(h_E, E_idx, inv_mapping):
#     orig_shape = h_E.shape
#     flattened = h_E.flatten(1, 2)
#     condensed = scatter_mean(flattened, inv_mapping, dim=1)
#     expanded_inv_mapping = inv_mapping.unsqueeze(-1).expand((-1, -1, orig_shape[-1]))
#     rescattered = torch.gather(condensed, dim=1, index=expanded_inv_mapping)
#     rescattered = rescattered.unflatten(1, (orig_shape[1], orig_shape[2]))
#     return rescattered

def merge_dups(h_E, inv_mapping):
    orig_shape = h_E.shape
    flattened = h_E.flatten(1, 2)
    # condensed = scatter_mean(flattened, inv_mapping, dim=1)
    expanded_inv_mapping = inv_mapping.unsqueeze(-1).expand((-1, -1, orig_shape[-1]))
    rescattered = torch.gather(flattened, dim=1, index=expanded_inv_mapping)
    rescattered = rescattered.unflatten(1, (orig_shape[1], orig_shape[2]))
    return rescattered

class AblatedPairEnergies(nn.Module):
    """Ablated GNN Potts Model Encoder

    Attributes
    ----------
    dev: str
        Device representing where the model is held
    hparams: dict
        Dictionary of parameter settings (see :code:`terminator/utils/model/default_hparams.py`)
    features : MultiChainProteinFeatures
        Module that featurizes a protein backbone (including multimeric proteins)
    W : nn.Linear
        Output layer that projects edge embeddings to proper output dimensionality
    """
    def __init__(self, hparams):
        """ Graph labeling network """
        super().__init__()
        hdim = hparams['energies_hidden_dim']
        self.hparams = hparams

        # Featurization layers
        self.features = MultiChainProteinFeatures(node_features=hdim,
                                                  edge_features=hdim,
                                                  top_k=hparams['k_neighbors'],
                                                  features_type=hparams['energies_protein_features'],
                                                  augment_eps=hparams['energies_augment_eps'],
                                                  dropout=hparams['energies_dropout'],
                                                  old=hparams['old'])

        self.W = nn.Linear(hparams['energies_input_dim'] * 3, hparams['energies_output_dim'])

    def forward(self, V_embed, E_embed, X, x_mask, chain_idx):
        """ Create kNN etab from TERM features, then project to proper output dimensionality.

        Args
        ----
        V_embed : torch.Tensor
            TERM node embeddings
            Shape: n_batch x n_res x n_hidden
        E_embed : torch.Tensor
            TERM edge embeddings
            Shape : n_batch x n_res x n_res x n_hidden
        X : torch.Tensor
            Backbone coordinates
            Shape: n_batch x n_res x 4 x 3
        x_mask : torch.ByteTensor
            Mask for X.
            Shape: n_batch x n_res
        chain_idx : torch.LongTensor
            Indices such that each chain is assigned a unique integer and each residue in that chain
            is assigned that integer.
            Shape: n_batch x n_res

        Returns
        -------
        etab : torch.Tensor
            Energy table in kNN dense form
            Shape: n_batch x n_res x k x n_hidden
        E_idx : torch.LongTensor
            Edge index for `etab`
            Shape: n_batch x n_res x k
        """
        # compute the kNN etab
        _, _, E_idx = self.features(X, chain_idx, x_mask)  # notably, we throw away the backbone features
        E_embed_neighbors = gather_edges(E_embed, E_idx)
        h_E = cat_edge_endpoints(E_embed_neighbors, V_embed, E_idx)
        etab = self.W(h_E)

        # merge duplicate pairEs
        n_batch, n_res, k, out_dim = etab.shape
        # ensure output etab is masked properly
        etab = etab * x_mask.view(n_batch, n_res, 1, 1)
        etab = etab.unsqueeze(-1).view(n_batch, n_res, k, 20, 20)
        etab[:, :, 0] = etab[:, :, 0] * torch.eye(20).to(etab.device) # zero off-diagonal energies
        etab = merge_duplicate_pairE(etab, E_idx)
        etab = etab.view(n_batch, n_res, k, out_dim)

        return etab, E_idx

class PairEnergies(nn.Module):
    """GNN Potts Model Encoder

    Attributes
    ----------
    dev: str
        Device representing where the model is held
    hparams: dict
        Dictionary of parameter settings (see :code:`terminator/utils/model/default_hparams.py`)
    features : MultiChainProteinFeatures
        Module that featurizes a protein backbone (including multimeric proteins)
    W_v : nn.Linear
        Embedding layer for incoming TERM node embeddings
    W_e : nn.Linear
        Embedding layer for incoming TERM edge embeddings
    edge_encoder : nn.ModuleList of EdgeTransformerLayer or EdgeMPNNLayer
        Edge graph update layers
    node_encoder : nn.ModuleList of NodeTransformerLayer or NodeMPNNLayer
        Node graph update layers
    W_out : nn.Linear
        Output layer that projects edge embeddings to proper output dimensionality
    W_proj : nn.Linear (optional)
        Output layer that projects node embeddings to proper output dimensionality.
        Enabled when :code:`hparams["node_self_sub"]=True`
    """
    def __init__(self, hparams, use_esm=False, edges_to_seq=False):
        """ Graph labeling network """
        super().__init__()
        self.hparams = hparams

        hdim = hparams['energies_hidden_dim']

        # Hyperparameters
        self.node_features = hdim
        self.edge_features = hdim
        self.input_dim = hdim
        hidden_dim = hdim
        output_dim = hparams['energies_output_dim']
        dropout = hparams['energies_dropout']
        esm_dropout = hparams['esm_dropout']
        num_encoder_layers = hparams['energies_encoder_layers']
        self.num_encoder_layers = num_encoder_layers
        self.center_node = hparams['center_node']
        self.center_node_ablation = hparams['center_node_ablation']
        self.center_node_only = hparams['center_node_only']
        self.top_k = hparams['k_neighbors']
        # Featurization layers
        if hparams['featurizer'] == 'multichain':
            E_feats = None
            if 'mpnn' in hparams['energies_protein_features']:
                E_feats = ProteinMPNNFeatures(node_features=hdim,
                                                edge_features=hdim,
                                                top_k=hparams['k_neighbors'],
                                                features_type=hparams['energies_protein_features'],
                                                augment_eps=hparams['energies_augment_eps'],
                                                dropout=hparams['energies_dropout'],
                                                esm_rep_feat_ins=hparams['esm_rep_feat_ins'],
                                                esm_rep_feat_outs=hparams['esm_rep_feat_outs'],
                                                esm_attn_feat_ins=hparams['esm_attn_feat_ins'],
                                                esm_attn_feat_outs=hparams['esm_attn_feat_outs'],
                                                only_E=True,
                                                random_type=hparams['random_type'])
            self.features = MultiChainProteinFeatures(node_features=hdim,
                                                    edge_features=hdim,
                                                    top_k=hparams['k_neighbors'],
                                                    features_type=hparams['energies_protein_features'],
                                                    augment_eps=hparams['energies_augment_eps'],
                                                    dropout=hparams['energies_dropout'],
                                                    esm_dropout=hparams['esm_dropout'],
                                                    esm_rep_feat_ins=hparams['esm_rep_feat_ins'],
                                                    esm_rep_feat_outs=hparams['esm_rep_feat_outs'],
                                                    esm_attn_feat_ins=hparams['esm_attn_feat_ins'],
                                                    esm_attn_feat_outs=hparams['esm_attn_feat_outs'],
                                                    old=hparams['old'],
                                                    E_feats=E_feats,
                                                    bias=hparams['bias'],
                                                    random_type=hparams['random_type'],
                                                    chain_handle=hparams['chain_handle'],
                                                    center_node=hparams['center_node'],
                                                    center_node_ablation=hparams['center_node_ablation'],
                                                    random_graph=hparams['random_graph'])
        elif hparams['featurizer'] == 'resseq':
            self.features = reSSeqMPNNFeatures(node_features=hdim,
                                                    edge_features=hdim,
                                                    top_k=hparams['k_neighbors'],
                                                    features_type=hparams['energies_protein_features'],
                                                    augment_eps=hparams['energies_augment_eps'],
                                                    dropout=hparams['energies_dropout'],
                                                    esm_rep_feat_ins=hparams['esm_rep_feat_ins'],
                                                    esm_rep_feat_outs=hparams['esm_rep_feat_outs'],
                                                    esm_attn_feat_ins=hparams['esm_attn_feat_ins'],
                                                    esm_attn_feat_outs=hparams['esm_attn_feat_outs'],
                                                    old=hparams['old'])
        elif hparams['featurizer'] == 'protein_mpnn':
            self.features = ProteinMPNNFeatures(node_features=hdim,
                                                edge_features=hdim,
                                                top_k=hparams['k_neighbors'],
                                                features_type=hparams['energies_protein_features'],
                                                augment_eps=hparams['energies_augment_eps'],
                                                dropout=hparams['energies_dropout'],
                                                esm_rep_feat_ins=hparams['esm_rep_feat_ins'],
                                                esm_rep_feat_outs=hparams['esm_rep_feat_outs'],
                                                esm_attn_feat_ins=hparams['esm_attn_feat_ins'],
                                                esm_attn_feat_outs=hparams['esm_attn_feat_outs'])
        self.nonlinear_features = hparams['nonlinear_features']


        # Embedding layers
        self.W_v = nn.Linear(hdim + hparams['energies_input_dim'], hdim, bias=True)
        self.W_e = nn.Linear(hdim + hparams['energies_input_dim'], hdim, bias=True)
        edge_layer = EdgeTransformerLayer if not hparams['energies_use_mpnn'] else EdgeMPNNLayer
        node_layer = NodeTransformerLayer if not hparams['energies_use_mpnn'] else NodeMPNNLayer

        # Encoder layers
        self.edge_encoder = nn.ModuleList(
            [edge_layer(hidden_dim, hidden_dim * 3, dropout=dropout) for _ in range(num_encoder_layers)])
        self.node_encoder = nn.ModuleList(
            [node_layer(hidden_dim, hidden_dim * 2, dropout=dropout) for _ in range(num_encoder_layers)])
        if self.center_node:
            self.center_node_encoder = nn.ModuleList(
                [node_layer(hidden_dim, hidden_dim * 2, dropout=dropout) for _ in range(num_encoder_layers)])

        # if enabled, generate self energies in etab from node embeddings
        if "node_self_sub" in hparams.keys() and hparams["node_self_sub"] is True:
            self.W_proj = nn.Linear(hidden_dim, 20)

        # project edges to proper output dimensionality
        self.W_out = nn.Linear(hidden_dim, output_dim, bias=True)
        self.use_esm = use_esm
        self.use_edges = hparams['use_edges_for_nodes']
        if self.use_esm:
            self.V_out = nn.Linear(hidden_dim, 20)
        self.edges_to_seq = edges_to_seq
        if edges_to_seq:
            self.E_outs = nn.ModuleList()
            ins = [600, 256, 64]
            outs = [256, 64, 1]
            for i in range(3):
                self.E_outs.append(nn.Linear(ins[i], outs[i]))
        self.ener_finetune_mlp = hparams['ener_finetune_mlp']
        if self.ener_finetune_mlp:
            self.ener_mlp = MultiLayerLinear(in_features=[400, 400, 400], out_features=[400, 400, 400], num_layers=3)

        self.post_esm_module = hparams['post_esm_module']
        self.esm_once = hparams['esm_once']            
        if self.post_esm_module:
            if self.esm_once:
                self.esm_edge_encoder = nn.ModuleList(
                [edge_layer(hidden_dim, hidden_dim * 3, dropout=esm_dropout) for _ in range(1)])
                self.esm_node_encoder = nn.ModuleList(
                [node_layer(hidden_dim, hidden_dim * 2, dropout=esm_dropout) for _ in range(1)])
            else:
                self.esm_edge_encoder = nn.ModuleList(
                [edge_layer(hidden_dim, hidden_dim * 3, dropout=esm_dropout) for _ in range(num_encoder_layers)])
                self.esm_node_encoder = nn.ModuleList(
                [node_layer(hidden_dim, hidden_dim * 2, dropout=esm_dropout) for _ in range(num_encoder_layers)])


        self.esm_module = hparams['esm_module']
        self.esm_module_only = hparams['esm_module_only']
        if self.esm_module:
            self.W_s = nn.Linear(hdim, hdim, bias=True)
            self.W_a = nn.Linear(hdim, hdim, bias=True)
            self.esm_edge_encoder = nn.ModuleList(
            [edge_layer(hidden_dim, hidden_dim * 3, dropout=esm_dropout) for _ in range(num_encoder_layers)])
            self.esm_node_encoder = nn.ModuleList(
            [node_layer(hidden_dim, hidden_dim * 2, dropout=esm_dropout) for _ in range(num_encoder_layers)])
            self.edge_combine = nn.ModuleList(
                [nn.Linear(hidden_dim*2, hidden_dim, bias=False) for _ in range(num_encoder_layers)]
            )
            self.node_combine = nn.ModuleList(
                [nn.Linear(hidden_dim*2, hidden_dim, bias=False) for _ in range(num_encoder_layers)]
            )


        self.side_chain_graph = hparams['side_chain_graph']
        if self.side_chain_graph:
            sc_hidden_dim = hparams['sc_hidden_dim']
            sc_hidden_dim = hdim
            sc_dropout = hparams['sc_dropout']
            self.sc_edge_encoder = nn.ModuleList(
                [edge_layer(sc_hidden_dim, sc_hidden_dim * 3, dropout=sc_dropout) for _ in range(num_encoder_layers)])
            self.sc_node_encoder = nn.ModuleList(
                [node_layer(sc_hidden_dim, sc_hidden_dim * 2, dropout=sc_dropout) for _ in range(num_encoder_layers)])
            self.sc_W_out = nn.Linear(sc_hidden_dim, hidden_dim)
            self.sc_V_out = nn.Linear(sc_hidden_dim, hidden_dim)
            self.sc_edge_combine = nn.ModuleList(
                [nn.Linear(sc_hidden_dim*2, sc_hidden_dim, bias=False) for _ in range(num_encoder_layers)]
            )
            self.sc_node_combine = nn.ModuleList(
                [nn.Linear(sc_hidden_dim*2, sc_hidden_dim, bias=False) for _ in range(num_encoder_layers)]
            )

        self.mpnn_decoder = hparams['mpnn_decoder']
        if self.mpnn_decoder:
            # ProteinMPNN code
            if not self.side_chain_graph:
                self.W_seq = nn.Embedding(22, hidden_dim)
            self.decoder_layers = nn.ModuleList([
                DecLayer(hidden_dim, hidden_dim*3, dropout=dropout)
                for _ in range(hparams['num_decoder_layers'])
            ])
            self.dec_out = nn.Linear(hidden_dim, 22, bias=True)
            self.teacher_forcing = hparams['teacher_forcing']
        else:
            self.teacher_forcing = False

        self.zero_node_features = hparams['zero_node_features']
        self.zero_edge_features = hparams['zero_edge_features']

        self.predict_confidence = hparams['predict_confidence']
        self.predict_blosum = hparams['predict_blosum']
        self.confidence_vector = hparams['confidence_vector']
        if self.predict_confidence:
            self.confidence_norm = torch.nn.LayerNorm([hidden_dim])
            self.confidence_module = MultiLayerLinear(in_features = [hidden_dim, hidden_dim, int(hidden_dim/2)], out_features=[hidden_dim, int(hidden_dim/2), hparams['confidence_vector_dim']], num_layers=3, last_activation=hparams['confidence_vector'], dropout=0)
            if not self.predict_blosum:
                self.confidence_activation = torch.nn.Sigmoid()
            if self.confidence_vector:
                self.confidence_activation = torch.nn.Softmax(dim=-1)

        self.pifold_decoder = hparams['pifold_decoder']
        self.restrict_output = hparams['restrict_pifold_output']
        if self.pifold_decoder:
            if self.restrict_output:
                self.dec_out = nn.Linear(hidden_dim, 20, bias=True)
            else:
                self.dec_out = nn.Linear(hidden_dim, 22, bias=True)


        self.W_dropout = torch.nn.Dropout(p=hparams['ft_dropout'])

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def adjust_edges(self, h_E, x_mask, E_idx):
        n_batch, n_res, k, out_dim = h_E.shape
        h_E = h_E * x_mask.view(n_batch, n_res, 1, 1) # ensure output etab is masked properly
        if self.hparams['energy_merge_fn'] == 'default':
            h_E = h_E.unsqueeze(-1).view(n_batch, n_res, k, 20, 20)
            h_E[:, :, 0] = h_E[:, :, 0] * torch.eye(20).to(h_E.device) # zero off-diagonal energies
            h_E = merge_duplicate_pairE(h_E, E_idx)
        elif self.hparams['energy_merge_fn'] == 'identical':
            inv_mapping, _ = get_merge_dups_mask(E_idx)
            h_E = merge_dups(h_E, E_idx, inv_mapping)
            h_E = h_E.unsqueeze(-1).view(n_batch, n_res, k, 20, 20)
            h_E[:, :, 0] = h_E[:, :, 0] * torch.eye(20).to(h_E.device) # zero off-diagonal energies

        # reshape to fit kNN output format
        h_E = h_E.view(n_batch, n_res, k, out_dim)
        # transform edges to seq shape and info
        if self.edges_to_seq:
            h_V = h_E.clone()
            n_batch, L, k, _ = h_V.shape
            h_V = h_V.unsqueeze(-1).view(n_batch, L, k, 20, 20)
            h_V = torch.transpose(h_V, 2, 3).reshape(n_batch, L, 20, k*20)
            for i, decoder in enumerate(self.E_outs):
                h_V = decoder(h_V)
                if i < len(self.E_outs):
                    h_V = gelu(h_V)
            h_V = h_V.squeeze(-1)
        return h_E

    def forward(self, V_embed, E_embed, X, S, x_mask, seq_lens, chain_idx, X_sc, sc_ids, sc_chi, x_mask_sc, sc_mask_full, esm_embs, esm_attns, chain_ends_info, use_input_decoding_order=False, decoding_order=None, finetune=False, use_transfer_model=False):
        """ Create kNN etab from backbone and TERM features, then project to proper output dimensionality.

        Args
        ----
        V_embed : torch.Tensor or None
            TERM node embeddings. None only accepted if :code:`hparams['energies_input_dim']=0`.
            Shape: n_batch x n_res x n_hidden
        E_embed : torch.Tensor or None
            TERM edge embeddings. None only accepted if :code:`hparams['energies_input_dim']=0`.
            Shape : n_batch x n_res x n_res x n_hidden
        X : torch.Tensor
            Backbone coordinates
            Shape: n_batch x n_res x 4 x 3
        x_mask : torch.ByteTensor
            Mask for X.
            Shape: n_batch x n_res
        chain_idx : torch.LongTensor
            Indices such that each chain is assigned a unique integer and each residue in that chain
            is assigned that integer.
            Shape: n_batch x n_res

        Returns
        -------
        etab : torch.Tensor
            Energy table in kNN dense form
            Shape: n_batch x n_res x k x n_hidden
        E_idx : torch.LongTensor
            Edge index for `etab`
            Shape: n_batch x n_res x k
        """
        # Prepare node and edge embeddings

        features = self.features(X, chain_idx, x_mask, seq_lens, X_sc, sc_ids, sc_chi, x_mask_sc, sc_mask_full, esm_embs, esm_attns, chain_ends_info, nonlinear=self.nonlinear_features)
        if self.side_chain_graph:
            V, E, h_SV, h_SE, E_idx = features
        elif self.post_esm_module or self.esm_module:
            V, E, E_idx, esm_embs, esm_attns = features
            if self.esm_module:
                h_S = self.W_s(esm_embs)
                h_A = self.W_a(esm_attns)
        else:
            V, E, E_idx = features
            h_S = None
            h_A = None
        if self.zero_edge_features:
            E = torch.zeros_like(E)
        if self.zero_node_features:
            V = torch.zeros_like(V)
        h_V = self.W_v(V)
        h_E = self.W_e(E)
        h_C = None
        # Graph updates
        if self.hparams['edge_merge_fn'] == 'scatter':
            merge_fn = merge_dups
            inv_mapping, _ = get_merge_dups_mask(E_idx)
        elif self.hparams['edge_merge_fn'] == 'default':
            merge_fn = merge_duplicate_edges
            inv_mapping = None
        if self.center_node:
            # x_mask = insert_vectors_2D(x_mask, torch.ones(1).to(dtype=x_mask.dtype, device=x_mask.device), seq_lens)
            if not self.center_node_ablation:
                x_mask = torch.cat([x_mask, torch.ones((x_mask.shape[0], 1), dtype=x_mask.dtype, device=x_mask.device)], dim=1)
            else:
                x_mask = torch.cat([x_mask, torch.zeros((x_mask.shape[0], 1), dtype=x_mask.dtype, device=x_mask.device)], dim=1)
        mask_attend = gather_nodes(x_mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = x_mask.unsqueeze(-1) * mask_attend
        if self.center_node:
            node_mask_attend = gather_nodes(x_mask.unsqueeze(-1), E_idx).squeeze(-1)
            node_mask_attend[torch.arange(len(seq_lens), device=mask_attend.device), seq_lens] = torch.ones(E_idx.shape[2], device=E_idx.device)
            node_mask_attend[:,-1] = 0
            if not self.center_node_ablation:
                mask_attend[torch.arange(len(seq_lens), device=mask_attend.device), seq_lens] = torch.ones(E_idx.shape[2], device=E_idx.device)
            else:
                mask_attend[torch.arange(len(seq_lens), device=mask_attend.device), seq_lens] = torch.zeros(E_idx.shape[2], device=E_idx.device)
            if self.center_node_only:
                node_mask_attend = torch.zeros_like(node_mask_attend)
                node_mask_attend[torch.arange(len(seq_lens), device=mask_attend.device), seq_lens] = torch.ones(E_idx.shape[2], device=E_idx.device)
                node_mask_attend[:,-1] = 0

                mask_attend = torch.zeros_like(mask_attend)
                mask_attend[torch.arange(len(seq_lens), device=mask_attend.device), seq_lens] = torch.ones(E_idx.shape[2], device=E_idx.device)
        if self.esm_module:
            h_V_comb = torch.zeros_like(h_V)
            h_E_comb = torch.zeros_like(h_E)
        for i, (edge_layer, node_layer) in enumerate(zip(self.edge_encoder, self.node_encoder)):
            h_EV_edges = cat_edge_endpoints(h_E, h_V, E_idx)
            h_E = edge_layer(h_E, h_EV_edges, E_idx, merge_fn, inv_mapping=inv_mapping, mask_E=x_mask, mask_attend=mask_attend)
            if not self.center_node:
                h_EV_nodes = cat_neighbors_nodes(h_V, h_E, E_idx)
                h_V = node_layer(h_V, h_EV_nodes, mask_V=x_mask, mask_attend=mask_attend)
            else:
                h_center = h_V[:,-1].unsqueeze(1).clone()
                h_EV_nodes = cat_neighbors_nodes(h_V, h_E, E_idx)
                h_V = node_layer(h_V, h_EV_nodes, mask_V=x_mask, mask_attend=node_mask_attend)
                h_EV_center = torch.cat([h_E[:,:,-1,:].unsqueeze(1), h_V.unsqueeze(1)], dim=-1)
                # h_center = h_V[torch.arange(seq_lens, device=h_V.device), seq_lens].unsqueeze(1)
                center_mask = torch.ones((h_center.shape[0], h_center.shape[1]), dtype=x_mask.dtype, device=h_center.device)
                center_attend_mask = x_mask.unsqueeze(1)
                h_V[:,-1] = self.center_node_encoder[i](h_center, h_EV_center, mask_V=center_mask, mask_attend=center_attend_mask).squeeze(1)
                
                # h_V[torch.arange(seq_lens, device=h_V.device), seq_lens] = self.center_node_encoder[i](h_center, h_EV_center, mask_V=center_mask, mask_attend=center_attend_mask)

            if self.esm_module:
                h_AS_edges = cat_edge_endpoints(h_A, h_S, E_idx)
                h_A = edge_layer(h_A, h_AS_edges, E_idx, merge_fn, inv_mapping=inv_mapping, mask_E=x_mask, mask_attend=mask_attend)
                h_AS_nodes = cat_neighbors_nodes(h_S, h_A, E_idx)
                h_S = node_layer(h_S, h_AS_nodes, mask_V=x_mask, mask_attend=mask_attend)

                if (not self.esm_once) or (self.esm_once and i == self.num_encoder_layers):
                    if self.esm_module_only:
                        h_E_comb = h_A
                        h_V_comb = h_S
                        # h_E = torch.zeros_like(h_E)
                        # h_V = torch.zeros_like(h_V)
                    else:
                        h_E_comb += self.node_combine[i](torch.cat([h_E, h_A], dim=-1))
                        h_V_comb += self.edge_combine[i](torch.cat([h_V, h_S], dim=-1))
            if self.post_esm_module and (self.esm_once and i == self.num_encoder_layers):
                h_EV_edges = cat_edge_endpoints(h_E, esm_embs, E_idx)
                h_E = self.esm_edge_encoder[i](h_E, h_EV_edges, E_idx, merge_fn, inv_mapping=inv_mapping, mask_E=x_mask, mask_attend=mask_attend)
                h_EV_nodes = cat_neighbors_nodes(esm_embs, h_E, E_idx)
                h_V = self.esm_node_encoder[i](h_V, h_EV_nodes, mask_V=x_mask, mask_attend=mask_attend)

            if self.side_chain_graph:
                
                h_SEV_edges = cat_edge_endpoints(h_SE, h_SV, E_idx)
                h_SE = self.sc_edge_encoder[i](h_SE, h_SEV_edges, E_idx, merge_fn, inv_mapping=inv_mapping, mask_E=x_mask, mask_attend=mask_attend)

                h_SEV_nodes = cat_neighbors_nodes(h_SV, h_SE, E_idx)
                h_SV = self.sc_node_encoder[i](h_SV, h_SEV_nodes, mask_V=x_mask, mask_attend=mask_attend)

                h_E = self.sc_edge_combine[i](torch.cat([h_E, h_SE], dim=-1))
                h_V = self.sc_node_combine[i](torch.cat([h_V, h_SV], dim=-1))

        # Save node embeddings if using transfer model
        if use_transfer_model:
            return h_E, h_V, h_C, E_idx
        
        if self.center_node:
            h_V = h_V[:,:-1]
            h_E = h_E[:,:-1,:-1]
            E_idx = E_idx[:,:-1,:-1]
            x_mask  = x_mask[:,:-1]


        if not self.esm_module:
            h_V_comb = h_V
            h_E_comb = h_E

        if self.mpnn_decoder:
            # ProteinMPNN code
            if not self.side_chain_graph:
                h_SV = self.W_seq(S)
            h_SEV = cat_neighbors_nodes(h_SV, h_E, E_idx)

            # Build encoder embeddings
            h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_SV), h_E, E_idx)
            h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
            if x_mask_sc.numel() > 0:
                chain_M = (1 - x_mask_sc) * x_mask
            else:
                chain_M = x_mask
            mask_1D = x_mask.view([x_mask.size(0), x_mask.size(1), 1, 1])
            if self.teacher_forcing:
                if not use_input_decoding_order:
                    decoding_order = torch.argsort((chain_M+0.0001)*(torch.abs(torch.randn(chain_M.shape, device=h_E.device)))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
                mask_size = E_idx.shape[1]
                permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
                order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size, device=h_E.device))), permutation_matrix_reverse, permutation_matrix_reverse)
                mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
                mask_bw = mask_1D * mask_attend
                mask_fw = mask_1D * (1. - mask_attend)
            else:
                mask_attend = torch.gather(chain_M.unsqueeze(-1).expand((E_idx.shape[0], E_idx.shape[1], E_idx.shape[1])), 2, E_idx).unsqueeze(-1)
                mask_fw = mask_1D * mask_attend
                mask_bw = mask_1D * (1 - mask_attend)
            
            h_EXV_encoder_fw = mask_fw * h_EXV_encoder
            for layer in self.decoder_layers:
                h_ESV = cat_neighbors_nodes(h_V, h_SEV, E_idx)
                h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
                h_V = torch.utils.checkpoint.checkpoint(layer, h_V, h_ESV, x_mask)

            # calculate confidence
            if self.predict_confidence:
                h_C = self.confidence_norm(h_V)
                h_C = self.confidence_module(h_C).squeeze(-1)
                if not self.predict_blosum or (False and self.confidence_vector):
                    h_C = self.confidence_activation(h_C)
            h_V = self.dec_out(h_V)
        elif self.pifold_decoder:
            # calculate confidence
            if self.predict_confidence:
                h_C = self.confidence_norm(h_V_comb)
                h_C = self.confidence_module(h_C).squeeze(-1)
                if not self.predict_blosum or (False and self.confidence_vector):
                    h_C = self.confidence_activation(h_C)

            # Save node embeddings if using transfer model
            # if use_transfer_model:
                # return h_E_comb, h_V_comb, h_E, h_V, h_A, h_S, h_C, E_idx
                # h_V_copy = copy.deepcopy(h_V)
                # print(h_V_copy.shape)

            h_V_comb = self.dec_out(h_V_comb)
            if self.restrict_output and (not self.hparams["node_self_sub"]):
                pad = (0, 2)
                h_V_comb = F.pad(h_V_comb, pad, "constant", 0)

        # project to output and merge duplicate pairEs
        if self.use_esm and ((not self.mpnn_decoder) and (not self.pifold_decoder)):
            if self.use_edges:
                h_V_comb = h_E_comb[:,:,0]
            h_V_comb = self.V_out(h_V_comb)

        # if use_transfer_model:
        #     # h_V = h_V_copy
        #     return h_E_comb, h_V_comb, h_E, h_V, h_A, h_S, h_C, E_idx
        if finetune:
            return h_E_comb, h_V_comb, h_C, E_idx
        
        h_E = self.W_dropout(h_E)
        h_E_comb = self.W_out(h_E_comb)
        # h_E = self.W_out_E(h_E)
        # h_A = self.W_out_A(h_A)

        h_E_comb = self.adjust_edges(h_E_comb, x_mask, E_idx)
        # h_E = self.adjust_edges(h_E, x_mask, E_idx)
        # h_A = self.adjust_edges(h_A, x_mask, E_idx)

        # if specified, generate self energies from node embeddings
        if "node_self_sub" in self.hparams.keys() and self.hparams["node_self_sub"]:
            # h_V = self.W_proj(h_V)
            h_E_comb[..., 0, :] = torch.diag_embed(h_V_comb, dim1=-2, dim2=-1).flatten(start_dim=-2, end_dim=-1)
            if self.restrict_output:
                pad = (0, 2)
                h_V_comb = F.pad(h_V_comb, pad, "constant", 0)


        if self.ener_finetune_mlp:
            h_E_comb = self.ener_mlp(h_E_comb)
        
        return h_E_comb, h_V_comb, h_C, E_idx

    ## Based on ProteinMPNN

    def log_probs(self, X, S, x_mask, chain_idx, X_sc, sc_ids, sc_chi, x_mask_sc, sc_mask_full, esm_embs, esm_attns, use_input_decoding_order=False, decoding_order=None):
        _, h_V, _, _ = self.forward(None, None, X, S, x_mask, chain_idx, X_sc, sc_ids, sc_chi, x_mask_sc, sc_mask_full, esm_embs, esm_attns, use_input_decoding_order, decoding_order)
        return F.log_softmax(h_V, dim=-1)


    def sample(self, X, S_true, x_mask, chain_idx, X_sc, sc_ids, sc_chi, x_mask_sc, sc_mask_full, esm_embs, esm_attns, chain_lens, temperature=1.0, omit_AAs_np=None, bias_AAs_np=None, omit_AA_mask=None, bias_by_res=None, pssm_coef=None, pssm_bias=None, pssm_multi=None, pssm_log_odds_flag=None, pssm_log_odds_mask=None, pssm_bias_flag=None, num_recycles=1, esm=None, batch_converter=None, esm_options={}, teacher_forcing=False):
        device = X.device
        # Prepare node and edge embeddings
        for num_recycle in range(num_recycles + 1):
            # print('esm emb A: ', torch.sum(esm_embs[0][:chain_lens[0][0]]))
            # print('esm emb B: ', torch.sum(esm_embs[0][:chain_lens[0][1]]))
            # print('x_mask_sc: ', x_mask_sc)
            features = self.features(X, chain_idx, x_mask, X_sc, sc_ids, sc_chi, x_mask_sc, sc_mask_full, esm_embs, esm_attns, nonlinear=self.nonlinear_features)
            if self.side_chain_graph:
                V, E, h_SV, h_SE, E_idx = features
            else:
                V, E, E_idx = features
            if self.zero_node_features:
                V = torch.zeros_like(V)
            if self.zero_edge_features:
                E = torch.zeros_like(E)
            h_V = self.W_v(V)
            h_E = self.W_e(E)
            # Graph updates
            if self.hparams['edge_merge_fn'] == 'scatter':
                merge_fn = merge_dups
                inv_mapping, _ = get_merge_dups_mask(E_idx)
            elif self.hparams['edge_merge_fn'] == 'default':
                merge_fn = merge_duplicate_edges
                inv_mapping = None
            mask_attend = gather_nodes(x_mask.unsqueeze(-1), E_idx).squeeze(-1)
            mask_attend = x_mask.unsqueeze(-1) * mask_attend
            for i, (edge_layer, node_layer) in enumerate(zip(self.edge_encoder, self.node_encoder)):
                h_EV_edges = cat_edge_endpoints(h_E, h_V, E_idx)
                h_E = edge_layer(h_E, h_EV_edges, E_idx, merge_fn, inv_mapping=inv_mapping, mask_E=x_mask, mask_attend=mask_attend)

                h_EV_nodes = cat_neighbors_nodes(h_V, h_E, E_idx)
                h_V = node_layer(h_V, h_EV_nodes, mask_V=x_mask, mask_attend=mask_attend)

                if self.side_chain_graph:
                    
                    h_SEV_edges = cat_edge_endpoints(h_SE, h_SV, E_idx)
                    h_SE = self.sc_edge_encoder[i](h_SE, h_SEV_edges, E_idx, merge_fn, inv_mapping=inv_mapping, mask_E=x_mask, mask_attend=mask_attend)

                    h_SEV_nodes = cat_neighbors_nodes(h_SV, h_SE, E_idx)
                    h_SV = self.sc_node_encoder[i](h_SV, h_SEV_nodes, mask_V=x_mask, mask_attend=mask_attend)

                    h_E = self.sc_edge_combine[i](torch.cat([h_E, h_SE], dim=-1))
                    h_V = self.sc_node_combine[i](torch.cat([h_V, h_SV], dim=-1))

            # Decoder uses masked self-attention
            if num_recycle > 0:
                x_mask_sc = orig_x_mask_sc
            if x_mask_sc.numel() > 0:
                chain_mask = (1 - x_mask_sc) * x_mask
            else:
                chain_mask = x_mask
            mask = x_mask
            randn = torch.randn(chain_mask.shape, device=h_E.device)
            decoding_order = torch.argsort((chain_mask+0.0001)*(torch.abs(randn))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
            mask_size = E_idx.shape[1]
            permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
            order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)
            mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
            mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
            mask_bw = mask_1D * mask_attend
            mask_fw = mask_1D * (1. - mask_attend)

            N_batch, N_nodes = X.size(0), X.size(1)
            log_probs = torch.zeros((N_batch, N_nodes, 22), device=device)
            all_probs = torch.zeros((N_batch, N_nodes, 22), device=device, dtype=torch.float32)
            h_S = torch.zeros_like(h_V, device=device)
            S = torch.zeros((N_batch, N_nodes), dtype=torch.int64, device=device)
            h_V_stack = [h_V] + [torch.zeros_like(h_V, device=device) for _ in range(len(self.decoder_layers))]
            constant = torch.tensor(omit_AAs_np, device=device)
            constant_bias = torch.tensor(bias_AAs_np, device=device)
            #chain_mask_combined = chain_mask*chain_M_pos 
            omit_AA_mask_flag = omit_AA_mask != None

            h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
            h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
            h_EXV_encoder_fw = mask_fw * h_EXV_encoder
            for t_ in range(N_nodes):
                t = decoding_order[:,t_] #[B]
                chain_mask_gathered = torch.gather(chain_mask, 1, t[:,None]) #[B]
                mask_gathered = torch.gather(mask, 1, t[:,None]) #[B]
                bias_by_res_gathered = torch.gather(bias_by_res, 1, t[:,None,None].repeat(1,1,22))[:,0,:] #[B, 21]
                if (mask_gathered==0).all(): #for padded or missing regions only
                    S_t = torch.gather(S_true, 1, t[:,None])
                else:
                    # Hidden layers
                    E_idx_t = torch.gather(E_idx, 1, t[:,None,None].repeat(1,1,E_idx.shape[-1]))
                    h_E_t = torch.gather(h_E, 1, t[:,None,None,None].repeat(1,1,h_E.shape[-2], h_E.shape[-1]))
                    h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
                    h_EXV_encoder_t = torch.gather(h_EXV_encoder_fw, 1, t[:,None,None,None].repeat(1,1,h_EXV_encoder_fw.shape[-2], h_EXV_encoder_fw.shape[-1]))
                    mask_t = torch.gather(mask, 1, t[:,None])
                    for l, layer in enumerate(self.decoder_layers):
                        # Updated relational features for future states
                        h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
                        h_V_t = torch.gather(h_V_stack[l], 1, t[:,None,None].repeat(1,1,h_V_stack[l].shape[-1]))
                        h_ESV_t = torch.gather(mask_bw, 1, t[:,None,None,None].repeat(1,1,mask_bw.shape[-2], mask_bw.shape[-1])) * h_ESV_decoder_t + h_EXV_encoder_t
                        h_V_stack[l+1].scatter_(1, t[:,None,None].repeat(1,1,h_V.shape[-1]), layer(h_V_t, h_ESV_t, mask_V=mask_t))
                    # Sampling step
                    h_V_t = torch.gather(h_V_stack[-1], 1, t[:,None,None].repeat(1,1,h_V_stack[-1].shape[-1]))[:,0]
                    logits = self.dec_out(h_V_t) / temperature
                    probs = F.softmax(logits-constant[None,:]*1e8+constant_bias[None,:]/temperature+bias_by_res_gathered/temperature, dim=-1)
                    # print(probs)
                    if pssm_bias_flag:
                        pssm_coef_gathered = torch.gather(pssm_coef, 1, t[:,None])[:,0]
                        pssm_bias_gathered = torch.gather(pssm_bias, 1, t[:,None,None].repeat(1,1,pssm_bias.shape[-1]))[:,0]
                        probs = (1-pssm_multi*pssm_coef_gathered[:,None])*probs + pssm_multi*pssm_coef_gathered[:,None]*pssm_bias_gathered
                    if pssm_log_odds_flag:
                        pssm_log_odds_mask_gathered = torch.gather(pssm_log_odds_mask, 1, t[:,None, None].repeat(1,1,pssm_log_odds_mask.shape[-1]))[:,0] #[B, 21]
                        probs_masked = probs*pssm_log_odds_mask_gathered
                        probs_masked += probs * 0.001
                        probs = probs_masked/torch.sum(probs_masked, dim=-1, keepdim=True) #[B, 21]
                    if omit_AA_mask_flag:
                        omit_AA_mask_gathered = torch.gather(omit_AA_mask, 1, t[:,None, None].repeat(1,1,omit_AA_mask.shape[-1]))[:,0] #[B, 21]
                        probs_masked = probs*(1.0-omit_AA_mask_gathered)
                        probs = probs_masked/torch.sum(probs_masked, dim=-1, keepdim=True) #[B, 21]
                    S_t = torch.multinomial(probs, 1)
                    all_probs.scatter_(1, t[:,None,None].repeat(1,1,22), (chain_mask_gathered[:,:,None,]*probs[:,None,:]).float())
                S_true_gathered = torch.gather(S_true, 1, t[:,None])
                S_t = (S_t*chain_mask_gathered+S_true_gathered*(1.0-chain_mask_gathered)).long()
                temp1 = self.W_seq(S_t)
                h_S.scatter_(1, t[:,None,None].repeat(1,1,temp1.shape[-1]), temp1)
                S.scatter_(1, t[:,None], S_t)
            if num_recycles == 0:
                continue
            orig_x_mask_sc = copy.deepcopy(x_mask_sc)
            if esm is not None:
                esm_embs = []
                for i in range(len(chain_lens)):
                    new_seq = _S_to_seq(S[i], torch.ones_like(chain_mask[i]))
                    new_seq = torch.tensor([ord(c) for c in new_seq]).to(device=X.device)
                    mask = x_mask_sc[i][:sum(chain_lens[i])].to(dtype=torch.bool)
                    base_seq = "".join(_S_to_seq(S_true[i][:sum(chain_lens[i])], torch.ones_like(chain_mask[i])))
                    base_seq = torch.tensor([ord(c) for c in base_seq]).to(device=X.device)
                    if teacher_forcing:
                        mask = torch.ones_like(mask)
                    # print('teacher forcing: ', teacher_forcing)
                    seq = torch.where(mask, base_seq, new_seq)
                    seq = ''.join([chr(int(c)) for c in seq])
                    # print('change region: ', mask)
                    # print('proposed seq: ', new_seq)
                    # print('base_seq: ', base_seq)
                    # print('updated seq: ', seq)
                    esm_emb, _ = _esm_featurize(chain_lens[i], seq, esm, batch_converter, esm_options['use_esm_attns'], esm_options['esm_embed_layer'], esm_options['from_rla'], esm_options['use_reps'], esm_options['connect_chains'], esm_options['one_hot'], dev=X.device)
                    esm_embs.append(esm_emb)
                esm_embs = pad_sequence(esm_embs, batch_first=True, padding_value = 0)
                S_new = S.to(dtype = S_true.dtype)
                # print('NSR: ', torch.sum(S_new == S_true) / torch.numel(S_new))
                
                x_mask_sc = torch.ones_like(x_mask_sc).to(device=x_mask_sc.device)
                # print(x_mask_sc == 1)
                S_true = torch.where(x_mask_sc == 1, S_true, S_new)
                
        output_dict = {"S": S, "probs": all_probs, "decoding_order": decoding_order}
        return output_dict
    
    def conditional_probs(self, X, S, x_mask, chain_idx, X_sc, sc_ids, sc_chi, x_mask_sc, sc_mask_full, esm_embs, esm_attns, backbone_only=False):
        if x_mask_sc.numel() > 0:
            chain_M = (1 - x_mask_sc) * x_mask
        else:
            chain_M = x_mask
        device = X.device
        # Prepare node and edge embeddings
        features = self.features(X, chain_idx, x_mask, X_sc, sc_ids, sc_chi, x_mask_sc, sc_mask_full, esm_embs, esm_attns, nonlinear=self.nonlinear_features)
        if self.side_chain_graph:
            V, E, h_SV, h_SE, E_idx = features
        else:
            V, E, E_idx = features
        h_V = self.W_v(V)
        h_E = self.W_e(E)
        # Graph updates
        if self.hparams['edge_merge_fn'] == 'scatter':
            merge_fn = merge_dups
            inv_mapping, _ = get_merge_dups_mask(E_idx)
        elif self.hparams['edge_merge_fn'] == 'default':
            merge_fn = merge_duplicate_edges
            inv_mapping = None
        mask_attend = gather_nodes(x_mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = x_mask.unsqueeze(-1) * mask_attend
        for i, (edge_layer, node_layer) in enumerate(zip(self.edge_encoder, self.node_encoder)):
            h_EV_edges = cat_edge_endpoints(h_E, h_V, E_idx)
            h_E = edge_layer(h_E, h_EV_edges, E_idx, merge_fn, inv_mapping=inv_mapping, mask_E=x_mask, mask_attend=mask_attend)

            h_EV_nodes = cat_neighbors_nodes(h_V, h_E, E_idx)
            h_V = node_layer(h_V, h_EV_nodes, mask_V=x_mask, mask_attend=mask_attend)

            if self.side_chain_graph:
                
                h_SEV_edges = cat_edge_endpoints(h_SE, h_SV, E_idx)
                h_SE = self.sc_edge_encoder[i](h_SE, h_SEV_edges, E_idx, merge_fn, inv_mapping=inv_mapping, mask_E=x_mask, mask_attend=mask_attend)

                h_SEV_nodes = cat_neighbors_nodes(h_SV, h_SE, E_idx)
                h_SV = self.sc_node_encoder[i](h_SV, h_SEV_nodes, mask_V=x_mask, mask_attend=mask_attend)

                h_E = self.sc_edge_combine[i](torch.cat([h_E, h_SE], dim=-1))
                h_V = self.sc_node_combine[i](torch.cat([h_V, h_SV], dim=-1))

        # Decoder uses masked self-attention
        
        if x_mask_sc.numel() > 0:
                chain_mask = (1 - x_mask_sc) * x_mask
        else:
            chain_mask = x_mask
        mask = x_mask
        randn = torch.randn(chain_mask.shape, device=h_E.device)

        # Concatenate sequence embeddings for autoregressive decoder
        h_S = self.W_seq(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
  
        chain_M_np = chain_M.cpu().numpy()
        idx_to_loop = np.argwhere(chain_M_np[0,:]==1)[:,0]
        log_conditional_probs = torch.zeros([X.shape[0], chain_M.shape[1], 22], device=device).float()

        for idx in idx_to_loop:
            h_V = torch.clone(h_V)
            order_mask = torch.zeros(chain_M.shape[1], device=device).float()
            if backbone_only:
                order_mask = torch.ones(chain_M.shape[1], device=device).float()
                order_mask[idx] = 0.
            else:
                order_mask = torch.zeros(chain_M.shape[1], device=device).float()
                order_mask[idx] = 1.
            decoding_order = torch.argsort((order_mask[None,]+0.0001)*(torch.abs(randn))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
            mask_size = E_idx.shape[1]
            permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
            order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)
            mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
            mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
            mask_bw = mask_1D * mask_attend
            mask_fw = mask_1D * (1. - mask_attend)

            h_EXV_encoder_fw = mask_fw * h_EXV_encoder
            for layer in self.decoder_layers:
                # Masked positions attend to encoder information, unmasked see. 
                h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
                h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
                h_V = layer(h_V, h_ESV, mask)

            logits = self.dec_out(h_V)
            log_probs = F.log_softmax(logits, dim=-1)
            log_conditional_probs[:,idx,:] = log_probs[:,idx,:]
        return log_conditional_probs


    def unconditional_probs(self,  X, S, x_mask, chain_idx, X_sc, sc_ids, sc_chi, x_mask_sc, sc_mask_full, esm_embs, esm_attns):
        """ Graph-conditioned sequence model """

        device = X.device
        # Prepare node and edge embeddings
        features = self.features(X, chain_idx, x_mask, X_sc, sc_ids, sc_chi, x_mask_sc, sc_mask_full, esm_embs, esm_attns, nonlinear=self.nonlinear_features)
        if self.side_chain_graph:
            V, E, h_SV, h_SE, E_idx = features
        else:
            V, E, E_idx = features
        h_V = self.W_v(V)
        h_E = self.W_e(E)
        # Graph updates
        if self.hparams['edge_merge_fn'] == 'scatter':
            merge_fn = merge_dups
            inv_mapping, _ = get_merge_dups_mask(E_idx)
        elif self.hparams['edge_merge_fn'] == 'default':
            merge_fn = merge_duplicate_edges
            inv_mapping = None
        mask_attend = gather_nodes(x_mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = x_mask.unsqueeze(-1) * mask_attend
        for i, (edge_layer, node_layer) in enumerate(zip(self.edge_encoder, self.node_encoder)):
            h_EV_edges = cat_edge_endpoints(h_E, h_V, E_idx)
            h_E = edge_layer(h_E, h_EV_edges, E_idx, merge_fn, inv_mapping=inv_mapping, mask_E=x_mask, mask_attend=mask_attend)

            h_EV_nodes = cat_neighbors_nodes(h_V, h_E, E_idx)
            h_V = node_layer(h_V, h_EV_nodes, mask_V=x_mask, mask_attend=mask_attend)

            if self.side_chain_graph:
                
                h_SEV_edges = cat_edge_endpoints(h_SE, h_SV, E_idx)
                h_SE = self.sc_edge_encoder[i](h_SE, h_SEV_edges, E_idx, merge_fn, inv_mapping=inv_mapping, mask_E=x_mask, mask_attend=mask_attend)

                h_SEV_nodes = cat_neighbors_nodes(h_SV, h_SE, E_idx)
                h_SV = self.sc_node_encoder[i](h_SV, h_SEV_nodes, mask_V=x_mask, mask_attend=mask_attend)

                h_E = self.sc_edge_combine[i](torch.cat([h_E, h_SE], dim=-1))
                h_V = self.sc_node_combine[i](torch.cat([h_V, h_SV], dim=-1))

        # Decoder uses masked self-attention
        
        if x_mask_sc.numel() > 0:
                chain_mask = (1 - x_mask_sc) * x_mask
        else:
            chain_mask = x_mask
        mask = x_mask
        randn = torch.randn(chain_mask.shape, device=h_E.device)

        # Concatenate sequence embeddings for autoregressive decoder
        h_S = self.W_seq(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        order_mask_backward = torch.zeros([X.shape[0], X.shape[1], X.shape[1]], device=device)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.decoder_layers:
            h_V = layer(h_V, h_EXV_encoder_fw, mask)

        logits = self.dec_out(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs