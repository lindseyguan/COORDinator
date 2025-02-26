""" GNN Potts Model Encoder modules

This file contains the GNN Potts Model Encoder, as well as an ablated version of
itself. """
from __future__ import print_function

import torch
from torch import nn
import torch.nn.functional as F
from terminator.models.layers.graph_features import MultiChainProteinFeatures, ProteinMPNNFeatures, MultiLayerLinear
from terminator.models.layers.s2s_modules import (EdgeMPNNLayer, EdgeTransformerLayer, NodeMPNNLayer,
                                                  NodeTransformerLayer, DecLayer)
from terminator.models.layers.utils import (cat_edge_endpoints, cat_neighbors_nodes, gather_edges, gather_nodes,
                                            merge_duplicate_pairE, merge_duplicate_edges, get_merge_dups_mask, gelu)

# pylint: disable=no-member, not-callable
from torch_scatter import scatter_mean


def merge_dups(h_E, E_idx, inv_mapping):
    orig_shape = h_E.shape
    flattened = h_E.flatten(1, 2)
    condensed = scatter_mean(flattened, inv_mapping, dim=1)
    expanded_inv_mapping = inv_mapping.unsqueeze(-1).expand((-1, -1, orig_shape[-1]))
    rescattered = torch.gather(condensed, dim=1, index=expanded_inv_mapping)
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
        num_encoder_layers = hparams['energies_encoder_layers']

        # Featurization layers
        if hparams['featurizer'] == 'multichain':
            self.features = MultiChainProteinFeatures(node_features=hdim,
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

        self.mpnn_decoder = hparams['mpnn_decoder']
        self.side_chain_graph = hparams['side_chain_graph']
        if self.side_chain_graph:
            sc_hidden_dim = hparams['sc_hidden_dim']
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

        if self.mpnn_decoder:
            # ProteinMPNN code
            self.decoder_layers = nn.ModuleList([
                DecLayer(hidden_dim, hidden_dim*3, dropout=dropout)
                for _ in range(hparams['num_decoder_layers'])
            ])
            self.dec_out = nn.Linear(hidden_dim, 20, bias=True)

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, V_embed, E_embed, X, x_mask, chain_idx, X_sc, sc_ids, sc_chi, x_mask_sc, sc_mask_full, esm_embs, esm_attns):
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
        V, E, h_SV, h_SE, E_idx = self.features(X, chain_idx, x_mask, X_sc, sc_ids, sc_chi, x_mask_sc, sc_mask_full, esm_embs, esm_attns, nonlinear=self.nonlinear_features)
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
                h_SE = self.sc_edge_encoder[i](h_SV, h_SEV_edges, E_idx, merge_fn, inv_mapping=inv_mapping, mask_E=x_mask, mask_attend=mask_attend)

                h_SEV_nodes = cat_neighbors_nodes(h_SV, h_SE, E_idx)
                h_SV = self.sc_node_encoder[i](h_SV, h_SEV_nodes, mask_V=x_mask, mask_attend=mask_attend)

                h_E = self.sc_edge_combine(torch.cat([h_E, h_SE], dim=-1))
                h_V = self.sc_node_combine(torch.cat([h_V, h_SV], dim=-1))

        
        if self.mpnn_decoder:
            # ProteinMPNN code

            h_SEV = cat_neighbors_nodes(h_SV, h_E, E_idx)

            # Build encoder embeddings
            h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_SV), h_E, E_idx)
            h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

            chain_M = (1 - x_mask_sc) * x_mask
            decoding_order = torch.argsort((chain_M+0.0001)*(torch.abs(torch.randn(chain_M.shape, device=h_E.device)))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
            mask_size = E_idx.shape[1]
            permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
            order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size, device=h_E.device))), permutation_matrix_reverse, permutation_matrix_reverse)
            mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
            mask_1D = x_mask.view([x_mask.size(0), x_mask.size(1), 1, 1])
            mask_bw = mask_1D * mask_attend
            mask_fw = mask_1D * (1. - mask_attend)
            
            h_EXV_encoder_fw = mask_fw * h_EXV_encoder
            for layer in self.decoder_layers:
                h_ESV = cat_neighbors_nodes(h_V, h_SEV, E_idx)
                h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
                h_V = torch.utils.checkpoint.checkpoint(layer, h_V, h_ESV, x_mask)

            logits = self.W_out(h_V)
            log_probs = F.log_softmax(logits, dim=-1)
        else:
            log_probs = None

        # project to output and merge duplicate pairEs
        if self.use_esm:
            if self.use_edges:
                h_V = h_E[:,:,0]
            h_V = self.V_out(h_V)
        h_E = self.W_out(h_E)
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

        # if specified, generate self energies from node embeddings
        if "node_self_sub" in self.hparams.keys() and self.hparams["node_self_sub"] is True:
            h_V = self.W_proj(h_V)
            h_E[..., 0, :, :] = torch.diag_embed(h_V, dim1=-2, dim2=-1)

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
        if self.ener_finetune_mlp:
            h_E = self.ener_mlp(h_E)
        if self.mpnn_decoder:
            return h_E, h_V, E_idx, log_probs
        return h_E, h_V, E_idx


# def run_epoch_ener_ft(model, optimizer, dataloader, pep_bind_ener, args, run_hparams, model_hparams, loss_fn, dev, grad, finetune, load_model):

#     # set grads properly
#     if grad:
#         model.train()
#         if finetune: # freeze all but the last output layer
#             for (name, module) in model.named_children():
#                 if name == "top":
#                     for (n, m) in module.named_children():
#                         if n == "W_out" or "V_out" in n:
#                             m.requires_grad = True
#                         else:
#                             m.requires_grad = False
#                 else:
#                     module.requires_grad = False
#         else:
#             torch.set_grad_enabled(True)
#     else:
#         model.eval()
#         torch.set_grad_enabled(False)
#     progress = tqdm(total=len(dataloader))
#     avg_loss = 0
#     i_item = 0
#     torch.autograd.set_detect_anomaly(True)
#     for data in dataloader:
#         _to_dev(data, dev)
#         pid = data['ids'][0]
#         wt_seq = "".join(ints_to_seq_torch(data['seqs'][0]))[-20:]
#         protein_seq = "".join(ints_to_seq_torch(data['seqs'][0]))[:-20]
#         prot_name = pid.split('_')[0]
#         seqs_to_score = []
#         seq_eners = {}
#         for i_seq, seq in enumerate(pep_bind_ener[prot_name+"_SORTCERY"]):
#             #dTERMen pep are shorter, so need to adjust
#             seq_to_score = str(seq)[2:CHAIN_LENGTH+2]
#             seqs_to_score.append(seq_to_score)
#             seq_eners[seq_to_score] = torch.from_numpy(np.array([pep_bind_ener[prot_name+"_SORTCERY"][str(seq)]])).to(device=dev, dtype=data['X'].dtype)[0]
#         if args.replace_muts:
#             eners = []
#             args.sc_mask = []
#             args.sc_mask_rate = 0
#             args.base_sc_mask = 0
#             args.subset_list = [pid]
#             args.data_only = True
#             args.no_mask = True
#             args.sc_screen = False
#             if model_hparams['esm_feats']:
#                 if model_hparams['esm_model'] == '650':
#                     esm, alphabet = esmlib.pretrained.esm2_t33_650M_UR50D()
#                 else:
#                     esm, alphabet = esmlib.pretrained.esm2_t30_150M_UR50D()
#                 esm = esm.to(device='cuda:0')
#                 args.batch_converter = alphabet.get_batch_converter()
#                 args.esm = esm.eval()
#             else:
#                 args.batch_converter = None
#                 args.esm = None
#             new_dataloader = load_model.load_model(args)
#             for new_data in new_dataloader:
#                 break
#             _to_dev(new_data, dev)
#             muts = []
#             masked_muts = []
#             sc_masks = []
#             for seq in seqs_to_score:
#                 mut_pos = {}
#                 for i, (mc, wc) in enumerate(zip(seq, wt_seq)):
#                     if mc != wc:
#                         mut_pos[i] = mc
#                 args.sc_mask = list(mut_pos.keys())
#                 sc_masks.append([len(protein_seq) + scm for scm in args.sc_mask])
#                 mut = list(copy.deepcopy(wt_seq))
#                 masked_mut = list(copy.deepcopy(wt_seq))
#                 for i, mc in mut_pos.items():
#                     mut[i] = mc
#                     masked_mut[i] = 'X'
#                 mut = "".join(mut)
#                 masked_mut = "".join(masked_mut)
#                 muts.append(mut)
#                 masked_muts.append(masked_mut)
                
#             avg_loss_pp = 0
#             for i_seq, seq in enumerate(seqs_to_score):
#                 cdata = copy.deepcopy(new_data)
#                 args.sc_mask = torch.from_numpy(np.array(sc_masks[i_seq]))
#                 esm_seq = protein_seq + masked_muts[i_seq]
#                 if model_hparams['esm_feats']:
#                     esm_emb, esm_attn = get_esm_embs(data['chain_lens'][0], data['X'][0],   
#                                                         esm_seq, args.esm, args.batch_converter, run_hparams['use_esm_attns'],
#                                                         model_hparams['esm_embed_layer'], model_hparams['rla_feats'], run_hparams['use_reps'], 
#                                                         run_hparams['connect_chains'], model_hparams['one_hot'], dev)
#                 if torch.numel(cdata['X_sc']) > 0:
#                     cdata['X_sc'][0][args.sc_mask] = -1
#                     cdata['sc_chi'][0][args.sc_mask] = -1000
#                     cdata['sc_ids'][0][args.sc_mask] = -1
#                     cdata['x_mask_sc'][0][args.sc_mask] = 0
#                     cdata['sc_mask_full'][0][args.sc_mask] = False
#                     if model_hparams['esm_feats']:
#                         if run_hparams['post_esm_mask']:
#                             esm_emb *= cdata['x_mask_sc'].unsqueeze(-1)
#                         cdata['esm_embs'] = esm_emb
#                         cdata['esm_attns'] = esm_attn
#                 etab, idx = get_out(model, cdata, dev=dev)
#                 eners, _, _ = calc_eners(etab, idx, data, all_peptide_seqs=[seq], all_ref_energies=[seq_eners[seq]])
#                 ener = eners[0]
#                 loss = loss_fn(ener, seq_eners[seq])
#                 avg_loss_pp += loss.item()
#                 if grad:
#                     optimizer.zero_grad()
#                     loss.backward(retain_graph=True)
#                     optimizer.step()
                              
#         else:
#             etab, idx = get_out(model, data, dev=dev)
            
#             avg_loss_pp = 0
#             for seq in seqs_to_score:
#                 eners, _, _ = calc_eners(etab, idx, data, all_peptide_seqs=[seq], all_ref_energies=[seq_eners[seq]])
#                 loss = loss_fn(eners[0], seq_eners[seq])
#                 avg_loss_pp += loss.item()
#                 if grad:
#                     optimizer.zero_grad()
#                     loss.backward(retain_graph=True)
#                     optimizer.step()
        
#         avg_loss_pp /= len(seqs_to_score)
#         i_item += 1
#         avg_loss += avg_loss_pp
#         progress.update(1)
#         progress.refresh()
#         progress.set_description_str(f'avg loss {avg_loss / i_item}')
#         progress.close()
            
#     return avg_loss / i_item