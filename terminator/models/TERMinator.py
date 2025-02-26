"""TERMinator models"""
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from .layers.energies.s2s import (AblatedPairEnergies, PairEnergies)
from .layers.utils import gather_edges, pad_sequence_12, expand_etab
from ..utils.model.transfer_model import TransferModel, PottsTransferModel


# pylint: disable=no-member, not-callable


class TERMinator(nn.Module):
    """TERMinator model for multichain proteins

    Attributes
    ----------
    dev: str
        Device representing where the model is held
    hparams: dict
        Dictionary of parameter settings (see :code:`terminator/utils/model/default_hparams.py`)
    bot: CondenseTERM
        TERM information condenser network
    top: PairEnergies (or appropriate variant thereof)
        GNN Potts Model Encoder network
    """
    def __init__(self, hparams, device='cuda:0', use_esm=False, edges_to_seq=False, esm=None, batch_converter=None):
        """
        Initializes TERMinator according to given parameters.

        Args
        ----
        hparams : dict
            Dictionary of parameter settings (see :code:`terminator/utils/model/default_hparams.py`)
        device : str
            Device to place model on
        """
        super().__init__()
        self.dev = device
        self.hparams = hparams
        self.use_esm = use_esm
        self.edges_to_seq = edges_to_seq
        self.hparams['energies_input_dim'] = 0

        if hparams['struct2seq_linear']:
            self.top = AblatedPairEnergies(hparams).to(self.dev)
        else:
            self.top = PairEnergies(hparams, use_esm, edges_to_seq).to(self.dev)

        if hparams['struct_predict']:
            from .layers.trunk import FoldingTrunk
            url = "https://dl.fbaipublicfiles.com/fair-esm/models/esmfold_structure_module_only_650M.pt"
            model_data = torch.hub.load_state_dict_from_url(url, progress=False, map_location="cpu")
            cfg = model_data["cfg"]["model"]
            if hparams['multimer_structure_module']:
                cfg['trunk']['structure_module']['is_multimer'] = True
            # cfg['trunk']['dropout'] = hparams['struct_dropout'] ## JFM
            # cfg['trunk']['node_dropout'] = hparams['struct_node_dropout'] ## JFM
            # cfg['trunk']['edge_dropout'] = hparams['struct_edge_dropout'] ## JFM
            self.struct_module = FoldingTrunk(**cfg.trunk).to(self.dev)

        if hparams['use_transfer_model']:
            if hparams['transfer_model'] == 'nodes': self.transfer_model = TransferModel(embeddings_dim=hparams['energies_hidden_dim'], out_dim=None)
            else: 
                self.transfer_model = PottsTransferModel(embeddings_dim=hparams['transfer_hidden_dim'], out_dim=hparams['transfer_hidden_dim'], mult_etabs=hparams['mult_etabs'], single_etab_dense=hparams['single_etab_dense'], use_light_attn=hparams['transfer_use_light_attn'], num_linear=hparams['transfer_num_linear'], use_out=hparams['transfer_use_out'], use_esm=hparams['transfer_use_esm'])

        # if hparams['run_discriminator']:
        #     from .layers.disc import SequenceDiscriminator
        #     self.disc = SequenceDiscriminator(seq_encoder=esm)

        if self.hparams['use_terms']:
            
            print(f'TERM information condenser hidden dimensionality is {self.bot.hparams["term_hidden_dim"]}')

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, data, max_seq_len, finetune=False, use_transfer_model=False):
        """Compute the Potts model parameters for the structure

        Runs the full TERMinator network for prediction.

        Args
        ----
        data : dict
            Contains the following keys:

            msas : torch.LongTensor
                Integer encoding of sequence matches.
                Shape: n_batch x n_term_res x n_matches
            features : torch.FloatTensor
                Featurization of match structural data.
                Shape: n_batch x n_term_res x n_matches x n_features(=9 by default)
            seq_lens : int np.ndarray
                1D Array of batched sequence lengths.
                Shape: n_batch
            focuses : torch.LongTensor
                Indices for TERM residues matches.
                Shape: n_batch x n_term_res
            term_lens : int np.ndarray
                2D Array of batched TERM lengths.
                Shape: n_batch x n_terms
            src_key_mask : torch.ByteTensor
                Mask for TERM residue positions padding.
                Shape: n_batch x n_term_res
            X : torch.FloatTensor
                Raw coordinates of protein backbones.
                Shape: n_batch x n_res x 4 x 3
            x_mask : torch.ByteTensor
                Mask for X.
                Shape: n_batch x n_res
            sequence : torch.LongTensor
                Integer encoding of ground truth native sequences.
                Shape: n_batch x n_res
            max_seq_len : int
                Max length of protein in the batch.
            ppoe : torch.FloatTensor
                Featurization of target protein structural data.
                Shape: n_batch x n_res x n_features(=9 by default)
            chain_idx : torch.LongTensor
                Integers indices that designate ever residue to a chain.
                Shape: n_batch x n_res
            contact_idx : torch.LongTensor
                Integers representing contact indices across all TERM residues.
                Shape: n_batch x n_term_res
            gvp_data : list of torch_geometric.data.Data
                Vector and scalar featurizations of the backbone, as required by GVP

        Returns
        -------
        etab : torch.FloatTensor
            Dense kNN representation of the energy table, with :code:`E_idx`
            denotating which energies correspond to which edge.
            Shape: n_batch x n_res x k(=30 by default) x :code:`hparams['energies_output_dim']` (=400 by default)
        E_idx : torch.LongTensor
            Indices representing edges in the kNN graph.
            Given node `res_idx`, the set of edges centered around that node are
            given by :code:`E_idx[b_idx][res_idx]`, with the `i`-th closest node given by
            :code:`E_idx[b_idx][res_idx][i]`.
            Shape: n_batch x n_res x k(=30 by default)
        """
        if self.hparams['use_terms']:
            node_embeddings, edge_embeddings = self.bot(data, max_seq_len)
        else:
            node_embeddings, edge_embeddings = None, None

        
        etab, h_V, h_C, E_idx = self.top(node_embeddings, edge_embeddings, data['X'], data['seqs'], data['x_mask'], data['seq_lens'], data['chain_idx'], data['X_sc'], data['sc_ids'], data['sc_chi'], data['x_mask_sc'], data['sc_mask_full'], data['esm_embs'], data['esm_attns'], data['chain_ends_info'], finetune=finetune, use_transfer_model=use_transfer_model)

        if self.hparams['struct_predict']:
            b, n, k, h = etab.shape
            h2 = int(np.sqrt(h))
            if self.hparams['struct_predict_pairs']:
                struct_etab = etab.clone().view(b, n, k, h2, h2)
                struct_etab = expand_etab(struct_etab, E_idx).reshape(b, n, n, h).squeeze(-1).to(torch.float32)
            else:
                struct_etab = torch.zeros((b, n, n, h, h)).to(torch.float32)
            if self.hparams['struct_predict_seq']:
                h_V_fold = h_V
            else:
                h_V_fold = torch.zeros_like(h_V)
            structure = self.struct_module(h_V_fold, struct_etab, 7*torch.ones(data['x_mask'].shape, dtype=torch.long, device=etab.device), data['pos_idx'].to(torch.long), data['x_mask'])
            frames = structure['frames']
            positions = structure['positions']
        else:
            frames = None
            positions = None

        # if self.hparams['run_discriminator']:
        #     real_prob, gen_prob = self.disc(h_V, data)
        
        if self.hparams['k_cutoff']:
            k = E_idx.shape[-1]
            k_cutoff = self.hparams['k_cutoff']
            assert k > k_cutoff > 0, f"k_cutoff={k_cutoff} must be greater than k"
            etab = etab[..., :k_cutoff, :]
            E_idx = E_idx[..., :k_cutoff]

        return etab, h_V, h_C, E_idx, frames, positions #, real_prob, gen_prob
    
    def sample(self, data):
        """
        Run ProteinMPNN-like autoregressive decoding
        """
        ouput_dict = self.top.sample(data['X'], data['seqs'], data['x_mask'], data['seq_lens'], data['chain_idx'], data['X_sc'], data['sc_ids'], data['sc_chi'], data['x_mask_sc'], data['sc_mask_full'], data['esm_embs'], data['esm_attns'])
        return ouput_dict

