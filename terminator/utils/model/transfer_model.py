
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from terminator.utils.model.loss_fn import calc_eners_stability
from terminator.models.layers.utils import expand_etab, expand_etab_4d, batch_score_seq_batch_pos, batch_score_seq_batch, batch_score_seq_batch_segment
import math

def gelu(x):
    """
    This is the gelu implementation from the original ESM repo. Using F.gelu yields subtly wrong results.
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class MultiLayerLinear(nn.Module):
    def __init__(self, in_features, out_features, num_layers, dropout=0):
        super(MultiLayerLinear, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Linear(in_features[i], out_features[i]))
        self.dropout_prob = dropout
        if dropout > 0:
            self.dropout = nn.Dropout(dropout, inplace=False)
            
    def forward(self, x):
        for layer in self.layers:  
            # print('\t', x.shape)
            x = layer(x)
            x = gelu(x)
        if self.dropout_prob > 0:
            x = self.dropout(x)
        return x


class TransferModel(nn.Module):

    def __init__(self, embeddings_dim, out_dim):
        super().__init__()
       
        print('MLP HIDDEN SIZES:', embeddings_dim)

        self.light_attention = LightAttention(embeddings_dim=embeddings_dim)

        # self.both_out = nn.Sequential()

        hid_sizes = [ embeddings_dim ]
        hid_sizes += [64, 32]
        hid_sizes += [ 22 ]

        self.both_out = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hid_sizes[0], hid_sizes[1]),
            nn.ReLU(),
            nn.Linear(hid_sizes[1], hid_sizes[2]),
            nn.ReLU(),
            nn.Linear(hid_sizes[2], hid_sizes[3])
        )

        # for sz1, sz2 in zip(hid_sizes, hid_sizes[1:]):
        #     self.both_out.append(nn.ReLU())
        #     self.both_out.append(nn.Linear(sz1, sz2))

        self.ddg_out = nn.Linear(1, 1)

    def forward(self, data, all_input, edges, E_idx):        

        all_out = []
        wt_seq = data['seqs'][0]
        for i, id in enumerate(data['ids']):
            pos_out = []
            mut_pos = int(id.split('-')[1]) - 1
            wt_aa_index = wt_seq[mut_pos]
            lin_input = all_input[i][mut_pos]
            # passing vector through lightattn
            lin_input = torch.unsqueeze(torch.unsqueeze(lin_input, -1), 0)
            lin_input = self.light_attention(lin_input, data['x_mask'])

            both_input = torch.unsqueeze(self.both_out(lin_input), -1)
            ddg_out = self.ddg_out(both_input)

            for mut_seq, mut_ener in zip(data['sortcery_seqs'][i], data['sortcery_nrgs'][i]):
                aa_index = mut_seq[mut_pos]
                if torch.isnan(mut_ener):
                    pos_out.append(torch.Tensor([torch.nan]).to(device=data['X'].device)[0])
                    continue
                ddg = ddg_out[aa_index][0] - ddg_out[wt_aa_index][0]
                pos_out.append(ddg)

            all_out.append(torch.stack(pos_out))
        
        all_out = torch.stack(all_out, dim=0)

        return all_out

def get_row_col_elements(tensor, i):
    # Get the i-th row and i-th column elements
    row_elements = tensor[i, :]
    col_elements = tensor[:, i]

    # Concatenate row and column elements, excluding the (i, i) element
    concatenated = torch.cat((tensor[i,i].unsqueeze(0), row_elements[:i], row_elements[i+1:], col_elements[:i], col_elements[i+1:]))
    return concatenated


class PottsTransferModel(nn.Module):
    def __init__(self, embeddings_dim, out_dim, mult_etabs=True, single_etab_dense=False, use_light_attn=True, num_linear=2, use_out=True, use_esm=False):
        super().__init__()
       
        print('MLP HIDDEN SIZES:', embeddings_dim)

        self.use_light_attn = use_light_attn
        if self.use_light_attn:
            self.light_attention = LightAttention(embeddings_dim=embeddings_dim)

        # self.both_out = nn.Sequential()

        if num_linear == 2:

            hid_sizes = [ embeddings_dim, embeddings_dim, out_dim ]

            self.both_out = nn.Sequential(
                nn.ReLU(),
                nn.Linear(hid_sizes[0], hid_sizes[1]),
                nn.ReLU(),
                nn.Linear(hid_sizes[1], hid_sizes[2]),
            )

        elif num_linear == 1:
            hid_sizes = [ embeddings_dim, out_dim ]

            self.both_out = nn.Sequential(
                nn.ReLU(),
                nn.Linear(hid_sizes[0], hid_sizes[1]),
            )

        self.use_esm = use_esm
        if use_esm:
            self.esm_condenser = MultiLayerLinear(in_features=[2560, 1280, 640, 320], out_features=[1280, 640, 320, embeddings_dim], num_layers=4, dropout=0.25).float()

        # for sz1, sz2 in zip(hid_sizes, hid_sizes[1:]):
        #     self.both_out.append(nn.ReLU())
        #     self.both_out.append(nn.Linear(sz1, sz2))
        
        self.use_out = use_out
        if self.use_out:
            self.ddg_out = nn.Linear(1, 1)
        self.mult_etabs = mult_etabs
        self.single_etab_dense = single_etab_dense

    def forward(self, data, nodes, batch_etab, E_idx):
        b, n, k, h = batch_etab.shape
        h = int(np.sqrt(h))
        batch_etab = batch_etab.view(b, n, k, h, h)
        pad = (0, 2, 0, 2)
        batch_etab = F.pad(batch_etab, pad, "constant", 0)
        
        sort_seqs = torch.cat([data['sortcery_seqs'], data['seqs'].unsqueeze(1)], dim=1)

        ddg_outs = []        
        pos_list = []

        if self.mult_etabs:
            batch_etab = batch_etab.reshape(b,n,k,(h+2)**2)
            for i, id in enumerate(data['ids']):
                mut_pos = int(id.split('-')[1]) - 1
                etab = expand_etab_4d(batch_etab[i].unsqueeze(0), E_idx[i].unsqueeze(0)).squeeze(0)
                lin_input = etab[mut_pos]
                # passing vector through lightattn
                # lin_input = torch.unsqueeze(torch.unsqueeze(lin_input, -1), 0)
                mask = lin_input[:,0] != 0
                lin_input = lin_input.transpose(-1,-2)
                lin_input = self.light_attention(lin_input.unsqueeze(0), mask.unsqueeze(0)).transpose(-1,-2)
                both_input = torch.unsqueeze(self.both_out(lin_input), -1)
                ddg_out = self.ddg_out(both_input).squeeze(-1)
                ddg_outs.append(ddg_out)
                pos_list.append(mut_pos)
            pos_list = torch.Tensor(pos_list).to(dtype=torch.int64, device=etab.device).unsqueeze(-1).unsqueeze(-1).expand([b,sort_seqs.shape[1],1])
            etab = torch.stack(ddg_outs, 0)
            b, n, h = etab.shape
            h = int(np.sqrt(h))
            etab = etab.view(b, n, h, h)
            batch_scores = batch_score_seq_batch_pos(etab, sort_seqs, pos_list)
        elif self.single_etab_dense:
            lin_input = batch_etab[0].reshape(n,k,(h+2)**2).to(dtype=torch.float32)
            if self.use_esm:
                esm_condensed = self.esm_condenser(data['esm_embs'][0]).unsqueeze(1)
                lin_input = torch.cat([lin_input, esm_condensed], 1)
            if self.use_light_attn:
                lin_input = lin_input.transpose(-1,-2)
                lin_input = self.light_attention(lin_input, None).transpose(-1,-2)
                if self.use_esm:
                    lin_input = lin_input[:,-1,:]
            both_input = torch.unsqueeze(self.both_out(lin_input), -1)
            if self.use_out:
                ddg_out = self.ddg_out(both_input).squeeze(-1)
            else:
                ddg_out = both_input.squeeze(-1)
            etab = ddg_out.unsqueeze(0)
            b, n, k, h = etab.shape
            h = int(np.sqrt(h))
            etab = etab.view(b, n, k, h, h)
            etab = expand_etab(etab, E_idx)
            batch_scores = batch_score_seq_batch(etab, sort_seqs)
            batch_scores = batch_scores.to(dtype=torch.float32)
        else:
            etab = expand_etab(batch_etab, E_idx).squeeze(0).to(dtype=torch.float32)
            etab = etab.reshape(n,n,(h+2)**2)
            ddg_outs = []
            for mut_pos in range(etab.shape[0]):
                lin_input = etab[mut_pos]
                mask = lin_input[:,0] != 0
                lin_input = lin_input.transpose(-1,-2)
                lin_input = self.light_attention(lin_input.unsqueeze(0), mask.unsqueeze(0)).transpose(-1,-2)
                both_input = torch.unsqueeze(self.both_out(lin_input), -1)
                ddg_out = self.ddg_out(both_input).squeeze(-1)
                ddg_outs.append(ddg_out)
            etab = torch.stack(ddg_outs, 0).unsqueeze(0)
            b, n, n, h = etab.shape
            h = int(np.sqrt(h))
            etab = etab.view(b, n, n, h, h)
            batch_scores = batch_score_seq_batch(etab, sort_seqs)
            batch_scores = batch_scores.to(dtype=torch.float32)
        
        all_scores = []
        for test_ind in range(batch_scores.shape[0]):
            test_scores = batch_scores[test_ind] - batch_scores[test_ind, -1]
            test_scores = test_scores[:-1] #[~torch.isnan(ref_energies[test_ind])]
            # test_scores = batch_scores[test_ind][ref_energies[test_ind] != torch.nan].cpu().numpy()
            #test_real = ref_energies[test_ind][~torch.isnan(ref_energies[test_ind])]
            all_scores.append(test_scores)
            #all_real.append(test_real)

        all_scores = torch.stack(all_scores, 0)
        #all_real = torch.cat(all_real, 0)
        return all_scores #, all_real


class LightAttention(nn.Module):
    """Source:
    Hannes Stark et al. 2022
    https://github.com/HannesStark/protein-localization/blob/master/models/light_attention.py
    """
    def __init__(self, embeddings_dim=1024, output_dim=11, dropout=0.25, kernel_size=9, conv_dropout: float = 0.25):
        super(LightAttention, self).__init__()

        self.feature_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                             padding=kernel_size // 2)
        self.attention_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                               padding=kernel_size // 2)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(conv_dropout)

    def forward(self, x: torch.Tensor, mask, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [batch_size, embeddings_dim, sequence_length] embedding tensor that should be classified
            mask: [batch_size, sequence_length] mask corresponding to the zero padding used for the shorter sequecnes in the batch. All values corresponding to padding are False and the rest is True.
        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """
        o = self.feature_convolution(x)  # [batch_size, embeddings_dim, sequence_length]
        
        o = self.dropout(o)  # [batch_gsize, embeddings_dim, sequence_length]

        attention = self.attention_convolution(x)  # [batch_size, embeddings_dim, sequence_length]
        if mask is not None:
            attention = attention.masked_fill(mask[:, None, :] == False, -1e9)
        
        o1 = o * self.softmax(attention)
        return torch.squeeze(o1)
