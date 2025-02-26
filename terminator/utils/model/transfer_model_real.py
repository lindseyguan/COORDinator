import torch
import torch.nn as nn
import esm as esmlib
import os
import math

HIDDEN_DIM = 128
EMBED_DIM = 128 #2560
VOCAB_DIM = 21
ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'

MLP = True
SUBTRACT_MUT = True

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

    def __init__(self):
        super().__init__()

        HIDDEN_DIM = 128

        hid_sizes = [ HIDDEN_DIM ]
        hid_sizes += [128, 64, 32]
        hid_sizes += [ VOCAB_DIM ]

        print('MLP HIDDEN SIZES:', hid_sizes)

        self.lightattn = True
        self.condense_esm = False
        self.light_attention = LightAttention(embeddings_dim=HIDDEN_DIM)

        self.both_out = nn.Sequential()

        for sz1, sz2 in zip(hid_sizes, hid_sizes[1:]):
            self.both_out.append(nn.ReLU())
            self.both_out.append(nn.Linear(sz1, sz2))

        self.ddg_out = nn.Linear(1, 1)

    def forward(self, all_input, data):        
        preds = []
        reals = []
        for mut_seq, mut_ener in zip(data['sortcery_seqs'][0], data['sortcery_nrgs'][0]):
            mut_ener = mut_ener.to(dtype=torch.float32)
            found_pos = False
            for pos, (wc, mc) in enumerate(zip(data['seqs'][0], mut_seq)):
                if wc != mc:
                    found_pos = True
                    break
            if not found_pos:
                continue

            lin_input = all_input[pos]

            lin_input = torch.unsqueeze(torch.unsqueeze(lin_input, -1), 0)
            lin_input = self.light_attention(lin_input)

            both_input = torch.unsqueeze(self.both_out(lin_input), -1)
            ddg_out = self.ddg_out(both_input)

            ddg = ddg_out[wc][0] - ddg_out[mc][0]
            
            preds.append(ddg.unsqueeze(0))
            reals.append(mut_ener.unsqueeze(0))
        return preds, reals

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

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
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
        
        o1 = o * self.softmax(attention)
        return torch.squeeze(o1)
