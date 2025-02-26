""" Util functions useful in TERMinator modules """
import sys
import time
import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import re

# pylint: disable=no-member
ALPHABET='ACDEFGHIKLMNPQRSTVWY-X'

def gelu(x):
    """
    This is the gelu implementation from the original ESM repo. Using F.gelu yields subtly wrong results.
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class MultiLayerLinear(nn.Module):
    def __init__(self, in_features, out_features, num_layers, activation_layers='relu', last_activation=True, dropout=0):
        super(MultiLayerLinear, self).__init__()
        self.activation_layers = activation_layers
        self.last_activation = last_activation
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Linear(in_features[i], out_features[i]))
        self.dropout_prob = dropout
        if dropout > 0:
            self.dropout = nn.Dropout(dropout, inplace=False)
            
    def forward(self, x):
        for i, layer in enumerate(self.layers):  
            # print('\t', x.shape)
            x = layer(x)
            if (i == len(self.layers) - 1) and (not self.last_activation):
                continue
            if self.activation_layers == 'relu':
                x = F.relu(x)
            else:
                x = gelu(x)
        if self.dropout_prob > 0:
            x = self.dropout(x)
        return x

def get_msa_paired_stats(msa, E_idx):
    pair_etab = torch.zeros((len(msa[0])), E_idx.shape[-1], 22, 22)
    for i_pos in range(len(msa[0])):
        for ii_pos, j_pos in enumerate(E_idx[i_pos]):
            dup = False
            if i_pos in E_idx[j_pos]:
                try:
                    jj_pos = (E_idx[j_pos] == i_pos).nonzero(as_tuple=True)[0].item()
                except Exception as e:
                    jj_pos = (E_idx[j_pos] == i_pos).nonzero(as_tuple=True)[0][0].item()
                dup = True
            if dup and j_pos > i_pos:
                continue
            cur_pos_etab = torch.zeros((22,22))
            for seq in msa:
                cur_pos_etab[seq[i_pos], seq[j_pos]] += 1
            cur_pos_etab = torch.div(cur_pos_etab, msa.shape[0])
            pair_etab[i_pos, ii_pos] = cur_pos_etab
            if dup:
                pair_etab[j_pos, jj_pos] = cur_pos_etab.transpose(-1,-2)
    return pair_etab

# batchify functions



def pad_sequence_12(sequences, padding_value=0):
    """Given a sequence of tensors, batch them together by pads both dims 1 and 2 to max length.

    Args
    ----
    sequences : list of torch.Tensor
        Sequence of tensors with number of axes `N >= 2`
    padding value : int, default=0
        What value to pad the tensors with

    Returns
    -------
    out_tensor : torch.Tensor
        Batched tensor with shape (n_batch, max_dim1, max_dim2, ...)
    """
    n_batches = len(sequences)
    out_dims = list(sequences[0].size())
    dim1, dim2 = 0, 1
    max_dim1 = max([s.size(dim1) for s in sequences])
    max_dim2 = max([s.size(dim2) for s in sequences])
    out_dims[dim1] = max_dim1
    out_dims[dim2] = max_dim2
    out_dims = [n_batches] + out_dims

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        len1 = tensor.size(0)
        len2 = tensor.size(1)
        # use index notation to prevent duplicate references to the tensor
        out_tensor[i, :len1, :len2, ...] = tensor

    return out_tensor

def insert_vectors_4D(tensor, vectors, positions):
    """
    Efficiently insert `b` vectors into a batched tensor at specified positions using torch.scatter.

    Args:
        tensor (torch.Tensor): The original tensor of shape (b, n, k, h).
        vectors (torch.Tensor): The vectors to insert of shape (b, k, h).
        positions (torch.Tensor): The positions to insert each vector, of shape (b,).

    Returns:
        torch.Tensor: The updated tensor with vectors inserted at specified positions.
    """
    tensor = F.pad(tensor, (0, 0, 0, 0, 0, 1), 'constant', 0)
    # Create index tensor for scatter
    batch_indices = torch.arange(len(positions), device=tensor.device)  # Shape: (n,)

    # Scatter vectors into the tensor
    tensor[batch_indices, positions] = vectors  # Direct assignment
    
    return tensor

def insert_vectors_3D(tensor, vectors, positions):
    """
    Efficiently insert `b` vectors into a batched tensor at specified positions using torch.scatter.

    Args:
        tensor (torch.Tensor): The original tensor of shape (b, n, h).
        vectors (torch.Tensor): The vectors to insert of shape (b, h).
        positions (torch.Tensor): The positions to insert each vector, of shape (b,).

    Returns:
        torch.Tensor: The updated tensor with vectors inserted at specified positions.
    """
    tensor = F.pad(tensor, (0, 0, 0, 1), 'constant', 0)
    # Create index tensor for scatter
    batch_indices = torch.arange(len(positions), device=tensor.device)  # Shape: (n,)

    # Scatter vectors into the tensor
    tensor[batch_indices, positions] = vectors  # Direct assignment
    
    return tensor

def insert_vectors_2D(tensor, vectors, positions):
    """
    Efficiently insert `b` vectors into a batched tensor at specified positions using torch.scatter.

    Args:
        tensor (torch.Tensor): The original tensor of shape (b, n).
        vectors (torch.Tensor): The vectors to insert of shape (b).
        positions (torch.Tensor): The positions to insert each vector, of shape (b,).

    Returns:
        torch.Tensor: The updated tensor with vectors inserted at specified positions.
    """
    tensor = F.pad(tensor, (0, 1), 'constant', 0)
    # Create index tensor for scatter
    batch_indices = torch.arange(len(positions), device=tensor.device)  # Shape: (n,)

    # Scatter vectors into the tensor
    tensor[batch_indices, positions] = vectors  # Direct assignment
    
    return tensor


def batchify(batched_flat_terms, term_lens):
    """ Take a flat representation of TERM information and batch them into a stacked representation.

    In the TERM information condensor, TERM information is initially stored by concatenating all
    TERM tensors side by side in one dimension. However, for message passing, it's convenient to batch
    these TERMs by splitting them and stacking them in a new dimension.

    Args
    ----
    batched_flat_terms : torch.Tensor
        Tensor with shape :code:`(n_batch, sum_term_len, ...)`
    term_lens : list of (list of int)
        Length of each TERM per protein

    Returns
    -------
    batchify_terms : torch.Tensor
        Tensor with shape :code:`(n_batch, max_num_terms, max_term_len, ...)`
    """
    n_batches = batched_flat_terms.shape[0]
    flat_terms = torch.unbind(batched_flat_terms)
    list_terms = [torch.split(flat_terms[i], term_lens[i]) for i in range(n_batches)]
    padded_terms = [pad_sequence(terms) for terms in list_terms]
    padded_terms = [term.transpose(0, 1) for term in padded_terms]
    batchify_terms = pad_sequence_12(padded_terms)
    return batchify_terms


# gather and cat functions
# struct level


def gather_edges(edges, neighbor_idx):
    """ Gather the edge features of the nearest neighbors.

    From https://github.com/jingraham/neurips19-graph-protein-design

    Args
    ----
    edges : torch.Tensor
        The edge features in dense form
        Shape: n_batch x n_res x n_res x n_hidden
    neighbor_idx : torch.LongTensor
        kNN sparse edge indices
        Shape : n_batch x n_res x k

    Returns
    -------
    edge_features : torch.Tensor
        The gathered edge features
        Shape : n_batch x n_res x k x n_hidden
    """
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features


def gather_nodes(nodes, neighbor_idx):
    """ Gather node features of nearest neighbors.

    From https://github.com/jingraham/neurips19-graph-protein-design

    Args
    ----
    nodes : torch.Tensor
        The node features for all nodes
        Shape: n_batch x n_res x n_hidden
    neighbor_idx : torch.LongTensor
        kNN sparse edge indices
        Shape : n_batch x n_res x k

    Returns
    -------
    neighbor_features : torch.Tensor
        The gathered neighbor node features
        Shape : n_batch x n_res x k x n_hidden
    """
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features

def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    """ Concatenate node features onto the ends of gathered edge features given kNN sparse edge indices

    From https://github.com/jingraham/neurips19-graph-protein-design

    Args
    ----
    h_nodes : torch.Tensor
        The node features for all nodes
        Shape: n_batch x n_res x n_hidden
    h_neighbors : torch.Tensor
        The gathered edge features
        Shape: n_batch x n_res x k x n_hidden
    E_idx : torch.LongTensor
        kNN sparse edge indices
        Shape : n_batch x n_res x k

    Returns
    -------
    h_nn : torch.Tensor
        The gathered concatenated node and edge features
        Shape : n_batch x n_res x k x n_hidden
    """
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn


def cat_edge_endpoints(h_edges, h_nodes, E_idx):
    """ Concatenate both node features onto the ends of gathered edge features given kNN sparse edge indices

    Args
    ----
    h_edges : torch.Tensor
        The gathered edge features
        Shape: n_batch x n_res x k x n_hidden
    h_nodes : torch.Tensor
        The node features for all nodes
        Shape: n_batch x n_res x n_hidden
    E_idx : torch.LongTensor
        kNN sparse edge indices
        Shape : n_batch x n_res x k

    Returns
    -------
    h_nn : torch.Tensor
        The gathered concatenated node and edge features
        Shape : n_batch x n_res x k x n_hidden
    """
    # Neighbor indices E_idx [B,N,K]
    # Edge features h_edges [B,N,N,C]
    # Node features h_nodes [B,N,C]
    k = E_idx.shape[-1]

    h_i_idx = E_idx[:, :, 0].unsqueeze(-1).expand(-1, -1, k).contiguous()
    h_j_idx = E_idx

    h_i = gather_nodes(h_nodes, h_i_idx)
    h_j = gather_nodes(h_nodes, h_j_idx)

    # output features [B, N, K, 3C]
    h_nn = torch.cat([h_i, h_j, h_edges], -1)
    return h_nn


def gather_pairEs(pairEs, neighbor_idx):
    """ Gather the pair energies features of the nearest neighbors.

    From https://github.com/jingraham/neurips19-graph-protein-design

    Args
    ----
    pairEs : torch.Tensor
        The pair energies in dense form
        Shape: n_batch x n_res x n_res x n_aa x n_aa
    neighbor_idx : torch.LongTensor
        kNN sparse edge indices
        Shape : n_batch x n_res x k

    Returns
    -------
    pairE_features : torch.Tensor
        The gathered pair energies
        Shape : n_batch x n_res x k x n_aa x n_aa
    """
    n_aa = pairEs.size(-1)
    neighbors = neighbor_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, n_aa, n_aa)
    pairE_features = torch.gather(pairEs, 2, neighbors)
    return pairE_features


# term level


def gather_term_nodes(nodes, neighbor_idx):
    """ Gather TERM node features of nearest neighbors.

    Adatped from https://github.com/jingraham/neurips19-graph-protein-design

    Args
    ----
    nodes : torch.Tensor
        The node features for all nodes
        Shape: n_batch x n_terms x n_res x n_hidden
    neighbor_idx : torch.LongTensor
        kNN sparse edge indices
        Shape : n_batch x n_terms x n_res x k

    Returns
    -------
    neighbor_features : torch.Tensor
        The gathered neighbor node features
        Shape : n_batch x n_terms x n_res x k x n_hidden
    """
    # Features [B,T,N,C] at Neighbor indices [B,T,N,K] => [B,T,N,K,C]
    # Flatten and expand indices per batch [B,T,N,K] => [B,T,NK] => [B,T,NK,C]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], neighbor_idx.shape[1], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, -1, nodes.size(3))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 2, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:4] + [-1])
    return neighbor_features


def gather_term_edges(edges, neighbor_idx):
    """ Gather the TERM edge features of the nearest neighbors.

    From https://github.com/jingraham/neurips19-graph-protein-design

    Args
    ----
    edges : torch.Tensor
        The edge features in dense form
        Shape: n_batch x n_terms x n_res x n_res x n_hidden
    neighbor_idx : torch.LongTensor
        kNN sparse edge indices
        Shape : n_batch x n_terms x n_res x k

    Returns
    -------
    edge_features : torch.Tensor
        The gathered edge features
        Shape : n_batch x n_terms x n_res x k x n_hidden
    """
    # Features [B,T,N,N,C] at Neighbor indices [B,T,N,K] => Neighbor features [B,T,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 3, neighbors)
    return edge_features


def cat_term_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    """ Concatenate node features onto the ends of gathered edge features given kNN sparse edge indices

    From https://github.com/jingraham/neurips19-graph-protein-design

    Args
    ----
    h_nodes : torch.Tensor
        The node features for all nodes
        Shape: n_batch x n_terms x n_res x n_hidden
    h_neighbors : torch.Tensor
        The gathered edge features
        Shape: n_batch x n_terms x n_res x k x n_hidden
    E_idx : torch.LongTensor
        kNN sparse edge indices
        Shape : n_batch x n_terms x n_res x k

    Returns
    -------
    h_nn : torch.Tensor
        The gathered concatenated node and edge features
        Shape : n_batch x n_terms x n_res x k x n_hidden
    """
    h_nodes = gather_term_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn


def cat_term_edge_endpoints(h_edges, h_nodes, E_idx):
    """ Concatenate both node features onto the ends of gathered edge features given kNN sparse edge indices

    Args
    ----
    h_edges : torch.Tensor
        The gathered edge features
        Shape: n_batch x n_terms x n_res x k x n_hidden
    h_nodes : torch.Tensor
        The node features for all nodes
        Shape: n_batch x n_terms x n_res x n_hidden
    E_idx : torch.LongTensor
        kNN sparse edge indices
        Shape : n_batch x n_terms x n_res x k

    Returns
    -------
    h_nn : torch.Tensor
        The gathered concatenated node and edge features
        Shape : n_batch x n_terms x n_res x k x n_hidden
    """
    # Neighbor indices E_idx [B,T,N,K]
    # Edge features h_edges [B,T,N,N,C]
    # Node features h_nodes [B,T,N,C]
    k = E_idx.shape[-1]

    h_i_idx = E_idx[:, :, :, 0].unsqueeze(-1).expand(-1, -1, -1, k).contiguous()
    h_j_idx = E_idx

    h_i = gather_term_nodes(h_nodes, h_i_idx)
    h_j = gather_term_nodes(h_nodes, h_j_idx)

    # e_ij = gather_edges(h_edges, E_idx)
    e_ij = h_edges

    # output features [B, T, N, K, 3C]
    h_nn = torch.cat([h_i, h_j, e_ij], -1)
    return h_nn

# Extract kNN info
def extract_knn(X, mask, eps, top_k):
    # Convolutional network on NCHW
    mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
    dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
    D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)

    # Identify k nearest neighbors (including self)
    D_max, _ = torch.max(D, -1, keepdim=True)
    D_adjust = D + (1. - mask_2D) * D_max
    D_neighbors, E_idx = torch.topk(D_adjust, top_k, dim=-1, largest=False)
    return mask_2D, D_neighbors, E_idx

def extract_edge_mapping(E_idx, mapping, max_num_edges):
    k = E_idx.shape[2]
    edge_idx = torch.zeros((mapping.shape[0], max_num_edges, 2*k), dtype=torch.int64)
    node_endpoint_idx = torch.zeros((mapping.shape[0], max_num_edges, 2*k), dtype=torch.int64)
    node_neighbors_idx = torch.zeros((mapping.shape[0], max_num_edges, 2*k), dtype=torch.int64)
    mapping_r = torch.reshape(mapping, E_idx.shape)
    for b in range(mapping_r.shape[0]):
        for i in range(mapping_r.shape[1]):
            for ij in range(mapping_r.shape[2]):
                j = E_idx[b,i,ij].item()
                mapping_i = mapping_r[b,i,ij]
                if i > j and i in E_idx[b,j]:
                    continue
                edge_idx[b, mapping_i, :k] = mapping_r[b,i]
                node_endpoint_idx[b, mapping_i, :k] = i*torch.ones(node_endpoint_idx[b, mapping_i,:k].shape)
                node_neighbors_idx[b, mapping_i, :k] = E_idx[b,i]
                edge_idx[b, mapping_i, k:] = mapping_r[b,j]
                node_endpoint_idx[b, mapping_i, k:] = j*torch.ones(node_endpoint_idx[b, mapping_i,:k].shape)
                node_neighbors_idx[b, mapping_i, k:] = E_idx[b,j]
    return edge_idx, node_endpoint_idx, node_neighbors_idx

# merge edge fns

def get_merge_dups_mask(E_idx):
    N = E_idx.shape[1]
    if E_idx.is_cuda:
        tens_place = torch.arange(N).cuda().unsqueeze(0).unsqueeze(-1)
    else:
        tens_place = torch.arange(N).unsqueeze(0).unsqueeze(-1)
    # tens_place = tens_place.unsqueeze(0).unsqueeze(-1)
    min_val = torch.minimum(E_idx, tens_place)
    max_val = torch.maximum(E_idx, tens_place)
    edge_indices = min_val*N + max_val
    edge_indices = edge_indices.flatten(1,2)
    unique_inv = []
    all_num_edges = []
    for b in range(len(edge_indices)):
        uniq, inv = torch.unique(edge_indices[b], return_inverse=True)
        unique_inv.append(inv)
        all_num_edges.append(len(uniq))
    unique_inv = torch.stack(unique_inv)
    return unique_inv, all_num_edges

def merge_duplicate_edges(h_E_update, E_idx, inv_mapping=None):
    """ Average embeddings across bidirectional edges.

    TERMinator edges are represented as two bidirectional edges, and to allow for
    communication between these edges we average the embeddings.

    Args
    ----
    h_E_update : torch.Tensor
        Update tensor for edges embeddings in kNN sparse form
        Shape : n_batch x n_res x k x n_hidden
    E_idx : torch.LongTensor
        kNN sparse edge indices
        Shape : n_batch x n_res x k

    Returns
    -------
    merged_E_updates : torch.Tensor
        Edge update with merged updates for bidirectional edges
        Shape : n_batch x n_res x k x n_hidden
    """
    k = E_idx.shape[-1]
    seq_lens = torch.ones(h_E_update.shape[0]).long().to(h_E_update.device) * h_E_update.shape[1]
    h_dim = h_E_update.shape[-1]
    h_E_geometric = h_E_update.view([-1, h_dim])
    split_E_idxs = torch.unbind(E_idx)
    offset = [seq_lens[:i].sum() for i in range(len(seq_lens))]
    split_E_idxs = [e.to(h_E_update.device) + o for e, o in zip(split_E_idxs, offset)]
    edge_index_row = torch.cat([e.view(-1) for e in split_E_idxs], dim=0)
    edge_index_col = torch.repeat_interleave(torch.arange(edge_index_row.shape[0] // k), k).to(h_E_update.device)
    edge_index = torch.stack([edge_index_row, edge_index_col])
    merge = merge_duplicate_edges_geometric(h_E_geometric, edge_index)
    merge = merge.view(h_E_update.shape)

    # dev = h_E_update.device
    # n_batch, n_nodes, _, hidden_dim = h_E_update.shape
    # # collect edges into NxN tensor shape
    # collection = torch.zeros((n_batch, n_nodes, n_nodes, hidden_dim)).to(dev)
    # neighbor_idx = E_idx.unsqueeze(-1).expand(-1, -1, -1, hidden_dim).to(dev)
    # collection.scatter_(2, neighbor_idx, h_E_update)
    # # transpose to get same edge in reverse direction
    # collection = collection.transpose(1, 2)
    # # gather reverse edges
    # reverse_E_update = gather_edges(collection, E_idx)
    # # average h_E_update and reverse_E_update at non-zero positions
    # merged_E_updates = torch.where(reverse_E_update != 0, (h_E_update + reverse_E_update) / 2, h_E_update)
    # assert (merge == merged_E_updates).all()

    return merge


def merge_duplicate_edges_geometric(h_E_update, edge_index):
    """ Average embeddings across bidirectional edges for Torch Geometric graphs

    TERMinator edges are represented as two bidirectional edges, and to allow for
    communication between these edges we average the embeddings.

    Args
    ----
    h_E_update : torch.Tensor
        Update tensor for edges embeddings in Torch Geometric sparse form
        Shape : n_edge x n_hidden
    edge_index : torch.LongTensor
        Torch Geometric sparse edge indices
        Shape : 2 x n_edge

    Returns
    -------
    merged_E_updates : torch.Tensor
        Edge update with merged updates for bidirectional edges
        Shape : n_edge x n_hidden
    """
    num_nodes = edge_index.max() + 1
    row_idx = edge_index[0] + edge_index[1] * num_nodes
    col_idx = edge_index[1] + edge_index[0] * num_nodes
    internal_idx = torch.arange(edge_index.shape[1]).to(edge_index.device)

    mapping = (torch.zeros(max(row_idx.max(), col_idx.max()) + 1).long() - 1).to(edge_index.device)
    mapping[col_idx] = internal_idx

    reverse_idx = mapping[row_idx]
    mask = (reverse_idx >= 0)
    reverse_idx = reverse_idx[mask]

    reverse_h_E = h_E_update[mask]
    h_E_update[reverse_idx] = (h_E_update[reverse_idx] + reverse_h_E)/2

    return h_E_update


def merge_duplicate_term_edges(h_E_update, E_idx):
    """ Average embeddings across bidirectional TERM edges.

    TERMinator edges are represented as two bidirectional edges, and to allow for
    communication between these edges we average the embeddings.

    Args
    ----
    h_E_update : torch.Tensor
        Update tensor for edges embeddings in kNN sparse form
        Shape : n_batch x n_terms x n_res x k x n_hidden
    E_idx : torch.LongTensor
        kNN sparse edge indices
        Shape : n_batch x n_terms x n_res x k

    Returns
    -------
    merged_E_updates : torch.Tensor
        Edge update with merged updates for bidirectional edges
        Shape : n_batch x n_terms x n_res x k x n_hidden
    """
    dev = h_E_update.device
    n_batch, n_terms, n_aa, _, hidden_dim = h_E_update.shape
    # collect edges into NxN tensor shape
    collection = torch.zeros((n_batch, n_terms, n_aa, n_aa, hidden_dim)).to(dev)
    neighbor_idx = E_idx.unsqueeze(-1).expand(-1, -1, -1, -1, hidden_dim).to(dev)
    collection.scatter_(3, neighbor_idx, h_E_update)
    # transpose to get same edge in reverse direction
    collection = collection.transpose(2, 3)
    # gather reverse edges
    reverse_E_update = gather_term_edges(collection, E_idx)
    # average h_E_update and reverse_E_update at non-zero positions
    merged_E_updates = torch.where(reverse_E_update != 0, (h_E_update + reverse_E_update) / 2, h_E_update)
    return merged_E_updates


def merge_duplicate_pairE(h_E, E_idx):
    """ Average pair energy tables across bidirectional edges.

    TERMinator edges are represented as two bidirectional edges, and to allow for
    communication between these edges we average the embeddings. In the case for
    pair energies, we transpose the tables to ensure that the pair energy table
    is symmetric upon inverse (e.g. the pair energy between i and j should be
    the same as the pair energy between j and i)

    Args
    ----
    h_E : torch.Tensor
        Pair energies in kNN sparse form
        Shape : n_batch x n_res x k x n_aa x n_aa
    E_idx : torch.LongTensor
        kNN sparse edge indices
        Shape : n_batch x n_res x k

    Returns
    -------
    torch.Tensor
        Pair energies with merged energies for bidirectional edges
        Shape : n_batch x n_res x k x n_aa x n_aa
    """
    try:
        k = E_idx.shape[-1]
        seq_lens = torch.ones(h_E.shape[0]).long().to(h_E.device) * h_E.shape[1]
        h_E_geometric = h_E.view([-1, 400])
        split_E_idxs = torch.unbind(E_idx)
        offset = [seq_lens[:i].sum() for i in range(len(seq_lens))]
        split_E_idxs = [e.to(h_E.device) + o for e, o in zip(split_E_idxs, offset)]
        edge_index_row = torch.cat([e.view(-1) for e in split_E_idxs], dim=0)
        edge_index_col = torch.repeat_interleave(torch.arange(edge_index_row.shape[0] // k), k).to(h_E.device)
        edge_index = torch.stack([edge_index_row, edge_index_col])
        merge = merge_duplicate_pairE_geometric(h_E_geometric, edge_index)
        merge = merge.view(h_E.shape)
        #old_merge = merge_duplicate_pairE_dense(h_E, E_idx)
        #assert (old_merge == merge).all(), (old_merge, merge)

        return merge
    except RuntimeError as err:
        print(err, file=sys.stderr)
        print("We're handling this error as if it's an out-of-memory error", file=sys.stderr)
        torch.cuda.empty_cache()  # this is probably unnecessary but just in case
        return merge_duplicate_pairE_sparse(h_E, E_idx)


def merge_duplicate_pairE_dense(h_E, E_idx):
    """ Dense method to average pair energy tables across bidirectional edges.

    TERMinator edges are represented as two bidirectional edges, and to allow for
    communication between these edges we average the embeddings. In the case for
    pair energies, we transpose the tables to ensure that the pair energy table
    is symmetric upon inverse (e.g. the pair energy between i and j should be
    the same as the pair energy between j and i)

    Args
    ----
    h_E : torch.Tensor
        Pair energies in kNN sparse form
        Shape : n_batch x n_res x k x n_aa x n_aa
    E_idx : torch.LongTensor
        kNN sparse edge indices
        Shape : n_batch x n_res x k

    Returns
    -------
    torch.Tensor
        Pair energies with merged energies for bidirectional edges
        Shape : n_batch x n_res x k x n_aa x n_aa
    """
    dev = h_E.device
    n_batch, n_nodes, _, n_aa, _ = h_E.shape
    # collect edges into NxN tensor shape
    collection = torch.zeros((n_batch, n_nodes, n_nodes, n_aa, n_aa)).to(dev)
    neighbor_idx = E_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, n_aa, n_aa).to(dev)
    collection.scatter_(2, neighbor_idx, h_E)
    # transpose to get same edge in reverse direction
    collection = collection.transpose(1, 2)
    # transpose each pair energy table as well
    collection = collection.transpose(-2, -1)
    # gather reverse edges
    reverse_E = gather_pairEs(collection, E_idx)
    # average h_E and reverse_E at non-zero positions
    merged_E = torch.where(reverse_E != 0, (h_E + reverse_E) / 2, h_E)
    return merged_E


# TODO: rigorous test that this is equiv to the dense version
def merge_duplicate_pairE_sparse(h_E, E_idx):
    """ Sparse method to average pair energy tables across bidirectional edges.

    Note: This method involves a significant slowdown so it's only worth using if memory is an issue.

    TERMinator edges are represented as two bidirectional edges, and to allow for
    communication between these edges we average the embeddings. In the case for
    pair energies, we transpose the tables to ensure that the pair energy table
    is symmetric upon inverse (e.g. the pair energy between i and j should be
    the same as the pair energy between j and i)

    Args
    ----
    h_E : torch.Tensor
        Pair energies in kNN sparse form
        Shape : n_batch x n_res x k x n_aa x n_aa
    E_idx : torch.LongTensor
        kNN sparse edge indices
        Shape : n_batch x n_res x k

    Returns
    -------
    torch.Tensor
        Pair energies with merged energies for bidirectional edges
        Shape : n_batch x n_res x k x n_aa x n_aa
    """
    dev = h_E.device
    n_batch, n_nodes, k, n_aa, _ = h_E.shape
    # convert etab into a sparse etab
    # self idx of the edge
    ref_idx = E_idx[:, :, 0:1].expand(-1, -1, k)
    # sparse idx
    g_idx = torch.cat([E_idx.unsqueeze(1), ref_idx.unsqueeze(1)], dim=1)
    sparse_idx = g_idx.view([n_batch, 2, -1])
    # generate a 1D idx for the forward and backward direction
    scaler = torch.ones_like(sparse_idx).to(dev)
    scaler = scaler * n_nodes
    scaler_f = scaler
    scaler_f[:, 0] = 1
    scaler_r = torch.flip(scaler_f, [1])
    batch_offset = torch.arange(n_batch).unsqueeze(-1).expand([-1, n_nodes * k]) * n_nodes * k
    batch_offset = batch_offset.to(dev)
    sparse_idx_f = torch.sum(scaler_f * sparse_idx, 1) + batch_offset
    flat_idx_f = sparse_idx_f.view([-1])
    sparse_idx_r = torch.sum(scaler_r * sparse_idx, 1) + batch_offset
    flat_idx_r = sparse_idx_r.view([-1])
    # generate sparse tensors
    flat_h_E_f = h_E.view([n_batch * n_nodes * k, n_aa**2])
    reverse_h_E = h_E.transpose(-2, -1).contiguous()
    flat_h_E_r = reverse_h_E.view([n_batch * n_nodes * k, n_aa**2])
    sparse_etab_f = torch.sparse_coo_tensor(flat_idx_f.unsqueeze(0), flat_h_E_f,
                                            (n_batch * n_nodes * n_nodes, n_aa**2))
    count_f = torch.sparse_coo_tensor(flat_idx_f.unsqueeze(0), torch.ones_like(flat_idx_f),
                                      (n_batch * n_nodes * n_nodes, ))
    sparse_etab_r = torch.sparse_coo_tensor(flat_idx_r.unsqueeze(0), flat_h_E_r,
                                            (n_batch * n_nodes * n_nodes, n_aa**2))
    count_r = torch.sparse_coo_tensor(flat_idx_r.unsqueeze(0), torch.ones_like(flat_idx_r),
                                      (n_batch * n_nodes * n_nodes, ))
    # merge
    sparse_etab = sparse_etab_f + sparse_etab_r
    sparse_etab = sparse_etab.coalesce()
    count = count_f + count_r
    count = count.coalesce()

    # this step is very slow, but implementing something faster is probably a lot of work
    # requires pytorch 1.10 to be fast enough to be usable
    collect = sparse_etab.index_select(0, flat_idx_f).to_dense()
    weight = count.index_select(0, flat_idx_f).to_dense()

    flat_merged_etab = collect / weight.unsqueeze(-1)
    merged_etab = flat_merged_etab.view(h_E.shape)
    return merged_etab


def merge_duplicate_pairE_geometric(h_E, edge_index):
    """ Sparse method to average pair energy tables across bidirectional edges with Torch Geometric.

    TERMinator edges are represented as two bidirectional edges, and to allow for
    communication between these edges we average the embeddings. In the case for
    pair energies, we transpose the tables to ensure that the pair energy table
    is symmetric upon inverse (e.g. the pair energy between i and j should be
    the same as the pair energy between j and i)

    This function assumes edge_index is sorted by columns, and will fail if
    this is not the case.

    Args
    ----
    h_E : torch.Tensor
        Pair energies in Torch Geometric sparse form
        Shape : n_edge x 400
    E_idx : torch.LongTensor
        Torch Geometric sparse edge indices
        Shape : 2 x n_edge

    Returns
    -------
    torch.Tensor
        Pair energies with merged energies for bidirectional edges
        Shape : n_edge x 400
    """
    num_nodes = edge_index.max() + 1
    row_idx = edge_index[0] + edge_index[1] * num_nodes
    col_idx = edge_index[1] + edge_index[0] * num_nodes
    internal_idx = torch.arange(edge_index.shape[1]).to(h_E.device)

    mapping = torch.zeros(max(row_idx.max(), col_idx.max()) + 1).long().to(h_E.device) - 1
    mapping[col_idx] = internal_idx

    reverse_idx = mapping[row_idx]
    mask = (reverse_idx >= 0)
    reverse_idx = reverse_idx[mask]

    reverse_h_E = h_E[mask]
    transpose_h_E = reverse_h_E.view([-1, 20, 20]).transpose(-1, -2).reshape([-1, 400])
    h_E[reverse_idx] = (h_E[reverse_idx] + transpose_h_E)/2

    return h_E


# edge aggregation fns


def aggregate_edges(edge_embeddings, E_idx, max_seq_len):
    """ Aggregate TERM edge embeddings into a sequence-level dense edge features tensor

    Args
    ----
    edge_embeddings : torch.Tensor
        TERM edge features tensor
        Shape : n_batch x n_terms x n_aa x n_neighbors x n_hidden
    E_idx : torch.LongTensor
        TERM edge indices
        Shape : n_batch x n_terms x n_aa x n_neighbors
    max_seq_len : int
        Max length of a sequence in the batch

    Returns
    -------
    torch.Tensor
        Dense sequence-level edge features
        Shape : n_batch x max_seq_len x max_seq_len x n_hidden
    """
    dev = edge_embeddings.device
    n_batch, _, _, n_neighbors, hidden_dim = edge_embeddings.shape
    # collect edges into NxN tensor shape
    collection = torch.zeros((n_batch, max_seq_len, max_seq_len, hidden_dim)).to(dev)
    # edge the edge indecies
    self_idx = E_idx[:, :, :, 0].unsqueeze(-1).expand(-1, -1, -1, n_neighbors)
    neighbor_idx = E_idx
    # tensor needed for accumulation
    layer = torch.arange(n_batch).view([n_batch, 1, 1, 1]).expand(neighbor_idx.shape).to(dev)
    # thicc index_put_
    collection.index_put_((layer, self_idx, neighbor_idx), edge_embeddings, accumulate=True)

    # we also need counts for averaging
    count = torch.zeros((n_batch, max_seq_len, max_seq_len)).to(dev)
    count_idx = torch.ones_like(neighbor_idx).float().to(dev)
    count.index_put_((layer, self_idx, neighbor_idx), count_idx, accumulate=True)

    # we need to set all 0s to 1s so we dont get nans
    count[count == 0] = 1

    return collection / count.unsqueeze(-1)


## From joint-protein-embs
def extract_idxs(E_idx, mask):
    ref_indexes = torch.arange(E_idx.shape[0])
    ref_indexes = ref_indexes.unsqueeze(-1)
    ref_indexes = ref_indexes.expand(E_idx.shape)
    ref_indexes = ref_indexes.to(E_idx.device)
    max_length = int(10**(np.ceil(np.log10(E_idx.shape[0]))))
    opt1 = torch.from_numpy(E_idx.numpy() / max_length + ref_indexes.numpy())
    opt2 = torch.from_numpy(E_idx.numpy() + ref_indexes.numpy() / max_length)
    E_idx_paired = torch.min(opt1, opt2)
    del opt1, opt2
    # E_idx_paired = torch.min(E_idx / max_length + ref_indexes, E_idx + ref_indexes / max_length)
    E_idx_paired_flat = torch.flatten(E_idx_paired)
    out, count = torch.unique(E_idx_paired_flat, sorted=True, return_inverse=False, return_counts=True, dim=0)

    out_dup = out[count > 1]
    o_size = out_dup.numel()
    v_size = E_idx_paired_flat.numel()

    o_expand = out_dup.unsqueeze(1).expand(o_size, v_size) 
    v_expand = E_idx_paired_flat.unsqueeze(0).expand(o_size, v_size)
    result = (o_expand - v_expand == 0).nonzero()[:,1]
    idxs_to_dup = result[::2]
    idxs_to_remove = result[1::2]

    result_singles = torch.arange(len(E_idx_paired_flat)).type(idxs_to_dup.dtype)
    base = -1 * torch.ones(len(E_idx_paired_flat)).type(idxs_to_dup.dtype)
    result_singles = result_singles.scatter_(0, idxs_to_dup, base)
    result_singles = result_singles.scatter_(0, idxs_to_remove, base)
    idxs_singles = result_singles[result_singles > 0]
        
    inds_reduce, mask_reduced_combs = per_node_to_all_comb_inds(idxs_to_remove, E_idx_paired.shape)
    inds_expand = all_comb_to_per_node_inds(idxs_to_remove, idxs_to_dup, E_idx_paired.shape, inds_reduce.shape)
    mask = mask[0].unsqueeze(1)
    mask_reduced_nodes = mask.expand(E_idx_paired.shape)
    mask_reduced_nodes = per_node_to_all_comb(mask_reduced_nodes, inds_reduce, batched=False)
    mask_reduced = torch.multiply(mask_reduced_combs, mask_reduced_nodes)

    return inds_reduce, inds_expand, idxs_to_remove, idxs_to_dup, idxs_singles, mask_reduced.to(torch.bool) 

def sync_inds_shape(inds, shape):
    i = len(inds.shape)
    while len(shape) > len(inds.shape):
        inds = torch.unsqueeze(inds, -1)
        new_shape = list(inds.shape)
        new_shape[i] = shape[i]
        inds = inds.expand(new_shape)
        i += 1
    return inds

def per_node_to_all_comb_inds(idxs_to_remove, per_node_shape):
    base = -1*torch.ones(per_node_shape[0] * per_node_shape[1], dtype=torch.int64)
    inds = torch.arange(len(base), dtype=torch.int64).scatter_(0, idxs_to_remove, base)
    all_inds = inds[inds > -1]
    mask = torch.logical_not(torch.isnan(all_inds))
    all_inds = torch.nan_to_num(all_inds)
    all_inds = all_inds.type(torch.int64)
    return all_inds, mask

def per_node_to_all_comb(per_node_tensor, inds_reduce, batched=True):
    if batched:
        begin_dim = 1
    else:
        begin_dim = 0

    all_comb_tensor = torch.flatten(per_node_tensor, begin_dim, begin_dim+1)
    inds_reduce = sync_inds_shape(inds_reduce, all_comb_tensor.shape)
    all_comb_tensor = torch.gather(all_comb_tensor, begin_dim, inds_reduce)
    return all_comb_tensor

# Function to create a one-hot encoding for a character
def char_to_1hot(char):
    char_idx = ALPHABET.find(char)
    if char_idx == -1:
        return None  # Character not found in the alphabet
    one_hot = np.zeros(len(ALPHABET))
    one_hot[char_idx] = 1
    return one_hot

def str_to_1hot(input_string):
    embedding = []
    for char in input_string:
        one_hot = char_to_1hot(char)
        if one_hot is not None:
            embedding.append(torch.from_numpy(one_hot))
    return torch.stack(embedding)

def str_to_list(s):
    pattern = r'<[^>]+>|.'
    
    return re.findall(pattern, s)

def _esm_featurize(chain_lens, base_seq, seq, esm, batch_converter, use_esm_attns, esm_embed_layer, from_rla=False, use_reps=True, connect_chains=False, one_hot=False, dev='cpu', return_logits=False, linker_length=25, from_pepmlm=False):
    with torch.no_grad():

        esm_data = []
        base_data_seq = []
        scl = 0
        if one_hot:
            return str_to_1hot("".join(seq)).to(dtype=torch.float), None
        if not connect_chains:
            for ic, cl in enumerate(chain_lens):
                esm_data.append((str(ic), "".join(seq[scl:scl+cl])))
                base_data_seq.append((str(ic), "".join(list(base_seq)[scl:scl+cl])))
                scl += cl
        else:
            esm_seq = []
            for ic, cl in enumerate(chain_lens):
                esm_seq.append("".join(seq[scl:scl+cl]))
                base_data_seq.append("".join(list(base_seq)[scl:scl+cl]))
                scl += cl
            mask_seq = (linker_length*"/").join(base_data_seq)
            esm_seq = (linker_length*"G").join(esm_seq)
            mask = torch.from_numpy(np.array(str_to_list(mask_seq)) != "/")
            esm_data.append(('prot', esm_seq))
        _, _, batch_tokens = batch_converter(esm_data)

        batch_tokens = batch_tokens.to(device=dev)
        with torch.no_grad():
            if from_pepmlm:
                results = esm(batch_tokens, output_hidden_states=True, output_attentions=use_esm_attns)
            elif not from_rla:
                results = esm(batch_tokens, repr_layers=[esm_embed_layer], return_contacts=use_esm_attns)
            else:
                results = esm(batch_tokens, output_attentions=use_esm_attns)
        
        if return_logits:
            if not connect_chains:
                logit_results = [emb_result[1:chain_lens[ic] + 1] for ic, emb_result in enumerate(results['logits'])]
                logits = torch.cat(logit_results)
            else:
                logits = results['logits'][0][1:-1]
                logits = logits[mask]
            return logits
            
        if use_reps:
            if from_pepmlm:
                emb_results = results['hidden_states'][esm_embed_layer]
                if use_esm_attns: results['attentions'] = results['attentions'][32]
            elif not from_rla:
                emb_results = results['representations'][esm_embed_layer]
            else:
                emb_results = results['last_hidden_state']
        else:
            emb_results = results['logits']

        if not connect_chains:
            emb_results = [emb_result[1:chain_lens[ic] + 1] for ic, emb_result in enumerate(emb_results)]
            embs = torch.cat(emb_results)
        else:
            embs = emb_results[0][1:-1]
            embs = embs[mask]
        if not use_esm_attns:
            return embs, None
        if not connect_chains:
            scl = 0
            if from_rla:
                attns = torch.cat(results['attentions'], 1)
                attns = [attn[:,1:chain_lens[ic] + 1, 1:chain_lens[ic] + 1] for ic, attn in enumerate(attns)]
                tot_len = sum([attns[i].size(1) for i in range(len(attns))])
                final_attns = torch.zeros(tot_len, tot_len, attns[0].size(0))
                for ic, attn in enumerate(attns):
                    final_attns[scl:scl+chain_lens[ic], scl:scl+chain_lens[ic], :] = attn.permute(1,2,0)
                    scl += chain_lens[ic]
            else:
                sha = results['attentions'].shape
                attns = results['attentions'].contiguous().view(sha[0], -1, sha[3], sha[4])
                # assert(attns.shape[0] == 1)
                attns = [attn[:,1:chain_lens[ic] + 1, 1:chain_lens[ic] + 1] for ic, attn in enumerate(attns)]
                tot_len = sum([attns[i].size(1) for i in range(len(attns))])
                final_attns = torch.zeros(tot_len, tot_len, attns[0].size(0))
                for ic, attn in enumerate(attns):
                    final_attns[scl:scl+chain_lens[ic], scl:scl+chain_lens[ic], :] = attn.permute(1,2,0)
                    scl += chain_lens[ic]
        else:
            if from_pepmlm:
                sha = results['attentions'].shape
                attns = results['attentions'].contiguous()[0,1:-1,1:-1]
                attns = attns.permute(1,2,0)
                attns = attns[mask]
                final_attns = attns[:,mask,:]
            elif not from_rla:
                sha = results['attentions'].shape
                attns = results['attentions'].contiguous().view(sha[0], -1, sha[3], sha[4])[0,:,1:-1,1:-1]
                attns = attns.permute(1,2,0)
                attns = attns[mask]
                final_attns = attns[:,mask,:]
            else:
                attns = torch.cat(results['attentions'], 1)[0,:,1:-1,1:-1].permute(1,2,0)
                attns = attns[mask]
                final_attns = attns[:,mask,:]
    return embs, final_attns
    

def expand_etab_single(etab, idxs):
    h = etab.shape[-1]
    tetab = etab.to(dtype=torch.float64)
    eidx = idxs.unsqueeze(-1).unsqueeze(-1).expand(etab.shape)
    netab = torch.zeros(tetab.shape[0], tetab.shape[0], h, h, dtype=torch.float64)
    netab.scatter_(1, eidx, tetab)
    cetab = netab.transpose(0,1).transpose(2,3)
    cetab.scatter_(1, eidx, tetab)
    return cetab

def expand_etab(etab, idxs):
    h = etab.shape[-1]
    tetab = etab.to(dtype=torch.float64)
    eidx = idxs.unsqueeze(-1).unsqueeze(-1).expand(etab.shape)
    netab = torch.zeros(tetab.shape[0], tetab.shape[1], tetab.shape[1], h, h, dtype=torch.float64, device=etab.device)
    netab.scatter_(2, eidx, tetab)
    cetab = netab.transpose(1,2).transpose(3,4)
    cetab.scatter_(2, eidx, tetab)
    return cetab

def expand_etab_4d(etab, idxs):
    h = etab.shape[-1]
    tetab = etab.to(dtype=torch.float32)
    eidx = idxs.unsqueeze(-1).expand(etab.shape)
    netab = torch.zeros(tetab.shape[0], tetab.shape[1], tetab.shape[1], h, dtype=torch.float32, device=etab.device)
    netab.scatter_(2, eidx, tetab)
    return netab

def condense_etab_expanded(e, protein_array, pep_len=20):
    h = e.shape[-1]
    prot_len = protein_array.shape[1]
    new = e[:, -1*pep_len:, -1*pep_len:,:, :]
    eners = e[:, :prot_len, prot_len:prot_len+pep_len]
    inds = protein_array[:, :, None, None, None].expand(new.shape[0], prot_len, pep_len, 1, h)
    geners = torch.gather(eners, 3, inds).squeeze(3)
    seners = torch.sum(geners, axis=1, keepdims=False)
    self_update_diag1 = torch.zeros_like(new[:,:,:,0])
    expand_diag_1 = torch.arange(new.shape[1]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(new.shape[0], -1, 1, new.shape[3]).to(dtype=inds.dtype, device=inds.device)
    self_update_diag1.scatter_(2, expand_diag_1, seners.unsqueeze(2))
    expand_diag_2 = torch.arange(new.shape[3]).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(new.shape[0], new.shape[1], new.shape[2], -1, 1).to(dtype=inds.dtype, device=inds.device)
    self_update = torch.zeros_like(new)
    self_update.scatter_(4, expand_diag_2, self_update_diag1.unsqueeze(-1))
    out = new + self_update
    return out

def score_seq_batch(etab, seq_ints):
    h = etab.shape[-1]
    mask = torch.triu(torch.ones(etab.shape[0], etab.shape[0]), diagonal=0).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, h).to(etab.device)
    etab *= mask
    n, l = seq_ints.shape
    batch_etab = etab.unsqueeze(0).expand(n, l, l, h, h)
    j_batch_seq = seq_ints.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).expand(-1, l, -1, h, 1)
    i_batch_seq = seq_ints.unsqueeze(2).unsqueeze(2).expand(n, l, l, 1)
    j_batch_eners = torch.gather(batch_etab, 4, j_batch_seq).squeeze(-1)
    batch_eners = torch.gather(j_batch_eners, 3, i_batch_seq).sum(dim=(1,2)).squeeze(-1)
    return batch_eners

def batch_score_seq_batch(etab, seq_ints):
    h = etab.shape[-1]
    mask = torch.triu(torch.ones(etab.shape[0], etab.shape[1], etab.shape[1]), diagonal=0).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, h, h).to(etab.device)
    etab *= mask
    b, n, l = seq_ints.shape
    batch_etab = etab.unsqueeze(1).expand(b, n, l, l, h, h)
    j_batch_seq = seq_ints.unsqueeze(2).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, l, -1, h, 1)
    i_batch_seq = seq_ints.unsqueeze(3).unsqueeze(3).expand(b, n, l, l, 1)
    j_batch_eners = torch.gather(batch_etab, 5, j_batch_seq).squeeze(-1)
    batch_eners = torch.gather(j_batch_eners, 4, i_batch_seq).sum(dim=(2,3)).squeeze(-1)
    return batch_eners

def batch_score_seq_batch_segment(etab, seq_ints, step=100):
    b, n, l = seq_ints.shape
    h = etab.shape[-1]
    mask = torch.triu(torch.ones(etab.shape[0], etab.shape[1], etab.shape[1]), diagonal=0).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, h, h).to(etab.device)
    mask = mask.to(device=etab.device)
    etab *= mask
    batch_eners = []
    for n_segment in range(0, n, step):
        seq_ints_segment = seq_ints
        batch_etab = etab.unsqueeze(1).expand(b, n, l, l, h, h)
        j_batch_seq = seq_ints.unsqueeze(2).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, l, -1, h, 1)
        i_batch_seq = seq_ints.unsqueeze(3).unsqueeze(3).expand(b, n, l, l, 1)
        j_batch_eners = torch.gather(batch_etab, 5, j_batch_seq).squeeze(-1)
        batch_eners = torch.gather(j_batch_eners, 4, i_batch_seq).sum(dim=(2,3)).squeeze(-1)
        
    return batch_eners

def batch_score_seq_batch_pos(etab, seq_ints, pos_list):
    h = etab.shape[-1]
    # mask = torch.triu(torch.ones(etab.shape[0], etab.shape[1], etab.shape[1]), diagonal=0).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, h, h).to(etab.device)
    # etab *= mask
    b, n, l = seq_ints.shape
    batch_etab = etab.unsqueeze(1).expand(b, n, l, h, h)
    j_batch_seq = seq_ints.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, h, 1)
    i_batch_seq = torch.gather(seq_ints, 2, pos_list).unsqueeze(2).expand(b, n, l, 1)
    j_batch_eners = torch.gather(batch_etab, 4, j_batch_seq).squeeze(-1)
    batch_eners = torch.gather(j_batch_eners, 3, i_batch_seq).sum(dim=(2)).squeeze(-1)
    return batch_eners

def sample_seqs(h_V, n_seqs=32, temperature=0.1, constant=None, constant_bias=None, bias_by_res_gathered=None):
    if constant is None:
        constant = torch.zeros(h_V.shape[2]).to(device=h_V.device)
    if constant_bias is None:
        constant_bias = torch.zeros(h_V.shape[2]).to(device=h_V.device)
        constant_bias[20:] = 1
    if bias_by_res_gathered is None:
        bias_by_res_gathered = torch.zeros_like(h_V.squeeze())

    logits = (h_V / temperature).squeeze()

    probs = F.softmax(logits-constant[None,:]*1e8+constant_bias[None,:]/temperature+bias_by_res_gathered/temperature, dim=-1)
    S_t = torch.multinomial(probs, n_seqs, replacement=True).squeeze(1)
    return S_t

def compute_pairwise_probabilities(input_tensor):
    # input_tensor: L x C
    L, C = input_tensor.shape 
    
    # Expand dimensions for pairwise computation
    P_i = input_tensor[:-1]  # L-1 x C
    P_j = input_tensor[1:]   # L-1 x C
    
    # Compute outer product for all class combinations
    pairwise_probs = torch.einsum('ij,ik->ijk', P_i, P_j)  # (L-1) x C x C
    
    # Reshape to C^2 x (L-1)
    pairwise_probs = pairwise_probs.view(-1, C**2).T  # C^2 x (L-1)
    
    # Pad the last column with zeros to make it C^2 x L
    padding = torch.zeros(C**2, 1).to(device=input_tensor.device)
    result = torch.cat((pairwise_probs, padding), dim=1)
    
    return result.sum(dim = 1) / result.sum()

def compute_pairwise_probabilities_batched(input_tensor):
    # input_tensor: B x L x C
    B, L, C = input_tensor.shape 
    
    # Expand dimensions for pairwise computation
    P_i = input_tensor[:, :-1]  # B x L-1 x C
    P_j = input_tensor[:, 1:]   # B x L-1 x C
    
    # Compute outer product for all class combinations
    pairwise_probs = torch.einsum('bij,bik->bijk', P_i, P_j)  # B x (L-1) x C x C
    
    # Reshape to B x C^2 x (L-1)
    pairwise_probs = pairwise_probs.view(B, -1, C**2).transpose(1,2)  # B x C^2 x (L-1)
    
    # Pad the last column with zeros to make it C^2 x L
    padding = torch.zeros(B, C**2, 1).to(device=input_tensor.device)
    result = torch.cat((pairwise_probs, padding), dim=2)
    
    return result.sum(dim = 2) / result.sum(dim=[1,2]).unsqueeze(1)

def compute_triplet_probabilities(input_tensor):
    # input_tensor: L x C
    L, C = input_tensor.shape  # number 1    

    # Select probabilities for consecutive positions
    P_i = input_tensor[:-2]   # L-2 x C
    P_j = input_tensor[1:-1]  # L-2 x C
    P_k = input_tensor[2:]    # L-2 x C
    
    # Compute outer product for all class triplet combinations
    triplet_probs = torch.einsum('ij,ik,il->ijkl', P_i, P_j, P_k)  # (L-2) x C x C x C
    
    
    # Reshape to C^3 x (L-2)
    triplet_probs = triplet_probs.view(-1, C**3).T  # C^3 x (L-2)
    # Pad the last two columns with zeros to make it C^3 x L
    padding = torch.zeros(C**3, 2).to(device=input_tensor.device)
    result = torch.cat((triplet_probs, padding), dim=1)
         
    return result.sum(dim = 1) / result.sum()

def compute_triplet_probabilities_batched(input_tensor):
    # input_tensor: B x L x C
    B, L, C = input_tensor.shape  # Batch size, sequence length, and number of classes

    # Select probabilities for consecutive positions
    P_i = input_tensor[:, :-2, :]   # B x (L-2) x C
    P_j = input_tensor[:, 1:-1, :]  # B x (L-2) x C
    P_k = input_tensor[:, 2:, :]    # B x (L-2) x C

    # Compute outer product for all class triplet combinations
    # Using einsum: 'bij,bik,bil->bijkl'
    triplet_probs = torch.einsum('bij,bik,bil->bijkl', P_i, P_j, P_k)  # B x (L-2) x C x C x C
    
    # Reshape to B x C^3 x (L-2)
    triplet_probs = triplet_probs.view(B, -1, C**3).transpose(1,2)  # B x C^3 x (L-2)

    # Pad the last two columns with zeros to make it B x C^3 x L
    padding = torch.zeros(B, C**3, 2, device=input_tensor.device)  # B x C^3 x 2
    triplet_probs_padded = torch.cat((triplet_probs, padding), dim=2)  # B x C^3 x L
    # Compute the sum across the sequence dimension (L) and normalize
    result = triplet_probs_padded.sum(dim=2) / triplet_probs_padded.sum(dim=[1,2]).unsqueeze(1)
    return result
