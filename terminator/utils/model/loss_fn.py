""" Loss functions for TERMinator, and a customizable loss function constructor built from the included components.

In order to use the customizable loss function constructor :code:`construct_loss_fn`, loss functions
must have the signature :code:`loss(etab, E_idx, data)`, where
    - :code:`loss` is the name of the loss fn
    - :code:`etab` is the outputted etab from TERMinator
    - :code:`E_idx` is the edge index outputted from TERMinator
    - :code:`data` is the training data dictionary
Additionally, the function must return two outputs :code:`loss_contribution, norm_count`, where
    - :code:`loss_contribution` is the computed loss contribution by the function
    - :code:`norm_count` is a normalizing constant associated with the loss (e.g. when averaging across losses in batches,
the average loss will be :math:`\\frac{\\sum_i loss_contribution}{\\sum_i norm_count}`)
"""
import sys

import torch
import torch.nn.functional as F
import numpy as np
import random
import os
import copy
# from torch_scatter import scatter_mean, scatter_add
from terminator.models.layers.utils import (get_merge_dups_mask, get_msa_paired_stats, extract_knn, expand_etab_single, expand_etab, condense_etab_expanded, score_seq_batch, batch_score_seq_batch, compute_pairwise_probabilities_batched, compute_triplet_probabilities_batched)
from terminator.utils.common import ints_to_seq, seq_to_ints
from terminator.utils.model.contact_utils import calc_attns, get_esm_contacts
import GPUtil
import math
import blosum as bl

matrix = bl.BLOSUM(62)
TORCH_BLOSUM = torch.zeros(22, 22)
for i in range(22):
    stri = ints_to_seq([i])[0]
    for j in range(i, 22):
        strj = ints_to_seq([j])[0]
        TORCH_BLOSUM[i, j] = matrix[stri][strj]
        TORCH_BLOSUM[j, i] = matrix[stri][strj]
TORCH_BLOSUM = torch.nan_to_num(TORCH_BLOSUM, neginf=-5)

TEST_MATRIX = torch.zeros(22, 22)
for i in range(22):
    TEST_MATRIX[i, i] = 2


TEST_MATRIX2 = torch.zeros(22, 22)
for i in range(22):
    TEST_MATRIX2[i, i] = 2
    for j in range(i+1, i + 4):
        if j > 21:
            continue
        TEST_MATRIX2[i, j] = 1
        TEST_MATRIX2[j, i] = 1

CUSTOM_MATRIX = torch.Tensor([
    [6., 3., 3., 2., 3., 5., 2., 5., 2., 4., 3., 3., 0., 3., 2., 4., 4., 5., 3., 2., 0., 0.],
    [3., 6., 3., 2., 1., 3., 2., 3., 2., 2., 3., 3., 0., 3., 2., 4., 4., 3., 1., 2., 0., 0.],
    [3., 3., 6., 4., 1., 3., 3., 3., 3., 3., 1., 3., 0., 3., 3., 4., 4., 3., 1., 2., 0., 0.],
    [2., 2., 4., 6., 2., 2., 2., 2., 4., 2., 2., 4., 0., 4., 4., 3., 3., 2., 2., 3., 0., 0.],
    [3., 1., 1., 2., 6., 3., 2., 3., 2., 2., 3., 3., 0., 3., 2., 2., 2., 3., 5., 4., 0., 0.],
    [5., 3., 3., 2., 3., 6., 2., 5., 2., 4., 3., 3., 0., 3., 2., 4., 4., 5., 3., 2., 0., 0.],
    [2., 2., 3., 2., 2., 2., 6., 2., 3., 2., 0., 2., 0., 2., 3., 3., 3., 2., 2., 3., 0., 0.],
    [5., 3., 3., 2., 3., 5., 2., 6., 2., 4., 3., 3., 0., 3., 2., 4., 4., 5., 3., 2., 0., 0.],
    [2., 2., 3., 4., 2., 2., 3., 2., 6., 2., 2., 4., 0., 4., 5., 3., 3., 2., 2., 3., 0., 0.],
    [4., 2., 3., 2., 2., 4., 2., 4., 2., 6., 2., 2., 0., 2., 2., 3., 3., 4., 2., 1., 0., 0.],
    [3., 3., 1., 2., 3., 3., 0., 3., 2., 2., 6., 3., 0., 3., 2., 2., 2., 3., 3., 2., 0., 0.],
    [3., 3., 3., 4., 3., 3., 2., 3., 4., 2., 3., 6., 0., 5., 4., 4., 4., 3., 3., 4., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 6., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [3., 3., 3., 4., 3., 3., 2., 3., 4., 2., 3., 5., 0., 6., 4., 4., 4., 3., 3., 4., 0., 0.],
    [2., 2., 3., 4., 2., 2., 3., 2., 5., 2., 2., 4., 0., 4., 6., 3., 3., 2., 2., 3., 0., 0.],
    [4., 4., 4., 3., 2., 4., 3., 4., 3., 3., 2., 4., 0., 4., 3., 6., 5., 4., 2., 3., 0., 0.],
    [4., 4., 4., 3., 2., 4., 3., 4., 3., 3., 2., 4., 0., 4., 3., 5., 6., 4., 2., 3., 0., 0.],
    [5., 3., 3., 2., 3., 5., 2., 5., 2., 4., 3., 3., 0., 3., 2., 4., 4., 6., 3., 2., 0., 0.],
    [3., 1., 1., 2., 5., 3., 2., 3., 2., 2., 3., 3., 0., 3., 2., 2., 2., 3., 6., 4., 0., 0.],
    [2., 2., 2., 3., 4., 2., 3., 2., 3., 1., 2., 4., 0., 4., 3., 3., 3., 2., 4., 6., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 6., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 6.]
        ])

SIMPLE_MATRIX = torch.Tensor([
        [2., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1., 1., 1.,
         0., 0., 0., 0.],
        [0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0.,
         0., 0., 0., 0.],
        [0., 0., 2., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0.,
         0., 0., 0., 0.],
        [0., 0., 1., 2., 0., 0., 0., 0., 1., 0., 0., 1., 0., 1., 1., 0., 0., 0.,
         0., 0., 0., 0.],
        [0., 0., 0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         1., 1., 0., 0.],
        [1., 0., 0., 0., 0., 2., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1., 1., 1.,
         0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 1., 0., 2., 0., 1., 0., 0., 0., 0., 0., 1., 1., 1.,
         0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 2., 0., 0., 1., 0., 1., 1., 0., 0., 0.,
         0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 1., 0., 1., 0., 2., 0., 0., 0., 0., 0., 0., 0., 1.,
         0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 2., 0., 1., 1., 1., 1., 0.,
         0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0.,
         0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 1., 0., 2., 1., 1., 1., 0.,
         0., 1., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 1., 0., 1., 2., 0., 0., 0.,
         0., 0., 0., 0.],
        [1., 1., 1., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1., 0., 2., 1., 1.,
         0., 0., 0., 0.],
        [1., 1., 1., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1., 0., 1., 2., 1.,
         0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1., 1., 2.,
         0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         2., 1., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0.,
         1., 2., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 2., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 2.]])


inter_types = {
    'charged-charged': [(2, 3), (14, 6, 8)],
    'polar-charged (+)': [(15, 16, 11, 13), (14, 6, 8)],
    'polar-charged (-)': [(15, 16, 11, 13), (2, 3)],
    'polar-polar': [(15, 16, 11, 13), (15, 16, 11, 13)],
    'cation-pi': [(14, 6, 8), (4, 19, 18)],
    'polar-pi': [(11, 13), (4, 19, 18)]
}

inter_array = 7*torch.ones(22,22)
for i in range(20):
    for j in range(20):
        found_inter = 6
        for i_inter, (inter_type, inter_combo) in enumerate(inter_types.items()):
            if (i in inter_combo[0] and j in inter_combo[1]) or (j in inter_combo[0] and i in inter_combo[1]):
                found_inter = i_inter
                break
        inter_array[i,j] = found_inter

binary_inter_array = 7*torch.ones(22,22)
for i in range(20):
    for j in range(20):
        found_inter = 6
        for i_inter, (inter_type, inter_combo) in enumerate(inter_types.items()):
            if (i in inter_combo[0] and j in inter_combo[1]) or (j in inter_combo[0] and i in inter_combo[1]):
                found_inter = 0
                break
        binary_inter_array[i,j] = found_inter


# pylint: disable=no-member

NOT_LOSS_FNS = ["_get_loss_fn", "construct_loss_fn"]

# def merge_dups(h_E, inv_mapping, shape="undirected", mode="average"):
#     orig_shape = h_E.shape
#     flattened = h_E.flatten(1, 2)
#     if mode == "average":
#         condensed = scatter_mean(flattened, inv_mapping, dim=1)
#     elif mode == "add":
#         condensed = scatter_add(flattened, inv_mapping, dim=1)
#     else:
#         condensed = flattened
#     expanded_inv_mapping = inv_mapping.unsqueeze(-1).expand((-1, -1, orig_shape[-1]))
#     rescattered = torch.gather(condensed, dim=1, index=expanded_inv_mapping)
#     if shape == "undirected":
#         if mode == "select":
#             return rescattered
#         return condensed  
#     rescattered = rescattered.unflatten(1, (orig_shape[1], orig_shape[2]))
#     return rescattered

def _calc_log_composite_prob_dist(etab, E_idx, ref_seqs, x_mask, k):

    # separate selfE and pairE since we have to treat selfE differently
    self_etab = etab[:, :, 0:1] # b x L x 1 x 22 x 22
    pair_etab = etab[:, :, 1:] # b x L x 29 x 22 x 22

    # gather 22 self energies by taking the diagonal of the etab
    self_nrgs_im = torch.diagonal(self_etab, offset=0, dim1=-2, dim2=-1) # b x L x 1 x 22
    self_nrgs_im_expand = self_nrgs_im.expand(-1, -1, k - 1, -1) # b x L x 29 x 22

    # E_idx for all but self
    E_idx_jn = E_idx[:, :, 1:] # b x L x 29

    # self Es gathered from E_idx_others
    E_idx_jn_expand = E_idx_jn.unsqueeze(-1).expand(-1, -1, -1, 22) # b x L x 29 x 22
    self_nrgs_jn = torch.gather(self_nrgs_im_expand, 1, E_idx_jn_expand) # b x L x 29 x 22

    # idx matrix to gather the identity at all other residues given a residue of focus
    try:
        E_aa = torch.gather(ref_seqs.unsqueeze(-1).expand(-1, -1, k - 1), 1, E_idx_jn) # b x L x 29
    except Exception as e:
        print(e)
        raise ValueError
    nrg_mask = torch.gather(x_mask.unsqueeze(-1).expand(-1,-1, k - 1), 1, E_idx_jn) # b x L x 29

    # expand the matrix so we can gather pair energies
    E_aa = E_aa.view(list(E_idx_jn.shape) + [1, 1]).expand(-1, -1, -1, 22, -1) # b x L x 29 x 22 x 1
    # gather the 22 energies for each edge based on E_aa
    pair_nrgs_jn = torch.gather(pair_etab, 4, E_aa).squeeze(-1) # b x L x 29 x 22
    pair_nrgs_jn *= nrg_mask.unsqueeze(-1)
    # sum_(u != n,m) E_p(a_i,n; A_u)
    sum_pair_nrgs_jn = torch.sum(pair_nrgs_jn, dim=2) # b x L x 22
    pair_nrgs_im_u = sum_pair_nrgs_jn.unsqueeze(2).expand(-1, -1, k - 1, -1) - pair_nrgs_jn # b x L x 29 x 22

    # get pair_nrgs_u_jn from pair_nrgs_im_u
    E_idx_imu_to_ujn = E_idx_jn.unsqueeze(-1).expand(pair_nrgs_im_u.shape) # b x L x 29 x 22
    pair_nrgs_u_jn = torch.gather(pair_nrgs_im_u, 1, E_idx_imu_to_ujn) # b x L x 29 x 22

    # start building this wacky energy table
    self_nrgs_im_expand = self_nrgs_im_expand.unsqueeze(-1).expand(-1, -1, -1, -1, 22) # b x L x 29 x 22 x 22
    self_nrgs_jn_expand = self_nrgs_jn.unsqueeze(-1).expand(-1, -1, -1, -1, 22).transpose(-2, -1) # b x L x 29 x 22 x 22
    pair_nrgs_im_expand = pair_nrgs_im_u.unsqueeze(-1).expand(-1, -1, -1, -1, 22) # b x L x 29 x 22 x 22
    pair_nrgs_jn_expand = pair_nrgs_u_jn.unsqueeze(-1).expand(-1, -1, -1, -1, 22).transpose(-2, -1) # b x L x 29 x 22 x 22
    # Fix pair nrgs jn
    # fixed_pair_nrgs_jn_expand = torch.zeros(pair_nrgs_jn_expand.shape).to(device=etab.device)
    # for b in range(E_idx.shape[0]):
    #     for i in range(E_idx.shape[1]):
    #         for ij in range(29):
    #             j = E_idx[b,i,ij+1].item()
    #             pair_nrg_jn = torch.sum(etab[b,j,1:], dim=0)[:,0]
    #             if i in E_idx[b,j]:
    #                 ji = (E_idx[b,j] == i).nonzero(as_tuple=True)[0][0].item()
    #                 pair_nrg_jn -= etab[b,j,ji,:,0]
    #             fixed_pair_nrgs_jn_expand[b,i,ij,:] = pair_nrg_jn * torch.ones(pair_nrgs_jn_expand.shape[-2]).to(device=fixed_pair_nrgs_jn_expand.device)
    composite_nrgs = (self_nrgs_im_expand + self_nrgs_jn_expand + pair_etab + pair_nrgs_im_expand +
                        pair_nrgs_jn_expand) # b x L x 29 x 21 x 21
    composite_nrgs *= nrg_mask.unsqueeze(-1).unsqueeze(-1)
   
    # composite_nrgs = (self_nrgs_im_expand + self_nrgs_jn_expand + pair_etab + pair_nrgs_im_expand +
    #                   pair_nrgs_jn_expand) # b x L x 29 x 22 x 22
    return composite_nrgs, E_aa

def _calc_log_probs_msa(composite_nrgs, n_batch, L, k, E_aa, ref_seqs, x_mask, is_valid_aa):
    # convert energies to probabilities
    composite_nrgs_reshape = composite_nrgs.view(n_batch, L, k - 1, 22 * 22, 1) # b x L x 29 x 484 x 1
    log_composite_prob_dist = torch.log_softmax(-composite_nrgs_reshape, dim=-2).view(n_batch, L, k - 1, 22, 22) # b x L x 29 x 22 x 22
    if torch.isnan(log_composite_prob_dist).any():
        print("Nans will cause problems")
    # get the probability of the sequence
    im_probs = torch.gather(log_composite_prob_dist, 4, E_aa).squeeze(-1) # b x L x 29 x 22
    ref_seqs_expand = ref_seqs.view(list(ref_seqs.shape) + [1, 1]).expand(-1, -1, k - 1, 1) # b x L x 29 x 1
    try:
        log_edge_probs = torch.gather(im_probs, 3, ref_seqs_expand).squeeze(-1) # b x L x 29
    except Exception as e:
        print(e)
        raise ValueError
    
    # reshape masks
    
    is_valid_aa = is_valid_aa.unsqueeze(-1) # b x L x 1
    full_mask = x_mask * is_valid_aa
    n_edges = torch.sum(full_mask.expand(log_edge_probs.shape))
    # n_edges = torch.sum(full_mask.expand(log_edge_probs.shape), dim=(1, 2))
    # n_edges[n_edges == 0] = 1
    # convert to nlcpl
    log_edge_probs *= full_mask  # zero out positions that don't have residues or where the native sequence is X


    # reshape data
    log_edge_probs_seqs = -torch.sum(log_edge_probs) / n_edges
    # log_edge_probs_seqs[log_edge_probs_seqs != log_edge_probs_seqs] = 0
    return log_edge_probs_seqs, n_edges

def _calc_log_composite_prob_dist_undirected(etab, edge_idx, node_endpoint_idx, node_neighbor_idx, node_idx, ref_seqs, k):

    # Set up indices and masks
    b, C, n_tok1, n_tok2 = etab.shape
    # Set up indices and masks
    seq_idx_i = torch.gather(ref_seqs, 1, node_idx[:,:,0])
    seq_idx_j = torch.gather(ref_seqs, 1, node_idx[:,:,1])
    node_endpoint_idx_transpose = torch.cat([node_endpoint_idx[:,:,k:], node_endpoint_idx[:,:,:k]], dim=-1)
    central_mask = (node_endpoint_idx_transpose == node_neighbor_idx).to(device=etab.device)

    # Get energy for other sequence identities at all pairs of positions    
    etab_condense_shape_i = (etab.shape[0], etab.shape[1], etab.shape[2], 1)
    etab_condense_shape_j = (etab.shape[0], etab.shape[1], 1, etab.shape[3])
    etab_other_i_seq = torch.gather(etab, 3, seq_idx_j.unsqueeze(-1).unsqueeze(-1).expand(etab_condense_shape_i))
    etab_other_i_seq = etab_other_i_seq[:,:,:,0]
    # etab_other_i_seq = torch.gather(etab_other_i_seq, -1, seq_idx_i.unsqueeze(-1).unsqueeze(-1).expand((etab.shape[0], etab.shape[1], etab.shape[2], 1))).squeeze(dim=-1)
    etab_other_j_seq = torch.gather(etab, 2, seq_idx_i.unsqueeze(-1).unsqueeze(-1).expand(etab_condense_shape_j))
    etab_other_j_seq = etab_other_j_seq[:,:,0]
    # etab_other_j_seq = torch.gather(etab_other_j_seq, -2, seq_idx_j.unsqueeze(-1).unsqueeze(-1).expand((etab.shape[0], etab.shape[1], 1, etab.shape[3]))).squeeze(dim=-2)
    etab_other_i_seq_e = etab_other_i_seq.unsqueeze(2).expand(-1,-1,2*k,-1)
    etab_other_j_seq_e = etab_other_j_seq.unsqueeze(2).expand(-1,-1,2*k,-1)
    edge_idx_e = edge_idx.unsqueeze(-1).expand(etab_other_i_seq_e.shape)
    neighbor_nrg_i = torch.gather(etab_other_i_seq_e, 1, edge_idx_e)
    neighbor_nrg_j = torch.gather(etab_other_j_seq_e, 1, edge_idx_e)

    #  Fix transpose issue
    first_node_endpoint_idx = node_endpoint_idx[:,:,0].unsqueeze(-1).expand(node_endpoint_idx.shape)
    neighbor_endpoint_first = torch.gather(first_node_endpoint_idx, 1, edge_idx)
    transpose_mask = (neighbor_endpoint_first == node_endpoint_idx).unsqueeze(-1).expand(neighbor_nrg_j.shape)
    neighbor_nrg_other = neighbor_nrg_i * transpose_mask + neighbor_nrg_j * (~transpose_mask)

    #  Handle central edge differently and sum over neighbors
    central_mask_e = central_mask.unsqueeze(-1).expand(neighbor_nrg_other.shape)
    neighbor_nrg_other *= ~central_mask_e
    neighbor_nrg_is = torch.sum(neighbor_nrg_other[:,:,1:k,:], dim=-2)
    neighbor_nrg_js = torch.sum(neighbor_nrg_other[:,:,k+1:,:], dim=-2)

    #  Handle self edges differently
    self_nrg = torch.diagonal(etab, dim1=-2, dim2=-1)
    self_nrg_e = self_nrg.unsqueeze(-2).expand(-1,-1,2*k,-1)
    edge_idx_se = torch.cat([edge_idx_e[:,:,0,:].unsqueeze(-2), edge_idx_e[:,:,k,:].unsqueeze(-2)], dim=-2)
    self_nrg = torch.gather(self_nrg_e, 1, edge_idx_se)
    self_nrg_i = self_nrg[:,:,0,:]
    self_nrg_j = self_nrg[:,:,1,:]

    # Combine energies to create energy table
    self_nrg_i_e = self_nrg_i.unsqueeze(-1).expand(-1, -1, -1, n_tok1) # b x C x 21 x 21
    self_nrg_j_e = self_nrg_j.unsqueeze(-1).expand(-1, -1, -1, n_tok1).transpose(-2, -1) # b x C x 21 x 21
    neighbor_nrg_is_e = neighbor_nrg_is.unsqueeze(-1).expand(-1, -1, -1, n_tok1) # b x C x 21 x 21
    neighbor_nrg_js_e = neighbor_nrg_js.unsqueeze(-1).expand(-1, -1, -1, n_tok1).transpose(-2, -1) # b x C x 21 x 21
    nrg = self_nrg_i_e + self_nrg_j_e + etab + neighbor_nrg_is_e + neighbor_nrg_js_e

    return nrg

def edge_loss(etab, E_idx, data, options={}):
    ref_seqs = data['seqs']
    x_mask = data['x_mask']
    n_batch, L, k, _ = etab.shape
    etab = torch.log_softmax(-etab, dim=-1)
    etab = etab.unsqueeze(-1).view(n_batch, L, k, 20, 20) # b x L x 30 x 20 x 20

    # X is encoded as 20 so lets just add an extra row/col of zeros
    pad = (0, 2, 0, 2)
    etab = F.pad(etab, pad, "constant", 0)
    isnt_x_aa = (torch.logical_and(ref_seqs != 20, ref_seqs != 21)).float() # b x L
    E_aa = torch.gather(ref_seqs.unsqueeze(-1).expand(-1, -1, 30), 1, E_idx[:, :, :])
    E_aa = E_aa.view(list(E_idx[:, :, :].shape) + [1, 1]).expand(-1, -1, -1, 22, -1)
    im_probs = torch.gather(etab, 4, E_aa).squeeze(-1) # b x L x 29 x 22
    ref_seqs_expand = ref_seqs.view(list(ref_seqs.shape) + [1, 1]).expand(-1, -1, k , 1) # b x L x 30 x 1
    log_edge_probs = torch.gather(im_probs, 3, ref_seqs_expand).squeeze(-1) # b x L x 29

    # reshape masks
    x_mask = x_mask.unsqueeze(-1) # b x L x 1
    isnt_x_aa = isnt_x_aa.unsqueeze(-1) # b x L x 1
    full_mask = x_mask * isnt_x_aa
    n_edges = torch.sum(full_mask.expand(log_edge_probs.shape))

    # convert to nlcpl
    log_edge_probs *= full_mask  # zero out positions that don't have residues or where the native sequence is X
    edge_return = -torch.sum(log_edge_probs) / n_edges
    return edge_return, int(n_edges)

def edge_loss_msa(etab, E_idx, data, options={}):
    ref_seqs = torch.transpose(data['seqs'], 1, 2)
    isnt_x_aa = (torch.logical_and(ref_seqs[:,0,:] != 20, ref_seqs[:,0,:] != 21)).float() # b x L
    pair_etab_no_gap = data['pair_etabs'][:,:,:,:20,:20]
    pair_etab_no_gap_reshape = pair_etab_no_gap.flatten(-2,-1)
    pair_etab_no_gap = torch.div(pair_etab_no_gap_reshape, torch.sum(pair_etab_no_gap_reshape, -1).unsqueeze(-1)).view(pair_etab_no_gap.shape)
    pair_etab = pair_etab_no_gap.to(dtype=etab.dtype, device=etab.device)
    x_mask = data['x_mask']
    n_batch, L, k, _ = etab.shape
    etab = torch.log_softmax(-etab, dim=-1)
    etab = etab.unsqueeze(-1).view(n_batch, L, k, 20, 20) # b x L x 30 x 20 x 20

    # X is encoded as 20 so lets just add an extra row/col of zeros
    pad = (0, 2, 0, 2)
    etab = F.pad(etab, pad, "constant", 0)
    pair_etab = F.pad(pair_etab, pad, "constant", 0)
    log_edge_probs = torch.mul(etab, pair_etab).flatten(-2, -1).sum(-1)

    # reshape masks
    x_mask = x_mask.unsqueeze(-1) # b x L x 1
    isnt_x_aa = isnt_x_aa.unsqueeze(-1) # b x L x 1
    full_mask = x_mask * isnt_x_aa
    n_edges = torch.sum(full_mask.expand(log_edge_probs.shape))

    # convert to nlcpl
    log_edge_probs *= full_mask  # zero out positions that don't have residues or where the native sequence is X
    edge_return = -torch.sum(log_edge_probs) / n_edges
    return edge_return, int(n_edges)


def nlpl(etab, E_idx, data, options={}):
    """ Negative log psuedo-likelihood
        Returns negative log psuedolikelihoods per residue, with padding residues masked """
    ref_seqs = data['seqs']
    x_mask = data['x_mask']
    n_batch, L, k, _ = etab.shape
    etab = etab.unsqueeze(-1).view(n_batch, L, k, 20, 20)

    # X is encoded as 20 so lets just add an extra row/col of zeros
    pad = (0, 2, 0, 2)
    etab = F.pad(etab, pad, "constant", 0)
    isnt_x_aa = (torch.logical_and(ref_seqs != 20, ref_seqs != 21)).float() # b x L

    # separate selfE and pairE since we have to treat selfE differently
    self_etab = etab[:, :, 0:1]
    pair_etab = etab[:, :, 1:]
    # idx matrix to gather the identity at all other residues given a residue of focus
    E_aa = torch.gather(ref_seqs.unsqueeze(-1).expand(-1, -1, k - 1), 1, E_idx[:, :, 1:])
    E_aa = E_aa.view(list(E_idx[:, :, 1:].shape) + [1, 1]).expand(-1, -1, -1, 22, -1)
    # gather the 22 energies for each edge based on E_aa
    pair_nrgs = torch.gather(pair_etab, 4, E_aa).squeeze(-1)
    # gather 22 self energies by taking the diagonal of the etab
    self_nrgs = torch.diagonal(self_etab, offset=0, dim1=-2, dim2=-1)
    # concat the two to get a full edge etab
    edge_nrgs = torch.cat((self_nrgs, pair_nrgs), dim=2)
    # get the avg nrg for 22 possible aa identities at each position
    aa_nrgs = torch.sum(edge_nrgs, dim=2)

    # convert energies to probabilities
    log_all_aa_probs = torch.log_softmax(-aa_nrgs, dim=2)
    # get the probability of the sequence
    log_seqs_probs = torch.gather(log_all_aa_probs, 2, ref_seqs.unsqueeze(-1)).squeeze(-1)

    full_mask = x_mask * isnt_x_aa
    n_res = torch.sum(x_mask * isnt_x_aa)

    # convert to nlpl
    log_seqs_probs *= full_mask  # zero out positions that don't have residues or where the native sequence is X
    nlpl_return = -torch.sum(log_seqs_probs) / n_res
    return nlpl_return, int(n_res)

def nlpl_msa(etab, E_idx, data, options={}):
    """ Negative log psuedo-likelihood
        Returns negative log psuedolikelihoods per residue, with padding residues masked """
    ref_seqs = data['seqs'][0]
    x_mask = data['x_mask']
    n_batch, L, k, _ = etab.shape
    etab = etab.unsqueeze(-1).view(n_batch, L, k, 20, 20)

    # X is encoded as 20 so lets just add an extra row/col of zeros
    pad = (0, 2, 0, 2)
    etab = F.pad(etab, pad, "constant", 0)
    isnt_x_aa = (torch.logical_and(ref_seqs != 20, ref_seqs != 21)).float() # b x L

    # separate selfE and pairE since we have to treat selfE differently
    self_etab = etab[:, :, 0:1]
    pair_etab = etab[:, :, 1:]
    # idx matrix to gather the identity at all other residues given a residue of focus
    E_aa = torch.gather(ref_seqs.unsqueeze(-1).expand(-1, -1, k - 1), 1, E_idx[:, :, 1:])
    E_aa = E_aa.view(list(E_idx[:, :, 1:].shape) + [1, 1]).expand(-1, -1, -1, 22, -1)
    # gather the 22 energies for each edge based on E_aa
    pair_nrgs = torch.gather(pair_etab, 4, E_aa).squeeze(-1)
    # gather 22 self energies by taking the diagonal of the etab
    self_nrgs = torch.diagonal(self_etab, offset=0, dim1=-2, dim2=-1)
    # concat the two to get a full edge etab
    edge_nrgs = torch.cat((self_nrgs, pair_nrgs), dim=2)
    # get the avg nrg for 22 possible aa identities at each position
    aa_nrgs = torch.sum(edge_nrgs, dim=2)

    # convert energies to probabilities
    log_all_aa_probs = torch.log_softmax(-aa_nrgs, dim=2)
    # get the probability of the sequence
    log_seqs_probs = torch.gather(log_all_aa_probs, 2, ref_seqs.unsqueeze(-1)).squeeze(-1)

    full_mask = x_mask * isnt_x_aa
    n_res = torch.sum(x_mask * isnt_x_aa)

    # convert to nlpl
    log_seqs_probs *= full_mask  # zero out positions that don't have residues or where the native sequence is X
    nlpl_return = -torch.sum(log_seqs_probs) / n_res
    return nlpl_return, int(n_res)

def nlcpl(etab, E_idx, data, options={}, use_sc_mask=False, per_protein_loss=False, per_residue_loss=False, weight_inter_types=None):
    """ Negative log composite psuedo-likelihood
        Averaged nlcpl per residue, across batches
        p(a_i,m ; a_j,n) =
            softmax [
                E_s(a_i,m) + E_s(a_j,n)
                + E_p(a_i,m ; a_j,n)
                + sum_(u != m,n) [
                    E_p(a_i,m; A_u)
                    + E_p(A_u, a_j,n)
                    ]
                ]

        Returns: log likelihoods per residue, as well as tensor mask
    """
    ref_seqs = data['seqs']
    x_mask = data['x_mask']
    x_mask_sc = data['x_mask_sc']

    n_batch, L, k, _ = etab.shape
    etab = etab.unsqueeze(-1).view(n_batch, L, k, 20, 20)

    # X is encoded as 20 so lets just add an extra row/col of zeros
    pad = (0, 2, 0, 2)
    etab = F.pad(etab, pad, "constant", 0)

    isnt_x_aa = (torch.logical_and(ref_seqs != 20, ref_seqs != 21)).float() # b x L

    # separate selfE and pairE since we have to treat selfE differently
    self_etab = etab[:, :, 0:1] # b x L x 1 x 22 x 22
    pair_etab = etab[:, :, 1:] # b x L x 29 x 22 x 22

    # gather 22 self energies by taking the diagonal of the etab
    self_nrgs_im = torch.diagonal(self_etab, offset=0, dim1=-2, dim2=-1) # b x L x 1 x 22
    self_nrgs_im_expand = self_nrgs_im.expand(-1, -1, k - 1, -1) # b x L x 29 x 22

    # E_idx for all but self
    E_idx_jn = E_idx[:, :, 1:] # b x L x 29

    # self Es gathered from E_idx_others
    E_idx_jn_expand = E_idx_jn.unsqueeze(-1).expand(-1, -1, -1, 22) # b x L x 29 x 22
    self_nrgs_jn = torch.gather(self_nrgs_im_expand, 1, E_idx_jn_expand) # b x L x 29 x 22

    # idx matrix to gather the identity at all other residues given a residue of focus
    E_aa = torch.gather(ref_seqs.unsqueeze(-1).expand(-1, -1, k - 1), 1, E_idx_jn) # b x L x 29
    # expand the matrix so we can gather pair energies
    E_aa = E_aa.view(list(E_idx_jn.shape) + [1, 1]).expand(-1, -1, -1, 22, -1) # b x L x 29 x 22 x 1
    # gather the 22 energies for each edge based on E_aa
    pair_nrgs_jn = torch.gather(pair_etab, 4, E_aa).squeeze(-1) # b x L x 29 x 22
    # sum_(u != n,m) E_p(a_i,n; A_u)
    sum_pair_nrgs_jn = torch.sum(pair_nrgs_jn, dim=2) # b x L x 22
    pair_nrgs_im_u = sum_pair_nrgs_jn.unsqueeze(2).expand(-1, -1, k - 1, -1) - pair_nrgs_jn # b x L x 29 x 22

    # get pair_nrgs_u_jn from pair_nrgs_im_u
    E_idx_imu_to_ujn = E_idx_jn.unsqueeze(-1).expand(pair_nrgs_im_u.shape) # b x L x 29 x 22
    pair_nrgs_u_jn = torch.gather(pair_nrgs_im_u, 1, E_idx_imu_to_ujn) # b x L x 29 x 22

    # start building this wacky energy table
    self_nrgs_im_expand = self_nrgs_im_expand.unsqueeze(-1).expand(-1, -1, -1, -1, 22) # b x L x 29 x 22 x 22
    self_nrgs_jn_expand = self_nrgs_jn.unsqueeze(-1).expand(-1, -1, -1, -1, 22).transpose(-2, -1) # b x L x 29 x 22 x 22
    pair_nrgs_im_expand = pair_nrgs_im_u.unsqueeze(-1).expand(-1, -1, -1, -1, 22) # b x L x 29 x 22 x 22
    pair_nrgs_jn_expand = pair_nrgs_u_jn.unsqueeze(-1).expand(-1, -1, -1, -1, 22).transpose(-2, -1) # b x L x 29 x 22 x 22

    composite_nrgs = (self_nrgs_im_expand + self_nrgs_jn_expand + pair_etab + pair_nrgs_im_expand +
                      pair_nrgs_jn_expand) # b x L x 29 x 21 x 21

    # convert energies to probabilities
    composite_nrgs_reshape = composite_nrgs.view(n_batch, L, k - 1, 22 * 22, 1) # b x L x 29 x 484 x 1
    log_composite_prob_dist = torch.log_softmax(-composite_nrgs_reshape, dim=-2).view(n_batch, L, k - 1, 22, 22) # b x L x 29 x 22 x 22
    # get the probability of the sequence
    im_probs = torch.gather(log_composite_prob_dist, 4, E_aa).squeeze(-1) # b x L x 29 x 22
    ref_seqs_expand = ref_seqs.view(list(ref_seqs.shape) + [1, 1]).expand(-1, -1, k - 1, 1) # b x L x 29 x 1
    log_edge_probs = torch.gather(im_probs, 3, ref_seqs_expand).squeeze(-1) # b x L x 29

    # reshape masks
    x_mask = x_mask.unsqueeze(-1) # b x L x 1
    isnt_x_aa = isnt_x_aa.unsqueeze(-1) # b x L x 1
    full_mask = x_mask * isnt_x_aa
    if use_sc_mask and x_mask_sc.numel() > 0:
        x_mask_sc = x_mask_sc.unsqueeze(-1)
        full_mask *= (~x_mask_sc.to(dtype=torch.bool)).to(dtype=full_mask.dtype)
        if torch.sum(full_mask) == 0:
            full_mask = torch.ones_like(full_mask).to(dtype=full_mask.dtype, device=full_mask.device)
    
    # convert to nlcpl
    log_edge_probs *= full_mask  # zero out positions that don't have residues or where the native sequence is X
    if per_residue_loss:
        return log_edge_probs, full_mask
    if weight_inter_types is not None:
        seq_self_expand = data['seqs'].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(data['seqs'].shape[0], data['seqs'].shape[1], 30, 22, 1)
        seq_neighbors = torch.gather(data['seqs'].unsqueeze(-1).expand(-1, -1, 30), 1, E_idx)
        seq_neighbors_expand = seq_neighbors.unsqueeze(-1)
        if weight_inter_types == 'all':
            inter_array_expand = inter_array.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(data['seqs'].shape[0], data['seqs'].shape[1], 30, 22, 22).to(device=log_edge_probs.device)
        elif weight_inter_types == 'binary':
            inter_array_expand = binary_inter_array.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(data['seqs'].shape[0], data['seqs'].shape[1], 30, 22, 22).to(device=log_edge_probs.device)
        inter_array_self = torch.gather(inter_array_expand, -1, seq_self_expand).squeeze(-1)
        inter_array_neighbors = torch.gather(inter_array_self, -1, seq_neighbors_expand).squeeze(-1)[:,:,1:].to(dtype=torch.int64)

        inter_counts = torch.bincount(inter_array_neighbors[inter_array_neighbors <= 6], minlength=7).to(dtype=torch.float64, device=log_edge_probs.device)
        total_count = inter_counts.sum()
        inter_ratios = total_count / inter_counts
        inter_weights = torch.zeros_like(inter_array_neighbors, dtype=torch.float64)
        inter_weights[inter_array_neighbors <= 6] = inter_ratios[inter_array_neighbors[inter_array_neighbors <= 6]]

        log_edge_probs_weighted = log_edge_probs * inter_weights
        log_edge_probs_weighted /= torch.sum(inter_weights)
        nlcpl_return = -1*torch.sum(log_edge_probs_weighted)
        n_edges = torch.sum(full_mask.expand(log_edge_probs.shape))
    else:
        if per_protein_loss:
            n_edges = torch.sum(full_mask.expand(log_edge_probs.shape), dim=(1,2))
            nlcpl_return = -1*torch.div(torch.sum(log_edge_probs, dim=(1,2)), n_edges)
            n_edges = torch.sum(n_edges)
            nlcpl_return = torch.mean(nlcpl_return)
        else:
            n_edges = torch.sum(full_mask.expand(log_edge_probs.shape))
            nlcpl_return = -1*torch.sum(log_edge_probs) / n_edges
    
    if nlcpl_return.isnan().item():
        print("ALARM")
        # print(nlcpl_return)
        # for i in range(data['x_mask_sc'].shape[0]):
        #     print('\t', torch.sum((1 -(data['x_mask_sc'][i])) * (data['x_mask'][i])).item())
        #     print('\t', torch.sum((1 -(data['x_mask_sc'][i])) * (data['x_mask'][i]) * (isnt_x_aa[i,:,0])).item())
        # print(n_edges)
        # print(data['ids'])
        # print(data['X'].shape, data['x_mask_sc'].shape)
        # print(data['seqs'])
        nlcpl_return = torch.Tensor([0])
        return nlcpl_return, -1

    return nlcpl_return, int(n_edges)

# def nlcpl_fixed(etab, E_idx, data, options={}):

#     edge_idx, node_endpoint_idx, node_neighbor_idx = data['edge_update_inds']
#     ref_seqs = data['seqs']
#     x_mask = data['x_mask']
#     inv_mapping = data['mapping']
#     etab = merge_dups(etab, inv_mapping, shape="undirected", mode="average")
#     n_batch, C, n_tok = etab.shape
#     L = ref_seqs.shape[1]
#     k = E_idx.shape[-1]
#     n_tok = int(np.sqrt(n_tok))
#     etab = etab.unsqueeze(-1).view(n_batch, C, n_tok, n_tok)
#     self_nrg_mask = node_endpoint_idx[:,:,0] == node_endpoint_idx[:,:,k]
#     x_mask_expand = x_mask.unsqueeze(-1).expand(n_batch, L, k)
#     x_mask_expand = torch.gather(x_mask_expand, -1, ref_seqs.unsqueeze(-1).expand(-1, -1, k))
#     x_mask = merge_dups(x_mask_expand.unsqueeze(-1).float(), inv_mapping, shape="undirected", mode="average").squeeze(-1)
#     x_mask = x_mask > 0.6

#     # X is encoded as 21 so lets just add an extra row/col of zeros
#     # - is encoded as 20 so add another extra row/col of zeros
#     pad = (0, 2, 0, 2)
#     etab = F.pad(etab, pad, "constant", 0)
#     n_tok += 2
#     node_idx = node_endpoint_idx[:,:,[0,k]]
#     is_valid_aa = (torch.logical_and(ref_seqs != 20, ref_seqs != 21)).float() # b x L
#     iva_i = torch.gather(is_valid_aa, 1, node_idx[:,:,0])
#     iva_j = torch.gather(is_valid_aa, 1, node_idx[:,:,1])
#     is_valid_aa = (iva_i * iva_j) # b x C

#     # weight edges
#     edge_weight = torch.ones([n_batch, L, k, 1]).to(device=etab.device)
#     edge_weight = merge_dups(edge_weight, inv_mapping, shape="undirected", mode="add").squeeze(-1)
#     current_min = edge_weight.min()
#     current_max = edge_weight.max()

#     new_min = 1
#     new_max = new_min * options['undirected_edge_scale']

#     # Rescale the values to the new range
#     edge_weight = ((edge_weight - current_max) / (current_min - current_max)) * (new_min - new_max) + new_max

#     # if options['edge_scale_inverse']:
#     # else:
#     #     edge_weight = ((edge_weight - current_min) / (current_max - current_min)) * (new_max - new_min) + new_min

#     # get composite energies
#     composite_nrgs = _calc_log_composite_prob_dist_undirected(etab, edge_idx, node_endpoint_idx, node_neighbor_idx, node_idx, ref_seqs, k) # b x C x 484 x 1
#     composite_nrgs *= edge_weight.unsqueeze(-1).unsqueeze(-1)

#     # convert energies to probabilities
#     composite_nrgs_reshape = composite_nrgs.view(n_batch, etab.shape[1], n_tok * n_tok, 1) # b x C x 484 x 1
#     log_composite_prob_dist = torch.log_softmax(-composite_nrgs_reshape, dim=-2).view(n_batch, etab.shape[1], n_tok, n_tok) # b x C x 22 x 22
#     seq_idx_i = torch.gather(ref_seqs, 1, node_idx[:,:,0])
#     seq_idx_j = torch.gather(ref_seqs, 1, node_idx[:,:,1])
#     log_edge_probs_part = torch.gather(log_composite_prob_dist, 2, seq_idx_i.unsqueeze(-1).unsqueeze(-1).expand(n_batch,etab.shape[1],1,n_tok)).squeeze(dim=-2)
#     log_edge_probs = torch.gather(log_edge_probs_part, 2, seq_idx_j.unsqueeze(-1)).squeeze(dim=-1)

    

#     # reshape masks

#     full_mask = x_mask * is_valid_aa * ~self_nrg_mask
#     # full_mask_expand = full_mask.expand(log_edge_probs.shape)
#     n_edges = torch.sum(full_mask)
#     # convert to nlcpl
#     log_edge_probs *= full_mask  # zero out positions that don't have residues or where the native sequence is X
#     nlcpl_return = -torch.sum(log_edge_probs) / n_edges

#     return nlcpl_return, int(n_edges)

def msa_av_nlcpl(etab, E_idx, data, options={}):
    """ Negative log composite psuedo-likelihood
        Averaged nlcpl per residue averaged over msa, across batches
        p(a_i,m ; a_j,n) =
            softmax [
                E_s(a_i,m) + E_s(a_j,n)
                + E_p(a_i,m ; a_j,n)
                + sum_(u != m,n) [
                    E_p(a_i,m; A_u)
                    + E_p(A_u, a_j,n)
                    ]
                ]

        Returns: log likelihoods per residue, as well as tensor mask
    """
    ref_seqs_msa = data['seqs']
    # max_d = ref_seqs_msa.shape[-1] 
    ref_seqs_msa_split = list(torch.tensor_split(ref_seqs_msa, ref_seqs_msa.shape[2], dim=2))
    try:
        gpu = etab.get_device()
        if gpu == -1:
            dev = "cpu"
        else:
            dev = f"cuda:{gpu}"
    except:
        dev = "cpu"
    x_mask = data['x_mask']

    n_batch, L, k, n_tok = etab.shape
    L = ref_seqs_msa_split[0].shape[1]
    k = 30

    n_tok = int(np.sqrt(n_tok))
    etab = etab.unsqueeze(-1).view(n_batch, L, k, n_tok, n_tok)
    
    # X is encoded as 21 so lets just add an extra row/col of zeros
    # - is encoded as 20 so add another extra row/col of zeros
    pad = (0, 2, 0, 2)
    etab = F.pad(etab, pad, "constant", 0)
    nlcpl_msa = 0
    # reshape masks
    x_mask = x_mask.unsqueeze(-1) # b x L x 1
    for i, ref_seqs in enumerate(ref_seqs_msa_split):
        ref_seqs = torch.squeeze(ref_seqs, dim=2)
        is_valid_aa = (torch.logical_and(ref_seqs != 20, ref_seqs != 21)).float().to(etab.device) # b x L
        full_mask = x_mask[:,:,0] * is_valid_aa
        # get composite energies and log probabilities
        composite_nrgs, E_aa = _calc_log_composite_prob_dist(etab, E_idx, ref_seqs, full_mask, k)
        log_edge_probs_seqs, n_edges = _calc_log_probs_msa(composite_nrgs, n_batch, L, k, E_aa, ref_seqs, x_mask, is_valid_aa)
        nlcpl_msa += log_edge_probs_seqs
        torch.cuda.empty_cache()
        if 'msa_depth_lim' in options and i > options['msa_depth_lim']:
            break
    nlcpl_return = torch.divide(nlcpl_msa, i+1)
    return nlcpl_return, int(n_edges)

def msa_weighted_nlcpl(etab, E_idx, data, options={}):
    """ Negative log composite psuedo-likelihood
        Averaged nlcpl per residue averaged over msa, across batches
        p(a_i,m ; a_j,n) =
            softmax [
                E_s(a_i,m) + E_s(a_j,n)
                + E_p(a_i,m ; a_j,n)
                + sum_(u != m,n) [
                    E_p(a_i,m; A_u)
                    + E_p(A_u, a_j,n)
                    ]
                ]

        Returns: log likelihoods per residue, as well as tensor mask
    """
    ref_seqs_msa = data['seqs']
    # max_d = ref_seqs_msa.shape[-1] 
    ref_seqs_msa_split = list(torch.tensor_split(ref_seqs_msa, ref_seqs_msa.shape[2], dim=2))
    x_mask = data['x_mask']
    pair_etab_no_gap = data['pair_etabs'][:,:,:,:20,:20]
    pair_etab_no_gap_reshape = pair_etab_no_gap.flatten(-2,-1)
    pair_etab_no_gap = torch.div(pair_etab_no_gap_reshape, torch.sum(pair_etab_no_gap_reshape, -1).unsqueeze(-1)).view(pair_etab_no_gap.shape)
    pair_stats = pair_etab_no_gap.to(dtype=etab.dtype, device=etab.device)

    n_batch, L, k, n_tok = etab.shape
    L = ref_seqs_msa_split[0].shape[1]
    k = 30

    n_tok = int(np.sqrt(n_tok))
    etab = etab.unsqueeze(-1).view(n_batch, L, k, n_tok, n_tok)
    
    # X is encoded as 21 so lets just add an extra row/col of zeros
    # - is encoded as 20 so add another extra row/col of zeros
    pad = (0, 2, 0, 2)
    etab = F.pad(etab, pad, "constant", 0)
    n_tok += 2
    pair_stats = F.pad(pair_stats, pad, "constant", 0)
    # reshape masks
    x_mask = x_mask.unsqueeze(-1) # b x L x 1
    
    composite_nrgs_av = torch.zeros((n_batch, L, k-1, n_tok, n_tok)).to(device=etab.device, dtype=etab.dtype)
    pair_stats = pair_stats[:,:,1:]
    for i, ref_seqs in enumerate(ref_seqs_msa_split):
        ref_seqs = torch.squeeze(ref_seqs, dim=2)
        is_valid_aa = (torch.logical_and(ref_seqs != 20, ref_seqs != 21)).float().to(etab.device) # b x L
        full_mask = x_mask[:,:,0] * is_valid_aa
        # get composite energies and log probabilities
        composite_nrgs, _ = _calc_log_composite_prob_dist(etab, E_idx, ref_seqs, full_mask, k)
        composite_nrgs_av += composite_nrgs
        torch.cuda.empty_cache()
        if 'msa_depth_lim' in options and i == options['msa_depth_lim'] - 1:
            break
    composite_nrgs_av = torch.div(composite_nrgs_av, i + 1).view((n_batch, L, k-1, n_tok**2))
    composite_log_probs = torch.log_softmax(-composite_nrgs_av, dim=-1).unsqueeze(-1).view(n_batch, L, k-1, n_tok, n_tok)
    composite_log_probs = torch.mul(composite_log_probs, pair_stats).flatten(-2, -1).sum(-1)
    n_edges = torch.sum(x_mask.expand(composite_log_probs.shape))
    composite_log_probs *= x_mask

    nlcpl_return = -torch.sum(composite_log_probs) / n_edges
    return nlcpl_return, int(n_edges)

def msa_av_split_nlcpl(etab, E_idx, data, options={}):
    """ Negative log composite psuedo-likelihood
        Averaged nlcpl per residue averaged over msa, across batches
        p(a_i,m ; a_j,n) =
            softmax [
                E_s(a_i,m) + E_s(a_j,n)
                + E_p(a_i,m ; a_j,n)
                + sum_(u != m,n) [
                    E_p(a_i,m; A_u)
                    + E_p(A_u, a_j,n)
                    ]
                ]

        Returns: log likelihoods per residue, as well as tensor mask
    """
    ref_seqs_msa = data['seqs']
    # max_d = ref_seqs_msa.shape[-1] 
    ref_seqs_msa_split = list(torch.tensor_split(ref_seqs_msa, ref_seqs_msa.shape[2], dim=2))
    try:
        gpu = etab.get_device()
        if gpu == -1:
            dev = "cpu"
        else:
            dev = f"cuda:{gpu}"
    except:
        dev = "cpu"
    x_mask = data['x_mask']

    n_batch, L, k, n_tok = etab.shape
    L = ref_seqs_msa_split[0].shape[1]
    k = 30

    n_tok = int(np.sqrt(n_tok))
    etab = etab.unsqueeze(-1).view(n_batch, L, k, n_tok, n_tok)
    
    # X is encoded as 21 so lets just add an extra row/col of zeros
    # - is encoded as 20 so add another extra row/col of zeros
    pad = (0, 2, 0, 2)
    etab = F.pad(etab, pad, "constant", 0)
    # reshape masks
    x_mask = x_mask.unsqueeze(-1) # b x L x 1
    ref_seqs_msa_split = ref_seqs_msa_split[:options['msa_depth_lim']]
    num_msa_seqs = len(ref_seqs_msa_split)
    ref_seqs_msa = torch.cat(ref_seqs_msa_split, dim=0)
    ref_seqs_msa = torch.squeeze(ref_seqs_msa, dim=2)
    etab_msa = etab.repeat_interleave(repeats = num_msa_seqs, dim=0)
    E_idx_msa = E_idx.repeat_interleave(repeats = num_msa_seqs, dim=0)
    x_mask_msa = x_mask.repeat_interleave(repeats = num_msa_seqs, dim=0)
    is_valid_aa = (torch.logical_and(ref_seqs_msa != 20, ref_seqs_msa != 21)).float().to(etab.device) # b x L
    full_mask = x_mask_msa[:,:,0] * is_valid_aa
    composite_nrgs, E_aa = _calc_log_composite_prob_dist(etab_msa, E_idx_msa, ref_seqs_msa, full_mask, k)
    nlcpl_return, n_edges = _calc_log_probs_msa(composite_nrgs, n_batch * num_msa_seqs, L, k, E_aa, ref_seqs_msa, x_mask_msa, is_valid_aa)
    torch.cuda.empty_cache()
    return nlcpl_return, int(n_edges)


def evcouplings_loss_corr(etab, E_idx, data, options={}):
    n_batch, L, k, n_tok = etab.shape
    n_tok = int(np.sqrt(n_tok))
    ev_etab = data['ev_etabs'].reshape(etab.shape)
    x_mask = data['x_mask']
    ref_seqs = data['seqs']
    is_valid_aa = (torch.logical_and(ref_seqs != 20, ref_seqs != 21)).float() # b x L
    ev_mask = ev_etab != 0
    # reshape masks
    full_mask = x_mask.unsqueeze(-1).unsqueeze(-1).expand(ev_mask.shape) * is_valid_aa.unsqueeze(-1).unsqueeze(-1).expand(ev_mask.shape) * ev_mask
    n_edges = torch.sum(full_mask)
    ev_etab_masked = ev_etab * full_mask
    etab_masked = etab * full_mask
    ev_etab_nz = torch.count_nonzero(ev_etab_masked, dim=-1)
    etab_nz = torch.count_nonzero(etab_masked, dim=-1)
    ev_etab_mean = torch.sum(ev_etab_masked, dim=-1) / ev_etab_nz
    etab_mean = torch.sum(etab_masked, dim=-1) / etab_nz
    ev_etab_norm = (ev_etab_masked - ev_etab_mean.unsqueeze(-1)) * full_mask
    etab_norm = (etab_masked - etab_mean.unsqueeze(-1)) * full_mask
    pearson = 1 - (torch.sum(ev_etab_norm * etab_norm, dim=-1) / (torch.sqrt(torch.sum(ev_etab_norm**2, dim=-1)) * torch.sqrt(torch.sum(etab_norm**2, dim=-1))))
    
    loss = torch.sum(pearson) / n_edges

    return loss, n_edges

def evcouplings_loss_mse(etab, E_idx, data, options={}):
    n_batch, L, k, n_tok = etab.shape
    n_tok = int(np.sqrt(n_tok))
    etab = etab.unsqueeze(-1).view(n_batch, L, k, n_tok, n_tok)
    loss = F.mse_loss(etab, data['ev_etabs'], reduction='none')
    x_mask = data['x_mask']
    ref_seqs = data['seqs']
    is_valid_aa = (torch.logical_and(ref_seqs != 20, ref_seqs != 21)).float() # b x L
    ev_mask = data['ev_etabs'] != 0
    # reshape masks
    full_mask = x_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(ev_mask.shape) * is_valid_aa.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(ev_mask.shape) * ev_mask
    n_edges = torch.sum(full_mask)
    loss = (loss * full_mask.float()).sum() / n_edges

    return loss, n_edges

def ints_to_seq_torch(seq):
    return "".join(ints_to_seq(seq.cpu().numpy()))

def esm_cmap_loss(nodes, data, esm, batch_converter):
    real_contacts = data['X'][:,:,1,:].unsqueeze(1) - data['X'][:,:,1,:].unsqueeze(2)
    real_contacts = (real_contacts**2).sum(dim=3)
    contacts_mask = (data['x_mask'].unsqueeze(1) * data['x_mask'].unsqueeze(2)).to(device=real_contacts.device)
    seqs = torch.argmax(torch.softmax(nodes, dim=-1), dim=-1) * data['x_mask']
    aa_seqs = [ints_to_seq_torch(seq) for seq in seqs]
    esm_data = []
    for i, aa_seq in enumerate(aa_seqs):
        esm_data.append(("protein_" + str(i), aa_seq[:data['seq_lens'][i]])) 
    _, _, batch_tokens = batch_converter(esm_data)
    with torch.no_grad():
        results = esm(batch_tokens, repr_layers=[30], return_contacts=True)
    pred_contacts = results["contacts"].to(real_contacts.device)
    diff = (pred_contacts - real_contacts) ** 2 * contacts_mask
    cmap_loss = torch.div(diff.sum(dim=(1,2)), contacts_mask.sum(dim=(1,2)))
    cmap_loss = cmap_loss.mean()
    cmap_loss.requires_grad = True
    return cmap_loss, -1

def make_mask(sz, thresh, inv=False):
    mask = torch.full((sz, sz), True, dtype=torch.bool)
    for i in range(sz):
        mask[i, i] = False
        for j in range(i - thresh, i + thresh + 1):
            if 0 <= j < sz and j != i:
                mask[i, j] = False
    if inv: mask = ~mask
    return mask.unsqueeze(0)

def apply_mask(loss, mask, mask_type=None):
    if mask_type == 'far':
        mask *= make_mask(loss.shape[1], 6).to(device=mask.device)
    elif mask_type == 'close':
        mask *= make_mask(loss.shape[1], 6, inv=True).to(device=mask.device)
    diag_indices = torch.arange(mask.shape[1]).to(mask.device)
    diag_mask = torch.abs(diag_indices.view(-1, 1) - diag_indices) > 6
    diag_mask = diag_mask.unsqueeze(0)
    mask *= diag_mask
    loss *= mask
    return -1 * loss.sum() / mask.sum(), mask.sum()

def nlll_loss(nodes, data, eps=1e-6, use_sc_mask=False):
    log_l = torch.softmax(nodes, dim=-1)
    criterion = torch.nn.NLLLoss(reduction='none')
    seqs_encoded = data['seqs']
    pad = (0, 2)
    log_l = F.pad(log_l, pad, "constant", 0)
    # seqs_encoded = torch.nn.functional.one_hot(data['seqs'], num_classes=20)
    loss = criterion(
        log_l.contiguous().view(-1, log_l.size(-1)), seqs_encoded.contiguous().view(-1)
    ).view(seqs_encoded.size())
    seqs = torch.argmax(log_l, dim=-1) * data['x_mask']
    x_mask = data['x_mask']
    x_mask_sc = data['x_mask_sc']
    isnt_x_aa = (torch.logical_and(data['seqs'] != 20, data['seqs'] != 21)).to(dtype=torch.bool) # b x L
    correct_seqs = seqs == data['seqs']
    correct_seqs *= (x_mask * isnt_x_aa).to(dtype=correct_seqs.dtype)
    nsr = torch.sum(correct_seqs) / torch.sum(x_mask * isnt_x_aa)
    full_mask = x_mask * isnt_x_aa
    if use_sc_mask and x_mask_sc.numel() > 0:
        full_mask *= (~x_mask_sc.to(dtype=torch.bool)).to(dtype=full_mask.dtype)
    loss *= full_mask
    loss = torch.sum(loss) / torch.sum(full_mask)
    n_edges = torch.sum(full_mask)
    return loss, n_edges, nsr, seqs 

def loss_smoothed(nodes, E_idx, data, eps=1e-6, weight=0.1, use_sc_mask=False, weight_inter_types=None):
    """ Negative log probabilities """
    try:
        log_probs = F.log_softmax(nodes, dim=-1)
    except Exception as e:
        print(e)
        print(data['ids'])
        nlcpl_return = torch.Tensor([0])
        return nlcpl_return, -1, 0, data['seqs']
    # pad = (0, 2)
    # log_probs = F.pad(log_probs, pad, "constant", 0)
    S = data['seqs']
    x_mask = data['x_mask']
    x_mask_sc = data['x_mask_sc']
    isnt_x_aa = (torch.logical_and(data['seqs'] != 20, data['seqs'] != 21)).to(dtype=torch.bool) # b x L
    mask = x_mask * isnt_x_aa
    if use_sc_mask and x_mask_sc.numel() > 0:
        mask *= (~x_mask_sc.to(dtype=torch.bool)).to(dtype=mask.dtype)
    
    S_onehot = torch.nn.functional.one_hot(S, 22).float()
    # Label smoothing
    S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)

    loss = -(S_onehot * log_probs).sum(-1)
    if weight_inter_types is not None:
        seq_self_expand = data['seqs'].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(data['seqs'].shape[0], data['seqs'].shape[1], 30, 22, 1)
        seq_neighbors = torch.gather(data['seqs'].unsqueeze(-1).expand(-1, -1, 30), 1, E_idx)
        seq_neighbors_expand = seq_neighbors.unsqueeze(-1)
        inter_array_expand = binary_inter_array.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(data['seqs'].shape[0], data['seqs'].shape[1], 30, 22, 22).to(device=loss.device)
        inter_array_self = torch.gather(inter_array_expand, -1, seq_self_expand).squeeze(-1)
        inter_array_neighbors = torch.gather(inter_array_self, -1, seq_neighbors_expand).squeeze(-1)[:,:,1:].to(dtype=torch.int64)
        polar_weights = (inter_array_neighbors == 0).sum(dim=-1) + 1
        loss_av = torch.sum((loss * polar_weights) / polar_weights.sum())
    else:
        loss_av = torch.sum(loss * mask) / torch.sum(mask) #fixed 2000.0 
    ret_seqs = torch.argmax(log_probs, dim=-1)
    seqs = torch.argmax(log_probs, dim=-1) * data['x_mask']
    correct_seqs = seqs == data['seqs']
    correct_seqs *= mask.to(dtype=correct_seqs.dtype)
    nsr = torch.sum(correct_seqs) / torch.sum(mask)
    # print('nsr: ', nsr)
    # print(seqs)
    # print(data['seqs'])
    # print(mask)
    # print(data['x_mask'])
    # print(isnt_x_aa)
    # print(x_mask_sc)
    # print('------------')
    n_edges = torch.sum(mask)

    return loss_av, n_edges, nsr, ret_seqs

def loss_spike(nodes, E_idx, data, eps=1e-6, weight=0.1, use_sc_mask=False, weight_inter_types=None, lambda_weight=1):
    """ Single max probabilities """
    try:
        probs = F.softmax(nodes, dim=-1)
    except Exception as e:
        print(e)
        print(data['ids'])
        nlcpl_return = torch.Tensor([0])
        return nlcpl_return, -1, 0, data['seqs']
    # pad = (0, 2)
    # log_probs = F.pad(log_probs, pad, "constant", 0)
    max_probs, _ = torch.max(probs, dim=-1)
    non_max_probs = probs.sum(dim=-1) - max_probs
    loss = -torch.log(max_probs) + lambda_weight * non_max_probs

    x_mask = data['x_mask']
    x_mask_sc = data['x_mask_sc']
    isnt_x_aa = (torch.logical_and(data['seqs'] != 20, data['seqs'] != 21)).to(dtype=torch.bool) # b x L
    mask = x_mask * isnt_x_aa
    if use_sc_mask and x_mask_sc.numel() > 0:
        mask *= (~x_mask_sc.to(dtype=torch.bool)).to(dtype=mask.dtype)

    n_edges = torch.sum(mask)
    loss_av = torch.sum(loss * mask) / torch.sum(mask)

    ret_seqs = torch.argmax(torch.log(probs), dim=-1)
    seqs = torch.argmax(torch.log(probs), dim=-1) * data['x_mask']
    correct_seqs = seqs == data['seqs']
    correct_seqs *= mask.to(dtype=correct_seqs.dtype)
    nsr = torch.sum(correct_seqs) / torch.sum(mask)

    return loss_av, n_edges, nsr, ret_seqs


def gumbel_softmax(logits, temperature=1.0):
    """
    Differentiable one-hot sampling using Gumbel-Softmax.

    Args:
    - logits (torch.Tensor): Logits (batch_size x seq_len x vocab_size).
    - temperature (float): Temperature parameter for controlling sharpness.

    Returns:
    - torch.Tensor: Differentiable one-hot-like outputs (batch_size x seq_len x vocab_size).
    """
    # Sample Gumbel noise
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
    # Add noise to logits and apply softmax with temperature
    gumbel_logits = (logits + gumbel_noise) / temperature
    return torch.softmax(gumbel_logits, dim=-1)


def straight_through_gumbel_softmax(logits, temperature=1.0):
    """
    Straight-through Gumbel-Softmax for discrete sampling with gradient flow.

    Args:
    - logits (torch.Tensor): Logits (batch_size x seq_len x vocab_size).
    - temperature (float): Temperature parameter for controlling sharpness.

    Returns:
    - torch.Tensor: Discrete one-hot predictions (batch_size x seq_len x vocab_size).
    """
    # Gumbel-Softmax probabilities
    gumbel_probs = gumbel_softmax(logits, temperature)
    # Hard one-hot encoding (discrete)
    hard_one_hot = torch.zeros_like(gumbel_probs)
    hard_one_hot.scatter_(-1, gumbel_probs.argmax(dim=-1, keepdim=True), 1.0)
    # Straight-through trick: use hard_one_hot in the forward pass, but gumbel_probs in the backward pass
    return (hard_one_hot - gumbel_probs).detach() + gumbel_probs

def diversity_loss(nodes, E_idx, data, eps=1e-6, weight=0.1, use_sc_mask=False, weight_inter_types=None, temperature=0.1):
    probs = F.softmax(nodes, dim=-1)
    pred_seq = straight_through_gumbel_softmax(probs, temperature)




def pssm_loss(nodes, data, eps=1e-6, weight=0.1, use_sc_mask=False):
    """ Negative log probabilities """
    
    log_probs = F.log_softmax(nodes, dim=-1)
    # pad = (0, 2)
    # log_probs = F.pad(log_probs, pad, "constant", 0)
    x_mask = data['x_mask']
    x_mask_sc = data['x_mask_sc']
    isnt_x_aa = (torch.logical_and(data['seqs'] != 20, data['seqs'] != 21)).to(dtype=torch.bool) # b x L
    mask = x_mask * isnt_x_aa
    if use_sc_mask and x_mask_sc.numel() > 0:
        mask *= (~x_mask_sc.to(dtype=torch.bool)).to(dtype=mask.dtype)
    
    S_onehot = data['pssm'].float()
    # Label smoothing
    S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)

    loss = -(S_onehot * log_probs).sum(-1)
    loss_av = torch.sum(loss * mask) / torch.sum(mask) #fixed 2000.0 
    seqs = torch.argmax(log_probs, dim=-1) * data['x_mask']
    correct_seqs = seqs == data['seqs']
    correct_seqs *= mask.to(dtype=correct_seqs.dtype)
    nsr = torch.sum(correct_seqs) / torch.sum(mask)
    # print('nsr: ', nsr)
    # print(seqs)
    # print(data['seqs'])
    # print(mask)
    # print(data['x_mask'])
    # print(isnt_x_aa)
    # print(x_mask_sc)
    # print('------------')
    n_edges = torch.sum(mask)

    return loss_av, n_edges, nsr, seqs
    
def confidence_loss(nodes, confidence, data, diff=False, vector=False, matrix_type=None, eps=1e-6):
    """Confidence loss of predictions"""

    log_probs = F.log_softmax(nodes, dim=-1)
    x_mask = data['x_mask']
    x_mask_sc = data['x_mask_sc']
    isnt_x_aa = (torch.logical_and(data['seqs'] != 20, data['seqs'] != 21)).to(dtype=torch.bool) # b x L
    mask = x_mask * isnt_x_aa
    if x_mask_sc.numel() > 0:
        mask *= (~x_mask_sc.to(dtype=torch.bool)).to(dtype=mask.dtype)

    seqs = torch.argmax(log_probs, dim=-1) * data['x_mask']
    correct_seqs = (seqs == data['seqs']).to(dtype=torch.float32)
    correct_seqs *= mask.to(dtype=correct_seqs.dtype)
    
    loss_fn = torch.nn.BCELoss(reduction='none')
    loss = loss_fn(confidence, correct_seqs)
    loss = torch.sum(loss * mask) / torch.sum(mask)
    n_edges = torch.sum(mask)

    return loss, n_edges

def confidence_loss_blosum(nodes, confidence, data, diff=False, vector=False, matrix_type='blosum', eps=1e-6):
    """Confidence loss of predictions"""

    if matrix_type == 'blosum':
        sim_matrix = TORCH_BLOSUM
        num_classes = 17
    elif matrix_type == 'test':
        sim_matrix = TEST_MATRIX
        num_classes = 2
    elif matrix_type == 'test2':
        sim_matrix = TEST_MATRIX2
        num_classes = 3
    elif matrix_type == 'custom':
        num_classes = 7
        sim_matrix = CUSTOM_MATRIX
    elif matrix_type == 'simple':
        num_classes = 3
        sim_matrix = SIMPLE_MATRIX

    log_probs = F.log_softmax(nodes, dim=-1)
    x_mask = data['x_mask']
    x_mask_sc = data['x_mask_sc']
    isnt_x_aa = (torch.logical_and(data['seqs'] != 20, data['seqs'] != 21)).to(dtype=torch.bool) # b x L
    mask = x_mask * isnt_x_aa
    if x_mask_sc.numel() > 0:
        mask *= (~x_mask_sc.to(dtype=torch.bool)).to(dtype=mask.dtype)

    seqs = torch.argmax(log_probs, dim=-1) * data['x_mask']
    correct_seqs = (seqs == data['seqs']).to(dtype=torch.float32)
    correct_seqs *= mask.to(dtype=correct_seqs.dtype)
    
    S_expand = data['seqs'].unsqueeze(-1).expand(data['seqs'].shape[0], data['seqs'].shape[1], sim_matrix.shape[0])
    blosum_expand = sim_matrix.unsqueeze(0).expand(data['seqs'].shape[0], sim_matrix.shape[0], sim_matrix.shape[1]).to(device=S_expand.device)
    S_expand_bl = torch.gather(blosum_expand, 1, S_expand).to(dtype=blosum_expand.dtype)
    seqs_expand = seqs.unsqueeze(-1)
    seq_bl = torch.gather(S_expand_bl, 2, seqs_expand.to(data['seqs'].dtype)).squeeze(-1)
    if diff:
        bl_base = torch.gather(S_expand_bl, 2, data['seqs'].unsqueeze(-1)).squeeze(-1)
        seq_bl = torch.abs(bl_base - seq_bl)
    if vector:
        if not diff and matrix_type == 'blosum':
            seq_bl += 5
        seq_bl = seq_bl.to(dtype = torch.int64)
        seq_bl = F.one_hot(seq_bl, num_classes=num_classes)
        # loss = -1*torch.mean(torch.mul(confidence, seq_bl), dim=-1)
        conf_for_loss = confidence.reshape((confidence.shape[0] * confidence.shape[1], confidence.shape[2]))
        bl_for_loss = seq_bl.to(dtype=torch.float64).reshape((seq_bl.shape[0] * seq_bl.shape[1], seq_bl.shape[2]))
        loss = F.cross_entropy(conf_for_loss, bl_for_loss, reduction='none').reshape((confidence.shape[0], confidence.shape[1]))
    else:
        loss_fn = torch.nn.MSELoss(reduction='none')
        loss = loss_fn(seq_bl, confidence)
    loss = torch.sum(loss * mask) / torch.sum(mask)
    n_edges = torch.sum(mask)
    return loss, n_edges

def complexity_loss(nodes, data, eps=1e-6, weight=0.1):
    """Complexity of generated sequences"""
    log_probs = F.log_softmax(nodes, dim=-1)
    S = data['seqs']
    x_mask = data['x_mask']
    x_mask_sc = data['x_mask_sc']
    isnt_x_aa = (torch.logical_and(data['seqs'] != 20, data['seqs'] != 21)).to(dtype=torch.bool) # b x L
    mask = x_mask * isnt_x_aa
    if x_mask_sc.numel() > 0:
        mask *= (~x_mask_sc.to(dtype=torch.bool)).to(dtype=mask.dtype)
    
    S_onehot = torch.nn.functional.one_hot(S, 22).float()
    # Label smoothing
    S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)

    loss = -(S_onehot * log_probs).sum(-1)

def esm_loss(nodes, data, esm, batch_converter, tokenizer, esm_type, converter, mask_type=None, eps=1e-6, eval=False, use_sc_mask=False):
    dev = 'cpu'# nodes.device
    # GT
    x_mask = data['x_mask']
    x_mask_sc = data['x_mask_sc']
    isnt_x_aa = (torch.logical_and(data['seqs'] != 20, data['seqs'] != 21)).to(dtype=torch.bool) # b x L
    full_mask = x_mask * isnt_x_aa
    if use_sc_mask and x_mask_sc.numel() > 0:
        full_mask *= (~x_mask_sc.to(dtype=torch.bool)).to(dtype=full_mask.dtype)

    # Set up ESM constant token embeddings
    if dev == 'cpu':
        esm = esm.cpu()
        converter = converter.cpu()
    esm_consts = esm.embed_tokens(torch.tensor([0, 2, 1]).to(dev)) #nodes.device
    esm_consts = esm_consts.unsqueeze(0).unsqueeze(0).expand((nodes.size(0), 1, -1, -1)).to(dev) #nodes.device
    eos_inds = copy.deepcopy(data['chain_eos_inds']).to(dev)
    eos_inds[eos_inds == -1] = eos_inds.shape[1] -1
    eos_inds[eos_inds == -2] = eos_inds.shape[1]

    # Predictions
    nodes = torch.softmax(nodes, dim=-1) #F.log_softmax(nodes, dim=-1)
    nodes_esm = converter(nodes.to(dev))
    nodes_esm = torch.cat([nodes_esm, esm_consts[:,:,1,:], esm_consts[:,:,2,:]], dim=1)
    nodes_esm = torch.gather(nodes_esm, 1, eos_inds.unsqueeze(-1).expand((eos_inds.size(0), eos_inds.size(1), nodes_esm.size(-1))))
    nodes_esm = torch.cat([esm_consts[:,:,0,:], nodes_esm], dim=1)
    
    ## Check esm conversion
    seqs = torch.argmax(torch.softmax(nodes, dim=-1), dim=-1) * data['x_mask']
    real_strseqs = []
    for i in range(seqs.shape[0]):
        strseq = ints_to_seq_torch(seqs[i])
        strseq = strseq[:sum(data['chain_lens'][i])]
        real_strseqs.append(strseq)
    esm_data = []
    for i_s, seq in enumerate(real_strseqs):
        esm_data.append((f"protein_{i_s}", seq))
    _, _, batch_tokens = batch_converter(esm_data)
    batch_tokens = batch_tokens.to(dev) #nodes.device
    esm_embs = esm.embed_tokens(batch_tokens)
    if eval:
        nodes_esm = esm_embs

    mse = torch.nn.MSELoss()(esm_embs, nodes_esm)
    if mse > 0.025:
        print(mse, 'SKIP!')
        return None, None
    
    
    eos_mask = torch.cat([torch.zeros_like(data['chain_eos_inds'][:,0]).unsqueeze(1), data['chain_eos_inds']], dim=1)
    eos_mask = eos_mask != -1
    results = esm(nodes_esm, repr_layers=[30], return_contacts=True, embed_model=True, custom_eos_mask=eos_mask)
    preds = results['representations'][30][:,1:-1].to(device=nodes.device)

    # Calc loss
    loss = F.cosine_similarity(data['esm_embs'], preds, dim=-1)
    loss *= full_mask
    loss = -1 * (loss.sum() / full_mask.sum())
    return loss, full_mask.sum()
    
# Function to append 0 to the beginning and 2 to the end of each row
def append_values(tensor):
    zeros_col = torch.zeros(tensor.size(0), 1, dtype=tensor.dtype, device=tensor.device)
    twos_col = 2 * torch.ones(tensor.size(0), 1, dtype=tensor.dtype, device=tensor.device)
    tensor_with_zero = torch.cat((zeros_col, tensor), dim=1)
    tensor_with_zero_and_two = torch.cat((tensor_with_zero, twos_col), dim=1)
    return tensor_with_zero_and_two

def soft_argmax(input_tensor, beta=1):
    soft_values = torch.softmax(beta * input_tensor, dim=-1)
    indices = torch.arange(input_tensor.size(-1), device=input_tensor.device)
    return torch.round(torch.sum(indices * soft_values, dim=-1))

def get_contacts(nodes, data, esm, batch_converter, converter, tokenizer, esm_type='esm', contact_head=None, mask_type=None, eps=1e-6, eval=False, check_conv=True, already_probs=False, use_sc_mask=False, dev='cpu', verbose=True):
    # Set up ESM constant token embeddings
    if dev == 'cpu':
        esm = esm.cpu()
        converter = converter.cpu()
    esm_consts = esm.embed_tokens(torch.tensor([0, 2, 1]).to(device=dev)) #nodes.device
    esm_consts = esm_consts.unsqueeze(0).unsqueeze(0).expand((nodes.size(0), 1, -1, -1)).to(device=dev)
    eos_inds = copy.deepcopy(data['chain_eos_inds']).to(device=dev)
    eos_inds[eos_inds == -1] = eos_inds.shape[1] -1
    eos_inds[eos_inds == -2] = eos_inds.shape[1]

    # GT
    X = data['X'][:,:,1,:]
    real_contacts = X.unsqueeze(1) - X.unsqueeze(2)
    real_contacts = (real_contacts**2).sum(dim=3)
    real_contacts = real_contacts < 64
    # Predictions
    if not already_probs: nodes = torch.softmax(nodes, dim=-1)
    nodes_esm = converter(nodes.to(device=dev))
    nodes_esm = torch.cat([nodes_esm, esm_consts[:,:,1,:], esm_consts[:,:,2,:]], dim=1)
    nodes_esm = torch.gather(nodes_esm, 1, eos_inds.unsqueeze(-1).expand((eos_inds.size(0), eos_inds.size(1), nodes_esm.size(-1))))
    nodes_esm = torch.cat([esm_consts[:,:,0,:], nodes_esm], dim=1)
    ## Check esm conversion
    if check_conv:
        seqs = torch.argmax(torch.softmax(nodes, dim=-1), dim=-1) * data['x_mask']
        real_strseqs = []
        for i in range(seqs.shape[0]):
            strseq = ints_to_seq_torch(seqs[i])
            strseq = strseq[:sum(data['chain_lens'][i])]
            real_strseqs.append(strseq)
        esm_data = []
        for i_s, seq in enumerate(real_strseqs):
            esm_data.append((f"protein_{i_s}", seq))
        _, _, batch_tokens = batch_converter(esm_data)
        batch_tokens = batch_tokens.to(device=dev)
        esm_embs = esm.embed_tokens(batch_tokens)
        if eval:
            nodes_esm = esm_embs
        mse = torch.nn.MSELoss()(esm_embs, nodes_esm)
        if mse > 0.025:
            if verbose: print(mse, 'SKIP!')
            return None, None, None
    
    ## Get contacts
    if esm_type == 'esm':
        pred_contacts = get_esm_contacts(esm, nodes_esm, data['chain_eos_inds'])
    else:
        attns = calc_attns(esm, seqs, tokenizer, data['x_mask'], dev=data['x_mask'].device)
    # Mask 
    x_mask = data['x_mask']
    x_mask_sc = data['x_mask_sc']
    ref_seqs = data['seqs']
    isnt_x_aa = (torch.logical_and(ref_seqs != 30, ref_seqs != 24)).to(dtype=torch.bool) # b x L
    x_mask = x_mask.to(dtype=torch.bool) # b x L
    full_mask = x_mask * isnt_x_aa
    if use_sc_mask and x_mask_sc.numel() > 0:
        x_mask_sc = x_mask_sc # b x L
        full_mask *= ~x_mask_sc.to(dtype=torch.bool)
    full_mask_rows = full_mask.unsqueeze(1)
    full_mask_cols = full_mask.unsqueeze(2)
    full_mask = full_mask_rows * full_mask_cols
    return real_contacts.to(device=pred_contacts.device), pred_contacts, full_mask.to(device=pred_contacts.device)

def esm_cmap_loss_tp(nodes, data, esm, batch_converter, tokenizer, esm_type, converter, mask_type=None, eps=1e-6, eval=False):
    real_contacts, pred_contacts, mask = get_contacts(nodes, data, esm, batch_converter, converter, tokenizer, esm_type, mask_type=mask_type, eps=eps, eval=eval)
    if real_contacts is None:
        return None, None
    pred_contacts = torch.log(pred_contacts.to(real_contacts.device) + eps)
    cmap_loss = (pred_contacts * real_contacts)
    mask *= (real_contacts != 0).to(dtype=mask.dtype)
    cmap_loss, num_pairs = apply_mask(cmap_loss, mask, mask_type)
    return cmap_loss, num_pairs

def esm_cmap_loss_all(nodes, data, esm, batch_converter, tokenizer, esm_type, converter, mask_type=None, eps=1e-6, eval=False):
    real_contacts, pred_cmap, mask = get_contacts(nodes, data, esm, batch_converter, converter, tokenizer, esm_type, mask_type=mask_type, eps=eps, eval=eval)
    pred_contacts = torch.log(pred_cmap + eps)
    log_inv_pred_cmap = torch.log(1 - pred_cmap + eps)
    cmap_loss = pred_contacts * real_contacts + log_inv_pred_cmap * ~real_contacts
    cmap_loss, num_pairs = apply_mask(cmap_loss, mask, mask_type)
    cmap_loss.requires_grad = True
    return cmap_loss, num_pairs

def esm_cmap_loss_tn(nodes, data, esm, batch_converter, tokenizer, esm_type, converter, mask_type=None, eps=1e-6, eval=False):
    real_contacts, pred_cmap, mask = get_contacts(nodes, data, esm, batch_converter, converter, tokenizer, esm_type, mask_type=mask_type, eps=eps, eval=eval)
    log_inv_pred_cmap = torch.log(1 - pred_cmap.to(real_contacts.device) + eps)
    cmap_loss = log_inv_pred_cmap * ~real_contacts
    mask *= (real_contacts == 0).to(dtype=mask.dtype)
    cmap_loss, num_pairs = apply_mask(cmap_loss, mask, mask_type)
    cmap_loss.requires_grad = True
    return cmap_loss, num_pairs

def global_monogram_loss(h_V, data, use_sc_mask=False):
    kl_loss = torch.nn.KLDivLoss(reduction="none", log_target=False)
    pred_probs = F.softmax(h_V, dim=-1)
    pred_probs = (pred_probs * data['x_mask'].unsqueeze(-1)).sum(dim=1) / ((pred_probs * data['x_mask'].unsqueeze(-1)).sum(dim=[1,2]) + 1e-5).unsqueeze(1)
    monogram_global_loss = kl_loss(torch.log(pred_probs + 1e-5), data['monogram_probs'].unsqueeze(0).expand(pred_probs.shape)).sum(dim=1).mean()

    return monogram_global_loss, 1

def monogram_loss(h_V, data, use_sc_mask=False):
    kl_loss = torch.nn.KLDivLoss(reduction="none", log_target=False)
    pred_probs = F.softmax(h_V, dim=-1)
    seq_mask = torch.ones(22).to(h_V.device)
    seq_mask[20:] = 0
    seq_one_hot = torch.nn.functional.one_hot(data['seqs'], num_classes=22)
    single_probs = seq_one_hot.sum(dim=1) / seq_one_hot.sum()
    pred_probs = (pred_probs * data['x_mask'].unsqueeze(-1)).sum(dim=1) / ((pred_probs * data['x_mask'].unsqueeze(-1)).sum(dim=[1,2]) + 1e-5).unsqueeze(1)
    single_probs *= seq_mask
    single_probs /= single_probs.sum(dim=1).unsqueeze(1)
    monogram_ngram_loss = torch.sum(kl_loss(torch.log( pred_probs + 1e-5), single_probs) * seq_mask.unsqueeze(0), dim=1)
    monogram_ngram_loss = monogram_ngram_loss.mean()

    return monogram_ngram_loss, 1


def bigram_loss(h_V, data, use_sc_mask=False):
    kl_loss = torch.nn.KLDivLoss(reduction="none", log_target=False)
    pred_probs = F.softmax(h_V, dim=-1)
    seq_mask = torch.ones(22).to(h_V.device)
    seq_mask[20:] = 0
    seq_one_hot = torch.nn.functional.one_hot(data['seqs'], num_classes=22) * seq_mask.unsqueeze(0)
    pred_bigrams = compute_pairwise_probabilities_batched(pred_probs * data['x_mask'].unsqueeze(-1))
    real_bigrams = compute_pairwise_probabilities_batched(seq_one_hot)
    bigram_ngram_loss = torch.sum(kl_loss(torch.log(pred_bigrams + 1e-5), real_bigrams), dim=1).mean()

    return bigram_ngram_loss, 1

def trigram_loss(h_V, data, use_sc_mask=False):
    kl_loss = torch.nn.KLDivLoss(reduction="none", log_target=False)
    pred_probs = F.softmax(h_V, dim=-1)
    seq_mask = torch.ones(22).to(h_V.device)
    seq_mask[20:] = 0
    seq_one_hot = torch.nn.functional.one_hot(data['seqs'], num_classes=22) * seq_mask.unsqueeze(0)
    pred_trigrams = compute_triplet_probabilities_batched(pred_probs * data['x_mask'].unsqueeze(-1))
    real_trigrams = compute_triplet_probabilities_batched(seq_one_hot)
    trigram_ngram_loss = torch.sum(kl_loss(torch.log(pred_trigrams + 1e-5), real_trigrams), dim=1).mean()

    return trigram_ngram_loss, 1

def etab_distill_loss(etab, data, distill_etab, use_sc_mask=False):
    x_mask = data['x_mask']
    ref_seqs = data['seqs']
    x_mask_sc = data['x_mask_sc']
    isnt_x_aa = (torch.logical_and(ref_seqs != 20, ref_seqs != 21)).float() # b x L

    x_mask = x_mask.unsqueeze(-1).unsqueeze(-1) # b x L x 1
    isnt_x_aa = isnt_x_aa.unsqueeze(-1).unsqueeze(-1) # b x L x 1
    full_mask = x_mask * isnt_x_aa
    if use_sc_mask and x_mask_sc.numel() > 0:
        x_mask_sc = x_mask_sc.unsqueeze(-1).unsqueeze(-1)
        full_mask *= (~x_mask_sc.to(dtype=torch.bool)).to(dtype=full_mask.dtype)
        if torch.sum(full_mask) == 0:
            full_mask = torch.ones_like(full_mask).to(dtype=full_mask.dtype, device=full_mask.device)

    distill_loss_fn = torch.nn.MSELoss(reduction='none')
    # distill_loss = distill_loss_fn(etab, data['distill_etab'])
    distill_loss = distill_loss_fn(etab, distill_etab)
    n_edges = full_mask.expand(etab.shape).sum()
    distill_loss = (distill_loss * full_mask).sum() / n_edges

    return distill_loss, full_mask.sum()


# def esm_cmap_loss_tp(nodes, data, esm, batch_converter, eps=1e-6):
#     real_contacts = data['X'][:,:,1,:].unsqueeze(1) - data['X'][:,:,1,:].unsqueeze(2)
#     real_contacts = (real_contacts**2).sum(dim=3)
#     real_cmap = (real_contacts < 64).to(dtype=torch.bool)
#     contacts_mask = (data['x_mask'].unsqueeze(1) * data['x_mask'].unsqueeze(2)).to(device=real_contacts.device, dtype=real_cmap.dtype)
#     real_cmap *= contacts_mask
#     seqs = torch.argmax(torch.softmax(nodes, dim=-1), dim=-1) * data['x_mask']
#     aa_seqs = [ints_to_seq_torch(seq) for seq in seqs]
#     esm_data = []
#     for i, aa_seq in enumerate(aa_seqs):
#         esm_data.append(("protein_" + str(i), aa_seq[:data['seq_lens'][i]])) 
#     _, _, batch_tokens = batch_converter(esm_data)
#     with torch.no_grad():
#         results = esm(batch_tokens, repr_layers=[33], return_contacts=True)
#     pred_contacts = torch.log(results["contacts"].to(real_contacts.device) + eps)
#     cmap_loss = (pred_contacts * real_cmap) * contacts_mask
#     cmap_loss = torch.div(cmap_loss.sum(dim=(1,2)), real_cmap.sum(dim=(1,2)))
#     cmap_loss = cmap_loss.mean()
#     cmap_loss.requires_grad = True
#     return cmap_loss, -1

# def esm_cmap_loss_all(nodes, data, esm, batch_converter, eps=1e-6):
#     real_contacts = data['X'][:,:,1,:].unsqueeze(1) - data['X'][:,:,1,:].unsqueeze(2)
#     real_contacts = (real_contacts**2).sum(dim=3)
#     real_cmap = (real_contacts < 64).to(dtype=torch.bool)
#     contacts_mask = (data['x_mask'].unsqueeze(1) * data['x_mask'].unsqueeze(2)).to(device=real_contacts.device, dtype=real_cmap.dtype)
    
#     seqs = torch.argmax(torch.softmax(nodes, dim=-1), dim=-1) * data['x_mask']
#     aa_seqs = [ints_to_seq_torch(seq) for seq in seqs]
#     esm_data = []
#     for i, aa_seq in enumerate(aa_seqs):
#         esm_data.append(("protein_" + str(i), aa_seq[:data['seq_lens'][i]])) 
#     _, _, batch_tokens = batch_converter(esm_data)
#     with torch.no_grad():
#         results = esm(batch_tokens, repr_layers=[33], return_contacts=True)
#     pred_cmap = results["contacts"].to(real_contacts.device)
#     pred_contacts = torch.log(pred_cmap + eps)
#     log_inv_pred_cmap = torch.log(1 - pred_cmap + eps)
#     cmap_loss = (pred_contacts * real_cmap + log_inv_pred_cmap * ~real_cmap) * contacts_mask
#     real_cmap *= contacts_mask
#     cmap_loss = torch.div(cmap_loss.sum(dim=(1,2)), real_cmap.sum(dim=(1,2)))
#     cmap_loss = cmap_loss.mean()
#     cmap_loss.requires_grad = True
#     return cmap_loss, -1


def nlcpl_test(etab, E_idx, data):
    """ Alias of nlcpl_full """
    return nlcpl_full(etab, E_idx, data)


def nlcpl_full(etab, E_idx, data):
    """ Negative log composite psuedo-likelihood
        Averaged nlcpl per residue, across batches
        p(a_i,m ; a_j,n) =

            softmax [
                E_s(a_i,m) + E_s(a_j,n)
                + E_p(a_i,m ; a_j,n)
                + sum_(u != m,n) [
                    E_p(a_i,m; A_u)
                    + E_p(A_u, a_j,n)
                    ]
                ]

        Returns: averaged log likelihood per residue pair,
        as well as the number of edges considered
    """

    ref_seqs = data['seqs']
    x_mask = data['x_mask']
    n_batch, L, k, _ = etab.shape
    etab = etab.unsqueeze(-1).view(n_batch, L, k, 20, 20)

    # X is encoded as 20 so lets just add an extra row/col of zeros
    pad = (0, 1, 0, 1)
    etab = F.pad(etab, pad, "constant", 0)

    isnt_x_aa = (ref_seqs != 20).float().to(etab.device)

    # separate selfE and pairE since we have to treat selfE differently
    self_etab = etab[:, :, 0:1]
    pair_etab = etab[:, :, 1:]

    # gather 22 self energies by taking the diagonal of the etab
    self_nrgs_im = torch.diagonal(self_etab, offset=0, dim1=-2, dim2=-1)
    self_nrgs_im_expand = self_nrgs_im.expand(-1, -1, k - 1, -1)

    # E_idx for all but self
    E_idx_jn = E_idx[:, :, 1:]

    # self Es gathered from E_idx_others
    E_idx_jn_expand = E_idx_jn.unsqueeze(-1).expand(-1, -1, -1, 21)
    self_nrgs_jn = torch.gather(self_nrgs_im_expand, 1, E_idx_jn_expand)

    # idx matrix to gather the identity at all other residues given a residue of focus
    E_aa = torch.gather(ref_seqs.unsqueeze(-1).expand(-1, -1, k - 1), 1, E_idx_jn)
    # expand the matrix so we can gather pair energies
    E_aa = E_aa.view(list(E_idx_jn.shape) + [1, 1]).expand(-1, -1, -1, 21, -1)
    # gather the 22 energies for each edge based on E_aa
    pair_nrgs_jn = torch.gather(pair_etab, 4, E_aa).squeeze(-1)
    # sum_(u != n,m) E_p(a_i,n; A_u)
    sum_pair_nrgs_jn = torch.sum(pair_nrgs_jn, dim=2)
    pair_nrgs_im_u = sum_pair_nrgs_jn.unsqueeze(2).expand(-1, -1, k - 1, -1) - pair_nrgs_jn

    # get pair_nrgs_u_jn from pair_nrgs_im_u
    E_idx_imu_to_ujn = E_idx_jn.unsqueeze(-1).expand(pair_nrgs_im_u.shape)
    pair_nrgs_u_jn = torch.gather(pair_nrgs_im_u, 1, E_idx_imu_to_ujn)

    # start building this wacky energy table
    self_nrgs_im_expand = self_nrgs_im_expand.unsqueeze(-1).expand(-1, -1, -1, -1, 21)
    self_nrgs_jn_expand = self_nrgs_jn.unsqueeze(-1).expand(-1, -1, -1, -1, 21).transpose(-2, -1)
    pair_nrgs_im_expand = pair_nrgs_im_u.unsqueeze(-1).expand(-1, -1, -1, -1, 21)
    pair_nrgs_jn_expand = pair_nrgs_u_jn.unsqueeze(-1).expand(-1, -1, -1, -1, 21).transpose(-2, -1)

    composite_nrgs = (self_nrgs_im_expand + self_nrgs_jn_expand + pair_etab + pair_nrgs_im_expand +
                      pair_nrgs_jn_expand)

    # convert energies to probabilities
    composite_nrgs_reshape = composite_nrgs.view(n_batch, L, k - 1, 21 * 21, 1)
    log_composite_prob_dist = torch.log_softmax(-composite_nrgs_reshape, dim=-2).view(n_batch, L, k - 1, 21, 21)
    # get the probability of the sequence
    im_probs = torch.gather(log_composite_prob_dist, 4, E_aa).squeeze(-1)
    ref_seqs_expand = ref_seqs.view(list(ref_seqs.shape) + [1, 1]).expand(-1, -1, k - 1, 1)
    log_edge_probs = torch.gather(im_probs, 3, ref_seqs_expand).squeeze(-1)

    # reshape masks
    x_mask = x_mask.unsqueeze(-1)
    isnt_x_aa = isnt_x_aa.unsqueeze(-1)
    full_mask = x_mask * isnt_x_aa
    n_self = torch.sum(full_mask)
    n_edges = torch.sum(full_mask.expand(log_edge_probs.shape))

    # compute probabilities for self edges
    self_nrgs = torch.diagonal(etab[:, :, 0], offset=0, dim1=-2, dim2=-1)
    log_self_probs_dist = torch.log_softmax(-self_nrgs, dim=-1) * full_mask
    log_self_probs = torch.gather(log_self_probs_dist, 2, ref_seqs.unsqueeze(-1))

    # convert to nlcpl
    log_edge_probs *= full_mask  # zero out positions that don't have residues or where the native sequence is X
    nlcpl_return = -(torch.sum(log_self_probs) + torch.sum(log_edge_probs)) / (n_self + n_edges)
    return nlcpl_return, int(n_self + n_edges)


# pylint: disable=unused-argument
def etab_norm_penalty(etab, E_idx, data):
    """ Take the norm of all etabs and scale it by the total number of residues involved """
    seq_lens = data['seq_lens']
    # etab_norm = torch.linalg.norm(etab.view([-1]))
    # return etab_norm / seq_lens.sum(), int(seq_lens.sum())
    etab_norm = torch.mean(torch.linalg.norm(etab, dim=(1,2,3)) / seq_lens)
    return etab_norm, int(seq_lens.sum())


# pylint: disable=unused-argument
def mindren_etab_norm_penalty(etab, E_idx, data):
    """ Take the norm of all etabs and scale it by the total number of residues involved """
    seq_lens = data['seq_lens']
    # etab_norm = torch.linalg.norm(etab.view([-1]))
    # return etab_norm / seq_lens.sum(), int(seq_lens.sum())
    etab_norm = torch.mean(torch.linalg.norm(etab, dim=(1,2,3)) / seq_lens)
    return etab_norm, int(seq_lens.sum())


# pylint: disable=unused-argument
def etab_l1_norm_penalty(etab, E_idx, data):
    """ Take the norm of all etabs and scale it by the total number of residues involved """
    seq_lens = data['seq_lens']
    # etab_norm = torch.linalg.norm(etab.view([-1]))
    # return etab_norm / seq_lens.sum(), int(seq_lens.sum())
    etab_norm = torch.mean(torch.sum(torch.abs(etab), dim=(1,2,3)) / seq_lens)
    return etab_norm, int(seq_lens.sum())


# pylint: disable=unused-argument
def pair_self_energy_ratio(etab, E_idx, data):
    """ Return the ratio of the scaled norm of pair energies vs self energies in an etab """
    n_batch, L, k, _ = etab.shape
    etab = etab.unsqueeze(-1).view(n_batch, L, k, 20, 20)
    self_etab = etab[:, :, 0:1]
    pair_etab = etab[:, :, 1:]

    # gather 22 self energies by taking the diagonal of the etab
    self_nrgs = torch.diagonal(self_etab, offset=0, dim1=-2, dim2=-1)
    # compute an "avg" by taking the mean of the magnitude of the values
    # then sqrt to get approx right scale for the energies
    self_nrgs_avg = self_nrgs[self_nrgs != 0].square().mean().sqrt()
    pair_nrgs_avg = pair_etab[pair_etab != 0].square().mean().sqrt()

    return pair_nrgs_avg / self_nrgs_avg, n_batch

def calc_eners_peptide(etab, E_idx, data, pep_len=20):
    ref_seqs = data['sortcery_seqs'][0]
    ref_energies = data['sortcery_nrgs'][0]
    cetab = condense_etab_expanded(expand_etab(etab, E_idx), data['seqs'][:,:-1*pep_len], pep_len=pep_len)
    batch_pred_E = score_seq_batch(cetab[0], data['sortcery_seqs'][0])
    return batch_pred_E, ref_seqs, ref_energies

def calc_eners_stability(etab, E_idx, sort_seqs, sort_nrgs, filter=True, expanded=False):
    if not expanded:
        etab = expand_etab(etab, E_idx)
    batch_scores = batch_score_seq_batch(etab, sort_seqs)
    ref_seqs = sort_seqs
    ref_energies = sort_nrgs
    if filter:
        mask = ref_energies != torch.nan
        ref_seqs = ref_seqs[mask]
        ref_energies = ref_energies[mask]
        batch_pred_E = batch_scores[mask]
        return batch_pred_E, ref_seqs, ref_energies
    else:
        return batch_scores, ref_seqs, ref_energies

def likelihood_ener_loss(h_V, data, ddg=True):
    if ddg:
        sort_seqs = torch.cat([data['seqs'].unsqueeze(1), data['sortcery_seqs']], dim=1)
        sort_eners = data['sortcery_nrgs']
        # sort_eners = torch.cat([torch.zeros_like(data['sortcery_nrgs'][:,0].unsqueeze(1)), data['sortcery_nrgs']], dim=1)
    else:
        sort_seqs = data['sortcery_seqs']
        sort_eners = data['sortcery_nrgs']
    log_probs = F.log_softmax(h_V, dim=-1).unsqueeze(1).expand((-1, sort_seqs.shape[1], -1, -1))
    batch_eners = torch.gather(log_probs, 3, sort_seqs.unsqueeze(-1)).squeeze(-1)
    batch_scores = batch_eners.sum(dim=-1)
    if ddg:
        batch_scores = batch_scores[:,0] - batch_scores 
        batch_scores = batch_scores[:,1:]
        sort_seqs = sort_seqs[:,1:]

    norm_pred = batch_scores - torch.mean(batch_scores) # n
    norm_ref = sort_eners - torch.mean(sort_eners) # n
    pearson = torch.sum(norm_pred * norm_ref) / (torch.sqrt(torch.sum(norm_pred**2)) * torch.sqrt(torch.sum(norm_ref**2)))
    if torch.isnan(pearson):
        print(norm_pred)
        print(norm_ref)
        print(torch.sqrt(torch.sum(norm_pred**2)))
        print(torch.sqrt(torch.sum(norm_ref**2)))
        raise ValueError
        return 0, -1
    return -pearson, len(batch_scores), batch_scores, sort_seqs, sort_eners

def stability_loss(base_etab, E_idx, data, max_tokens=20000, use_sc_mask=False):
    ''' Compute the correlation between etab's predicted energies and experimental stability energies.
    '''
    # TODO hacky fix to avoid problems when without SORTCERY data

    if data["sortcery_seqs"].numel() < 5:
        return 0, -1
    
    b, n, k, h = base_etab.shape
    h = int(np.sqrt(h))
    etab = base_etab.view(b, n, k, h, h)
    pad = (0, 2, 0, 2)
    etab = F.pad(etab, pad, "constant", 0)
    predicted_E, ref_seqs, ref_energies = calc_eners_stability(etab, E_idx, data['sortcery_seqs'], data['sortcery_nrgs'])

    # normalize values around 0 for pearson correlation calculation
    norm_pred = predicted_E - torch.mean(predicted_E) # n
    norm_ref = ref_energies - torch.mean(ref_energies) # n

    pearson = torch.sum(norm_pred * norm_ref) / (torch.sqrt(torch.sum(norm_pred**2)) * torch.sqrt(torch.sum(norm_ref**2)))
    if torch.isnan(pearson):
        return 0, -1
    return -pearson, len(ref_seqs) # scalar; negate, since we want to minimize our loss function

def stability_loss_loop(base_etab, E_idx, data, max_tokens=20000, use_sc_mask=False, return_preds=False):
    ''' Compute the correlation between etab's predicted energies and experimental stability energies.
    '''
    # TODO hacky fix to avoid problems when without SORTCERY data

    if data["sortcery_seqs"].numel() < 5:
        return 0, -1
    
    b, n, k, h = base_etab.shape
    h = int(np.sqrt(h))
    etab = base_etab.view(b, n, k, h, h)
    pad = (0, 2, 0, 2)
    etab = F.pad(etab, pad, "constant", 0)
    if n*data['sortcery_nrgs'].shape[1] > max_tokens:
        batch_size = int(max_tokens / n)
    else:
        batch_size = data['sortcery_nrgs'].shape[1]
    all_preds = []
    all_refs = []
    for batch in range(0, data['sortcery_nrgs'].shape[1], batch_size):
        predicted_E, _, ref_energies = calc_eners_stability(etab, E_idx, data['sortcery_seqs'][:,batch:batch+batch_size], data['sortcery_nrgs'][:,batch:batch+batch_size])
        all_preds.append(predicted_E)
        all_refs.append(ref_energies)

    # normalize values around 0 for pearson correlation calculation
    predicted_E = torch.cat(all_preds, dim=0)
    ref_energies = torch.cat(all_refs, dim=0)
    norm_pred = predicted_E - torch.mean(predicted_E) # n
    norm_ref = ref_energies - torch.mean(ref_energies) # n

    pearson = torch.sum(norm_pred * norm_ref) / (torch.sqrt(torch.sum(norm_pred**2)) * torch.sqrt(torch.sum(norm_ref**2)))
    if return_preds:
        return -pearson, predicted_E, ref_energies
    if torch.isnan(pearson):
        return 0, -1
    return -pearson, data['sortcery_nrgs'].shape[1] # scalar; negate, since we want to minimize our loss function

def stability_loss_loop_ddg(base_etab, E_idx, data, max_tokens=20000, use_sc_mask=False, return_preds=False, return_norm=True, prob_calc=False, prob_out=None, multiseq=False):
    ''' Compute the correlation between etab's predicted energies and experimental stability energies.
    '''
    # TODO hacky fix to avoid problems when without SORTCERY data

    if data["sortcery_seqs"].numel() < 5:
        return 0, -1
    
    b, n, k, h = base_etab.shape
    h = int(np.sqrt(h))
    etab = base_etab.view(b, n, k, h, h)
    pad = (0, 2, 0, 2)
    etab = F.pad(etab, pad, "constant", 0)

    ## Add WT
    seqs = torch.cat([data['seqs'].unsqueeze(1), data['sortcery_seqs']], dim=1)
    nrgs = F.pad(data['sortcery_nrgs'], (1, 0), "constant", 0)

    if n*nrgs.shape[1] > max_tokens:
        batch_size = int(max_tokens / n)
    else:
        batch_size = nrgs.shape[1]
    all_preds = []
    all_refs = []
    all_seqs = []
    
    for batch in range(0, nrgs.shape[1], batch_size):
        predicted_E, cur_seqs, ref_energies = calc_eners_stability(etab, E_idx, seqs[:,batch:batch+batch_size], nrgs[:,batch:batch+batch_size], multiseq=multiseq)
        all_preds.append(predicted_E)
        all_refs.append(ref_energies)
        all_seqs.append(cur_seqs)

    predicted_E = torch.cat(all_preds, dim=0)
    ref_energies = torch.cat(all_refs, dim=0) 
    all_seqs = torch.cat(all_seqs, dim=0)

    # Normalize to WT
    predicted_E = predicted_E[1:] - predicted_E[0]
    ref_energies = ref_energies[1:]
    all_seqs = all_seqs[1:]

    # Normalize values around 0 for pearson correlation calculation
    norm_pred = predicted_E - torch.mean(predicted_E) # n
    norm_ref = ref_energies - torch.mean(ref_energies) # n

    pearson = torch.sum(norm_pred * norm_ref) / (torch.sqrt(torch.sum(norm_pred**2)) * torch.sqrt(torch.sum(norm_ref**2)))
    if return_preds:
        if not return_norm:
            return -pearson, predicted_E, ref_energies
        return -pearson, norm_pred, norm_ref
    if torch.isnan(pearson):
        return 0, -1
    
    return -pearson, data['sortcery_nrgs'].shape[1] # scalar; negate, since we want to minimize our loss function


def setup_etab(etab, E_idx):
    b, n, k, h = etab.shape
    h = int(np.sqrt(h))
    etab = etab.view(b, n, k, h, h)
    pad = (0, 2, 0, 2)
    etab = F.pad(etab, pad, "constant", 0)
    etab = expand_etab(etab, E_idx)
    return etab

def stability_loss_diff(base_etab, E_idx, data, target_etab, target_E_idx, binder_etab, binder_E_idx, target_first=True, max_tokens=20000, use_sc_mask=False):
    ''' Compute the correlation between etab's predicted energies and experimental stability energies.
    '''
    # TODO hacky fix to avoid problems when without SORTCERY data

    if data["sortcery_seqs"].numel() < 5:
        return 0, -1
    
    n = base_etab.shape[1]
    target_n = target_etab.shape[1]
    binder_n = binder_etab.shape[1]
    if target_first:
        place_n = target_n
    else:
        place_n = binder_n

    etab = setup_etab(base_etab, E_idx)
    split_etabs = [setup_etab(target_etab, target_E_idx), setup_etab(binder_etab, binder_E_idx)]
    stitched_etab = torch.zeros_like(etab)
    stitched_etab[:,:place_n, :place_n] = split_etabs[int(not target_first)]
    stitched_etab[:,place_n:, place_n:] = split_etabs[int(target_first)]
    etab = etab - stitched_etab

    predicted_E, ref_seqs, ref_energies = calc_eners_stability(etab, E_idx, data['sortcery_seqs'], data['sortcery_nrgs'], expanded=True)

    # normalize values around 0 for pearson correlation calculation
    norm_pred = predicted_E - torch.mean(predicted_E) # n
    norm_ref = ref_energies - torch.mean(ref_energies) # n

    pearson = torch.sum(norm_pred * norm_ref) / (torch.sqrt(torch.sum(norm_pred**2)) * torch.sqrt(torch.sum(norm_ref**2)))
    if torch.isnan(pearson):
        return 0, -1
    return -pearson, len(ref_seqs) # scalar; negate, since we want to minimize our loss function


def stability_loss_diff_loop(base_etab, E_idx, data, target_etab, target_E_idx, binder_etab, binder_E_idx, target_first=True, max_tokens=20000, use_sc_mask=False, return_preds=False):
    ''' Compute the correlation between etab's predicted energies and experimental stability energies.
    '''
    # TODO hacky fix to avoid problems when without SORTCERY data

    if data["sortcery_seqs"].numel() < 5:
        return 0, -1
    
    n = base_etab.shape[1]
    target_n = target_etab.shape[1]
    binder_n = binder_etab.shape[1]
    if target_first:
        place_n = target_n
    else:
        place_n = binder_n

    etab = setup_etab(base_etab, E_idx)
    split_etabs = [setup_etab(target_etab, target_E_idx), setup_etab(binder_etab, binder_E_idx)]
    stitched_etab = torch.zeros_like(etab)
    stitched_etab[:,:place_n, :place_n] = split_etabs[int(not target_first)]
    stitched_etab[:,place_n:, place_n:] = split_etabs[int(target_first)]
    etab = etab - stitched_etab
    
    if n*data['sortcery_nrgs'].shape[1] > max_tokens:
        batch_size = int(max_tokens / n)
    else:
        batch_size = data['sortcery_nrgs'].shape[1]
    all_preds = []
    all_refs = []
    for batch in range(0, data['sortcery_nrgs'].shape[1], batch_size):
        predicted_E, _, ref_energies = calc_eners_stability(etab, E_idx, data['sortcery_seqs'][:,batch:batch+batch_size], data['sortcery_nrgs'][:,batch:batch+batch_size], expanded=True)
        all_preds.append(predicted_E)
        all_refs.append(ref_energies)

    # normalize values around 0 for pearson correlation calculation
    predicted_E = torch.cat(all_preds, dim=0)
    ref_energies = torch.cat(all_refs, dim=0)
    norm_pred = predicted_E - torch.mean(predicted_E) # n
    norm_ref = ref_energies - torch.mean(ref_energies) # n

    pearson = torch.sum(norm_pred * norm_ref) / (torch.sqrt(torch.sum(norm_pred**2)) * torch.sqrt(torch.sum(norm_ref**2)))
    if torch.isnan(pearson):
        return 0, -1
    if return_preds:
        return -pearson, predicted_E, ref_energies
    return -pearson, data['sortcery_nrgs'].shape[1] # scalar; negate, since we want to minimize our loss function


def stability_loss_mse(base_etab, E_idx, data, max_tokens=20000, use_sc_mask=False):
    ''' Compute the correlation between etab's predicted energies and experimental stability energies.
    '''
    # TODO hacky fix to avoid problems when without SORTCERY data
    loss_fn = torch.nn.MSELoss()
    if data["sortcery_seqs"].numel() == 0:
        # print(data["sortcery_seqs"].numel())
        # raise ValueError
        return 0, -1
    
    b, n, k, h = base_etab.shape
    h = int(np.sqrt(h))
    etab = base_etab.view(b, n, k, h, h)
    pad = (0, 2, 0, 2)
    etab = F.pad(etab, pad, "constant", 0)
    # data['sortcery_seqs'] = data['sortcery_seqs'].to(dtype=torch.float32)
    predicted_E, ref_seqs, ref_energies = calc_eners_stability(etab, E_idx, data['sortcery_seqs'], data['sortcery_nrgs'])
    predicted_E = predicted_E.to(dtype=torch.float32)

    # normalize values around 0
    norm_pred = predicted_E - torch.mean(predicted_E) # n
    norm_ref = ref_energies - torch.mean(ref_energies) # n

    mse = loss_fn(norm_pred, norm_ref)
    if torch.isnan(mse):
        # print(mse)
        # print(norm_pred)
        # print(norm_ref)
        # raise ValueError
        return 0, -1
    return mse, len(ref_seqs) # scalar; negate, since we want to minimize our loss function

def stability_loss_mse_raw(base_etab, E_idx, data, max_tokens=20000, use_sc_mask=False):
    ''' Compute the correlation between etab's predicted energies and experimental stability energies.
    '''
    # TODO hacky fix to avoid problems when without SORTCERY data
    loss_fn = torch.nn.MSELoss()
    if data["sortcery_seqs"].numel() == 0:
        # print(data["sortcery_seqs"].numel())
        # raise ValueError
        return 0, -1
    
    b, n, k, h = base_etab.shape
    h = int(np.sqrt(h))
    etab = base_etab.view(b, n, k, h, h)
    pad = (0, 2, 0, 2)
    etab = F.pad(etab, pad, "constant", 0)
    # data['sortcery_seqs'] = data['sortcery_seqs'].to(dtype=torch.float32)
    predicted_E, ref_seqs, ref_energies = calc_eners_stability(etab, E_idx, data['sortcery_seqs'], data['sortcery_nrgs'])
    predicted_E = predicted_E.to(dtype=torch.float32) / 1000

    mse = loss_fn(predicted_E, ref_energies)
    if torch.isnan(mse):
        # print(mse)
        # print(norm_pred)
        # print(norm_ref)
        # raise ValueError
        return 0, -1
    return mse, len(ref_seqs) # scalar; negate, since we want to minimize our loss function

def stability_loss_mse_loop(base_etab, E_idx, data, max_tokens=20000, use_sc_mask=False):
    ''' Compute the correlation between etab's predicted energies and experimental stability energies.
    '''
    # TODO hacky fix to avoid problems when without SORTCERY data
    loss_fn = torch.nn.MSELoss()
    if data["sortcery_seqs"].numel() == 0:
        # print(data["sortcery_seqs"].numel())
        # raise ValueError
        return 0, -1
    base_etab = base_etab.to(dtype=torch.float32)
    b, n, k, h = base_etab.shape
    h = int(np.sqrt(h))
    etab = base_etab.view(b, n, k, h, h)
    pad = (0, 2, 0, 2)
    etab = F.pad(etab, pad, "constant", 0)
    if n*data['sortcery_nrgs'].shape[1] > max_tokens:
        batch_size = int(max_tokens / n)
    else:
        batch_size = data['sortcery_nrgs'].shape[1]
    done_num = 0
    all_preds = []
    all_refs = []
    for batch in range(0, data['sortcery_nrgs'].shape[1], batch_size):
        predicted_E, _, ref_energies = calc_eners_stability(etab, E_idx, data['sortcery_seqs'][:,done_num:batch+batch_size], data['sortcery_nrgs'][:,done_num:batch+batch_size])
        all_preds.append(predicted_E)
        all_refs.append(ref_energies)
        done_num = batch

    # normalize values around 0 for pearson correlation calculation
    predicted_E = torch.cat(all_preds, dim=0)
    ref_energies = torch.cat(all_refs, dim=0)
    # norm_pred = predicted_E - torch.mean(predicted_E) # n
    # norm_ref = ref_energies - torch.mean(ref_energies) # n

    mse = loss_fn(predicted_E, ref_energies)
    if torch.isnan(mse):
        # print(mse)
        # print(norm_pred)
        # print(norm_ref)
        # raise ValueError
        return 0, -1
    return mse, data['sortcery_nrgs'].shape[1] # scalar; negate, since we want to minimize our loss function

def cluster_interface_loss(base_etab, E_idx, data, max_tokens=20000, use_sc_mask=False):
    # b, n, k, h = base_etab.shape
    # h = int(np.sqrt(h))
    # etab = base_etab.view(b, n, k, h, h)
    # pad = (0, 2, 0, 2)
    # etab = F.pad(etab, pad, "constant", 0)
    # # data['sortcery_seqs'] = torch.cat([data['sortcery_seqs'], data['seqs'].unsqueeze(1)], dim=1)
    # # pad = (0, 1, 0, 0)
    # # data['sortcery_nrgs'] = torch.nn.functional.pad(data['sortcery_nrgs'], pad)
    # batch_scores, _, ref_energies = calc_eners_peptide(etab, E_idx, data)
    # all_scores = []
    # all_real = []

    # for test_ind in range(batch_scores.shape[0]):
    #     test_scores = torch.log_softmax(-1*batch_scores[test_ind], dim=0)
    #     test_scores = test_scores[~torch.isnan(ref_energies[test_ind])]
    #     # test_scores = batch_scores[test_ind][ref_energies[test_ind] != torch.nan].cpu().numpy()
    #     test_real = ref_energies[test_ind][~torch.isnan(ref_energies[test_ind])]
    #     all_scores.append(test_scores)
    #     all_real.append(test_real)

    # all_scores = torch.cat(all_scores, 0)
    # all_real = torch.cat(all_real, 0)

    # all_scores = all_scores - torch.mean(all_scores)
    # all_real = all_real - torch.mean(all_real)
    
    # pearson = torch.sum(all_scores * all_real) / (torch.sqrt(torch.sum(all_scores**2)) * torch.sqrt(torch.sum(all_real**2)))

    # print(pearson)
    # print(all_scores)
    # print(all_real)

    # if torch.isnan(pearson):
    #     return 0, -1

    # return -pearson, len(ref_energies)
    b, n, k, h = base_etab.shape
    h = int(np.sqrt(h))
    etab = base_etab.view(b, n, k, h, h)
    pad = (0, 2, 0, 2)
    etab = F.pad(etab, pad, "constant", 0)
    # pad = (0, 1, 0, 0)
    # data['sortcery_nrgs'] = torch.nn.functional.pad(data['sortcery_nrgs'], pad)
    batch_scores, _, ref_energies = calc_eners_stability(etab, E_idx, data['sortcery_seqs'], data['sortcery_nrgs'], filter=False)
    all_scores = []
    all_real = []

    for test_ind in range(batch_scores.shape[0]):
        test_scores = torch.log_softmax(-1*batch_scores[test_ind], dim=0)
        test_scores = test_scores[~torch.isnan(ref_energies[test_ind])]
        # test_scores = batch_scores[test_ind][ref_energies[test_ind] != torch.nan].cpu().numpy()
        test_real = ref_energies[test_ind][~torch.isnan(ref_energies[test_ind])]
        all_scores.append(test_scores)
        all_real.append(test_real)

    all_scores = torch.cat(all_scores, 0)
    all_real = torch.cat(all_real, 0)

    all_scores = all_scores - torch.mean(all_scores)
    all_real = all_real - torch.mean(all_real)
    
    pearson = torch.sum(all_scores * all_real) / (torch.sqrt(torch.sum(all_scores**2)) * torch.sqrt(torch.sum(all_real**2)))
    if torch.isnan(pearson):
        return 0, -1

    return -pearson, len(ref_energies)


def cluster_stability_loss(base_etab, E_idx, data, use_sc_mask=False):
    b, n, k, h = base_etab.shape
    h = int(np.sqrt(h))
    etab = base_etab.view(b, n, k, h, h)
    pad = (0, 2, 0, 2)
    etab = F.pad(etab, pad, "constant", 0)
    data['sortcery_seqs'] = torch.cat([data['sortcery_seqs'], data['seqs'].unsqueeze(1)], dim=1)
    # pad = (0, 1, 0, 0)
    # data['sortcery_nrgs'] = torch.nn.functional.pad(data['sortcery_nrgs'], pad)
    batch_scores, _, ref_energies = calc_eners_stability(etab, E_idx, data['sortcery_seqs'], data['sortcery_nrgs'], filter=False)
    all_scores = []
    all_real = []

    for test_ind in range(batch_scores.shape[0]):
        test_scores = torch.log_softmax(-1*batch_scores[test_ind], dim=0)
        test_scores = test_scores[:-1][~torch.isnan(ref_energies[test_ind])]
        # test_scores = batch_scores[test_ind][ref_energies[test_ind] != torch.nan].cpu().numpy()
        test_real = ref_energies[test_ind][~torch.isnan(ref_energies[test_ind])]
        all_scores.append(test_scores)
        all_real.append(test_real)

    all_scores = torch.cat(all_scores, 0)
    all_real = torch.cat(all_real, 0)

    all_scores = all_scores - torch.mean(all_scores)
    all_real = all_real - torch.mean(all_real)
    
    pearson = torch.sum(all_scores * all_real) / (torch.sqrt(torch.sum(all_scores**2)) * torch.sqrt(torch.sum(all_real**2)))
    if torch.isnan(pearson):
        return 0, -1

    data['sortcery_seqs'] = data['sortcery_seqs'][:,:-1]
    return -pearson, len(ref_energies)

def cluster_stability_loss_diff(base_etab, E_idx, data, use_sc_mask=False):
    b, n, k, h = base_etab.shape
    h = int(np.sqrt(h))
    etab = base_etab.view(b, n, k, h, h)
    pad = (0, 2, 0, 2)
    etab = F.pad(etab, pad, "constant", 0)
    data['sortcery_seqs'] = torch.cat([data['sortcery_seqs'], data['seqs'].unsqueeze(1)], dim=1)
    # pad = (0, 1, 0, 0)
    # data['sortcery_nrgs'] = torch.nn.functional.pad(data['sortcery_nrgs'], pad)
    batch_scores, _, ref_energies = calc_eners_stability(etab, E_idx, data['sortcery_seqs'], data['sortcery_nrgs'], filter=False)
    all_scores = []
    all_real = []

    for test_ind in range(batch_scores.shape[0]):
        test_scores = batch_scores[test_ind] - batch_scores[test_ind, -1]
        test_scores = test_scores[:-1][~torch.isnan(ref_energies[test_ind])]
        # test_scores = batch_scores[test_ind][ref_energies[test_ind] != torch.nan].cpu().numpy()
        test_real = ref_energies[test_ind][~torch.isnan(ref_energies[test_ind])]
        all_scores.append(test_scores)
        all_real.append(test_real)

    all_scores = torch.cat(all_scores, 0)
    all_real = torch.cat(all_real, 0)

    all_scores = all_scores - torch.mean(all_scores)
    all_real = all_real - torch.mean(all_real)
    
    pearson = torch.sum(all_scores * all_real) / (torch.sqrt(torch.sum(all_scores**2)) * torch.sqrt(torch.sum(all_real**2)))
    if torch.isnan(pearson):
        return 0, -1

    data['sortcery_seqs'] = data['sortcery_seqs'][:,:-1]
    return -pearson, len(ref_energies)

def cluster_stability_loss_diff_mse(base_etab, E_idx, data, use_sc_mask=False):
    loss_fn = torch.nn.MSELoss()
    b, n, k, h = base_etab.shape
    h = int(np.sqrt(h))
    etab = base_etab.view(b, n, k, h, h)
    pad = (0, 2, 0, 2)
    etab = F.pad(etab, pad, "constant", 0)
    data['sortcery_seqs'] = torch.cat([data['sortcery_seqs'], data['seqs'].unsqueeze(1)], dim=1)
    # pad = (0, 1, 0, 0)
    # data['sortcery_nrgs'] = torch.nn.functional.pad(data['sortcery_nrgs'], pad)
    batch_scores, _, ref_energies = calc_eners_stability(etab, E_idx, data['sortcery_seqs'], data['sortcery_nrgs'], filter=False)
    all_scores = []
    all_real = []

    for test_ind in range(batch_scores.shape[0]):
        test_scores = batch_scores[test_ind] - batch_scores[test_ind, -1]
        test_scores = test_scores[:-1][~torch.isnan(ref_energies[test_ind])]
        # test_scores = batch_scores[test_ind][ref_energies[test_ind] != torch.nan].cpu().numpy()
        test_real = ref_energies[test_ind][~torch.isnan(ref_energies[test_ind])]
        all_scores.append(test_scores)
        all_real.append(test_real)

    all_scores = torch.cat(all_scores, 0)
    all_real = torch.cat(all_real, 0)

    loss = loss_fn(all_scores, all_real)

    data['sortcery_seqs'] = data['sortcery_seqs'][:,:-1]
    return loss, len(ref_energies)


def cluster_stability_loss_log_diff(base_etab, E_idx, data, use_sc_mask=False):
    b, n, k, h = base_etab.shape
    h = int(np.sqrt(h))
    etab = base_etab.view(b, n, k, h, h)
    pad = (0, 2, 0, 2)
    etab = F.pad(etab, pad, "constant", 0)
    data['sortcery_seqs'] = torch.cat([data['sortcery_seqs'], data['seqs'].unsqueeze(1)], dim=1)
    # pad = (0, 1, 0, 0)
    # data['sortcery_nrgs'] = torch.nn.functional.pad(data['sortcery_nrgs'], pad)
    batch_scores, _, ref_energies = calc_eners_stability(etab, E_idx, data['sortcery_seqs'], data['sortcery_nrgs'], filter=False)
    all_scores = []
    all_real = []

    for test_ind in range(batch_scores.shape[0]):
        test_scores = torch.log(batch_scores[test_ind]) - torch.log(batch_scores[test_ind, -1])
        test_scores = test_scores[:-1][~torch.isnan(ref_energies[test_ind])]
        # test_scores = batch_scores[test_ind][ref_energies[test_ind] != torch.nan].cpu().numpy()
        test_real = ref_energies[test_ind][~torch.isnan(ref_energies[test_ind])]
        all_scores.append(test_scores)
        all_real.append(test_real)

    all_scores = torch.cat(all_scores, 0)
    all_real = torch.cat(all_real, 0)

    all_scores = all_scores - torch.mean(all_scores)
    all_real = all_real - torch.mean(all_real)
    
    pearson = torch.sum(all_scores * all_real) / (torch.sqrt(torch.sum(all_scores**2)) * torch.sqrt(torch.sum(all_real**2)))
    if torch.isnan(pearson):
        return 0, -1

    data['sortcery_seqs'] = data['sortcery_seqs'][:,:-1]
    return -pearson, len(ref_energies)


def cluster_stability_loss_mse(base_etab, E_idx, data, use_sc_mask=False):
    loss_fn = torch.nn.MSELoss()
    b, n, k, h = base_etab.shape
    h = int(np.sqrt(h))
    etab = base_etab.view(b, n, k, h, h)
    pad = (0, 2, 0, 2)
    etab = F.pad(etab, pad, "constant", 0)
    data['sortcery_seqs'] = torch.cat([data['sortcery_seqs'], data['seqs'].unsqueeze(1)], dim=1)
    # pad = (0, 1, 0, 0)
    # data['sortcery_nrgs'] = torch.nn.functional.pad(data['sortcery_nrgs'], pad)
    batch_scores, _, ref_energies = calc_eners_stability(etab, E_idx, data['sortcery_seqs'], data['sortcery_nrgs'], filter=False)
    all_scores = []
    all_real = []
    for test_ind in range(batch_scores.shape[0]):
        # test_scores = torch.log_softmax(-1*batch_scores[test_ind], dim=0)
        test_scores = batch_scores[test_ind] - batch_scores[test_ind, -1]
        test_scores = test_scores[:-1][~torch.isnan(ref_energies[test_ind])]
        # test_scores = batch_scores[test_ind][ref_energies[test_ind] != torch.nan].cpu().numpy()
        test_real = ref_energies[test_ind][~torch.isnan(ref_energies[test_ind])]
        all_scores.append(test_scores)
        all_real.append(test_real)

    all_scores = torch.cat(all_scores, 0)
    all_real = torch.cat(all_real, 0)

    loss = loss_fn(all_scores, all_real)
    # loss = torch.mean((all_scores - all_real)**2)
    data['sortcery_seqs'] = data['sortcery_seqs'][:,:-1]
    return loss, len(ref_energies)

def thermo_loss_mse(ref_energies, pred_energies):
    loss_fn = torch.nn.MSELoss(reduction='none')
    pred_energies = pred_energies[~torch.isnan(ref_energies)]
    ref_energies = ref_energies[~torch.isnan(ref_energies)]
    loss = torch.nanmean(loss_fn(pred_energies, ref_energies.to(torch.float32)))
    count = torch.sum(~torch.isnan(ref_energies))
    return loss, count

def thermo_loss_corr(ref_energies, pred_energies):
    loss_fn = torch.nn.MSELoss(reduction='none')
    pred_energies = pred_energies[~torch.isnan(ref_energies)]
    ref_energies = ref_energies[~torch.isnan(ref_energies)]
    pred_energies = pred_energies - torch.mean(pred_energies)
    ref_energies = ref_energies - torch.mean(ref_energies)    
    pearson = torch.sum(pred_energies * ref_energies) / (torch.sqrt(torch.sum(pred_energies**2)) * torch.sqrt(torch.sum(ref_energies**2)))
    count = torch.sum(~torch.isnan(ref_energies))
    return -pearson, count

def cluster_stability_loss_old(base_etab, E_idx, data, use_sc_mask=False):
    ''' Compute the correlation between etab's predicted energies and experimental stability energies.
    '''
    # TODO hacky fix to avoid problems when without SORTCERY data

    if data["sortcery_nrgs"].numel() < 5:
        return 0, -1
    
    b, n, k, h = base_etab.shape
    h = int(np.sqrt(h))
    etab = base_etab.view(b, n, k, h, h)
    pad = (0, 2, 0, 2)
    etab = F.pad(etab, pad, "constant", 0)
    predicted_E, all_ref_seqs, ref_energies = calc_eners_stability(etab, E_idx, data['sortcery_seqs'], data['sortcery_nrgs'], filter=False)

    # normalize values around 0 for pearson correlation calculation
    pearson_loss = 0
    num_ref_seqs = 0
    for pred_E, ref_E, ref_seqs in zip(predicted_E, ref_energies, all_ref_seqs):
        mask = ref_E != torch.nan
        ref_E = ref_E[mask]
        pred_E = pred_E[mask]
        num_ref_seqs += len(ref_seqs[mask])
        norm_pred = pred_E - torch.mean(pred_E) # n
        norm_ref = ref_E - torch.mean(ref_E) # n

        pearson = torch.sum(norm_pred * norm_ref) / (torch.sqrt(torch.sum(norm_pred**2)) * torch.sqrt(torch.sum(norm_ref**2)))
        if torch.isnan(pearson):
            return 0, -1
        pearson_loss += pearson
    pearson_loss /= len(predicted_E)
    return -pearson_loss, num_ref_seqs # scalar; negate, since we want to minimize our loss function

# def stability_loss_mse(base_etab, E_idx, data, max_tokens=20000, use_sc_mask=False):
#     ''' Compute the correlation between etab's predicted energies and experimental stability energies.
#     '''
#     # TODO hacky fix to avoid problems when without SORTCERY data
#     if data["sortcery_seqs"].numel() < 5:
#         return 0, -1
    
#     b, n, k, h = base_etab.shape
#     h = int(np.sqrt(h))
#     etab = base_etab.view(b, n, k, h, h)
#     pad = (0, 2, 0, 2)
#     etab = F.pad(etab, pad, "constant", 0)
#     predicted_E, ref_seqs, ref_energies = calc_eners_stability(etab, E_idx, data['sortcery_seqs'], data['sortcery_nrgs'])

#     # normalize values around 0 for pearson correlation calculation
#     norm_pred = predicted_E - torch.mean(predicted_E) # n
#     norm_ref = ref_energies - torch.mean(ref_energies) # n

#     loss = torch.mean((norm_pred - norm_ref)**2)
#     return loss, len(ref_seqs) # scalar;


def sortcery_loss(base_etab, E_idx, data, max_tokens=20000, use_sc_mask=False):
    ''' Compute the mean squared error between the etab's predicted energies for peptide-protein complexes and experimental energies derived from SORTCERY.
    '''
    # TODO hacky fix to avoid problems when without SORTCERY data
    if len(data["sortcery_seqs"]) == 0:
        return 0, -1
    
    b, n, k, h = base_etab.shape
    h = int(np.sqrt(h))
    etab = base_etab.view(b, n, k, h, h)
    pad = (0, 2, 0, 2)
    etab = F.pad(etab, pad, "constant", 0)
    
    predicted_E, ref_seqs, ref_energies = calc_eners_peptide(etab, E_idx, data, pep_len=data['chain_lens'][0][-1])

    # normalize values around 0 for pearson correlation calculation
    norm_pred = predicted_E - torch.mean(predicted_E) # n
    norm_ref = ref_energies - torch.mean(ref_energies) # n
    # np_pred = predicted_E.cpu().numpy()
    # np_pred = (np_pred - np.mean(np_pred)) / np.std(np_pred)
    # if data['ids'][0] == 'B2LA1_HUMAN_5UUL_A_holo':
    #     print(list(np_pred))

    pearson = torch.sum(norm_pred * norm_ref) / (torch.sqrt(torch.sum(norm_pred**2)) * torch.sqrt(torch.sum(norm_ref**2)))
    return -pearson, len(ref_seqs) # scalar; negate, since we want to minimize our loss function

def sortcery_loss_mult_bb(etab, E_idx, data, max_tokens=20000, use_sc_mask=False):
    ''' Compute the mean squared error between the etab's predicted energies for peptide-protein complexes and experimental energies derived from SORTCERY.
    '''
    # TODO hacky fix to avoid problems when without SORTCERY data
    if len(data["sortcery_seqs"]) == 0:
        return 0, 1
    
    # predicted_E, ref_seqs, ref_energies = calc_eners_batched(etab, E_idx, data)
    # predicted_E = (predicted_E - torch.mean(predicted_E, 1, keepdim=True)) / torch.std(predicted_E, 1, keepdim=True)
    # predicted_E = torch.mean(predicted_E, 0)

    predicted_Es = []
    for i in range(etab.shape[0]):
        datac = copy.deepcopy(data)
        lenc = datac['chain_lens'][i][0] + datac['chain_lens'][i][1]
        etabc = etab[i,:lenc].unsqueeze(0)
        datac['seqs'] = datac['seqs'][i,:lenc].unsqueeze(0)
        datac['x_mask'] = datac['x_mask'][i,:lenc].unsqueeze(0)
        X_ca = datac['X'][i,:lenc,1,:].unsqueeze(0)
        _, _, E_idxc = extract_knn(X_ca, datac['x_mask'], eps=1E-6, top_k=30)
        predicted_E, ref_seqs, ref_energies = calc_eners_peptide(etabc, E_idxc, datac)

        # ### TEST ###
        # norm_pred_test = predicted_E - torch.mean(predicted_E)
        # norm_ref_test = ref_energies - torch.mean(ref_energies)
        # pearson_test = torch.sum(norm_pred_test * norm_ref_test) / (torch.sqrt(torch.sum(norm_pred_test**2)) * torch.sqrt(torch.sum(norm_ref_test**2)))
        # print(pearson_test)
        # ### END TEST ###

        predicted_E = (predicted_E - torch.mean(predicted_E)) / torch.std(predicted_E)
        predicted_Es.append(predicted_E)
    predicted_E = torch.stack(predicted_Es, dim=0)
    predicted_E = torch.mean(predicted_E, 0)

    # normalize values around 0 for pearson correlation calculation
    norm_pred = predicted_E - torch.mean(predicted_E) # n
    norm_ref = ref_energies - torch.mean(ref_energies) # n

    pearson = torch.sum(norm_pred * norm_ref) / (torch.sqrt(torch.sum(norm_pred**2)) * torch.sqrt(torch.sum(norm_ref**2)))
    return -pearson, len(ref_seqs) # scalar; negate, since we want to minimize our loss function

def calc_eners_nodes(log_probs, data):
    ref_seqs = data["seqs"][0]
    all_peptide_seqs = torch.stack([torch.from_numpy(np.array(seq_to_ints(seq))) for seq in all_peptide_seqs], dim=0).to(device=etab.device)
    if len(all_peptide_seqs.shape) == 1:
        all_peptide_seqs = all_peptide_seqs.unsqueeze(0)
    all_ref_energies = torch.Tensor([ener for ener in all_ref_energies])
    # all_ref_energies = all_ref_energies.unsqueeze(-1)
    # # for both memory constraints and improved performance, can randomly sample from the SORTCERY data if desired
    if all_peptide_seqs.shape[0] > 160000:
        inds = random.sample(list(np.arange(all_peptide_seqs.shape[0])), 1600)
        peptide_seqs = all_peptide_seqs[inds]
        ref_energies = all_ref_energies[inds]
    else:
        peptide_seqs = all_peptide_seqs
        ref_energies = all_ref_energies

    peptide_shape = peptide_seqs.shape # n, c; c is the length of the chain
    protein_seq = ref_seqs[:-peptide_shape[1]] # (L - c), remove the native peptide to obtain only the protein sequences, so we can replace the peptides with SORTCERY peptides

    return predicted_E, ref_seqs, ref_energies

def sortcery_loss_nodes_mult_bb(nodes, E_idx, data, max_tokens=20000, use_sc_mask=False):
    ''' Compute the mean squared error between the etab's predicted energies for peptide-protein complexes and experimental energies derived from SORTCERY.
    '''
    # TODO hacky fix to avoid problems when without SORTCERY data
    if len(data["sortcery_seqs"]) == 0:
        return 0, 1
    
    # predicted_E, ref_seqs, ref_energies = calc_eners_batched(etab, E_idx, data)
    # predicted_E = (predicted_E - torch.mean(predicted_E, 1, keepdim=True)) / torch.std(predicted_E, 1, keepdim=True)
    # predicted_E = torch.mean(predicted_E, 0)

    predicted_Es = []
    for i in range(etab.shape[0]):
        datac = copy.deepcopy(data)
        lenc = datac['chain_lens'][i][0] + datac['chain_lens'][i][1]
        nodesc = nodes[i,:lenc].unsqueeze(0)
        datac['seqs'] = datac['seqs'][i,:lenc].unsqueeze(0)
        datac['x_mask'] = datac['x_mask'][i,:lenc].unsqueeze(0)
        X_ca = datac['X'][i,:lenc,1,:].unsqueeze(0)
        _, _, E_idxc = extract_knn(X_ca, datac['x_mask'], eps=1E-6, top_k=30)
        predicted_E, ref_seqs, ref_energies = calc_eners_nodes(nodesc, E_idxc, datac)

        # ### TEST ###
        # norm_pred_test = predicted_E - torch.mean(predicted_E)
        # norm_ref_test = ref_energies - torch.mean(ref_energies)
        # pearson_test = torch.sum(norm_pred_test * norm_ref_test) / (torch.sqrt(torch.sum(norm_pred_test**2)) * torch.sqrt(torch.sum(norm_ref_test**2)))
        # print(pearson_test)
        # ### END TEST ###

        predicted_E = (predicted_E - torch.mean(predicted_E)) / torch.std(predicted_E)
        predicted_Es.append(predicted_E)
    predicted_E = torch.stack(predicted_Es, dim=0)
    predicted_E = torch.mean(predicted_E, 0)

    # normalize values around 0 for pearson correlation calculation
    norm_pred = predicted_E - torch.mean(predicted_E) # n
    norm_ref = ref_energies - torch.mean(ref_energies) # n

    pearson = torch.sum(norm_pred * norm_ref) / (torch.sqrt(torch.sum(norm_pred**2)) * torch.sqrt(torch.sum(norm_ref**2)))
    return -pearson, len(ref_seqs) # scalar; negate, since we want to minimize our loss function

def sortcery_loss_mse(etab, E_idx, datamax_tokens=20000, use_sc_mask=False):
    ''' Compute the mean squared error between the etab's predicted energies for peptide-protein complexes and experimental energies derived from SORTCERY.
    '''
    # TODO hacky fix to avoid problems when without SORTCERY data
    if len(data["sortcery_seqs"]) == 0:
        return 0, 1
    
    predicted_E, ref_seqs, ref_energies = calc_eners_peptide(etab, E_idx, data)

    # normalize values around 0 for pearson correlation calculation
    norm_pred = predicted_E - torch.mean(predicted_E) # n
    norm_ref = ref_energies - torch.mean(ref_energies) # n
    norm_ref = norm_ref.to(dtype=etab.dtype)
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(norm_pred, norm_ref)
    return loss, len(ref_seqs) # scalar; negate, since we want to minimize our loss function

def rocklin_loss(etab, E_idx, data):
    n_batch, n, k, n_tok = etab.shape
    n_tok = int(np.sqrt(n_tok))
    etab = etab[0].unsqueeze(-1).view(n, k, n_tok, n_tok)
    E_idx = E_idx[0]
    x_mask = data["x_mask"][0]
    rocklin_seqs = data['sortcery_seqs'][0]
    rocklin_energies = -1*data['sortcery_nrgs'][0]
    pred_energies = []
    if len(rocklin_seqs) <= 1:
        return 0, 1
    
    # X is encoded as 20 so lets just add an extra row/col of zeros
    pad = (0, 1, 0, 1)
    etab = F.pad(etab, pad, "constant", 0) # L x 30 x 21 x 21
    etab = etab.unsqueeze(0).expand(len(rocklin_seqs), -1, -1, -1, -1) # num_seqs x L x 30 x 21 x 21

    # identity of all residues
    seq_residues = rocklin_seqs.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, k, 21).unsqueeze(-1) # num_seqs x n x 30 x 21 x 1
    E_neighbors = torch.gather(etab, -1, seq_residues).squeeze(-1) # num_seqs x n x 30 x 21, gather etabs for each known residue in seqs

    # identity of all kNN
    E_idx_expanded = E_idx.unsqueeze(0).expand(len(rocklin_seqs), -1, -1)
    E_aa = torch.gather(rocklin_seqs.unsqueeze(-1).expand(-1, -1, k), 1, E_idx_expanded).unsqueeze(-1) # num_seqs x n x 30 x 1
    E_seqs = torch.gather(E_neighbors, -1, E_aa).squeeze(-1) # num_seqs x n x 30

    # halve the energies of all bidirectional edges to avoid double counting
    adj_matrix = torch.zeros((n,n), device=etab.device)
    residueNums = E_idx[:,0].unsqueeze(-1).expand(-1, k)
    edgeIndices = torch.cat((residueNums.reshape(-1,1), E_idx.reshape(-1,1)), dim=-1)
    adj_matrix[edgeIndices[:,0], edgeIndices[:,1]] = 1
    adj_matrix = adj_matrix + adj_matrix.t() # add the adjacency matrix to its transpose; bidirectional edges will have value 2
    bidir_mask = torch.gather(adj_matrix, -1, E_idx)
    bidir_mask[:,0] = 1 # self edges will be double counted, so we reset; then we take the reciprocal so that bidirectional edges have value 1/2 in the mask
    bidir_mask = (1/bidir_mask).unsqueeze(0).expand(len(rocklin_seqs), -1, -1) # n x L x 30

    # masks for residues with identity X (isnt_x_aa) and unmodeled residues (x_mask)
    isnt_x_aa = (rocklin_seqs != 20).float().to(etab.device) # num_seqs x n
    x_mask = x_mask.expand(len(rocklin_seqs), -1).unsqueeze(-1) # num_seqs x n x 1
    isnt_x_aa = isnt_x_aa.unsqueeze(-1) # num_seqs x n x 1
    full_mask = x_mask * isnt_x_aa # num_seqs x n x 1

    E_seqs *= full_mask * bidir_mask

    # compute predicted energy for each sequence
    pred_energies = E_seqs.sum(dim=(-2,-1)) # num_sequences, sum across all pairs for all n residues

    if len(pred_energies) == 0:
        return 0, 1
    pred_energies = pred_energies.to(device=rocklin_energies.device)
    norm_pred = pred_energies - torch.mean(pred_energies) # n
    norm_ref = rocklin_energies - torch.mean(rocklin_energies) # n
    pearson = torch.sum(norm_pred * norm_ref) / (torch.sqrt(torch.sum(norm_pred**2)) * torch.sqrt(torch.sum(norm_ref**2)))
    return -pearson, len(rocklin_seqs)


def structure_loss(frames, data):
    from openfold.utils.loss import backbone_loss

    loss = backbone_loss(data['backbone_4x4'], data['x_mask'], traj=frames[-1])

    return loss, torch.sum(data['x_mask'])


# Loss function construction


def _get_loss_fn(fn_name):
    """ Retrieve a loss function from this file given the function name """
    try:
        if fn_name in NOT_LOSS_FNS:  # prevent recursive and unexpected behavior
            raise NameError
        return getattr(sys.modules[__name__], fn_name)
    except NameError as ne:
        raise ValueError(f"Loss fn {fn_name} not found in {__name__}") from ne


def construct_loss_fn(hparams, aux_loss=False):
    """ Construct a combined loss function based on the inputted hparams

    Args
    ----
    hparams : dict
        The fully constructed hparams (see :code:`terminator/utils/model/default_hparams.py`). It should
        contain an entry for 'loss_config' in the format {loss_fn_name : scaling_factor}. For example,
        .. code-block :
            {
                'nlcpl': 1,
                'etab_norm_penalty': 0.01
            }

    Returns
    -------
    _loss_fn
        The constructed loss function
    """
    if aux_loss:
        loss_config = hparams['aux_loss']
    else:
        loss_config = hparams['loss_config']

    def _loss_fn(etab, h_V, h_C, pred_ener, E_idx, frames, data, esm, batch_converter, tokenizer, esm_type, converter, eval, distill_etab, save_seqs=False):
        """ The returned loss function """
        loss_dict = {}
        for loss_fn_name, scaling_factor in loss_config.items():
            subloss_fn = _get_loss_fn(loss_fn_name)
            seqs = torch.Tensor([0])
            nsr = torch.Tensor([0])
            if 'esm' in loss_fn_name:
                loss, count = subloss_fn(h_V, data, esm, batch_converter, tokenizer, esm_type, converter, eval=eval, use_sc_mask=hparams['use_sc_mask'])
            elif 'nlll' in loss_fn_name or loss_fn_name == 'loss_smoothed' or loss_fn_name == 'pssm_loss' or loss_fn_name == 'loss_spike':
                loss, count, nsr, seqs = subloss_fn(h_V, E_idx, data, use_sc_mask=hparams['use_sc_mask'], weight_inter_types=hparams['weight_inter_types'])
            elif 'confidence' in loss_fn_name:
                loss, count = subloss_fn(h_V, h_C, data, diff=hparams['blosum_diff'], vector=hparams['confidence_vector'], matrix_type=hparams['confidence_matrix_type'])
            elif 'rocklin' in loss_fn_name or 'sortcery' in loss_fn_name or 'stability' in loss_fn_name or 'interface' in loss_fn_name:
                loss, count = subloss_fn(etab, E_idx, data, max_tokens=hparams['max_loop_tokens'], use_sc_mask=hparams['use_sc_mask'])
            elif 'nlcpl' in loss_fn_name:
                loss, count = subloss_fn(etab, E_idx, data, hparams, use_sc_mask=hparams['use_sc_mask'], per_protein_loss=hparams['per_protein_loss'], weight_inter_types=hparams['weight_inter_types'])
            elif 'structure' in loss_fn_name:
                loss, count = subloss_fn(frames, data)
            elif 'thermo' in loss_fn_name:
                loss, count = subloss_fn(data['sortcery_nrgs'], pred_ener)
            elif 'gram' in loss_fn_name:
                loss, count = subloss_fn(h_V, data, use_sc_mask=hparams['use_sc_mask'])
            elif 'distill' in loss_fn_name:
                loss, count = subloss_fn(etab, data, distill_etab, use_sc_mask=hparams['use_sc_mask'])
            elif 'likelihood' in loss_fn_name:
                loss, count, _, _, _ = subloss_fn(h_V, data)
            else:
                loss, count, _, _, _ = subloss_fn(etab, E_idx, data, hparams, use_sc_mask=hparams['use_sc_mask'])
            if loss is None:
                continue
            if save_seqs:
                loss_dict[loss_fn_name] = {"loss": loss, "count": count, "scaling_factor": scaling_factor, "seqs": seqs, "nsr": nsr}
            else:
                loss_dict[loss_fn_name] = {"loss": loss, "count": count, "scaling_factor": scaling_factor, "nsr": nsr}
        return loss_dict

    return _loss_fn
