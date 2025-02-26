""" Backbone featurization modules

This file contains modules which featurize the protein backbone graph via
its backbone coordinates. Adapted from https://github.com/jingraham/neurips19-graph-protein-design
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import copy
import random
from .utils import gather_edges, gather_nodes, gelu, insert_vectors_3D, insert_vectors_4D

# pylint: disable=no-member

class MultiLayerLinear(nn.Module):
    def __init__(self, in_features, out_features, num_layers, activation_layers='relu', dropout=0):
        super(MultiLayerLinear, self).__init__()
        self.activation_layers = activation_layers
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
            if self.activation_layers == 'relu':
                x = F.relu(x)
            else:
                x = gelu(x)
        if self.dropout_prob > 0:
            x = self.dropout(x)
        return x
    
class VoxelModel(nn.Module):
    """Model with same padding
    Conv5 uses a large filter size to aggregate the features from the whole box"""

    def __init__(self):
        super(VoxelModel, self).__init__()
        self.conv1 = nn.Conv3d(8, 32, 3, padding="same")
        self.conv2 = nn.Conv3d(32, 64, 3, padding="same")
        self.conv3 = nn.Conv3d(64, 80, 3, padding="same")
        self.conv4 = nn.Conv3d(80, 20, 3, padding="same")
        self.conv5 = nn.Conv3d(20, 20, 20, padding="same")
        self.conv6 = nn.Conv3d(20, 16, 3, padding="same")
        self.conv7 = nn.Conv3d(16, 1, 3, padding="same")
        self.dropout1 = nn.Dropout(0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = F.relu(x)

        x = self.conv5(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.conv6(x)
        x = F.relu(x)

        x = self.conv7(x)
        x = torch.sigmoid(x)
        return x

class PositionalEncodings(nn.Module):
    """ Module to generate differential positional encodings for protein graph edges """
    def __init__(self, num_embeddings):
        super().__init__()
        self.num_embeddings = num_embeddings

    def forward(self, E_idx):
        """ Generate directional differential positional encodings for edges

        Args
        ----
        E_idx : torch.LongTensor
            Protein kNN edge indices
            Shape: n_batches x seq_len x k

        Returns
        -------
        E : torch.Tensor
            Directional Diffential positional encodings for edges
            Shape: n_batches x seq_len x k x num_embeddings
        """
        dev = E_idx.device
        # i-j
        N_nodes = E_idx.size(1)
        ii = torch.arange(N_nodes, dtype=torch.float32).view((1, -1, 1)).to(dev)
        d = (E_idx.float() - ii).unsqueeze(-1)
        # Original Transformer frequencies
        frequency = torch.exp(
            torch.arange(0, self.num_embeddings, 2, dtype=torch.float32) *
            -(np.log(10000.0) / self.num_embeddings)).to(dev)
        # Grid-aligned
        # frequency = 2. * np.pi * torch.exp(
        #     -torch.linspace(
        #         np.log(self.period_range[0]),
        #         np.log(self.period_range[1]),
        #         self.num_embeddings / 2
        #     )
        # )
        angles = d * frequency.view((1, 1, 1, -1))
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E


class ProteinFeatures(nn.Module):
    """ Protein backbone featurization based on Ingraham et al NeurIPS

    Attributes
    ----------
    embeddings : PositionalEncodings
        Module to generate differential positional embeddings for edges
    dropout : nn.Dropout
        Dropout module
    node_embeddings, edge_embeddings : nn.Linear
        Embedding layers for nodes and edges
    norm_nodes, norm_edges : nn.LayerNorm
        Normalization layers for node and edge features
    """
    def __init__(self,
                 edge_features,
                 node_features,
                 num_positional_embeddings=16,
                 num_rbf=16,
                 top_k=30,
                 random_type='',
                 random_alpha=3.0,
                 deterministic=False,
                 deterministic_seed=10,
                 random_temperature=1.0,
                 features_type='full',
                 augment_eps=0.,
                 dropout=0.1,
                 esm_dropout=0.1,
                 num_dihedrals=11,
                 esm_rep_feat_ins=[640],
                 esm_rep_feat_outs=[24],
                 esm_attn_feat_ins=[600],
                 esm_attn_feat_outs=[20],
                 raw=False,
                 old=False,
                 bias=True,
                 chain_handle='',
                 center_node=False,
                 center_node_ablation=False,
                 random_graph=False):
        """ Extract protein features """
        super().__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.random_type = random_type
        self.random_alpha = random_alpha
        self.deterministic = deterministic
        self.deterministic_seed = deterministic_seed
        self.random_temperature = random_temperature
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.raw = raw
        self.old = old
        self.chain_handle = chain_handle
        self.center_node = center_node
        self.center_node_ablation = center_node_ablation
        self.random_graph = random_graph

        # Feature types
        self.features_type = features_type
        self.feature_dimensions = {
            'coarse': (3, num_positional_embeddings + num_rbf + 7),
            'full': (6, num_positional_embeddings + num_rbf + 7),
            'dist': (6, num_positional_embeddings + num_rbf),
            'hbonds': (3, 2 * num_positional_embeddings),
            'side_chain': (26, num_positional_embeddings + num_rbf + 8),
            'side_chain_ids': (16, num_positional_embeddings + num_rbf + 7),
            'side_chain_angs': (36, num_positional_embeddings + num_rbf + 9),
            'side_chain_orient': (26, num_positional_embeddings + num_rbf + 8),
            'side_chain_dihedral': (26, num_positional_embeddings + num_rbf + 7),
            'side_chain_esm': (6 + 33, num_positional_embeddings + num_rbf + 7 + 20),
            'side_chain_esm_embs': (6 + 33, num_positional_embeddings + num_rbf + 7),
            'side_chain_esm_attns': (6, num_positional_embeddings + num_rbf + 7 + esm_attn_feat_outs[-1]),
            'side_chain_esm_reps': (6 + esm_rep_feat_outs[-1], num_positional_embeddings + num_rbf + 7),
            'side_chain_esm_reps_attns': (6 + esm_rep_feat_outs[-1], num_positional_embeddings + num_rbf + 7 + esm_attn_feat_outs[-1]),
            'side_chain_esm_reps_mpnn': (6 + esm_rep_feat_outs[-1], 416),
            'side_chain_esm_reps_mpnn_post': (6, 416),
            'side_chain_esm_reps_mpnn_attns': (6 + esm_rep_feat_outs[-1], 416 + esm_attn_feat_outs[-1]),
            'full_mpnn': (6, 416),
            'full_mpnn_esm_module': (6, 416)
        }

        self.node_layer_dims = {
            'full': [6],
            'side_chain': [6, 10, 10],
            'side_chain_ids': [6, 10],
            'side_chain_angs': [6, 10, 10, num_dihedrals],
            'side_chain_orient': [6, 10, num_dihedrals],
            'side_chain_dihedral': [6, num_dihedrals],
            'side_chain_esm': [6, 33],
            'side_chain_esm_embs': [6, 33],
            'side_chain_esm_reps': [6, esm_rep_feat_outs[-1]],
            'side_chain_esm_attns': [6],
            'side_chain_esm_reps_attns': [6, esm_rep_feat_outs[-1]],
            'side_chain_esm_reps_mpnn': [6, esm_rep_feat_outs[-1]],
            'side_chain_esm_reps_mpnn_post': [6],
            'side_chain_esm_reps_mpnn_attns': [6, esm_rep_feat_outs[-1]],
            'full_mpnn': [6],
            'full_mpnn_esm_module': [6]
        }
        node_layers = self.node_layer_dims[features_type]
        full_edge_dim = num_positional_embeddings + num_rbf + 7
        self.edge_layer_dims = {
            'full': [full_edge_dim],
            'side_chain': [full_edge_dim + 1],
            'side_chain_ids': [full_edge_dim],
            'side_chain_angs': [full_edge_dim + 2],
            'side_chain_orient': [full_edge_dim + 1],
            'side_chain_dihedral': [full_edge_dim],
            'side_chain_esm': [full_edge_dim, 20],
            'side_chain_esm_embs': [full_edge_dim],
            'side_chain_esm_attns': [full_edge_dim, esm_attn_feat_outs[-1]],
            'side_chain_esm_reps': [full_edge_dim],
            'side_chain_esm_reps_attns': [full_edge_dim, esm_attn_feat_outs[-1]],
            'side_chain_esm_reps_mpnn': [416],
            'side_chain_esm_reps_mpnn_post': [416],
            'side_chain_esm_reps_mpnn_attns': [416, esm_attn_feat_outs[-1]],
            'full_mpnn': [416],
            'full_mpnn_esm_module': [416]
        }
        edge_layers = self.edge_layer_dims[features_type]

        # Positional encoding
        self.embeddings = PositionalEncodings(num_positional_embeddings)
        self.dropout = nn.Dropout(dropout)

        # Normalization and embedding
        node_in, edge_in = self.feature_dimensions[features_type]
        if not old:
            self.node_layers = nn.ModuleList()
            if len(node_layers) == 0: node_layers = [node_in]
            if len(edge_layers) == 0: edge_layers = [edge_in]
            for i in range(len(node_layers)):
                self.node_layers.append(nn.Linear(node_layers[i], node_features, bias=bias))
            self.edge_layers = nn.ModuleList()
            for i in range(len(edge_layers)):
                self.edge_layers.append(nn.Linear(edge_layers[i], edge_features, bias=bias))
            self.node_embedding = nn.Linear(len(node_layers) * node_features, node_features, bias=bias)
            self.edge_embedding = nn.Linear(len(edge_layers) * edge_features, edge_features, bias=bias)
            if self.center_node:
                self.center_node_embedding = nn.Linear(3, node_features, bias=bias)
                self.center_edge_embedding = nn.Linear(35, edge_features, bias=bias)
                self.center_node_norm = nn.LayerNorm(node_features)
                self.center_edge_norm = nn.LayerNorm(edge_features)
        else:
            self.node_embedding = nn.Linear(node_in, node_features, bias=bias)
            self.edge_embedding = nn.Linear(edge_in, edge_features, bias=bias)
        self.norm_nodes = nn.LayerNorm(node_features)  # Normalize(node_features)
        self.norm_edges = nn.LayerNorm(edge_features)  # Normalize(edge_features)
        if 'esm' in features_type or 'esm_module' in features_type:
            self.esm_attns_embedding = MultiLayerLinear(in_features=esm_attn_feat_ins, out_features=esm_attn_feat_outs, num_layers=len(esm_attn_feat_outs), activation_layers='gelu', dropout=esm_dropout).float()
        if 'reps' in features_type or 'esm_module' in features_type:
            self.esm_reps_embedding = MultiLayerLinear(in_features=esm_rep_feat_ins, out_features=esm_rep_feat_outs, num_layers=len(esm_rep_feat_outs), activation_layers='gelu', dropout=esm_dropout).float()

    def _perturb_distances(self, D, criterion):
        # Replace distance by log-propensity
        # Adapted from https://github.com/generatebio/chroma
        if criterion == "random_log":
            logp_edge = -3 * torch.log(D)
        elif criterion == "random_linear":
            logp_edge = -D / self.random_alpha
        elif criterion == "random_uniform":
            logp_edge = D * 0
        else:
            return D
        
        if not self.deterministic:
            Z = torch.rand_like(D)
        else:
            with torch.random.fork_rng():
                torch.random.manual_seed(self.deterministic_seed)
                Z_shape = [1] + list(D.shape)[1:]
                Z = torch.rand(Z_shape, device=D.device)

        # Sample Gumbel noise
        G = -torch.log(-torch.log(Z))

        # Negate because are doing argmin instead of argmax
        D_key = -(logp_edge / self.random_temperature + G)

        return D_key
    
    def sample_neighbors(self, distance_matrix, num_neighbors=30):
        """
        Samples `num_neighbors` neighboring points for each point in the input distance matrix,
        where selection probability is proportional to the inverse square of the distance.

        Args:
            distance_matrix (torch.Tensor): Tensor of shape [b, n, n] representing pairwise distances.
            num_neighbors (int): Number of neighbors to sample for each point.

        Returns:
            torch.Tensor: Indices of sampled neighbors of shape [b, n, num_neighbors].
        """
        b, n, _ = distance_matrix.shape

        # Avoid self-edges being chosen
        distance_matrix = distance_matrix.clone()
        distance_matrix.diagonal(dim1=1, dim2=2).fill_(float('inf'))

        # Compute inverse square of the distances
        inv_square_dist = 1.0 / (distance_matrix ** 4 + 1e-8)  # Add small term for numerical stability

        # Normalize to get probabilities (sum along last dim)
        probs = inv_square_dist / inv_square_dist.sum(dim=-1, keepdim=True)

        # Sample indices based on probabilities
        sampled_indices = torch.multinomial(probs.view(b * n, n), num_neighbors, replacement=False)
        sampled_indices = sampled_indices.view(b, n, num_neighbors)  # Reshape back to [b, n, num_neighbors]

        sampled_distances = torch.gather(distance_matrix, dim=-1, index=sampled_indices)
        
        return sampled_distances, sampled_indices 

    def _dist(self, X, mask, eps=1E-6):
        """ Pairwise euclidean distances """
        # Convolutional network on NCHW
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)

        # Identify k nearest neighbors (including self)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        if self.random_type != '':
            top_k = self.top_k // 2
        else:
            top_k = self.top_k
        top_k = min(top_k, D.shape[1])
        if self.random_graph:
            D_neighbors, E_idx = self.sample_neighbors(D_adjust, top_k-1)
            D_neighbors = torch.cat([torch.zeros((D.shape[0], D.shape[1], 1), dtype=D_neighbors.dtype, device=D_neighbors.device), D_neighbors], dim=2)
            E_idx = torch.cat([torch.arange(D.shape[1]).unsqueeze(0).unsqueeze(-1).expand((D.shape[0], D.shape[1], -1)).to(dtype=E_idx.dtype, device=E_idx.device), E_idx], dim=2)
        else:
            D_neighbors, E_idx = torch.topk(D_adjust, top_k, dim=-1, largest=False)
        mask_neighbors = gather_edges(mask_2D.unsqueeze(-1), E_idx)
        if self.random_type != '':
            mask_remaining = (1.0 - mask_neighbors).squeeze(-1).to(mask_neighbors.dtype)
            mask_2D_remaining = torch.ones_like(mask_2D).scatter(2, E_idx, mask_remaining)
            mask_2D *= mask_2D_remaining
            D *= mask_2D_remaining
            D = self._perturb_distances(D, self.random_type)
            D_max, _ = torch.max(D, -1, keepdim=True)
            D_adjust = D + (1. - mask_2D) * D_max
            D_neighbors_rand, E_idx_rand = torch.topk(D_adjust, top_k, dim=-1, largest=False)
            mask_neighbors_rand = gather_edges(mask_2D.unsqueeze(-1), E_idx_rand)
            D_neighbors = torch.cat([D_neighbors, D_neighbors_rand], 2)
            E_idx = torch.cat([E_idx, E_idx_rand], 2)
            mask_neighbors = torch.cat([mask_neighbors, mask_neighbors_rand], 2)
            # for ib, b in enumerate(E_idx):
            #     for ii, i in enumerate(b):
            #         if mask[ib][ii] == 0:
            #             continue
            #         for ij, j in enumerate(i[:top_k]):
            #             if j in i[top_k:]:
            #                 print(i)
            #                 print(j, ib, ii)
            #                 raise ValueError


        return D_neighbors, E_idx, mask_neighbors

    def _rbf(self, D, D_max=20., num_dims=4, D_count=None):
        dev = D.device
        # Distance radial basis function
        D_min = 0.
        if D_count is None:
            D_count = self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=dev)
        if num_dims == 4:
            D_mu = D_mu.view([1, 1, 1, -1])
        else:
            D_mu = D_mu.view([1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)

        # for i in range(D_count):
        #     fig = plt.figure(figsize=(4,4))
        #     ax = fig.add_subplot(111)
        #     rbf_i = RBF.data.numpy()[0,i,:,:]
        #     # rbf_i = D.data.numpy()[0,0,:,:]
        #     plt.imshow(rbf_i, aspect='equal')
        #     plt.axis('off')
        #     plt.tight_layout()
        #     plt.savefig('rbf{}.pdf'.format(i))
        #     print(np.min(rbf_i), np.max(rbf_i), np.mean(rbf_i))
        # exit(0)
        return RBF

    def _quaternions(self, R, eps=1e-10):
        """ Convert a batch of 3D rotations [R] to quaternions [Q]
            R [...,3,3]
            Q [...,4]
        """
        def _R(i, j):
            return R[:, :, :, i, j]

        # Simple Wikipedia version
        # en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        # For other options see math.stackexchange.com/questions/2074316/calculating-rotation-axis-from-rotation-matrix
        diag = torch.diagonal(R, dim1=-2, dim2=-1)
        Rxx, Ryy, Rzz = diag.unbind(-1)
        magnitudes = 0.5 * torch.sqrt(
            torch.abs(1 + torch.stack([Rxx - Ryy - Rzz, -Rxx + Ryy - Rzz, -Rxx - Ryy + Rzz], -1) + eps))
        signs = torch.sign(torch.stack([_R(2, 1) - _R(1, 2), _R(0, 2) - _R(2, 0), _R(1, 0) - _R(0, 1)], -1))
        xyz = signs * magnitudes
        # The relu enforces a non-negative trace
        w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
        Q = torch.cat((xyz, w), -1)
        Q = F.normalize(Q, dim=-1)

        return Q

    def _contacts(self, D_neighbors, mask_neighbors, cutoff=8):
        """ Contacts """
        D_neighbors = D_neighbors.unsqueeze(-1)
        neighbor_C = mask_neighbors * (D_neighbors < cutoff).type(torch.float32)
        return neighbor_C

    def _hbonds(self, X, E_idx, mask_neighbors, eps=1E-3):
        """ Hydrogen bonds and contact map
        """
        X_atoms = dict(zip(['N', 'CA', 'C', 'O'], torch.unbind(X, 2)))

        # Virtual hydrogens
        X_atoms['C_prev'] = F.pad(X_atoms['C'][:, 1:, :], (0, 0, 0, 1), 'constant', 0)
        X_atoms['H'] = X_atoms['N'] + F.normalize(
            F.normalize(X_atoms['N'] - X_atoms['C_prev'], -1) + F.normalize(X_atoms['N'] - X_atoms['CA'], -1), -1)

        def _distance(X_a, X_b):
            return torch.norm(X_a[:, None, :, :] - X_b[:, :, None, :], dim=-1)

        def _inv_distance(X_a, X_b):
            return 1. / (_distance(X_a, X_b) + eps)

        # DSSP vacuum electrostatics model
        U = (0.084 * 332) * (_inv_distance(X_atoms['O'], X_atoms['N']) + _inv_distance(X_atoms['C'], X_atoms['H']) -
                             _inv_distance(X_atoms['O'], X_atoms['H']) - _inv_distance(X_atoms['C'], X_atoms['N']))

        HB = (U < -0.5).type(torch.float32)
        neighbor_HB = mask_neighbors * gather_edges(HB.unsqueeze(-1), E_idx)
        # print(HB)
        # HB = F.sigmoid(U)
        # U_np = U.cpu().data.numpy()
        # # plt.matshow(np.mean(U_np < -0.5, axis=0))
        # plt.matshow(HB[0,:,:])
        # plt.colorbar()
        # plt.show()
        # D_CA = _distance(X_atoms['CA'], X_atoms['CA'])
        # D_CA = D_CA.cpu().data.numpy()
        # plt.matshow(D_CA[0,:,:] < contact_D)
        # # plt.colorbar()
        # plt.show()
        # exit(0)
        return neighbor_HB
    
    def _expand_chain_ends_info(self, info, chain_handle, end_dir):
        if chain_handle == 'mask':
            roll_dir = 1 if end_dir == 'begin' else -1
            mask = info.roll(roll_dir) != 0
            info *= mask
        elif chain_handle == 'replace':
            r_roll_dir = -1 if end_dir == 'begin' else 2
            l_roll_dir = -2 if end_dir == 'begin' else 1
            r_mask = info == info.roll(1)
            l_mask = info == info.roll(-1)
            info[r_mask] = info.roll(r_roll_dir)[r_mask]
            info[l_mask] = info.roll(l_roll_dir)[l_mask]
        return info

    def _orientations_coarse(self, X, E_idx, chain_ends_info, eps=1e-6):
        # Pair features
        # Shifted slices of unit vectors
        dX = X[:, 1:, :] - X[:, :-1, :]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:, :-2, :]
        u_1 = U[:, 1:-1, :]
        u_0 = U[:, 2:, :]
        
        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1, dim=2), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0, dim=2), dim=-1)
        # Bond angle calculation
        cosA = -(u_1 * u_0).sum(-1)
        cosA = torch.clamp(cosA, -1 + eps, 1 - eps)
        A = torch.acos(cosA)
        # Angle between normals
        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)
        # Backbone features
        AD_features = torch.stack((torch.cos(A), torch.sin(A) * torch.cos(D), torch.sin(A) * torch.sin(D)), 2)
        AD_features = F.pad(AD_features, (0, 0, 1, 2), 'constant', 0)

        # Build relative orientations
        o_1 = F.normalize(u_2 - u_1, dim=-1)
        O = torch.stack((o_1, n_2, torch.cross(o_1, n_2, dim=2)), 2)
        O = O.view(list(O.shape[:2]) + [9])
        O = F.pad(O, (0, 0, 1, 2), 'constant', 0)

        # Get chain begin/end index info
        if self.chain_handle:
            chain_begin, chain_end, singles = chain_ends_info['begin'], chain_ends_info['end'], chain_ends_info['singles']
            expand_chain_begin = self._expand_chain_ends_info(chain_begin.clone(), self.chain_handle, 'begin').unsqueeze(-1)
            expand_chain_end = self._expand_chain_ends_info(chain_end.clone(), self.chain_handle, 'end').unsqueeze(-1)
            singles = singles.unsqueeze(-1)
            if self.chain_handle == 'mask':
                O *= expand_chain_begin * singles
                O *= expand_chain_end * singles
            elif self.chain_handle == 'replace':
                O = torch.gather(O, 1, expand_chain_begin.expand(O.shape)) * singles
                O = torch.gather(O, 1, expand_chain_end.expand(O.shape)) * singles

        O_neighbors = gather_nodes(O, E_idx)
        X_neighbors = gather_nodes(X, E_idx)

        # Re-view as rotation matrices
        O = O.view(list(O.shape[:2]) + [3, 3])
        O_neighbors = O_neighbors.view(list(O_neighbors.shape[:3]) + [3, 3])

        # Rotate into local reference frames
        dX = X_neighbors - X.unsqueeze(-2)
        dU = torch.matmul(O.unsqueeze(2), dX.unsqueeze(-1)).squeeze(-1)
        dU = F.normalize(dU, dim=-1)
        R = torch.matmul(O.unsqueeze(2).transpose(-1, -2), O_neighbors)
        Q = self._quaternions(R)

        # Orientation features
        O_features = torch.cat((dU, Q), dim=-1)

        return AD_features, O_features

    def _dihedrals(self, X, chain_ends_info, eps=1e-7):
        # First 3 coordinates are N, CA, C
        X = X[:, :, :3, :].reshape(X.shape[0], 3 * X.shape[1], 3)

        # Shifted slices of unit vectors
        dX = X[:, 1:, :] - X[:, :-1, :]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:, :-2, :]
        u_1 = U[:, 1:-1, :]
        u_0 = U[:, 2:, :]
        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1, dim=2), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0, dim=2), dim=-1)

        # Angle between normals
        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, (1, 2), 'constant', 0)
        D = D.view((D.size(0), int(D.size(1) / 3), 3))

        # Get chain begin/end index info
        if self.chain_handle:
            chain_begin, chain_end, singles = chain_ends_info['begin'], chain_ends_info['end'], chain_ends_info['singles']
            if self.chain_handle == 'mask':
                D[:,:,0] *= chain_begin * singles
                D[:,:,1] *= chain_end * singles
                D[:,:,2] *= chain_end * singles
            elif self.chain_handle == 'replace':
                D[:,:,0] = torch.gather(D[:,:,0], 1, chain_begin) * singles
                D[:,:,1] = torch.gather(D[:,:,1], 1, chain_end) * singles
                D[:,:,2] = torch.gather(D[:,:,2], 1, chain_end) * singles

        # Lift angle representations to the circle
        D_features = torch.cat((torch.cos(D), torch.sin(D)), 2)
        return D_features
    
    def _sc_dists(self, X_sc, X, sc_mask_full, E_idx):
        X_sc_Cb = X_sc[:,:,0]
        X_ca = X[:,:,0,:]
        mask = torch.all(X_sc_Cb == -1, dim=2)
        X_sc_Cb[mask] = X_ca[mask]
        X_sc[:,:,0] = X_sc_Cb
        X_sc_r = X_sc.view(X_sc.size(0), X_sc.size(1), -1)
        X_sc_re = gather_nodes(X_sc_r, E_idx)
        X_sc_e = X_sc_re.view(X_sc_re.size(0), X_sc_re.size(1), X_sc_re.size(2), X_sc.size(2), X_sc.size(3))
        X_sc_e_base = copy.deepcopy(X_sc_e)
        X_sc_c = X_sc.unsqueeze(2).expand(X_sc_e.shape)
        X_sc_mask_c = sc_mask_full.unsqueeze(2).expand(X_sc_e.shape)
        X_sc_dist_opts = []
        for i in range(10):
            X_sc_e = torch.cat((X_sc_e_base[:, :, :, i:], X_sc_e_base[:, :, :, :i]), dim=3)
            X_sc_dist = torch.sqrt(torch.sum((X_sc_e - X_sc_c)**2, dim=-1))
            X_sc_dist[~X_sc_mask_c[:,:,:,:,0]] = 1000
            X_sc_dist, _ = torch.min(X_sc_dist, dim=-1)
            X_sc_dist_opts.append(X_sc_dist.unsqueeze(-1))
        X_sc_dist, _ = torch.min(torch.cat(X_sc_dist_opts, dim=-1), dim=-1)
        X_sc_dist = X_sc_dist.unsqueeze(-1)
        return X_sc_dist
    
    def _sc_angles(self, X_sc, X, sc_mask_full, E_idx):
        X_sc_Cb = X_sc[:,:,0]
        X_ca = X[:,:,1,:]
        d = X[:,:,1,:] - X[:,:,0,:]
        c = X[:,:,2,:] - X[:,:,1,:]
        a = torch.cross(d, c, dim=-1)
        Cb = -0.58273431*a + 0.56802827*d - 0.54067466*c + X[:,:,1,:]
        mask = torch.all(X_sc_Cb == -1, dim=2)
        X_sc_Cb[mask] = Cb[mask]
        X_sc[:,:,0] = X_sc_Cb

        X_ca_e = X_ca.unsqueeze(2).expand(X_sc.shape)
        X_sc_vects = X_sc - X_ca_e
        X_sc_dists = torch.sqrt(torch.sum((X_sc_vects)**2, dim=-1))
        X_sc_dists[sc_mask_full[:,:,:,0]] = 0
        _, X_sc_inds = torch.max(X_sc_dists, dim=-1)
        X_sc_indices = X_sc_inds.unsqueeze(-1).unsqueeze(-1).expand(X_sc_vects.shape)[:,:,0,:].unsqueeze(2)  # b x L x 10 x 3
        X_sc_vects = torch.gather(X_sc_vects, 2, X_sc_indices).squeeze(2)
        X_sc_vects_n = gather_nodes(X_sc_vects, E_idx)
        X_sc_cos_sim = F.cosine_similarity(X_sc_vects_n, X_sc_vects.unsqueeze(2).expand(X_sc_vects_n.shape), dim=-1)
        return X_sc_cos_sim.unsqueeze(-1)

    def compute_gyration_tensor(self, coords, mask):
        """
        Compute the gyration tensor for a set of alpha-carbon coordinates.
        """
        com = torch.sum(coords * mask.unsqueeze(-1), dim=1) / mask.sum(dim=1, keepdim=True)
        centered_coords = coords - com.unsqueeze(1)
        gyration_tensor = torch.einsum('bij,bik->bjk', centered_coords, centered_coords * mask.unsqueeze(-1))
        gyration_tensor /= mask.sum(dim=1, keepdim=True).unsqueeze(-1)
        return gyration_tensor

    def compute_radius_of_gyration(self, coords, mask):
        """Compute the radius of gyration for each protein in the batch."""
        com = torch.sum(coords * mask.unsqueeze(-1), dim=1) / mask.sum(dim=1, keepdim=True)
        squared_distances = torch.sum((coords - com.unsqueeze(1))**2, dim=-1)
        R_g = torch.sqrt(torch.sum(squared_distances * mask, dim=1) / mask.sum(dim=1))
        return R_g

    def compute_sphericity(self, gyration_tensor):
        """Compute the sphericity from the gyration tensor (ratio of smallest to largest eigenvalue)."""
        eigenvalues = torch.linalg.eigvalsh(gyration_tensor)  # Compute eigenvalues (sorted)
        sphericity = eigenvalues[..., 0] / eigenvalues[..., -1]  # Ratio of smallest to largest
        return sphericity

    def calculate_sphericity_and_radius_of_gyration(self, coords, lengths):
        """
        Calculate the sphericity and radius of gyration for a batch of proteins using only alpha-carbons (Cα).
        
        Args:
            coords (torch.Tensor): Tensor of shape (b, L, 4, 3) containing 3D coordinates.
            lengths (list or torch.Tensor): List of actual lengths for each protein in the batch.
        
        Returns:
            torch.Tensor: Sphericity for each protein in the batch.
            torch.Tensor: Radius of gyration for each protein in the batch.
        """
        b, L, _, _ = coords.shape
        
        # Extract only the alpha-carbon (Cα) coordinates at index 1 along the second dimension (L x 4 x 3 -> L x 3)
        coords_ca = coords[:, :, 1, :]  # Shape: (b, L, 3)
        
        # Create a mask for valid alpha-carbons based on lengths
        mask = torch.arange(L).unsqueeze(0).to(coords.device) < lengths.unsqueeze(1).to(coords.device)
        mask = mask.float()  # Shape: (b, L)
        
        # Compute the radius of gyration
        R_g = self.compute_radius_of_gyration(coords_ca, mask)
        
        # Compute the gyration tensor and sphericity
        gyration_tensor = self.compute_gyration_tensor(coords_ca, mask)
        sphericity = self.compute_sphericity(gyration_tensor)
        
        return sphericity, R_g

    def detect_alpha_helix(self, ca_coords, helix_min_length=5):
        """
        Detect alpha-helical residues using Cα coordinates.

        Args:
            ca_coords (torch.Tensor): Tensor of shape (b, L, 3) representing Cα coordinates for b proteins.
            helix_min_length (int): Minimum number of consecutive residues to consider as an alpha helix.

        Returns:
            torch.Tensor: Binary mask of shape (b, L) indicating alpha-helical residues (1 for helix, 0 otherwise).
        """
        b, L, _ = ca_coords.shape
        if L < helix_min_length:
            return torch.zeros((b, L), dtype=torch.int)

        # Calculate distance between consecutive Cα atoms
        distances = torch.norm(ca_coords[:, 1:] - ca_coords[:, :-1], dim=-1)  # (b, L-1)
        
        # Calculate angles between triplets of Cα atoms
        v1 = ca_coords[:, 1:-1] - ca_coords[:, :-2]
        v2 = ca_coords[:, 2:] - ca_coords[:, 1:-1]
        angles = torch.acos(
            torch.clamp(
                torch.sum(v1 * v2, dim=-1) / (torch.norm(v1, dim=-1) * torch.norm(v2, dim=-1) + 1e-8), -1.0, 1.0
            )
        )  # (b, L-2)

        # Identify residues with distances and angles consistent with alpha helices
        helix_mask = (distances[:, :-1] < 4.0) & (angles > 1.2) & (angles < 1.9)  # Helical geometry conditions

        # Create a mask for residues that are part of consecutive stretches of helix_min_length or more
        helix_mask = F.conv1d(
            helix_mask.float().unsqueeze(1), torch.ones((1, 1, helix_min_length), device=helix_mask.device, dtype=helix_mask.float().dtype), padding=helix_min_length // 2
        ) > (helix_min_length - 1)
        helix_mask = helix_mask.squeeze(1).int()  # (b, L)

        return helix_mask

    def compute_ca_dists_and_vects(self, backbone_coords, mask):
        """
        Compute the re-oriented vector from each Cα to the center of mass in the residue's local reference frame.

        Args:
            backbone_coords (torch.Tensor): Tensor of shape (b, L, 4, 3), representing backbone atoms N, Cα, C, and O.

        Returns:
            torch.Tensor: Tensor of shape (b, L) containing distances and orientated vectors
        """
        ca_coords = backbone_coords[:, :, 1, :]  # Extract Cα coordinates (shape: b x L x 3)
        com = ca_coords.mean(dim=1, keepdim=True)  # Center of mass (shape: b x 1 x 3)
        distances = torch.norm(ca_coords - com, dim=-1)
        ca_to_com = com - ca_coords  # Vector from each Cα to the center of mass (shape: b x L x 3)

        # Compute local reference frame
        N_coords = backbone_coords[:, :, 0, :]  # N atom coordinates (shape: b x L x 3)
        C_coords = backbone_coords[:, :, 2, :]  # C atom coordinates (shape: b x L x 3)
        
        v1 = N_coords - ca_coords  # Vector from Cα to N
        v2 = C_coords - ca_coords  # Vector from Cα to C
        
        # Normalize v1 and v2
        n1 = F.normalize(torch.cross(v2, v1, dim=2), dim=-1)
        n2 = F.normalize(v2 - v1, dim=-1)
        n3 = torch.cross(n1, n2, dim=2)
        rot_matr = torch.stack([n1, n2, n3], dim=-1)
        
        rotated_vectors = F.normalize(torch.einsum('blij,blj->bli', rot_matr, ca_to_com), dim=-1)
        
        return distances * mask, rotated_vectors * mask.unsqueeze(-1)

    def forward(self, X, mask):
        """ Featurize coordinates as an attributed graph

        Args
        ----
        X : torch.Tensor
            Backbone coordinates
            Shape: n_batch x seq_len x 4 x 3
        mask : torch.ByteTensor
            Mask for residues
            Shape: n_batch x seq_len

        Returns
        -------
        V : torch.Tensor
            Node embeddings
            Shape: n_batches x seq_len x n_hidden
        E : torch.Tensor
            Edge embeddings in kNN dense form
            Shape: n_batches x seq_len x k x n_hidden
        E_idx : torch.LongTensor
            Edge indices
            Shape: n_batches x seq_len x k x n_hidden
        """
        # Data augmentation
        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)

        # Build k-Nearest Neighbors graph
        X_ca = X[:, :, 1, :]
        D_neighbors, E_idx, mask_neighbors = self._dist(X_ca, mask)

        # Pairwise features
        AD_features, O_features = self._orientations_coarse(X_ca, E_idx)
        RBF = self._rbf(D_neighbors)

        # Pairwise embeddings
        E_positional = self.embeddings(E_idx)

        if self.features_type == 'coarse':
            # Coarse backbone features
            V = AD_features
            E = torch.cat((E_positional, RBF, O_features), -1)
        elif self.features_type == 'hbonds':
            # Hydrogen bonds and contacts
            neighbor_HB = self._hbonds(X, E_idx, mask_neighbors)
            neighbor_C = self._contacts(D_neighbors, mask_neighbors)
            # Dropout
            neighbor_C = self.dropout(neighbor_C)
            neighbor_HB = self.dropout(neighbor_HB)
            # Pack
            V = mask.unsqueeze(-1) * torch.ones_like(AD_features)
            neighbor_C = neighbor_C.expand(-1, -1, -1, int(self.num_positional_embeddings / 2))
            neighbor_HB = neighbor_HB.expand(-1, -1, -1, int(self.num_positional_embeddings / 2))
            E = torch.cat((E_positional, neighbor_C, neighbor_HB), -1)
        elif self.features_type == 'full':
            # Full backbone angles
            V = self._dihedrals(X)
            E = torch.cat((E_positional, RBF, O_features), -1)
        elif self.features_type == 'dist':
            # Full backbone angles
            V = self._dihedrals(X)
            E = torch.cat((E_positional, RBF), -1)

        # Embed the nodes
        V = self.node_embedding(V)
        V = self.norm_nodes(V)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)

        # DEBUG
        # U = (np.nan * torch.zeros(X.size(0),X.size(1),X.size(1),3)).scatter(2, E_idx.unsqueeze(-1).expand(-1,-1,-1,3), E[:,:,:,:3])
        # plt.imshow(U.data.numpy()[0,:,:,0])
        # plt.show()
        # exit(0)
        return V, E, E_idx


class IndexDiffEncoding(nn.Module):
    """ Module to generate differential positional encodings for multichain protein graph edges

    Similar to ProteinFeatures, but zeros out features between interchain interactions """
    def __init__(self, num_embeddings):
        super().__init__()
        self.num_embeddings = num_embeddings

    def forward(self, E_idx, chain_idx):
        """ Generate directional differential positional encodings for edges

        Args
        ----
        E_idx : torch.LongTensor
            Protein kNN edge indices
            Shape: n_batches x seq_len x k
        chain_idx : torch.LongTensor
            Indices for residues such that each chain is assigned a unique integer
            and each residue in that chain is assigned that integer
            Shape: n_batches x seq_len

        Returns
        -------
        E : torch.Tensor
            Directional Diffential positional encodings for edges
            Shape: n_batches x seq_len x k x num_embeddings
        """
        dev = E_idx.device
        # i-j
        N_batch = E_idx.size(0)
        N_terms = E_idx.size(1)
        N_nodes = E_idx.size(2)
        N_neighbors = E_idx.size(3)
        ii = torch.arange(N_nodes, dtype=torch.float32).view((1, -1, 1)).to(dev)
        d = (E_idx.float() - ii).unsqueeze(-1)

        # Original Transformer frequencies
        frequency = torch.exp(
            torch.arange(0, self.num_embeddings, 2, dtype=torch.float32) *
            -(np.log(10000.0) / self.num_embeddings)).to(dev)
        # Grid-aligned
        # frequency = 2. * np.pi * torch.exp(
        #     -torch.linspace(
        #         np.log(self.period_range[0]),
        #         np.log(self.period_range[1]),
        #         self.num_embeddings / 2
        #     )
        # )
        angles = d * frequency.view((1, 1, 1, -1))
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)

        # we zero out positional frequencies from inter-chain edges
        # the idea is, the concept of "sequence distance"
        # between two residues in different chains doesn't
        # make sense :P
        chain_idx_expand = chain_idx.view(N_batch, 1, -1, 1).expand((-1, N_terms, -1, N_neighbors))
        E_chain_idx = torch.gather(chain_idx_expand.to(dev), 2, E_idx)
        same_chain = (E_chain_idx == E_chain_idx[:, :, :, 0:1]).to(dev)

        E *= same_chain.unsqueeze(-1)
        return E


class MultiChainProteinFeatures(ProteinFeatures):
    """ Protein backbone featurization which accounts for differences
    between inter-chain and intra-chain interactions.

    Attributes
    ----------
    embeddings : IndexDiffEncoding
        Module to generate differential positional embeddings for edges
    dropout : nn.Dropout
        Dropout module
    node_embeddings, edge_embeddings : nn.Linear
        Embedding layers for nodes and edges
    norm_nodes, norm_edges : nn.LayerNorm
        Normalization layers for node and edge features
    """
    def __init__(self,
                 edge_features,
                 node_features,
                 num_positional_embeddings=16,
                 num_rbf=16,
                 top_k=30,
                 features_type='full',
                 augment_eps=0.,
                 dropout=0.1,
                 esm_dropout=0.0,
                 esm_rep_feat_ins=[640],
                 esm_rep_feat_outs=[32],
                 esm_attn_feat_ins=[600, 100],
                 esm_attn_feat_outs=[100, 20],
                 raw=False,
                 old=False,
                 E_feats=None,
                 bias=True,
                 random_type='',
                 random_alpha=3.0,
                 deterministic=False,
                 deterministic_seed=10,
                 random_temperature=1.0,
                 chain_handle='',
                 center_node=False,
                 center_node_ablation=False,
                 random_graph=False):
        """ Extract protein features """
        super().__init__(edge_features,
                         node_features,
                         num_positional_embeddings=num_positional_embeddings,
                         num_rbf=num_rbf,
                         top_k=top_k,
                         features_type=features_type,
                         augment_eps=augment_eps,
                         dropout=dropout,
                         esm_dropout=esm_dropout,
                         esm_rep_feat_ins=esm_rep_feat_ins,
                         esm_rep_feat_outs=esm_rep_feat_outs,
                         esm_attn_feat_ins=esm_attn_feat_ins,
                         esm_attn_feat_outs=esm_attn_feat_outs,
                         raw=raw,
                         old=old,
                         bias=bias,
                         random_type=random_type,
                         random_alpha=random_alpha,
                         deterministic = deterministic,
                         deterministic_seed = deterministic_seed,
                         random_temperature = random_temperature,
                         chain_handle=chain_handle,
                         center_node=center_node,
                         center_node_ablation=center_node_ablation,
                         random_graph=random_graph)

        # so uh this is designed to work on the batched TERMS
        # but if we just treat the whole sequence as one big TERM
        # the math is the same so i'm not gonna code a new module lol
        self.embeddings = IndexDiffEncoding(num_positional_embeddings)
        self.E_feats = E_feats

    # pylint: disable=arguments-differ
    def forward(self, X, chain_idx, mask, seq_lens, X_sc, sc_ids, sc_chi, x_mask_sc, sc_mask_full, esm_embs, esm_attns, chain_ends_info, nonlinear=False):
        """ Featurize coordinates as an attributed graph

        Args
        ----
        X : torch.Tensor
            Backbone coordinates
            Shape: n_batch x seq_len x 4 x 3
        chain_idx : torch.LongTensor
            Indices for residues such that each chain is assigned a unique integer
            and each residue in that chain is assigned that integer
            Shape: n_batches x seq_len
        mask : torch.ByteTensor
            Mask for residues
            Shape: n_batch x seq_len

        Returns
        -------
        V : torch.Tensor
            Node embeddings
            Shape: n_batches x seq_len x n_hidden
        E : torch.Tensor
            Edge embeddings in kNN dense form
            Shape: n_batches x seq_len x k x n_hidden
        E_idx : torch.LongTensor
            Edge indices
            Shape: n_batches x seq_len x k x n_hidden
        """
        # Data augmentation
        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)

        # Build k-Nearest Neighbors graph
        X_ca = X[:, :, 1, :]
        D_neighbors, E_idx, mask_neighbors = self._dist(X_ca, mask)
        # Pairwise features
        AD_features, O_features = self._orientations_coarse(X_ca, E_idx, chain_ends_info)
        RBF = self._rbf(D_neighbors)
        # Pairwise embeddings
        # we unsqueeze to generate "1 TERM" per sequence,
        # then squeeze it back to get rid of it
        E_positional = self.embeddings(E_idx.unsqueeze(1), chain_idx).squeeze(1)
        if self.features_type == 'side_chain_esm' or 'attns' in self.features_type or self.features_type == 'full_mpnn_esm_module':
            esm_attns = self.esm_attns_embedding(esm_attns)
        if 'reps' in self.features_type or self.features_type == 'full_mpnn_esm_module':
            esm_embs = self.esm_reps_embedding(esm_embs)
        if self.features_type == 'coarse':
            # Coarse backbone features
            V = AD_features
            E = torch.cat((E_positional, RBF, O_features), -1)
        elif self.features_type == 'hbonds':
            # Hydrogen bonds and contacts
            neighbor_HB = self._hbonds(X, E_idx, mask_neighbors)
            neighbor_C = self._contacts(D_neighbors, mask_neighbors)
            # Dropout
            neighbor_C = self.dropout(neighbor_C)
            neighbor_HB = self.dropout(neighbor_HB)
            # Pack
            V = mask.unsqueeze(-1) * torch.ones_like(AD_features)
            neighbor_C = neighbor_C.expand(-1, -1, -1, int(self.num_positional_embeddings / 2))
            neighbor_HB = neighbor_HB.expand(-1, -1, -1, int(self.num_positional_embeddings / 2))
            E = torch.cat((E_positional, neighbor_C, neighbor_HB), -1)
        elif self.features_type == 'full':
            # Full backbone angles
            if self.old:
                V = self._dihedrals(X, chain_ends_info)
                E = torch.cat((E_positional, RBF, O_features), -1)
            else:
                V = self._dihedrals(X, chain_ends_info)
                V = self.node_layers[0](V)
                E = torch.cat((E_positional, RBF, O_features), -1)
                E = self.edge_layers[0](E)
        elif self.features_type == 'dist':
            # Full backbone angles
            V = self._dihedrals(X, chain_ends_info)
            E = torch.cat((E_positional, RBF), -1)
        elif self.features_type == 'side_chain':
            V = self._dihedrals(X, chain_ends_info)
            sc_dist = X_sc - X[:,:,1,:].unsqueeze(2).expand(X_sc.shape)
            sc_dist = torch.sqrt(torch.sum(sc_dist**2, dim=-1))
            sc_dist *= sc_mask_full[:,:,:,0]
            sc_ids = (1 + sc_ids) * sc_mask_full[:,:,:,0]
            V = [V, sc_dist, sc_ids]
            for i, v in enumerate(V):
                V[i] = self.node_layers[i](v)
            V = torch.cat((V[0], V[1], V[2]), dim=-1)
            sc_min_dists = self._sc_dists(X_sc, X, sc_mask_full, E_idx)
            X_sc_mask_e = gather_nodes(x_mask_sc.unsqueeze(-1), E_idx).squeeze(-1).to(dtype=torch.bool)
            sc_min_dists[~X_sc_mask_e] = 1000
            E = torch.cat((E_positional, RBF, O_features, sc_min_dists), -1)
            E = self.edge_layers[0](E)
        elif self.features_type == 'side_chain_ids':
            V = self._dihedrals(X, chain_ends_info)
            sc_ids = (1 + sc_ids) * sc_mask_full[:,:,:,0]
            V = [V, sc_ids]
            for i, v in enumerate(V):
                V[i] = self.node_layers[i](v)
            V = torch.cat((V[0], V[1]), dim=-1)
            E = torch.cat((E_positional, RBF, O_features), -1)
            E = self.edge_layers[0](E)
        elif self.features_type == 'side_chain_angs':
            V = self._dihedrals(X, chain_ends_info)
            sc_dist = X_sc - X[:,:,1,:].unsqueeze(2).expand(X_sc.shape)
            sc_dist = torch.sqrt(torch.sum(sc_dist**2, dim=-1))
            sc_dist *= sc_mask_full[:,:,:,0]
            sc_ids = (1 + sc_ids) * sc_mask_full[:,:,:,0]
            V = [V, sc_dist, sc_ids, sc_chi]
            for i, v in enumerate(V):
                V[i] = self.node_layers[i](v)
            V = torch.cat((V[0], V[1], V[2], V[3]), dim=-1)
            sc_min_dists = self._sc_dists(X_sc, X, sc_mask_full, E_idx)
            sc_angs = self._sc_angles(X_sc, X, sc_mask_full, E_idx)
            X_sc_mask_e = gather_nodes(x_mask_sc.unsqueeze(-1), E_idx).squeeze(-1).to(dtype=torch.bool)
            sc_min_dists[~X_sc_mask_e] = 1000
            E = torch.cat((E_positional, RBF, O_features, sc_min_dists, sc_angs), -1)
            E = self.edge_layers[0](E)
        elif self.features_type == 'side_chain_orient':
            V = self._dihedrals(X, chain_ends_info)
            sc_ids = (1 + sc_ids) * sc_mask_full[:,:,:,0]
            V = [V, sc_ids, sc_chi]
            for i, v in enumerate(V):
                V[i] = self.node_layers[i](v)
            V = torch.cat((V[0], V[1], V[2]), dim=-1)
            sc_angs = self._sc_angles(X_sc, X, sc_mask_full, E_idx)
            X_sc_mask_e = gather_nodes(x_mask_sc.unsqueeze(-1), E_idx).squeeze(-1).to(dtype=torch.bool)
            E = torch.cat((E_positional, RBF, O_features, sc_angs), -1)
            E = self.edge_layers[0](E)
        elif self.features_type == 'side_chain_dihedral':
            V = self._dihedrals(X, chain_ends_info)
            V = [V, sc_chi]
            for i, v in enumerate(V):
                V[i] = self.node_layers[i](v)
            V = torch.cat((V[0], V[1]), dim=-1)
            E = torch.cat((E_positional, RBF, O_features), -1)
            E = self.edge_layers[0](E)
        elif self.features_type == 'side_chain_esm':
            V = self._dihedrals(X, chain_ends_info)
            V = [V, esm_embs]
            for i, v in enumerate(V):
                V[i] = self.node_layers[i](v)
            V = torch.cat((V[0], V[1]), dim=-1)
            E = torch.cat((E_positional, RBF, O_features), -1)
            E = [E, esm_attns]
            for i, e in enumerate(E):
                E[i] = self.edge_layers[i](e)
            E = torch.cat((E[0], E[1]), -1)
        elif self.features_type == 'side_chain_esm_embs':
            V = self._dihedrals(X, chain_ends_info)
            V = [V, esm_embs]
            for i, v in enumerate(V):
                V[i] = self.node_layers[i](v)
            V = torch.cat((V[0], V[1]), dim=-1)
            E = torch.cat((E_positional, RBF, O_features), -1)
            E = self.edge_layers[0](E)
        elif self.features_type == 'side_chain_esm_attns':
            V = self._dihedrals(X, chain_ends_info)
            V = self.node_layers[0](V)
            E = torch.cat((E_positional, RBF, O_features), -1)
            E = [E, esm_attns]
            for i, e in enumerate(E):
                E[i] = self.edge_layers[i](e)
            E = torch.cat((E[0], E[1]), -1)
        elif self.features_type == 'side_chain_esm_reps':
            V = self._dihedrals(X, chain_ends_info)
            V = [V, esm_embs]
            for i, v in enumerate(V):
                V[i] = self.node_layers[i](v)
            V = torch.cat((V[0], V[1]), dim=-1)
            E = torch.cat((E_positional, RBF, O_features), -1)
            E = self.edge_layers[0](E)
        elif self.features_type == 'side_chain_esm_reps_attns':
            V = self._dihedrals(X, chain_ends_info)
            V = [V, esm_embs]
            for i, v in enumerate(V):
                V[i] = self.node_layers[i](v)
            V = torch.cat((V[0], V[1]), dim=-1)
            E = torch.cat((E_positional, RBF, O_features), -1)
            E = [E, esm_attns]
            for i, e in enumerate(E):
                E[i] = self.edge_layers[i](e)
            E = torch.cat((E[0], E[1]), -1)
        elif self.features_type == 'side_chain_esm_reps_mpnn':
            V = self._dihedrals(X, chain_ends_info)
            V = [V, esm_embs]
            for i, v in enumerate(V):
                V[i] = self.node_layers[i](v)
            V = torch.cat((V[0], V[1]), dim=-1)
            E = self.E_feats(X, chain_idx, mask, E_idx, D_neighbors)
            E = self.edge_layers[0](E)
        elif self.features_type == 'side_chain_esm_reps_mpnn_post':
            V = self._dihedrals(X, chain_ends_info)
            V = self.node_layers[0](V)
            E = self.E_feats(X, chain_idx, mask, E_idx, D_neighbors)
            E = self.edge_layers[0](E)  
        elif self.features_type == 'side_chain_esm_reps_mpnn_attns':
            V = self._dihedrals(X, chain_ends_info)
            V = [V, esm_embs]
            for i, v in enumerate(V):
                V[i] = self.node_layers[i](v)
            V = torch.cat((V[0], V[1]), dim=-1)
            E = self.E_feats(X, chain_idx, mask, E_idx, D_neighbors)
            E = [E, esm_attns]
            for i, e in enumerate(E):
                E[i] = self.edge_layers[i](e)
            E = torch.cat((E[0], E[1]), -1)
        elif self.features_type == 'full_mpnn' or self.features_type == 'full_mpnn_esm_module':
            V = self._dihedrals(X, chain_ends_info)
            V = self.node_layers[0](V)
            E = self.E_feats(X, chain_idx, mask, E_idx, D_neighbors)
            E = self.edge_layers[0](E)  
            if self.center_node:
                sphericity, radius_of_gyration = self.calculate_sphericity_and_radius_of_gyration(X, seq_lens)
                helix_mask = self.detect_alpha_helix(X_ca)
                helix_percentage = torch.sum(helix_mask, dim=-1) / seq_lens
                center_V = torch.stack([sphericity, radius_of_gyration, helix_percentage], dim=-1)
                center_V = self.center_node_embedding(center_V)

                center_V = torch.randn_like(center_V) ## JFM TEST

                center_dists, center_vecs = self.compute_ca_dists_and_vects(X, mask)
                center_RBF = self._rbf(center_dists, D_max=50, num_dims=3, D_count=32)
                center_E = torch.cat([center_RBF, center_vecs], dim=-1) * mask.unsqueeze(-1)
                center_E = self.center_edge_embedding(center_E)

                center_E = torch.randn_like(center_E) ## JFM_TEST
                

        # Embed the nodes
        if self.raw:
            return V, E, E_idx
        V = self.node_embedding(V)
        if nonlinear:
            V = gelu(V)
        V = self.norm_nodes(V)
        E = self.edge_embedding(E)
        if nonlinear:
            E = gelu(E)
        E = self.norm_edges(E)

        if self.center_node:
            if nonlinear:
                center_V = gelu(center_V)
                center_E = gelu(center_E)
            center_V = self.center_node_norm(center_V)
            center_E = self.center_edge_norm(center_E)
            ## TESTING CENTER NODE ABLATION 
            if self.center_node_ablation:
                center_V = torch.zeros_like(center_V)
                center_E = torch.zeros_like(center_E)
            ## END TESTING
            V = torch.cat([V, center_V.unsqueeze(1)], dim=1)
            # V = insert_vectors_3D(V, center_V, seq_lens)
            E = torch.cat([E, center_E.unsqueeze(2)], dim=2)
            E = torch.cat([E, torch.ones((E.shape[0], 1, E.shape[2], E.shape[3]), dtype=E.dtype, device=E.device)], dim=1)
            # E = insert_vectors_4D(E, torch.ones((E.shape[0], E.shape[2], E.shape[3]), dtype=E.dtype, device=E.device), seq_lens)
            E_idx = torch.cat([E_idx, seq_lens.unsqueeze(-1).unsqueeze(-1).expand(E_idx.shape[0], E_idx.shape[1], 1).clone()], dim=2)
            E_idx = torch.cat([E_idx, seq_lens.unsqueeze(-1).unsqueeze(-1).expand((E_idx.shape[0], 1, E_idx.shape[2]))], dim=1)
            # E_idx = insert_vectors_3D(E_idx, seq_lens.unsqueeze(1).expand((E_idx.shape[0], E_idx.shape[2])), seq_lens)

        # DEBUG
        # U = (np.nan * torch.zeros(X.size(0),X.size(1),X.size(1),3)).scatter(2, E_idx.unsqueeze(-1).expand(-1,-1,-1,3), E[:,:,:,:3])
        # plt.imshow(U.data.numpy()[0,:,:,0])
        # plt.show()
        # exit(0)
        if 'post' in self.features_type or 'module' in self.features_type:
            return V, E, E_idx, esm_embs, esm_attns
        return V, E, E_idx
    
class reSSeqMPNNFeatures(ProteinFeatures):
    """ Protein backbone featurization which accounts for differences
    between inter-chain and intra-chain interactions.

    Attributes
    ----------
    embeddings : IndexDiffEncoding
        Module to generate differential positional embeddings for edges
    dropout : nn.Dropout
        Dropout module
    node_embeddings, edge_embeddings : nn.Linear
        Embedding layers for nodes and edges
    norm_nodes, norm_edges : nn.LayerNorm
        Normalization layers for node and edge features
    """
    def __init__(self,
                 edge_features,
                 node_features,
                 num_positional_embeddings=16,
                 num_rbf=16,
                 top_k=30,
                 features_type='full',
                 sc_features_type='side_chain_esm_reps_attns',
                 augment_eps=0.,
                 dropout=0.1,
                 esm_rep_feat_ins=[640],
                 esm_rep_feat_outs=[32],
                 esm_attn_feat_ins=[600, 100],
                 esm_attn_feat_outs=[100, 20],
                 raw=False,
                 old=False):
        """ Extract protein features """
        super().__init__(edge_features,
                         node_features,
                         num_positional_embeddings=num_positional_embeddings,
                         num_rbf=num_rbf,
                         top_k=top_k,
                         features_type=features_type,
                         augment_eps=augment_eps,
                         dropout=dropout,
                         esm_rep_feat_ins=esm_rep_feat_ins,
                         esm_rep_feat_outs=esm_rep_feat_outs,
                         esm_attn_feat_ins=esm_attn_feat_ins,
                         esm_attn_feat_outs=esm_attn_feat_outs,
                         raw=raw,
                         old=True)

        # so uh this is designed to work on the batched TERMS
        # but if we just treat the whole sequence as one big TERM
        # the math is the same so i'm not gonna code a new module lol
        
        if 'esm' in sc_features_type:
            self.esm_attns_embedding = MultiLayerLinear(in_features=esm_attn_feat_ins, out_features=esm_attn_feat_outs, num_layers=len(esm_attn_feat_outs), activation_layers='gelu', dropout=0).float()
        if 'reps' in sc_features_type:
            self.esm_reps_embedding = MultiLayerLinear(in_features=esm_rep_feat_ins, out_features=esm_rep_feat_outs, num_layers=len(esm_rep_feat_outs), activation_layers='gelu', dropout=0).float()

        self.sc_features_type=sc_features_type
        self.embeddings = IndexDiffEncoding(num_positional_embeddings)
        self.sc_node_embedding = nn.Linear(esm_rep_feat_outs[-1], node_features)
        self.sc_edge_embedding = nn.Linear(esm_attn_feat_outs[-1], edge_features)
        self.sc_norm_nodes = nn.LayerNorm(node_features) 
        self.sc_norm_edges = nn.LayerNorm(edge_features)
        self.features_type = features_type

    # pylint: disable=arguments-differ
    def forward(self, X, chain_idx, mask, X_sc, sc_ids, sc_chi, x_mask_sc, sc_mask_full, esm_embs, esm_attns, nonlinear=False):
        """ Featurize coordinates as an attributed graph

        Args
        ----
        X : torch.Tensor
            Backbone coordinates
            Shape: n_batch x seq_len x 4 x 3
        chain_idx : torch.LongTensor
            Indices for residues such that each chain is assigned a unique integer
            and each residue in that chain is assigned that integer
            Shape: n_batches x seq_len
        mask : torch.ByteTensor
            Mask for residues
            Shape: n_batch x seq_len

        Returns
        -------
        V : torch.Tensor
            Node embeddings
            Shape: n_batches x seq_len x n_hidden
        E : torch.Tensor
            Edge embeddings in kNN dense form
            Shape: n_batches x seq_len x k x n_hidden
        E_idx : torch.LongTensor
            Edge indices
            Shape: n_batches x seq_len x k x n_hidden
        """

        # Data augmentation
        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)

        # Build k-Nearest Neighbors graph
        X_ca = X[:, :, 1, :]
        D_neighbors, E_idx, mask_neighbors = self._dist(X_ca, mask)

        # Pairwise features
        AD_features, O_features = self._orientations_coarse(X_ca, E_idx)
        RBF = self._rbf(D_neighbors)

        # Pairwise embeddings
        # we unsqueeze to generate "1 TERM" per sequence,
        # then squeeze it back to get rid of it
        E_positional = self.embeddings(E_idx.unsqueeze(1), chain_idx).squeeze(1)

        if self.sc_features_type == 'side_chain_esm' or 'attns' in self.sc_features_type:
            esm_attns = self.esm_attns_embedding(esm_attns)
        if 'reps' in self.sc_features_type:
            esm_embs = self.esm_reps_embedding(esm_embs)

        if self.features_type == 'coarse':
            # Coarse backbone features
            V = AD_features
            E = torch.cat((E_positional, RBF, O_features), -1)
        elif self.features_type == 'hbonds':
            # Hydrogen bonds and contacts
            neighbor_HB = self._hbonds(X, E_idx, mask_neighbors)
            neighbor_C = self._contacts(D_neighbors, mask_neighbors)
            # Dropout
            neighbor_C = self.dropout(neighbor_C)
            neighbor_HB = self.dropout(neighbor_HB)
            # Pack
            V = mask.unsqueeze(-1) * torch.ones_like(AD_features)
            neighbor_C = neighbor_C.expand(-1, -1, -1, int(self.num_positional_embeddings / 2))
            neighbor_HB = neighbor_HB.expand(-1, -1, -1, int(self.num_positional_embeddings / 2))
            E = torch.cat((E_positional, neighbor_C, neighbor_HB), -1)
        elif self.features_type == 'full':
            # Full backbone angles
            V = self._dihedrals(X)
            E = torch.cat((E_positional, RBF, O_features), -1)
        if self.sc_features_type == 'side_chain_esm_reps_attns':
            SV = self.sc_node_embedding(esm_embs)
            SE = self.sc_edge_embedding(esm_attns)

        # Embed the nodes
        V = self.node_embedding(V)
        if nonlinear:
            V = gelu(V)
        V = self.norm_nodes(V)
        E = self.edge_embedding(E)
        if nonlinear:
            E = gelu(E)
        E = self.norm_edges(E)

        SV = self.sc_norm_nodes(gelu(SV))
        SE = self.sc_norm_edges(gelu(SE))

        # DEBUG
        # U = (np.nan * torch.zeros(X.size(0),X.size(1),X.size(1),3)).scatter(2, E_idx.unsqueeze(-1).expand(-1,-1,-1,3), E[:,:,:,:3])
        # plt.imshow(U.data.numpy()[0,:,:,0])
        # plt.show()
        # exit(0)
        return V, E, SV, SE, E_idx

class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2*max_relative_feature+1+1, num_embeddings)

    def forward(self, offset, mask):
        d = torch.clip(offset + self.max_relative_feature, 0, 2*self.max_relative_feature)*mask + (1-mask)*(2*self.max_relative_feature+1)
        d_onehot = torch.nn.functional.one_hot(d, 2*self.max_relative_feature+1+1)
        E = self.linear(d_onehot.float())
        return E

class ProteinMPNNFeatures(nn.Module):
    def __init__(self, edge_features, node_features, num_positional_embeddings=16,
        num_rbf=16, top_k=30, augment_eps=0., features_type='include_nodes', dropout=0, esm_rep_feat_ins=[640], esm_rep_feat_outs=[32], esm_attn_feat_ins=[600,100], esm_attn_feat_outs=[100,20], only_E = True, random_type='', random_alpha=3.0, deterministic=False, deterministic_seed=10, random_temperature=1.0):
        """ Extract protein features """
        super(ProteinMPNNFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps 
        self.features_type = features_type
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        node_in, edge_in = 6, num_positional_embeddings + num_rbf*25
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)

        self.node_embedding = nn.Linear(node_in, node_features, bias=True)
        self.norm_nodes = nn.LayerNorm(node_features)  # Normalize(node_features)
        self.only_E = only_E
        self.random_type = random_type
        self.random_alpha = random_alpha
        self.deterministic = deterministic
        self.deterministic_seed = deterministic_seed
        self.random_temperature = random_temperature

    def _perturb_distances(self, D, criterion):
        # Replace distance by log-propensity
        # Adapted from https://github.com/generatebio/chroma
        if criterion == "random_log":
            logp_edge = -3 * torch.log(D)
        elif criterion == "random_linear":
            logp_edge = -D / self.random_alpha
        elif criterion == "random_uniform":
            logp_edge = D * 0
        else:
            return D
        
        if not self.deterministic:
            Z = torch.rand_like(D)
        else:
            with torch.random.fork_rng():
                torch.random.manual_seed(self.deterministic_seed)
                Z_shape = [1] + list(D.shape)[1:]
                Z = torch.rand(Z_shape, device=D.device)

        # Sample Gumbel noise
        G = -torch.log(-torch.log(Z))

        # Negate because are doing argmin instead of argmax
        D_key = -(logp_edge / self.random_temperature + G)

        return D_key

    def _dist(self, X, mask, eps=1E-6):
        # Convolutional network on NCHW
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)

        # Identify k nearest neighbors (including self)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        if self.random_type == '':
            top_k = self.top_k // 2
        top_k = min(top_k, D.shape[1])
        D_neighbors, E_idx = torch.topk(D_adjust, top_k, dim=-1, largest=False)
        mask_neighbors = gather_edges(mask_2D.unsqueeze(-1), E_idx)

        if self.random_type != '':
            mask_remaining = (1.0 - mask_neighbors).squeeze(-1).to(mask_neighbors.dtype)
            mask_2D_remaining = torch.ones_like(mask_2D).scatter(2, E_idx, mask_remaining)
            D *= mask_2D_remaining
            mask_2D *= mask_2D_remaining
            D = self._perturb_distances(D, self.random_type)
            D_max, _ = torch.max(D, -1, keepdim=True)
            D_adjust = D + (1. - mask_2D) * D_max
            D_neighbors_rand, E_idx_rand = torch.topk(D_adjust, top_k, dim=-1, largest=False)
            D_neighbors = torch.cat([D_neighbors, D_neighbors_rand], 2)
            E_idx = torch.cat([E_idx, E_idx_rand], 2)

        return D_neighbors, E_idx

    def _rbf(self, D, D_max=22.):
        device = D.device
        D_min, D_count = 2., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1,1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(torch.sum((A[:,:,None,:] - B[:,None,:,:])**2,-1) + 1e-6) #[B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:,:,:,None], E_idx)[:,:,:,0] #[B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def _dihedrals(self, X, eps=1e-7):
        # First 3 coordinates are N, CA, C
        X = X[:, :, :3, :].reshape(X.shape[0], 3 * X.shape[1], 3)

        # Shifted slices of unit vectors
        dX = X[:, 1:, :] - X[:, :-1, :]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:, :-2, :]
        u_1 = U[:, 1:-1, :]
        u_0 = U[:, 2:, :]
        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, (1, 2), 'constant', 0)
        D = D.view((D.size(0), int(D.size(1) / 3), 3))

        # Lift angle representations to the circle
        D_features = torch.cat((torch.cos(D), torch.sin(D)), 2)
        return D_features

    def forward(self, X, chain_idx, mask, E_idx, D_neighbors):     
        B = X.shape[0]
        L_max = X.shape[1]                  
        residue_idx = -100*np.ones([B, L_max], dtype=np.int32) #residue idx with jumps across chains
        for i_batch in range(B):
            c = 1
            l0 = 0
            l1 = 0
            _, chain_lens = torch.unique_consecutive(chain_idx[i_batch], return_counts=True)
            for chain_len in chain_lens[:-1]:
                l1 += chain_len.cpu().numpy()
                residue_idx[i_batch, l0:l1] = 100*(c-1)+np.arange(l0, l1)
                l0 += chain_len.cpu().numpy()
                c+=1
        residue_idx = torch.from_numpy(residue_idx).to(dtype=X.dtype, device=X.device)
        b = X[:,:,1,:] - X[:,:,0,:]
        c = X[:,:,2,:] - X[:,:,1,:]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + X[:,:,1,:]
        Ca = X[:,:,1,:]
        N = X[:,:,0,:]
        C = X[:,:,2,:]
        O = X[:,:,3,:]
 
        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors)) #Ca-Ca
        RBF_all.append(self._get_rbf(N, N, E_idx)) #N-N
        RBF_all.append(self._get_rbf(C, C, E_idx)) #C-C
        RBF_all.append(self._get_rbf(O, O, E_idx)) #O-O
        RBF_all.append(self._get_rbf(Cb, Cb, E_idx)) #Cb-Cb
        RBF_all.append(self._get_rbf(Ca, N, E_idx)) #Ca-N
        RBF_all.append(self._get_rbf(Ca, C, E_idx)) #Ca-C
        RBF_all.append(self._get_rbf(Ca, O, E_idx)) #Ca-O
        RBF_all.append(self._get_rbf(Ca, Cb, E_idx)) #Ca-Cb
        RBF_all.append(self._get_rbf(N, C, E_idx)) #N-C
        RBF_all.append(self._get_rbf(N, O, E_idx)) #N-O
        RBF_all.append(self._get_rbf(N, Cb, E_idx)) #N-Cb
        RBF_all.append(self._get_rbf(Cb, C, E_idx)) #Cb-C
        RBF_all.append(self._get_rbf(Cb, O, E_idx)) #Cb-O
        RBF_all.append(self._get_rbf(O, C, E_idx)) #O-C
        RBF_all.append(self._get_rbf(N, Ca, E_idx)) #N-Ca
        RBF_all.append(self._get_rbf(C, Ca, E_idx)) #C-Ca
        RBF_all.append(self._get_rbf(O, Ca, E_idx)) #O-Ca
        RBF_all.append(self._get_rbf(Cb, Ca, E_idx)) #Cb-Ca
        RBF_all.append(self._get_rbf(C, N, E_idx)) #C-N
        RBF_all.append(self._get_rbf(O, N, E_idx)) #O-N
        RBF_all.append(self._get_rbf(Cb, N, E_idx)) #Cb-N
        RBF_all.append(self._get_rbf(C, Cb, E_idx)) #C-Cb
        RBF_all.append(self._get_rbf(O, Cb, E_idx)) #O-Cb
        RBF_all.append(self._get_rbf(C, O, E_idx)) #C-O
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        offset = residue_idx[:,:,None]-residue_idx[:,None,:]
        offset = gather_edges(offset[:,:,:,None], E_idx)[:,:,:,0] #[B, L, K]

        d_chains = ((chain_idx[:, :, None] - chain_idx[:,None,:])==0).long() #find self vs non-self interaction
        E_chains = gather_edges(d_chains[:,:,:,None], E_idx)[:,:,:,0]
        E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all), -1)
        if self.only_E:
            return E
        E = self.edge_embedding(E)
        E = self.norm_edges(E)

        if self.features_type.find('include_nodes') > -1:
            V = self._dihedrals(X, eps=1e-7)
            V = self.node_embedding(V)
            V = self.norm_nodes(V)
        elif self.features_type.find('blank_nodes') > -1:
            V = torch.zeros(X.shape[0], X.shape[1], self.node_features).to(device=X.device, dtype=X.dtype)
        return V, E, E_idx