"""Datasets and dataloaders for loading pdbs.

This file is based on code from ProteinMPNN
https://github.com/dauparas/ProteinMPNN
"""
import torch
from torch.utils.data import DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence
import csv
from dateutil import parser
from Bio.PDB import PDBParser
import numpy as np
import time
import random
import os
import copy
from terminator.data.noise import generate_noise
from terminator.data.data import extract_knn, _apply_seq_mask, _esm_featurize, find_interface
from terminator.utils.common import esm_convert, ints_to_seq_torch, esm_ints_to_seq_torch, seq_to_ints
from terminator.models.layers.utils import gather_edges
import terminator.data.data_utils as data_utils

AA_ATOM_IDS = {'M': np.array([ 0.,  0.,  3.,  0., -1., -1., -1., -1., -1., -1.]),
 'S': np.array([ 0.,  2., -1., -1., -1., -1., -1., -1., -1., -1.]),
 'I': np.array([ 0.,  0.,  0.,  0., -1., -1., -1., -1., -1., -1.]),
 'A': np.array([ 0., -1., -1., -1., -1., -1., -1., -1., -1., -1.]),
 'N': np.array([ 0.,  0.,  1.,  2., -1., -1., -1., -1., -1., -1.]),
 'E': np.array([ 0.,  0.,  0.,  2.,  2., -1., -1., -1., -1., -1.]),
 'G': np.array([-1., -1., -1., -1., -1., -1., -1., -1., -1., -1.]),
 'F': np.array([ 0.,  0.,  0.,  0.,  0.,  0.,  0., -1., -1., -1.]),
 'L': np.array([ 0.,  0.,  0.,  0., -1., -1., -1., -1., -1., -1.]),
 'D': np.array([ 0.,  0.,  2.,  2., -1., -1., -1., -1., -1., -1.]),
 'K': np.array([ 0.,  0.,  0.,  0.,  1., -1., -1., -1., -1., -1.]),
 'R': np.array([ 0.,  0.,  0.,  1.,  0.,  1.,  1., -1., -1., -1.]),
 'P': np.array([ 0.,  0.,  0., -1., -1., -1., -1., -1., -1., -1.]),
 'T': np.array([ 0.,  0.,  2., -1., -1., -1., -1., -1., -1., -1.]),
 'C': np.array([ 0.,  3., -1., -1., -1., -1., -1., -1., -1., -1.]),
 'V': np.array([ 0.,  0.,  0., -1., -1., -1., -1., -1., -1., -1.]),
 'H': np.array([ 0.,  0.,  0.,  1.,  0.,  1., -1., -1., -1., -1.]),
 'Y': np.array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  2., -1., -1.]),
 'Q': np.array([ 0.,  0.,  0.,  1.,  2., -1., -1., -1., -1., -1.]),
 'W': np.array([0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]),
 'X': np.array([-1., -1., -1., -1., -1., -1., -1., -1., -1., -1.]),
 '-': np.array([-1., -1., -1., -1., -1., -1., -1., -1., -1., -1.])}

def dihedral_angles(points_matrix):
    """
    Calculate dihedral angles for consecutive series of 4 points in each list of points.
    
    Parameters:
    - points_matrix: An Lx14x3 matrix representing L lists of 14 points in 3D space.
    
    Returns:
    - An Lx11 matrix containing dihedral angles for each consecutive series of 4 points.
    """
    L, _, _ = points_matrix.shape
    
    
    p0 = points_matrix[:, :11]
    p1 = points_matrix[:, 1:12]
    p2 = points_matrix[:, 2:13]
    p3 = points_matrix[:, 3:14]
    mask = ~((p0.sum(axis=2) > -3) * (p1.sum(axis=2) > -3) * (p2.sum(axis=2) > -3) * (p3.sum(axis=2) > -3))

    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    b1 /= np.linalg.norm(b1, axis=2, keepdims=True)

    v = b0 - np.sum(b0 * b1, axis=2)[:, :, np.newaxis] * b1
    w = b2 - np.sum(b2 * b1, axis=2)[:, :, np.newaxis] * b1

    x = np.sum(v * w, axis=2)
    y = np.sum(np.cross(b1, v) * w, axis=2)

    dihedral_matrix = np.degrees(np.arctan2(y, x))
    dihedral_matrix[mask] = np.nan
    return dihedral_matrix

def make_sc_mask(coords, chain_lens, mask, seq_mask, sc_mask_schedule, base_sc_mask, sc_mask_rate, mask_neighbors, mask_interface, half_interface, inter_cutoff, name, epoch):
    mask = mask * seq_mask
    pos_rows = list(np.nonzero(mask))[0]
    gen_sc_mask = True
    if mask_interface:
        sc_mask = list(find_interface(coords, chain_lens, half=half_interface, inter_cutoff=inter_cutoff))
        gen_sc_mask = (len(sc_mask) == 0) or len(set(pos_rows).intersection(set(sc_mask))) == 0
    if gen_sc_mask:
        if sc_mask_schedule:
            sc_mask_rate = np.concatenate([np.linspace(base_sc_mask, sc_mask_rate, 50), sc_mask_rate*np.ones(50)])[epoch]
            if epoch > 50 and sc_mask_rate == 1 and random.random() > 0.75:
                sc_mask_rate = random.random() * sc_mask_rate
        random.seed(epoch + sum([ord(char) for char in name]))
        num_rows = int(sc_mask_rate * len(pos_rows))
        if num_rows == 0:
            num_rows = len(pos_rows)
        sc_mask = list(random.sample(list(pos_rows), num_rows))
    if len(sc_mask) == 0:
        sc_mask = list(range(coords.shape[0]))

    if mask_neighbors:
        X = np.expand_dims(coords, 0)
        X = torch.from_numpy(X[:,:,1,:])
        mask = torch.ones(X.shape[:-1])
        _, _, E_idx = extract_knn(X, mask, eps=1E-6, top_k=30)
        neighbors_to_mask = set(sc_mask)
        for i in sc_mask:
            neighbors_to_mask = neighbors_to_mask.union(set(E_idx[0,i].cpu().numpy()))
        sc_mask = list(neighbors_to_mask)

    return sc_mask

def featurize(batch, device, augment_type, augment_eps, replicate, epoch, mask_options=None, esm=None, batch_converter=None, noise_lim=2, use_sc=False, convert_to_esm=False, use_esm_attns=False, esm_embed_layer=30, from_rla=False, use_reps=False, connect_chains=False, one_hot=False, post_esm_mask=True):
    epoch = 2
    alphabet = 'ACDEFGHIKLMNPQRSTVWY-X'
    B = len(batch)
    lengths = np.array([len(b['seq']) for b in batch], dtype=np.int32) #sum of chain seq lengths
    L_max = max([len(b['seq']) for b in batch])
    X = np.zeros([B, L_max, 4, 3])
    X_sc = np.zeros([B, L_max, 10, 3])
    Chi_sc = np.zeros([B, L_max, 11])
    Ids_sc = np.zeros([B, L_max, 10])
    esm_embs = []
    esm_attns = []
    residue_idx = -100*np.ones([B, L_max], dtype=np.int32) #residue idx with jumps across chains
    chain_M = np.zeros([B, L_max], dtype=np.int32) #1.0 for the bits that need to be predicted, 0.0 for the bits that are given
    mask_self = np.ones([B, L_max, L_max], dtype=np.int32) #for interface loss calculation - 0.0 for self interaction, 1.0 for other
    chain_encoding_all = np.zeros([B, L_max], dtype=np.int32) #integer encoding for chains 0, 0, 0,...0, 1, 1,..., 1, 2, 2, 2...
    S = np.zeros([B, L_max], dtype=np.int32) #sequence AAs integers
    all_chain_lens = []
    init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H', 'I', 'J','K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T','U', 'V','W','X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j','k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't','u', 'v','w','x', 'y', 'z']
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_letters = init_alphabet + extra_alphabet
    names = []
    chain_eos_inds = []
    sc_mask_list = []
    for i, b in enumerate(batch):
        names.append(b['name'])
        chain_lens = []
        masked_chains = b['masked_list']
        visible_chains = b['visible_list']
        all_chains = masked_chains + visible_chains
        visible_temp_dict = {}
        masked_temp_dict = {}
        for step, letter in enumerate(all_chains):
            chain_seq = b[f'seq_chain_{letter}']
            if letter in visible_chains:
                visible_temp_dict[letter] = chain_seq
            elif letter in masked_chains:
                masked_temp_dict[letter] = chain_seq
        for km, vm in masked_temp_dict.items():
            for kv, vv in visible_temp_dict.items():
                if vm == vv:
                    if kv not in masked_chains:
                        masked_chains.append(kv)
                    if kv in visible_chains:
                        visible_chains.remove(kv)
        all_chains = masked_chains + visible_chains
        random.shuffle(all_chains) #randomly shuffle chain order
        num_chains = b['num_of_chains']
        mask_dict = {}
        x_chain_list = []
        chain_mask_list = []
        x_sc_chain_list = []
        chain_seq_list = []
        chain_encoding_list = []
        x_sc_chi_list = []
        x_sc_ids_list = []
        c = 1
        l0 = 0
        l1 = 0
        for step, letter in enumerate(all_chains):
            if letter in visible_chains:
                chain_seq = b[f'seq_chain_{letter}']
                chain_length = len(chain_seq)
                chain_coords = b[f'coords_chain_{letter}'] #this is a dictionary
                chain_mask = np.zeros(chain_length) #0.0 for visible chains
                x_chain = np.stack([chain_coords[c] for c in [f'N_chain_{letter}', f'CA_chain_{letter}', f'C_chain_{letter}', f'O_chain_{letter}']], 1) #[chain_length,4,3]
                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c*np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
                mask_self[i, l0:l1, l0:l1] = np.zeros([chain_length, chain_length])
                residue_idx[i, l0:l1] = 100*(c-1)+np.arange(l0, l1)
                l0 += chain_length
                c+=1
            elif letter in masked_chains: 
                chain_seq = b[f'seq_chain_{letter}']
                chain_length = len(chain_seq)
                chain_coords = b[f'coords_chain_{letter}'] #this is a dictionary
                chain_mask = np.ones(chain_length) #0.0 for visible chains
                x_chain = np.stack([chain_coords[c] for c in [f'N_chain_{letter}', f'CA_chain_{letter}', f'C_chain_{letter}', f'O_chain_{letter}']], 1) #[chain_lenght,4,3]
                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c*np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
                mask_self[i, l0:l1, l0:l1] = np.zeros([chain_length, chain_length])
                residue_idx[i, l0:l1] = 100*(c-1)+np.arange(l0, l1)
                l0 += chain_length
                c+=1
            chain_lens.append(chain_length)
            
            sc_coords = chain_coords[f'sc_{letter}']
            sc_chi = dihedral_angles(np.concatenate([x_chain, sc_coords], 1))
            sc_ids = np.stack([x for x in map(lambda x: AA_ATOM_IDS[x], chain_seq)])
            
            x_sc_chi_list.append(sc_chi)
            x_sc_chain_list.append(sc_coords)
            x_sc_ids_list.append(sc_ids)
            
        chain_eos_inds.append(torch.cat([torch.arange(sum(chain_lens)), torch.tensor([-1])], dim=-1))
        all_chain_lens.append(chain_lens)
        x = np.concatenate(x_chain_list,0) #[L, 4, 3]
        all_sequence = "".join(chain_seq_list)
        m = np.concatenate(chain_mask_list,0) #[L,], 1.0 for places that need to be predicted
        chain_encoding = np.concatenate(chain_encoding_list,0)

        

        l = len(all_sequence)
        x_pad = np.pad(x, [[0,L_max-l], [0,0], [0,0]], 'constant', constant_values=(np.nan, ))
        X[i,:,:,:] = x_pad

        if use_sc:
            x_sc = np.concatenate(x_sc_chain_list, 0)
            x_sc_pad = np.pad(x_sc, [[0,L_max-l], [0,0], [0,0]], 'constant', constant_values=(np.nan, ))
            X_sc[i,:,:,:] = x_sc_pad
            chi_sc = np.concatenate(x_sc_chi_list, 0)
            chi_sc_pad = np.pad(chi_sc, [[0,L_max-l], [0,0]], 'constant', constant_values=(np.nan, ))
            Chi_sc[i,:,:] = chi_sc_pad
            sc_ids = np.concatenate(x_sc_ids_list, 0)
            sc_ids_pad = np.pad(sc_ids, [[0,L_max-l], [0,0]], 'constant', constant_values=(np.nan, ))
            Ids_sc[i,:,:] = sc_ids_pad
            temp_mask = np.isfinite(np.sum(x,(1,2))).astype(np.float32)
            temp_seq_mask = np.array(list(all_sequence))
            temp_seq_mask = (~(temp_seq_mask == 'X') * ~(temp_seq_mask == '-')).astype(np.float32)
            sc_mask = make_sc_mask(x, chain_lens, temp_mask, temp_seq_mask, mask_options['sc_mask_schedule'], mask_options['base_sc_mask'], mask_options['sc_mask_rate'], mask_options['mask_neighbors'], mask_options['mask_interface'], mask_options['half_interface'], mask_options['inter_cutoff'], b['name'], epoch)
            X_sc[i,sc_mask, ...] = np.nan
            Chi_sc[i,sc_mask, ...] = np.nan
            Ids_sc[i,sc_mask, ...] = np.nan
            sc_mask_list.append(sc_mask)

            if esm is not None:
                seq_ints = torch.from_numpy(np.array(seq_to_ints(copy.deepcopy(all_sequence))))
                esm_seq = _apply_seq_mask(copy.deepcopy(seq_ints), None, None, sc_mask, None, None, None, None, None, None, converted_to_esm=convert_to_esm)
                esm_emb, esm_attn = _esm_featurize(chain_lens, esm_seq, esm, batch_converter, use_esm_attns, esm_embed_layer, from_rla=from_rla, use_reps=use_reps, connect_chains=connect_chains, one_hot=one_hot, dev=device)
                if use_esm_attns:
                    esm_attn = esm_attn.unsqueeze(0)
                    X_esm = np.expand_dims(x, 0)
                    X_esm = torch.from_numpy(X_esm[:,:,1,:])
                    mask = torch.ones(X_esm.shape[:-1])
                    _, _, E_idx = extract_knn(X_esm, mask, eps=1E-6, top_k=30)
                    esm_attn = gather_edges(esm_attn, E_idx).squeeze(0)
                    esm_attns.append(esm_attn)
                esm_embs.append(esm_emb)
     
        else:
            X_sc, x_mask_sc, Ids_sc, sc_mask_full, Chi_sc, sc_chi_mask = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
            esm_embs, esm_attns = np.array([]), np.array([])

        m_pad = np.pad(m, [[0,L_max-l]], 'constant', constant_values=(0.0, ))
        chain_M[i,:] = m_pad

        chain_encoding_pad = np.pad(chain_encoding, [[0,L_max-l]], 'constant', constant_values=(0.0, ))
        chain_encoding_all[i,:] = chain_encoding_pad

        # Convert to labels
        indices = np.asarray([alphabet.index(a) for a in all_sequence], dtype=np.int32)
        S[i, :l] = indices

    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    X[isnan] = 0.
    if use_sc:
        x_mask_sc = copy.deepcopy(mask)
        for i, sc_mask in enumerate(sc_mask_list):
            x_mask_sc[i, sc_mask] = 0
        X_sc[np.isnan(X_sc)] = -1
        Chi_sc[np.isnan(Chi_sc)] = -1000
        Ids_sc[np.isnan(Ids_sc)] = -1

    

    # Conversion
    residue_idx = torch.from_numpy(residue_idx).to(dtype=torch.long)
    S = torch.from_numpy(S).to(dtype=torch.long)
    X = torch.from_numpy(X).to(dtype=torch.float32)
    mask = torch.from_numpy(mask).to(dtype=torch.float32)
    mask_self = torch.from_numpy(mask_self).to(dtype=torch.float32)
    chain_M = torch.from_numpy(chain_M).to(dtype=torch.float32)
    chain_encoding_all = torch.from_numpy(chain_encoding_all).to(dtype=torch.long)
    chain_eos_inds = pad_sequence(chain_eos_inds, batch_first=True, padding_value=-2)
    X_sc = torch.from_numpy(X_sc).to(dtype=torch.float32)
    x_mask_sc = torch.from_numpy(x_mask_sc).to(dtype=torch.float32)
    Ids_sc = torch.from_numpy(Ids_sc).to(dtype=torch.float32)
    Chi_sc = torch.from_numpy(Chi_sc).to(dtype=torch.float32)
    
    if use_sc:
        sc_mask_full = (((Ids_sc != -1) * x_mask_sc.unsqueeze(-1)).unsqueeze(-1).expand(X_sc.shape)).to(dtype=torch.bool)
        

        if esm is not None:
            esm_embs = pad_sequence(esm_embs, batch_first=True, padding_value = 0)
            if post_esm_mask:
                esm_embs *= x_mask_sc.unsqueeze(-1)
            if use_esm_attns:
                esm_attns = pad_sequence(esm_attns, batch_first=True, padding_value = 0)
            else:
                esm_attns = torch.Tensor([])
        else:
            esm_embs, esm_attns = torch.Tensor([]), torch.Tensor([])
    else:
        esm_embs, esm_attns = torch.Tensor([]), torch.Tensor([])
        sc_mask_full = torch.Tensor([])

    # Add noise if needed
    if augment_type == 'atomic':
        X = X + augment_eps * torch.randn(X.shape)
    elif augment_type.find('torsion') > -1:
        for i in range(X.shape[0]):
            X[i] += generate_noise(augment_type, augment_eps, names[i], replicate, epoch, X[i], noise_lim=noise_lim, chain_lens=all_chain_lens[i], mask=mask[i])

    return {
        'X': X,
        'x_mask': mask,
        'seqs': S,
        'seq_lens': lengths,
        'ids': names,
        'chain_idx': chain_encoding_all,
        'chain_lens': all_chain_lens,
        'chain_eos_inds': chain_eos_inds,
        'X_sc': X_sc,
        'x_mask_sc': x_mask_sc,
        'sc_chi': Chi_sc,
        'sc_mask_full': sc_mask_full,
        'sc_ids': Ids_sc,
        'esm_embs': esm_embs,
        'esm_attns': esm_attns,
        }

class StructureDataset():
    def __init__(self, pdb_dict_list, verbose=True, truncate=None, max_length=100,
        alphabet='ACDEFGHIKLMNPQRSTVWYX'):
        alphabet_set = set([a for a in alphabet])
        discard_count = {
            'bad_chars': 0,
            'too_long': 0,
            'bad_seq_length': 0
        }

        self.data = []

        start = time.time()
        for i, entry in enumerate(pdb_dict_list):
            seq = entry['seq']
            name = entry['name']

            bad_chars = set([s for s in seq]).difference(alphabet_set)
            if len(bad_chars) == 0:
                if len(entry['seq']) <= max_length:
                    self.data.append(entry)
                else:
                    discard_count['too_long'] += 1
            else:
                #print(name, bad_chars, entry['seq'])
                discard_count['bad_chars'] += 1

            # Truncate early
            if truncate is not None and len(self.data) == truncate:
                return

            if verbose and (i + 1) % 1000 == 0:
                elapsed = time.time() - start
                #print('{} entries ({} loaded) in {:.1f} s'.format(len(self.data), i+1, elapsed))

            #print('Discarded', discard_count)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return [self.data[i] for i in idx]
        return self.data[idx]

class StructureSampler(Sampler):
    def __init__(self, dataset, batch_size=100, device='cpu', flex_type="", augment_eps=0, replicate=1, mask_options=None, esm=None, batch_converter=None, noise_lim=2, use_sc=False, convert_to_esm=False, use_esm_attns=False, esm_embed_layer=30, from_rla=False, use_reps=False, connect_chains=False, one_hot=False, post_esm_mask=True):
        self.size = len(dataset)
        self.lengths = [len(dataset[i]['seq']) for i in range(self.size)]
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.flex_type = flex_type
        self.augment_eps = augment_eps
        self.replicate = replicate
        self.epoch = -1
        self.mask_options = mask_options
        self.esm = esm
        self.batch_converter = batch_converter
        self.noise_lim = noise_lim
        self.use_sc = use_sc
        self.convert_to_esm = convert_to_esm
        self.use_esm_attns = use_esm_attns
        self.esm_embed_layer = esm_embed_layer
        self.from_rla = from_rla
        self.use_reps = use_reps
        self.connect_chains = connect_chains
        self.one_hot = one_hot
        self.post_esm_mask = post_esm_mask
        self._cluster()

    def _set_epoch(self, epoch):
        self.epoch = epoch

    def _cluster(self):
        sorted_ix = np.argsort(self.lengths)

        # Cluster into batches of similar sizes
        clusters, batch = [], []
        batch_max = 0
        for ix in sorted_ix:
            size = self.lengths[ix]
            if size * (len(batch) + 1) <= self.batch_size:
                batch.append(ix)
                batch_max = size
            else:
                clusters.append(batch)
                batch, batch_max = [], 0
        if len(batch) > 0:
            clusters.append(batch)
        self.clusters = clusters

    def package(self, b_idx):
        return featurize(b_idx, self.device, self.flex_type, self.augment_eps, self.replicate, self.epoch, self.mask_options, self.esm, self.batch_converter, self.noise_lim, self.use_sc, self.convert_to_esm, self.use_esm_attns, self.esm_embed_layer, self.from_rla, self.use_reps, self.connect_chains, self.one_hot, self.post_esm_mask)

    def __len__(self):
        return len(self.clusters)

    def __iter__(self):
        np.random.shuffle(self.clusters)
        for b_idx in self.clusters:
            # print(b_idx)
            # batch = [self.dataset[i] for i in b_idx]
            # print(batch[0].keys())
            yield b_idx



class StructureLoader():
    def __init__(self, dataset, batch_size=100, shuffle=True,
        collate_fn=lambda x:x, drop_last=False):
        self.dataset = dataset
        self.size = len(dataset)
        self.lengths = [len(dataset[i]['seq']) for i in range(self.size)]
        self.batch_size = batch_size
        sorted_ix = np.argsort(self.lengths)

        # Cluster into batches of similar sizes
        clusters, batch = [], []
        batch_max = 0
        for ix in sorted_ix:
            size = self.lengths[ix]
            if size * (len(batch) + 1) <= self.batch_size:
                batch.append(ix)
                batch_max = size
            else:
                clusters.append(batch)
                batch, batch_max = [], 0
        if len(batch) > 0:
            clusters.append(batch)
        self.clusters = clusters

    def __len__(self):
        return len(self.clusters)

    def __iter__(self):
        np.random.shuffle(self.clusters)
        for b_idx in self.clusters:
            batch = [self.dataset[i] for i in b_idx]
            yield batch


def worker_init_fn(worker_id):
    np.random.seed()

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer, step):
        self.optimizer = optimizer
        self._step = step
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    @property
    def param_groups(self):
        """Return param_groups."""
        return self.optimizer.param_groups

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()

def get_std_opt(parameters, d_model, step):
    return NoamOpt(
        d_model, 2, 4000, torch.optim.Adam(parameters, lr=0, betas=(0.9, 0.98), eps=1e-9), step
    )

def get_pdbs(data_loader, repeat=1, max_length=10000, num_units=1000000, prune_length=True):
    init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H', 'I', 'J','K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T','U', 'V','W','X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j','k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't','u', 'v','w','x', 'y', 'z']
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_alphabet = init_alphabet + extra_alphabet
    c = 0
    c1 = 0
    pdb_dict_list = []
    t0 = time.time()
    for _ in range(repeat):
        for step,t in enumerate(data_loader):
            # print(step)
            t = {k:v[0] for k,v in t.items()}
            c1 += 1
            if 'label' in list(t):
                my_dict = {}
                s = 0
                concat_seq = ''
                concat_N = []
                concat_CA = []
                concat_C = []
                concat_O = []
                concat_mask = []
                coords_dict = {}
                mask_list = []
                visible_list = []
                if len(list(np.unique(t['idx']))) < 352:
                    for idx in list(np.unique(t['idx'])):
                        letter = chain_alphabet[idx]
                        res = np.argwhere(t['idx']==idx)
                        initial_sequence= "".join(list(np.array(list(t['seq']))[res][0,]))
                        if initial_sequence[-6:] == "HHHHHH":
                            res = res[:,:-6]
                        if initial_sequence[0:6] == "HHHHHH":
                            res = res[:,6:]
                        if initial_sequence[-7:-1] == "HHHHHH":
                           res = res[:,:-7]
                        if initial_sequence[-8:-2] == "HHHHHH":
                           res = res[:,:-8]
                        if initial_sequence[-9:-3] == "HHHHHH":
                           res = res[:,:-9]
                        if initial_sequence[-10:-4] == "HHHHHH":
                           res = res[:,:-10]
                        if initial_sequence[1:7] == "HHHHHH":
                            res = res[:,7:]
                        if initial_sequence[2:8] == "HHHHHH":
                            res = res[:,8:]
                        if initial_sequence[3:9] == "HHHHHH":
                            res = res[:,9:]
                        if initial_sequence[4:10] == "HHHHHH":
                            res = res[:,10:]
                        if res.shape[1] < 4:
                            pass
                        else:
                            my_dict['seq_chain_'+letter]= "".join(list(np.array(list(t['seq']))[res][0,]))
                            concat_seq += my_dict['seq_chain_'+letter]
                            if idx in t['masked']:
                                mask_list.append(letter)
                            else:
                                visible_list.append(letter)
                            coords_dict_chain = {}
                            all_atoms = np.array(t['xyz'][res,])[0,] #[L, 14, 3]
                            coords_dict_chain['N_chain_'+letter]=all_atoms[:,0,:].tolist()
                            coords_dict_chain['CA_chain_'+letter]=all_atoms[:,1,:].tolist()
                            coords_dict_chain['C_chain_'+letter]=all_atoms[:,2,:].tolist()
                            coords_dict_chain['O_chain_'+letter]=all_atoms[:,3,:].tolist()
                            coords_dict_chain['sc_'+letter] = all_atoms[:,4:,:]
                            my_dict['coords_chain_'+letter]=coords_dict_chain

                    my_dict['name']= t['label']
                    my_dict['masked_list']= mask_list
                    my_dict['visible_list']= visible_list
                    my_dict['num_of_chains'] = len(mask_list) + len(visible_list)
                    my_dict['seq'] = concat_seq
                    if prune_length and len(my_dict['seq']) < 30:
                        continue
                    if len(concat_seq) <= max_length:
                        pdb_dict_list.append(my_dict)
                    if len(pdb_dict_list) >= num_units:
                        break
    return pdb_dict_list



class PDB_dataset(torch.utils.data.Dataset):
    def __init__(self, IDs, loader, train_dict, params):
        self.IDs = IDs
        self.train_dict = train_dict
        self.loader = loader
        self.params = params

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        ID = self.IDs[index]
        sel_idx = np.random.randint(0, len(self.train_dict[ID]))
        out = self.loader(self.train_dict[ID][sel_idx], self.params)
        return out



def loader_pdb(item,params):
    pdbid,chid = item[0].split('_')
    PREFIX = "%s/pdb/%s/%s"%(params['DIR'],pdbid[1:3],pdbid)
    
    # load metadata
    if not os.path.isfile(PREFIX+".pt"):
        return {'seq': np.zeros(5)}
    meta = torch.load(PREFIX+".pt")
    asmb_ids = meta['asmb_ids']
    asmb_chains = meta['asmb_chains']
    chids = np.array(meta['chains'])

    # find candidate assemblies which contain chid chain
    asmb_candidates = set([a for a,b in zip(asmb_ids,asmb_chains)
                           if chid in b.split(',')])
    # if the chains is missing is missing from all the assemblies
    # then return this chain alone
    if len(asmb_candidates)<1:
        chain = torch.load("%s_%s.pt"%(PREFIX,chid))
        L = len(chain['seq'])
        return {'seq'    : chain['seq'],
                'xyz'    : chain['xyz'],
                'idx'    : torch.zeros(L).int(),
                'masked' : torch.Tensor([0]).int(),
                'label'  : item[0]}

    # randomly pick one assembly from candidates
    asmb_i = random.sample(list(asmb_candidates), 1)

    # indices of selected transforms
    idx = np.where(np.array(asmb_ids)==asmb_i)[0]

    # load relevant chains
    chains = {c:torch.load("%s_%s.pt"%(PREFIX,c))
              for i in idx for c in asmb_chains[i]
              if c in meta['chains']}

    # generate assembly
    asmb = {}
    for k in idx:

        # pick k-th xform
        xform = meta['asmb_xform%d'%k]
        u = xform[:,:3,:3]
        r = xform[:,:3,3]

        # select chains which k-th xform should be applied to
        s1 = set(meta['chains'])
        s2 = set(asmb_chains[k].split(','))
        chains_k = s1&s2

        # transform selected chains 
        for c in chains_k:
            try:
                xyz = chains[c]['xyz']
                xyz_ru = torch.einsum('bij,raj->brai', u, xyz) + r[:,None,None,:]
                asmb.update({(c,k,i):xyz_i for i,xyz_i in enumerate(xyz_ru)})
            except KeyError:
                return {'seq': np.zeros(5)}

    # select chains which share considerable similarity to chid
    seqid = meta['tm'][chids==chid][0,:,1]
    homo = set([ch_j for seqid_j,ch_j in zip(seqid,chids)
                if seqid_j>params['HOMO']])
    # stack all chains in the assembly together
    seq,xyz,idx,masked = "",[],[],[]
    seq_list = []
    for counter,(k,v) in enumerate(asmb.items()):
        seq += chains[k[0]]['seq']
        seq_list.append(chains[k[0]]['seq'])
        xyz.append(v)
        idx.append(torch.full((v.shape[0],),counter))
        if k[0] in homo:
            masked.append(counter)
    return {'seq'    : seq,
            'xyz'    : torch.cat(xyz,dim=0),
            'idx'    : torch.cat(idx,dim=0),
            'masked' : torch.Tensor(masked).int(),
            'label'  : item[0]}

def loader_gen_pdb(pdb_file):
    parser = PDBParser()
    pdb_out = data_utils.process_pdb_raw(parser, pdb_file, all=True)
    seq = "".join(pdb_out[-2])
    xyz = torch.from_numpy(np.concatenate(pdb_out[0]))
    idx = []
    for counter in range(len(pdb_out[0])):
        idx.append(torch.full((pdb_out[0][counter].shape[0],),counter))
    idx = torch.cat(idx, dim=0)
    masked = []
    return {'seq'    : seq,
            'xyz'    : xyz,
            'idx'    : idx,
            'masked' : torch.Tensor(masked).int(),
            'label'  : os.path.basename(pdb_file)}

def build_training_clusters(params, debug):
    val_ids = set([int(l) for l in open(params['VAL']).readlines()])
    test_ids = set([int(l) for l in open(params['TEST']).readlines()])
   
    if debug:
        val_ids = []
        test_ids = []
 
    # read & clean list.csv
    with open(params['LIST'], 'r') as f:
        reader = csv.reader(f)
        next(reader)
        rows = [[r[0],r[3],int(r[4])] for r in reader
                if float(r[2])<=params['RESCUT'] and
                parser.parse(r[1])<=parser.parse(params['DATCUT'])]
    
    # compile training and validation sets
    train = {}
    valid = {}
    test = {}

    if debug:
        rows = rows[:20]
    for r in rows:
        if r[2] in val_ids:
            if r[2] in valid.keys():
                valid[r[2]].append(r[:2])
            else:
                valid[r[2]] = [r[:2]]
        elif r[2] in test_ids:
            if r[2] in test.keys():
                test[r[2]].append(r[:2])
            else:
                test[r[2]] = [r[:2]]
        else:
            if r[2] in train.keys():
                train[r[2]].append(r[:2])
            else:
                train[r[2]] = [r[:2]]
    if debug:
        valid=train       
    return train, valid, test

def build_dataset_from_pdb(pdbs, list='/data1/groups/keatinglab/ProteinMPNN/data/pdb_2021aug02/list.csv', rescut=3.5):
    # read & clean list.csv
    with open(list, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        rows = [[r[0],r[3],int(r[4])] for r in reader
                if float(r[2])<=rescut]
    
    # compile training and validation sets
    dataset = {}

    for r in rows:
        if r[0] in pdbs:
            dataset[r[2]] = r[:2]
           
    return dataset
