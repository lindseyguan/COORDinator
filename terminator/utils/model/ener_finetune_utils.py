import numpy as np
import torch
import sys
import os
import copy as vcopy
import re
from terminator.models.layers.utils import _esm_featurize, extract_knn, gather_edges
import terminator.utils.model.loop_utils as loop_utils
## Constants
myAmino = ["R","H","K","D","E","S","T","N","Q","C","G","P","A","V","I","L","M","F","Y","W"]
FullAmino = ["ARG","HIS","LYS","ASP","GLU","SER","THR","ASN","GLN","CYS","GLY","PRO","ALA","VAL","ILE","LEU","MET","PHE","TYR","TRP"]
aminos = {FullAmino[i]:myAmino[i] for i in range(len(myAmino))}
CHAIN_LENGTH = 20
# zero is used as padding
AA_to_int = {'A': 1, 'ALA': 1, 'C': 2, 'CYS': 2, 'D': 3, 'ASP': 3, 'E': 4, 'GLU': 4, 'F': 5, 'PHE': 5, 'G': 6, 'GLY': 6, 'H': 7, 'HIS': 7, 'I': 8, 'ILE': 8,
    'K': 9, 'LYS': 9, 'L': 10, 'LEU': 10, 'M': 11, 'MET': 11, 'N': 12, 'ASN': 12, 'P': 13, 'PRO': 13, 'Q': 14, 'GLN': 14, 'R': 15, 'ARG': 15, 'S': 16, 'SER': 16,
    'T': 17, 'THR': 17, 'V': 18, 'VAL': 18, 'W': 19, 'TRP': 19, 'Y': 20, 'TYR': 20, 'X': 21
}
## amino acid to integer
atoi = {key: val - 1 for key, val in AA_to_int.items()}
## integer to amino acid
iota = {y: x for x, y in atoi.items() if len(x) == 1}

dim12 = np.repeat(np.arange(20), 20)
dim34 = np.tile(np.arange(20), 20)

def _to_dev(data_dict, dev):
    """ Push all tensor objects in the dictionary to the given device.

    Args
    ----
    data_dict : dict
        Dictionary of input features to TERMinator
    dev : str
        Device to load tensors onto
    """
    for key, value in data_dict.items():
        if isinstance(value, torch.Tensor):
            data_dict[key] = value.to(dev)
        if key == 'gvp_data':
            data_dict['gvp_data'] = [data.to(dev) for data in data_dict['gvp_data']]
        if key == 'edge_update_inds':
            data_dict['edge_update_inds'] = [data.to(dev) for data in data_dict['edge_update_inds']]

def get_out(model, data, loss_fn, teacher_forcing, num_recycles, keep_seq, esm, batch_converter, esm_type, esm_options, converter, dev, grad):
    _to_dev(data, dev)
    max_seq_len = max(data['seq_lens'].tolist())
    try:
        _, dump, _ = loop_utils.run_iter(model=model, optimizer=None, loss_fn=loss_fn, data=data, teacher_forcing=teacher_forcing, num_recycles=num_recycles, keep_seq=keep_seq, keep_sc_mask_recycle=True, running_loss_dict={},
                             dump=[], esm=esm, batch_converter=batch_converter, tokenizer=None, esm_type=esm_type, esm_options=esm_options, converter=converter, grad=grad, epoch=-1, test=True,
                             stall=False, dev=dev, from_ener = True)
        etab = dump[0]['out']
        E_idx = dump[0]['idx']
    except Exception as e:
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
    return etab, E_idx
            
def get_full_native_eval(model, pdb_dir="/data1/groups/keatinglab/mlu/bcl2/clean_pdb/"):
    pdb = os.path.join(pdb_dir, model + '.pdb')
    lines = [line.rstrip('\n') for line in open(pdb)]

    seq = {"A": "", "B": ""}
    index = ""
    chain = ""
    for l in lines:
        sp = re.split("\s+",l)
        if sp[0] != "ATOM":
            continue
        if sp[5] != index or sp[4] != chain:
            index = sp[5]
            chain = sp[4]
            seq[chain] += aminos[sp[3]]
    return seq

def expand_etab(etab, idxs):
    tetab = torch.from_numpy(etab).to(dtype=torch.float64)
    eidx = torch.from_numpy(idxs).unsqueeze(-1).unsqueeze(-1).expand(etab.shape)
    netab = torch.zeros(tetab.shape[0], tetab.shape[0], 20, 20, dtype=torch.float64)
    netab.scatter_(1, eidx, tetab)
    cetab = netab.transpose(0,1).transpose(2,3)
    cetab.scatter_(1, eidx, tetab)
    return cetab.numpy()


def condense_etab_expanded(e, pdb, protein):

    new = e[-1*20:, -1*20:,:, :]

    protein_array = np.array([atoi[aa] for aa in protein])
    eners = e[:len(protein), len(protein):len(protein)+20]
    inds = np.broadcast_to(protein_array[:,np.newaxis,np.newaxis,np.newaxis], (len(protein_array), 20, 1, 20))
    geners = np.take_along_axis(eners,inds, 2).squeeze(2)
    seners = np.sum(geners, axis=0, keepdims=False)
    new[(dim12, dim12, dim34, dim34)] += seners.flatten()
    return new

# Get ESM embeddings
def get_esm_embs(chain_lens, X, esm_seq, esm, batch_converter, use_esm_attns, esm_embed_layer, from_rla, use_reps, connect_chains, one_hot, dev):
    esm_emb, esm_attn = _esm_featurize(chain_lens, esm_seq, esm, batch_converter, use_esm_attns, esm_embed_layer, from_rla=from_rla, use_reps=use_reps, connect_chains=connect_chains, one_hot=one_hot, dev=dev)
    if use_esm_attns:
        esm_attn = esm_attn.unsqueeze(0)
        X = np.expand_dims(X, 0)
        X = torch.from_numpy(X[:,:,1,:])
        mask = torch.ones(X.shape[:-1])
        _, _, E_idx = extract_knn(X, mask, eps=1E-6, top_k=30)
        esm_attn = gather_edges(esm_attn, E_idx).squeeze(0)
        esm_attn = esm_attn.unsqueeze(0)
    return esm_emb.unsqueeze(0), esm_attn

def score_sequence(seq,mat):
    sc = 0.0
    chain_size = len(seq)
    if type(seq[0]) == str:
        seq = [atoi[s] for s in seq]
    for i in range(0, chain_size):
        amino1 = seq[i]
        for j in range(i, chain_size):
            amino2 = seq[j]
            sc += mat[i][j][amino1][amino2]
    return sc