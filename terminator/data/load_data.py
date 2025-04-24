import numpy as np
import pandas as pd
import gzip
import warnings
from pathlib import Path
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB import PDBParser
from Bio import SeqIO
from Bio.SeqUtils import seq1, seq3
import webdataset as wds
import os
from tqdm import tqdm
import copy
import torch
from terminator.utils.common import seq_to_ints, ints_to_seq
np.seterr(all='ignore')

def convert_seq_df_to_data(df, data, seq_col='seqs', ener_col='ener'):
    seq = "".join(ints_to_seq(data['sequence']))
    df = df[df[seq_col] != seq]
    sort_seqs, sort_nrgs = [], []
    for mut_seq, ener in zip(df[seq_col], df[ener_col]):
        sort_seqs.append(torch.from_numpy(np.array(seq_to_ints(mut_seq))))
        sort_nrgs.append(torch.from_numpy(np.array(-1*ener)))
    return torch.stack(sort_seqs), torch.Tensor(sort_nrgs)  


def convert_df_to_data(df, data, pos_col='pos', wt_col='wildtype', mutant_col='mutant', label_col='mutation', chain_col='chain', ener_col='ener', site_split=None, mut_split=None, adj_index=True, pos_by_chain=True):
    sort_seqs = []
    sort_nrgs = []
    seq = "".join(ints_to_seq(data['sequence']))
    for pos_list, wt_list, mutant_list, chain_list, label_list, ener in zip(df[pos_col].values, df[wt_col].values, df[mutant_col].values, df[chain_col].values, df[label_col].values, df[ener_col].values):
        skip = False
        mut_seq = copy.deepcopy(list(seq))
        for pos, wt, mutant, label, chain in zip(str(pos_list).split(';'), str(wt_list).split(';'), str(mutant_list).split(';'), str(label_list).split(';'), str(chain_list).split(';')):
            pos = int(pos)
            if site_split is not None and pos not in site_split:
                skip = True
                break
            if mut_split is not None and label not in mut_split:
                skip = True
                break
            if mutant == '*' or pd.isna(ener):
                skip = True
                break
            rcl = 0
            for cid, cl in zip(data['chain_ids'], data['chain_lens']):
                if cid == chain:
                    break
                rcl += cl
            res_ids = data['res_ids'][rcl:rcl+cl]
            chain_seq = seq[rcl:rcl+cl]
            # print(chain_seq)
            pos = int(pos)
            if adj_index:
                seq_ind = np.where(res_ids == pos)[0]
                if len(seq_ind) == 0:
                    for res_id, chain_char in zip(res_ids, chain_seq):
                        if res_id == pos:
                            print('\t', res_id, chain_char)
                    skip = True
                    break
                seq_ind = seq_ind.item()
            elif pos_by_chain:
                seq_ind = pos
            else:
                seq_ind = pos - rcl
            try:
                if chain_seq[seq_ind] != wt:
                    print('ERROR', rcl,data['chain_lens'], pos, seq_ind, chain_seq, wt, label, df['protein'].values[0])
                    skip = True
                    break
            except Exception as e:
                print(seq_ind, len(chain_seq), pos, label)
                print(e)
                raise ValueError
                continue
            mut_seq[seq_ind + rcl] = mutant
        if skip:
            continue
        mut_seq = "".join(mut_seq)
        sort_seqs.append(torch.from_numpy(np.array(seq_to_ints(mut_seq))))
        sort_nrgs.append(torch.from_numpy(np.array(-1*ener)))
    return torch.stack(sort_seqs), torch.Tensor(sort_nrgs)

def infer_oxygen_position(n_coord, ca_coord, c_coord):
    """
    Infer the position of the oxygen atom (O) based on N, CA, and C coordinates.
    
    Args:
        n_coord (np.array): Coordinates of the N atom.
        ca_coord (np.array): Coordinates of the CA atom.
        c_coord (np.array): Coordinates of the C atom.
    
    Returns:
        np.array: Estimated coordinates of the O atom.
    """
    # Vector from C to CA
    ca_to_c = c_coord - ca_coord
    ca_to_n = n_coord - ca_coord

    # Compute a normal vector to the plane formed by CA, C, and N
    normal_vector = np.cross(ca_to_c, ca_to_n)
    normal_vector /= np.linalg.norm(normal_vector)

    # Approximate oxygen position by projecting a vector from C along the normal direction
    oxygen_position = c_coord + 1.22 * normal_vector  # 1.22 Å is the approximate C=O bond length

    return oxygen_position

def process_residue(residue):
    atoms = ['N', 'CA', 'C', 'O']
    coordinates = []
    for r in atoms:
        coord = residue.child_dict.get(r, None)
        if coord is None:
            if r == 'O':
                coord = residue.child_dict.get('OXT', None)
                if coord is None:
                    # Infer oxygen position if it's missing
                    n_coord = residue.child_dict['N'].get_coord()
                    ca_coord = residue.child_dict['CA'].get_coord()
                    c_coord = residue.child_dict['C'].get_coord()
                    coord = infer_oxygen_position(n_coord, ca_coord, c_coord)
                    coordinates.append(np.array(coord))
                    continue
        coordinates.append(np.array(coord.get_coord()))
    return np.stack(coordinates), seq1(residue.resname), residue.id[1]

def process_chain(chain, add_gap=False):
    coordinates = []
    seq = []
    residue_ids = []
    prev_id = None
    for r in chain:
        if len(r.get_id()[0].strip()) > 0:
            continue
        if add_gap and (prev_id is not None):
            for i in range(r.id[1] - (prev_id + 1)):
                coordinates.append(np.zeros((4,3)))
                seq.append('-')
                residue_ids.append(r.id[1]+i+1)
        output, residue_name, residue_id = process_residue(r)
        if output is not None:
            coordinates.append(output)
            seq.append(residue_name)
            residue_ids.append(residue_id)
            prev_id = r.id[1]
    if len(coordinates) == 0:
        return None, None, None
    coordinates = np.stack(coordinates)
    seq = ''.join(seq)
    return coordinates, seq, residue_ids

def process_chains(chains, pep=False, prot=False, add_gap=False):
    if pep or prot:
        chain_lens = []
        chain_ids = []
        for chain in chains:
            for i, res in enumerate(chain):
                continue
            chain_lens.append(i)
            chain_ids.append(chain.id)
        if chain_lens[0] < chain_lens[1]:
            pep_id = chain_ids[0]
            prot_id = chain_ids[1]
        else:
            pep_id = chain_ids[1]
            prot_id = chain_ids[0]
        if pep and isinstance(pep, str): pep_id == pep
        if prot and isinstance(prot, str): prot_id == prot
    output = []
    chain_ids = []
    residue_ids = []
    for chain in chains:
        if (pep and chain.id != pep_id) or (prot and chain.id != prot_id):
            continue
        out = process_chain(chain, add_gap)
        if out is not None:
            output.append(out)
            chain_ids.append(chain.id)
    coords = [u[0] for u in output]
    seqs = [u[1] for u in output]
    residue_ids = [u[2] for u in output]
    return coords, seqs, chain_ids, residue_ids

def process_structure(structure, pep=False, prot=False, add_gap=False):
    for s in structure: # only one structure
        return process_chains(s, pep, prot, add_gap)
    return None

# +
def process_pdb(parser, pdb_filename, add_gap=False):
    # print(pdb_filename)
    with gzip.open(pdb_filename, "rt") as file_handle:
        structure = parser.get_structure("?", file_handle)
        date = structure.header['deposition_date']
        return process_structure(structure, add_gap), date
    
def process_pdb_raw(parser, pdb_filename, pep=False, prot=False, add_gap=False):
    s = parser.get_structure("?", pdb_filename)
    return process_structure(s, pep, prot, add_gap)


def parse_PDB_biounits(x, atoms=['N','CA','C'], chain=None, add_gap=False):
  '''
  input:  x = PDB filename
          atoms = atoms to extract (optional)
  output: (length, atoms, coords=(x,y,z)), sequence
  '''

  alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")
  states = len(alpha_1)
  alpha_3 = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
             'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','GAP']
  
  aa_1_N = {a:n for n,a in enumerate(alpha_1)}
  aa_3_N = {a:n for n,a in enumerate(alpha_3)}
  aa_N_1 = {n:a for n,a in enumerate(alpha_1)}
  aa_1_3 = {a:b for a,b in zip(alpha_1,alpha_3)}
  aa_3_1 = {b:a for a,b in zip(alpha_1,alpha_3)}
  
  def AA_to_N(x):
    # ["ARND"] -> [[0,1,2,3]]
    x = np.array(x);
    if x.ndim == 0: x = x[None]
    return [[aa_1_N.get(a, states-1) for a in y] for y in x]
  
  def N_to_AA(x):
    # [[0,1,2,3]] -> ["ARND"]
    x = np.array(x);
    if x.ndim == 1: x = x[None]
    return ["".join([aa_N_1.get(a,"-") for a in y]) for y in x]

  xyz,seq,min_resn,max_resn = {},{},1e6,-1e6
  for line in open(x,"rb"):
    line = line.decode("utf-8","ignore").rstrip()

    if line[:6] == "HETATM" and line[17:17+3] == "MSE":
      line = line.replace("HETATM","ATOM  ")
      line = line.replace("MSE","MET")

    if line[:4] == "ATOM":
      ch = line[21:22]
      if ch == chain or chain is None:
        atom = line[12:12+4].strip()
        resi = line[17:17+3]
        resn = line[22:22+5].strip()
        x,y,z = [float(line[i:(i+8)]) for i in [30,38,46]]

        if resn[-1].isalpha(): 
            resa,resn = resn[-1],int(resn[:-1])-1
        else: 
            resa,resn = "",int(resn)-1
#         resn = int(resn)
        if resn < min_resn: 
            min_resn = resn
        if resn > max_resn: 
            max_resn = resn
        if resn not in xyz: 
            xyz[resn] = {}
        if resa not in xyz[resn]: 
            xyz[resn][resa] = {}
        if resn not in seq: 
            seq[resn] = {}
        if resa not in seq[resn]: 
            seq[resn][resa] = resi

        if atom not in xyz[resn][resa]:
          xyz[resn][resa][atom] = np.array([x,y,z])

  # convert to numpy arrays, fill in missing values
  seq_,xyz_ = [],[]
  try:
      for resn in range(min_resn,max_resn+1):
        if resn in seq:
          for k in sorted(seq[resn]): seq_.append(aa_3_N.get(seq[resn][k],20))
        elif add_gap: seq_.append(20)
        if resn in xyz:
          for k in sorted(xyz[resn]):
            for atom in atoms:
              if atom in xyz[resn][k]: xyz_.append(xyz[resn][k][atom])
              else: xyz_.append(np.full(3,np.nan))
        elif add_gap:
          for atom in atoms: xyz_.append(np.full(3,np.nan))
      return np.array(xyz_).reshape(-1,len(atoms),3), N_to_AA(np.array(seq_))
  except TypeError:
      return 'no_chain', 'no_chain'

def parse_PDB(path_to_pdb, input_chain_list=None, ca_only=False, add_gap=False):
    c=0
    pdb_dict_list = []
    init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H', 'I', 'J','K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T','U', 'V','W','X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j','k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't','u', 'v','w','x', 'y', 'z']
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_alphabet = init_alphabet + extra_alphabet
     
    if input_chain_list:
        chain_alphabet = input_chain_list  
 
    biounit_names = [path_to_pdb]
    for biounit in biounit_names:
        my_dict = {}
        s = 0
        concat_seq = ''
        concat_N = []
        concat_CA = []
        concat_C = []
        concat_O = []
        concat_mask = []
        coords_dict = {}
        for letter in chain_alphabet:
            if ca_only:
                sidechain_atoms = ['CA']
            else:
                sidechain_atoms = ['N', 'CA', 'C', 'O']
            xyz, seq = parse_PDB_biounits(biounit, atoms=sidechain_atoms, chain=letter, add_gap=add_gap)
            if type(xyz) != str:
                concat_seq += seq[0]
                my_dict['seq_chain_'+letter]=seq[0]
                coords_dict_chain = {}
                if ca_only:
                    coords_dict_chain['CA_chain_'+letter]=xyz.tolist()
                else:
                    coords_dict_chain['N_chain_' + letter] = xyz[:, 0, :].tolist()
                    coords_dict_chain['CA_chain_' + letter] = xyz[:, 1, :].tolist()
                    coords_dict_chain['C_chain_' + letter] = xyz[:, 2, :].tolist()
                    coords_dict_chain['O_chain_' + letter] = xyz[:, 3, :].tolist()
                my_dict['coords_chain_'+letter]=coords_dict_chain
                s += 1
        fi = biounit.rfind("/")
        my_dict['name']=biounit[(fi+1):-4]
        my_dict['num_of_chains'] = s
        my_dict['seq'] = concat_seq
        if s <= len(chain_alphabet):
            pdb_dict_list.append(my_dict)
            c+=1
    return pdb_dict_list

def process_data(pdb_data, pdb_dict_list, x_chain, pdb, site_split):
    seq = ''
    for chain_seq in pdb_data[1]:
        seq += chain_seq
    if len(seq) != x_chain.shape[0]:
        print(pdb, len(seq), x_chain.shape[0])
        return None
    batch = {}
    batch['pdb'] = pdb
    coords = []
    for chain in pdb_data[2]:
        coords.append( np.stack([np.array(pdb_dict_list[0][f'coords_chain_{chain}'][f'N_chain_{chain}']), np.array(pdb_dict_list[0][f'coords_chain_{chain}'][f'CA_chain_{chain}']),
                        np.array(pdb_dict_list[0][f'coords_chain_{chain}'][f'C_chain_{chain}']), np.array(pdb_dict_list[0][f'coords_chain_{chain}'][f'O_chain_{chain}'])], axis=1))
    batch['coords'] = np.concatenate(coords, axis=0)
    batch['ppoe'] = np.array([[[0.]]])
    batch['features'] = np.array([[[0.]]])
    batch['msas'] = np.array([[0.]])
    batch['focuses'] = np.array([[0.]])
    batch['contact_idxs'] = np.array([[0.]])
    batch['term_lens'] = np.array([1])
    batch['sequence'] = np.array(seq_to_ints("".join([pdb_dict_list[0][f'seq_chain_{chain}'] for chain in pdb_data[2]])))
    batch['seq_id'] = None
    batch['seq_len'] = len(seq)
    batch['chain_ids'] = pdb_data[2]
    batch['chain_lens'] = [len(pdb_dict_list[0][f'seq_chain_{chain}']) for chain in pdb_data[2]]
    batch['res_info'] = []
    for chain_len, chain_id in zip(batch['chain_lens'], pdb_data[2]):
        batch['res_info'] += [(chain_id, int(i_res)) for i_res in range(chain_len)]
    batch['sc_coords'] = np.zeros((batch['coords'].shape[0], 10, 3))
    batch['sc_chi'] = np.zeros((batch['coords'].shape[0], 10))
    batch['sc_ids'] = np.zeros((batch['coords'].shape[0], 10))
    batch['res_ids'] = np.concatenate([np.array(chain_info) for chain_info in pdb_data[3]])

    return batch

def load_data(pdb_dir, pdb_list, add_gap, ener_col='ener', sep_complex=False, ener_data=None, adj_index=True, seq_df=False, seq_col='seqs', site_split=None, pos_by_chain=True):
    parser = PDBParser(QUIET=True)
    dataset = []
    if sep_complex:
        target_dataset = []
        binder_dataset = []
    for pdb in pdb_list:
        pdb_path = os.path.join(pdb_dir, pdb)
        if '.pdb' not in pdb_path:
            pdb_path += '.pdb'
        if not os.path.exists(pdb_path):
            continue
        try:
            pdb_data = process_pdb_raw(parser, pdb_path, add_gap=add_gap)
            if pdb_data is None:
                continue
            pdb_dict_list = parse_PDB(pdb_path, input_chain_list=pdb_data[2], add_gap=add_gap)
            x_protein = np.concatenate(pdb_data[0], 0)

            batch = process_data(pdb_data, pdb_dict_list, x_protein, pdb, site_split)

        except:
            continue
        if batch is None:
            continue
        if ener_data is not None:
            try:
                ener_df = pd.read_csv(ener_data)
                if 'protein' in ener_df.columns:
                    ener_df = ener_df[ener_df['protein'] == os.path.splitext(pdb)[0]]
            except:
                print("Energy file not found. If none is required, use ener_data=None")
                raise ValueError
            try:
                print('seqy seqy: ', seq_df)
                if seq_df:
                    sort_seqs, sort_nrgs = convert_seq_df_to_data(ener_df, batch, seq_col=seq_col, ener_col=ener_col)
                else:
                    sort_seqs, sort_nrgs = convert_df_to_data(ener_df, batch, ener_col=ener_col, adj_index=adj_index, site_split=site_split, pos_by_chain=pos_by_chain)
            except Exception as e:
                print(pdb)
                print(e)
                raise e
            batch['sortcery_seqs'] = sort_seqs
            batch['sortcery_nrgs'] = sort_nrgs
        dataset.append([batch])
        if sep_complex:
            target_num = np.argmax(batch['chain_lens'])
            for chain_num, chain_id in enumerate(batch['chain_ids']):
                chain_data = [pdb_info[chain_num] for pdb_info in pdb_data]
                chain_data[3] = [chain_data[3]]
                chain_dict_list = [{f'seq_chain_{chain_id}': pdb_dict_list[0][f'seq_chain_{chain_id}'], 
                                    f'coords_chain_{chain_id}': pdb_dict_list[0][f'coords_chain_{chain_id}'],
                                    'name': pdb_dict_list[0]['name'], 'num_of_chains': 1, 'seq': pdb_dict_list[0][f'seq_chain_{chain_id}']
                                    }]
                x_chain = pdb_data[0][chain_num]
                batch = process_data(chain_data, chain_dict_list, x_chain, pdb + '_' + chain_id, site_split)
                if chain_num == target_num:
                    target_dataset.append([batch])
                else:
                    binder_dataset.append([batch])
    if not sep_complex:
        return dataset, None, None
    return dataset, target_dataset, binder_dataset
