import numpy as np
import os
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.SeqUtils import seq1
import gzip
warnings.simplefilter('ignore', PDBConstructionWarning)

def process_residue(residue):
    atoms = ['N', 'CA', 'C', 'O']
    coordinates = []
    for r in atoms:
        coord = residue.child_dict.get(r, None)
        if coord is None:
            if r == 'O':
                coord = residue.child_dict.get('OXT', None)
            if coord is None:
                return None, None
        coordinates.append(np.array(coord.get_coord()))
    return np.stack(coordinates), seq1(residue.resname)

def process_residue_all(residue):
    coordinates = []
    for atom in residue.get_atoms():
        coordinates.append(np.array(atom.get_coord()))
    while len(coordinates) < 14:
        coordinates.append(np.array([-1, -1, -1]))
    return np.stack(coordinates), seq1(residue.resname)

def process_chain(chain, all=False):
    coordinates = []
    seq = []
    for r in chain:
        if all:
            output, residue_name = process_residue_all(r)
        else:
            output, residue_name = process_residue(r)
        if output is not None:
            coordinates.append(output)
            seq.append(residue_name)
    if len(coordinates) == 0:
        return None
    coordinates = np.stack(coordinates)
    seq = ''.join(seq)
    return coordinates, seq

def process_chains(chains, pep=False, prot=False, all=False):
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
    for chain in chains:
        if (pep and chain.id != pep_id) or (prot and chain.id != prot_id):
            continue
        out = process_chain(chain, all=all)
        if out is not None:
            output.append(out)
            chain_ids.append(chain.id)
    coords = [u[0] for u in output]
    seqs = [u[1] for u in output]
    return coords, seqs, chain_ids

def process_structure(structure, pep=False, prot=False, all=False):
    for s in structure: # only one structure
        return process_chains(s, pep, prot, all=all)
    return None

# +
def process_pdb(parser, pdb_filename):
    # print(pdb_filename)
    with gzip.open(pdb_filename, "rt") as file_handle:
        structure = parser.get_structure("?", file_handle)
        date = structure.header['deposition_date']
        return process_structure(structure), date
    
def process_pdb_raw(parser, pdb_filename, pep=False, prot=False, all=False):
    s = parser.get_structure("?", pdb_filename)
    return process_structure(s, pep, prot, all=all)

def read_input_ids(index_file):
    input_ids = []
    with open(os.path.join(index_file), 'r') as f:
        for line in f:
            input_ids += [line.strip()]
    return np.array(input_ids)