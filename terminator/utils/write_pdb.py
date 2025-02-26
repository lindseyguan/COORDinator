import webdataset as wds
import os
import numpy as np
import torch
from terminator.utils.common import ints_to_seq_torch



def write_pdb(d, pdb_path, overwrite=True):

    def int_to_letters(n):
        """
        Convert a positive integer to a capital letter string (1-based indexing).
        Args:
            n: Integer (1 or higher).
        Returns:
            str: Corresponding letter string.
        """
        if n < 1:
            raise ValueError("Input must be a positive integer.")

        result = []
        while n > 0:
            n -= 1  # Adjust for 1-based indexing
            result.append(chr(n % 26 + ord('A')))
            n //= 26

        return ''.join(reversed(result))

    one_to_three = {
        'A': 'ALA',
        'C': 'CYS',
        'D': 'ASP',
        'E': 'GLU',
        'F': 'PHE',
        'G': 'GLY',
        'H': 'HIS',
        'I': 'ILE',
        'K': 'LYS',
        'L': 'LEU',
        'M': 'MET',
        'N': 'ASN',
        'P': 'PRO',
        'Q': 'GLN',
        'R': 'ARG',
        'S': 'SER',
        'T': 'THR',
        'V': 'VAL',
        'W': 'TRP',
        'Y': 'TYR',
        '-': '---',
        'U': 'SEC',
        'O': 'PYL',
        'B': 'ASX',
        'Z': 'GLX', 
        'X': 'XAA',
        'J': 'XLE'
    }
    atom_to_atom = {'N': 'N', 'CA': 'C', 'C': 'C', 'O': 'O'}
    
    pdb_id = d['ids'][0]
    if '.pdb' in pdb_id:
        pdb_file = os.path.join(pdb_path, pdb_id)
    else:
        pdb_file = os.path.join(pdb_path, pdb_id + '.pdb')
    if os.path.exists(pdb_file) and not overwrite:
        return False 
    
    seq = "".join(ints_to_seq_torch(d['seqs'][0]))

    with open(pdb_file, 'w') as f:
        f.write('MODEL        1\n')
        atom_count = 1
        res_count = 1
        prev_chain = None
        for res_num, chain_id in enumerate(d['chain_idx'][0].cpu().numpy()):
            if prev_chain is None:
                prev_chain = chain_id
            elif prev_chain != chain_id:
                prev_chain = chain_id
                f.write('TER\n')
            for atom_id, atom in enumerate(['N', 'CA', 'C', 'O']):
                line = ['ATOM']
                atom_count_gap = 7 - len(str(atom_count))
                line.append(' '*atom_count_gap)
                line.append(str(atom_count))
                atom_count += 1
                line.append('  ')
                line.append(atom + ' '*(4 - len(atom)))
                line.append(one_to_three[seq[res_num]])
                line.append(' ')
                line.append(int_to_letters(int(chain_id)))
                res_count_gap = 4 - len(str(res_count))
                line.append(' '*res_count_gap)
                line.append(str(res_count))
                x, y, z = np.round(d['X'][0, res_num, atom_id].cpu().numpy(), 3)
                x = str(x)
                y = str(y)
                z = str(z)
                zero_pad = 3 - len(x.split('.')[1])
                x += '0'*zero_pad
                zero_pad = 3 - len(y.split('.')[1])
                y += '0'*zero_pad
                zero_pad = 3 - len(z.split('.')[1])
                z += '0'*zero_pad
                x_gap = 12 - len(x)
                line.append(' '*x_gap)
                line.append(x)
                y_gap = 8 - len(y)
                line.append(' '*y_gap)
                line.append(y)
                z_gap = 8 - len(z)
                line.append(' '*z_gap)
                line.append(z)
                line.append('  1.00  0.00           ')
                line.append(atom_to_atom[atom])
                line.append('  \n')
                f.write("".join(line))
            res_count += 1
        f.write('ENDMDL\n')
    return True
            
                

if __name__ == '__main__':
    cols = ['inp.pyd']
    dataset = wds.WebDataset('/home/fosterb/rla/wds_dir/capri/4JW3.wds').decode().to_tuple(*cols)

    for d in dataset:
        write_pdb(d, '/mnt/shared/foldseek_trainset/4JW3')



# def write_pdb(d, pdb_path):
#     three_to_one = {
#         'A': 'ALA',
#         'C': 'CYS',
#         'D': 'ASP',
#         'E': 'GLU',
#         'F': 'PHE',
#         'G': 'GLY',
#         'H': 'HIS',
#         'I': 'ILE',
#         'K': 'LYS',
#         'L': 'LEU',
#         'M': 'MET',
#         'N': 'ASN',
#         'P': 'PRO',
#         'Q': 'GLN',
#         'R': 'ARG',
#         'S': 'SER',
#         'T': 'THR',
#         'V': 'VAL',
#         'W': 'TRP',
#         'Y': 'TYR',
#         '-': '---',
#         'U': 'SEC',
#         'O': 'PYL',
#         'B': 'ASX',
#         'Z': 'GLX', 
#         'X': 'XAA',
#         'J': 'XLE'
#     }
#     atom_to_atom = {'N': 'N', 'CA': 'C', 'C': 'C', 'O': 'O'}
    
#     pdb_id = d[0]['pdb_id']
#     if '.pdb' in pdb_id:
#         pdb_file = os.path.join(pdb_path, pdb_id)
#     else:
#         pdb_file = os.path.join(pdb_path, pdb_id + '.pdb')
#     if os.path.exists(pdb_file):
#         return False 
#     with open(pdb_file, 'w') as f:
#         f.write('MODEL        1\n')
#         atom_count = 1
#         res_count = 1
#         for chain_num, chain_id in enumerate(d[0]['chain_ids']):
#             for res in range(len(d[0]['seqs'][chain_num])):
#                 for atom_id, atom in enumerate(['N', 'CA', 'C', 'O']):
#                     line = ['ATOM']
#                     atom_count_gap = 7 - len(str(atom_count))
#                     line.append(' '*atom_count_gap)
#                     line.append(str(atom_count))
#                     atom_count += 1
#                     line.append('  ')
#                     line.append(atom + ' '*(4 - len(atom)))
#                     try:
#                         line.append(three_to_one[d[0]['seqs'][chain_num][res]])
#                     except:
#                         line.append('XAA')
#                         print(len(d[0]['seqs'][chain_num]))
#                         print(d[0]['coords'][chain_num].shape)
#                         print(d[0]['pdb_id'])
#                         raise ValueError
#                     line.append(' ')
#                     line.append(chain_id)
#                     res_count_gap = 4 - len(str(res_count))
#                     line.append(' '*res_count_gap)
#                     line.append(str(res_count))
#                     x, y, z = d[0]['coords'][chain_num][res][atom_id]
#                     x = str(x)
#                     y = str(y)
#                     z = str(z)
#                     zero_pad = 3 - len(x.split('.')[1])
#                     x += '0'*zero_pad
#                     zero_pad = 3 - len(y.split('.')[1])
#                     y += '0'*zero_pad
#                     zero_pad = 3 - len(z.split('.')[1])
#                     z += '0'*zero_pad
#                     x_gap = 12 - len(x)
#                     line.append(' '*x_gap)
#                     line.append(x)
#                     y_gap = 8 - len(y)
#                     line.append(' '*y_gap)
#                     line.append(y)
#                     z_gap = 8 - len(z)
#                     line.append(' '*z_gap)
#                     line.append(z)
#                     line.append('  1.00  0.00           ')
#                     line.append(atom_to_atom[atom])
#                     line.append('  \n')
#                     f.write("".join(line))
#                 res_count += 1
#             f.write('TER')
#             atom_count_gap = 8 - len(str(atom_count))
#             f.write(' '*atom_count_gap)
#             f.write(str(atom_count))
#             f.write('      ')
#             try:
#                 line.append(three_to_one[d[0]['seqs'][chain_num][res]])
#             except:
#                 line.append('XAA')
#                 print(len(d[0]['seqs'][chain_num]))
#                 print(d[0]['coords'][chain_num].shape)
#                 print(d[0]['pdb_id'])
#                 raise ValueError
#             f.write(' ')
#             f.write(chain_id)
#             res_count_gap = 4 - len(str(res_count))
#             f.write(' '*res_count_gap)
#             f.write(str(res_count) + '\n')
#         f.write('ENDMDL\n')
#     return True