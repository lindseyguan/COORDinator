import sys
import os
import re
import copy as vcopy
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import argparse
import shutil
import copy
import datetime
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import esm as esmlib
from finetune import _load_checkpoint
from train import _setup_hparams, _setup_dataloaders_wds, _setup_model
from terminator.utils.common import ints_to_seq_torch
from terminator.models.layers.utils import _esm_featurize, extract_knn, gather_edges
from terminator.utils.model.optim import get_std_opt
from terminator.utils.model.loop_utils import run_epoch_ener_ft
import terminator.models.load_model as load_model


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

## Create converter
class Converter(nn.Module):
    def __init__(self, h_in, h_out):
        super(Converter, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(h_in, h_out))
        self.layers.append(nn.Linear(h_out, h_out))
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x
    

class model_args():
    def __init__(self, dataset, model_dir, dev='cpu', subset = False, subset_list=[], sc_mask_rate=0, sc_mask=[], mask_neighbors=False, use_sc=True, mpnn_dihedrals=True, no_mask=False, sc_screen=False, sc_screen_range=[], replace_muts=False, data_only=False, verbose=False):
        self.dataset = dataset
        self.model_dir = model_dir
        self.dev = dev
        self.subset = subset
        self.subset_list = subset_list
        self.sc_mask = sc_mask
        self.sc_mask_rate = sc_mask_rate
        self.base_sc_mask = 0.0
        self.mask_neighbors = mask_neighbors
        self.use_sc = use_sc
        self.mpnn_dihedrals = mpnn_dihedrals
        self.no_mask = no_mask
        self.sc_info = 'all'
        self.sc_screen = sc_screen
        self.sc_screen_range = sc_screen_range
        self.replace_muts = replace_muts
        self.data_only = data_only
        self.verbose = verbose
        self.checkpoint = 'net_best_checkpoint.pt'

# Run finetuning based on deviation between predicted energies and measured energies for bcl2 data
def run_finetune(args):
    if torch.cuda.device_count() == 0:
        dev = "cpu"
    else:
        dev = "cuda:0"
    ## Set up reading of experimental data
    

    # zero is used as padding
    AA_to_int = {'A': 1,'ALA': 1,'C': 2,'CYS': 2,'D': 3,'ASP': 3,'E': 4,'GLU': 4,'F': 5,'PHE': 5,'G': 6,'GLY': 6,'H': 7,'HIS': 7,'I': 8,'ILE': 8,'K': 9,'LYS': 9,'L': 10,'LEU': 10,'M': 11,'MET': 11,'N': 12,'ASN': 12,'P': 13,'PRO': 13,'Q': 14,'GLN': 14,'R': 15,'ARG': 15,'S': 16,'SER': 16,'T': 17,'THR': 17,'V': 18,
    'VAL': 18,'W': 19,'TRP': 19,'Y': 20,'TYR': 20,'X': 21}
    ## amino acid to integer
    atoi = {key: val - 1 for key, val in AA_to_int.items()}

    #Benchmark dataset from SORTCERY
    all_name = ["x1","m1","f100"]
    #New name
    name_test = ["B2CL1_SORTCERY","MCL1_SORTCERY","B2LA1_SORTCERY"]
    

    # load data
    kwargs = {}
    kwargs['num_workers'] = 12
    model_hparams, run_hparams = _setup_hparams(args)
    esm_options = {'use_esm_attns': run_hparams['use_esm_attns'], 'esm_embed_layer': model_hparams['esm_embed_layer'], 'from_rla': model_hparams['rla_feats'],
                   'use_reps': run_hparams['use_reps'], 'connect_chains': run_hparams['connect_chains'], 'one_hot': model_hparams['one_hot']
                   }
    train_dataset, train_batch_sampler, val_dataset, val_batch_sampler, test_dataset, test_batch_sampler = _setup_dataloaders_wds(args, run_hparams, model_hparams,dataset_name=run_hparams['dataset_name'], suffix=run_hparams['suffix'])
    train_dataloader = DataLoader(train_dataset,
                                  batch_sampler=train_batch_sampler,
                                  collate_fn=train_batch_sampler.package,
                                  pin_memory=True,
                                  **kwargs)
    val_dataloader = DataLoader(val_dataset,
                                batch_sampler=val_batch_sampler,
                                collate_fn=val_batch_sampler.package,
                                pin_memory=True,
                                **kwargs)
    
    #Dictionnary of energy (exp or predicted)
    if run_hparams['data_source'] == 'bcl2':
        pep_bind_ener = dict()
        bench_dir = "/home/gridsan/fbirnbaum/TERMinator/analysis/energetics_and_finetuning/bcl2_benchmark/SORTCERY_data"

        #Get experimental energy values from sortcery
        for (t,name) in zip(all_name,name_test):
            subdf = pd.read_csv(os.path.join(bench_dir,t+"_merged.csv"))
            if name not in pep_bind_ener:
                pep_bind_ener[name] = dict()
            for (seq,val) in zip(subdf["protein"],subdf[t+"_mean_ener"]):
                pep_bind_ener[name][seq] = float(val)
        train_dataset_path = '/data1/groups/keatinglab/fosterb/bcl2_data/bcl2_train_sort.wds'
        val_dataset_path = '/data1/groups/keatinglab/fosterb/bcl2_data/bcl2_val_sort.wds'
    elif run_hparams['data_source'] == 'rocklin':
        dataset_file = '/data1/groups/keating_madry/rocklin_data/Processed_K50_dG_datasets/K50_dG_Dataset1_Dataset2.csv'
        pep_bind_ener = pd.read_csv(dataset_file)
        pep_bind_ener = pep_bind_ener[pep_bind_ener['dG_ML']!='-'].copy(deep=True) # Drop unreliable data
        pep_bind_ener = pep_bind_ener[pep_bind_ener['ddG_ML']!='-'].copy(deep=True)
        pep_bind_ener = pep_bind_ener[pep_bind_ener['Stabilizing_mut']!='-'].copy(deep=True) # JFM
        pep_bind_ener['dG_ML'] = -1*pd.to_numeric(pep_bind_ener['dG_ML'])
        pep_bind_ener['ddG_ML'] = -1*pd.to_numeric(pep_bind_ener['ddG_ML'])
        pep_bind_ener = pep_bind_ener[~pep_bind_ener['name'].str.contains('ins')].copy(deep=True) # Drop indels 
        pep_bind_ener = pep_bind_ener[~pep_bind_ener['name'].str.contains('del')].copy(deep=True)
        pep_bind_ener = pep_bind_ener[~pep_bind_ener['name'].str.contains('_con')].copy(deep=True) # Drop not modeled proteins
        train_dataset_path = '/data1/groups/keatinglab/fosterb/rocklin_data_2022/rocklin_train_thermo_single.wds'
        val_dataset_path = '/data1/groups/keatinglab/fosterb/rocklin_data_2022/rocklin_val_thermo_single.wds'

    # load checkpoint
    use_esm = model_hparams['nodes_to_probs'] or model_hparams['edges_to_seq']
    load_esm = False
    for loss_key in run_hparams['loss_config'].keys():
        if loss_key.find('esm_cmap') > -1 or loss_key.find('esm_loss') > -1:
            use_esm = not model_hparams['mpnn_decoder']
            load_esm = model_hparams['mpnn_decoder']
            break
    print('use esm: ', use_esm)
    print('finetune: ', run_hparams['finetune'])
    checkpoint_dict = _load_checkpoint(args.run_dir, dev, model_hparams, True, top_nodes=use_esm, use_esm=use_esm)
    best_validation = checkpoint_dict["best_validation"]
    best_checkpoint = checkpoint_dict["best_checkpoint"]
    if 'top.features.node_layers.0.weight' not in best_checkpoint.keys():
        model_hparams['old'] = True
    else:
        model_hparams['old'] = False
    last_optim_state = checkpoint_dict["last_optim_state"]
    writer = checkpoint_dict["writer"]
    training_curves = checkpoint_dict["training_curves"]

    isDataParallel = True if torch.cuda.device_count() > 1 and dev != "cpu" else False
    finetune = run_hparams["finetune"]

    # construct terminator, loss fn, and optimizer
    
    terminator, terminator_module = _setup_model(model_hparams, run_hparams, best_checkpoint, dev, use_esm, model_hparams['edges_to_seq'])
    loss_fn = nn.MSELoss()
    optimizer = get_std_opt(terminator.parameters(),
                            d_model=model_hparams['energies_hidden_dim'],
                            regularization=run_hparams['regularization'],
                            state=None,
                            finetune=finetune,
                            finetune_lr=run_hparams["finetune_lr"],
                            warmup=run_hparams['warmup'],
                            lr_multiplier=run_hparams['lr_multiplier'])
    best_val_loss = np.Inf
    iters_since_best_loss = 0
     
    if model_hparams['esm_feats']:
        if model_hparams['esm_model'] == '650':
            esm, alphabet = esmlib.pretrained.esm2_t33_650M_UR50D()
        else:
            esm, alphabet = esmlib.pretrained.esm2_t30_150M_UR50D()
        esm = esm.to(device='cuda:0')

    train_args = model_args(dataset = train_dataset_path,
                    model_dir = args.run_dir,
                    sc_screen=run_hparams['sc_screen'],
                    sc_screen_range=run_hparams['sc_screen_range'],
                    sc_mask_rate=run_hparams['sc_mask_rate'],
                    sc_mask=run_hparams['sc_mask'],
                    replace_muts=run_hparams['replace_muts'],
                    mask_neighbors=run_hparams['mask_neighbors'],
                    dev='cuda:0',
                     ) 
    valid_args = copy.deepcopy(train_args)
    valid_args.dataset = val_dataset_path

    if use_esm or run_hparams['num_recycles'] > 0 or load_esm:
        esm, alphabet = esmlib.pretrained.esm2_t30_150M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        converter_state = torch.load(args.converter_path)
        converter = Converter(22, 640)
        converter.load_state_dict(converter_state)
        if dev == 'cpu':
            esm = esm.float()  # convert to fp32 as ESM-2 in fp16 is not supported on CPU
            esm = esm.cpu()
            converter = converter.cpu()
        else:
            esm = esm.float()
            esm = esm.to(device='cuda:0')
            converter = converter.to(device='cuda:0')
            # esm = esm.to('cpu')
        esm.eval()
        converter.eval()
    else:
        esm = None
        batch_converter = None
        converter = None
    start_epoch = 0

    try:
        for epoch in range(args.epochs):
            print('epoch: ', epoch)
            train_batch_sampler._set_epoch(epoch)
            val_batch_sampler._set_epoch(epoch)
            test_batch_sampler._set_epoch(epoch)

            # Train
            epoch_loss = run_epoch_ener_ft(terminator, optimizer, train_dataloader, pep_bind_ener, train_args, run_hparams, model_hparams, loss_fn, dev, grad=True, finetune=run_hparams['finetune'], load_model=load_model, epoch=epoch, test=False, num_recycles=run_hparams['num_recycles'], keep_seq=run_hparams['keep_seq_recycle'], keep_sc_mask_recycle=run_hparams['keep_sc_mask_recycle'], data_source=run_hparams['data_source'], esm=esm, batch_converter=batch_converter, converter=converter, esm_options=esm_options, ener_finetune_mlp=model_hparams['ener_finetune_mlp'])
            
            print('train loss: ', epoch_loss)
            writer.add_scalar('training loss', epoch_loss)
            training_curves["train_loss"].append((epoch_loss, None))

            # Validate
            val_loss = run_epoch_ener_ft(terminator, optimizer, val_dataloader, pep_bind_ener, valid_args, run_hparams, model_hparams, loss_fn, dev, grad=False, finetune=run_hparams['finetune'], load_model=load_model, epoch=epoch, test=False, num_recycles=run_hparams['num_recycles'], keep_seq=run_hparams['keep_seq_recycle'], keep_sc_mask_recycle=run_hparams['keep_sc_mask_recycle'], data_source=run_hparams['data_source'], esm=esm, batch_converter=batch_converter, converter=converter, esm_options=esm_options, ener_finetune_mlp=model_hparams['ener_finetune_mlp'])
            print('val loss: ', val_loss)

            writer.add_scalar('val loss', val_loss)
            training_curves["val_loss"].append((val_loss, None))

            comp = (val_loss < best_validation)
            if comp: # if (val_loss < best_validation)
                best_validation = val_loss
                


            # save a state checkpoint
            checkpoint_state = {
                'epoch': epoch,
                'state_dict': terminator_module.state_dict(),
                'best_model': comp, # (val_loss < best_validation)
                'val_loss': best_validation,
                'optimizer_state': optimizer.state_dict(),
                'training_curves': training_curves
            }
            torch.save(checkpoint_state, os.path.join(args.run_dir, 'net_last_checkpoint.pt'))
            if comp:
                best_checkpoint = copy.deepcopy(terminator_module.state_dict())
                torch.save(checkpoint_state, os.path.join(args.run_dir, 'net_best_checkpoint.pt'))

            if val_loss < best_val_loss:
                val_loss = best_val_loss
                iters_since_best_loss = 0
            else:
                iters_since_best_loss += 1
            if iters_since_best_loss > run_hparams['patience']:
                print("Exiting due to early stopping")
                break




    except KeyboardInterrupt:
        pass

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train TERMinator!')
    parser.add_argument('--dataset', help='input folder .features files in proper directory structure.', required=True)
    parser.add_argument('--model_hparams', help='file path for model hparams', required=True)
    parser.add_argument('--run_hparams', help='file path for run hparams', required=True)
    parser.add_argument('--run_dir', help='path to place folder to store model files', required=True)
    parser.add_argument('--train', help='file with training dataset split')
    parser.add_argument('--validation', help='file with validation dataset split')
    parser.add_argument('--test', help='file with test dataset split')
    parser.add_argument('--out_dir',
                        help='path to place test set eval results (e.g. net.out). If not set, default to --run_dir')
    parser.add_argument('--dev', help='device to train on', default='cuda:0')
    parser.add_argument('--epochs', help='number of epochs to train for', default=100, type=int)
    parser.add_argument('--lazy', help="use lazy data loading", action='store_true')
    parser.add_argument('--n_nodes', help="number of cores for use in ddp", default=1, type=int)
    parser.add_argument("--backend", help="Backend for DDP", type=str, default="gloo")
    parser.add_argument('--orig_model', help='Baseline model to finetune', default=None)
    parser.add_argument('--n_trials', help="number of trials for optuna optimization", default=0, type=int)
    parser.add_argument('--converter_path', help='Path to model converting coord to esm embeddings', default='/home/gridsan/fbirnbaum/TERMinator/analysis/converter_model_coord.pt')
    parsed_args = parser.parse_args()
    print('args: ')
    print(parsed_args)

    shutil.copy2(parsed_args.orig_model, os.path.join(parsed_args.run_dir, 'net_original.pt'))

    # by default, if no splits are provided, read the splits from the dataset folder
    if parsed_args.train is None:
        parsed_args.train = os.path.join(parsed_args.dataset, 'train.in')
    if parsed_args.validation is None:
        parsed_args.validation = os.path.join(parsed_args.dataset, 'validation.in')
    if parsed_args.test is None:
        parsed_args.test = os.path.join(parsed_args.dataset, 'test.in')


    # setup ddp
    if parsed_args.dev == "cpu":
        local_rank = -1
    else:
        local_rank = int(os.environ["LOCAL_RANK"])
    print(f'using {parsed_args.n_nodes} nodes, local rank {local_rank}')
    if parsed_args.n_nodes > 1 and local_rank != -1:
        dist.init_process_group(backend=parsed_args.backend, init_method='env://', timeout=datetime.timedelta(0, 72000))
        torch.cuda.set_device(local_rank)
        torch.backends.cudnn.benchmark = True
    if dist.is_initialized():
        parsed_args.dev = torch.device("cuda")
        print(f"Local rank: {local_rank}")
        print(f"Device rank: {dist.get_rank()}")
        print(f"World size: {dist.get_world_size()}")
        parsed_args.workers = dist.get_world_size()
        device_rank = dist.get_rank()
        parsed_args.distributed = True
    else:
        parsed_args.workers = 1
        device_rank = -1
        parsed_args.distributed = False

    run_finetune(parsed_args)