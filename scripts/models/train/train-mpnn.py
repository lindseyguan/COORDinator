import argparse
import os.path
import json, time, os, sys, glob
import shutil
import warnings
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
torch.multiprocessing.set_sharing_strategy('file_system')
import queue
import copy
import torch.nn as nn
import torch.nn.functional as F
import random
import os.path
import esm as esmlib
import subprocess
from concurrent.futures import ProcessPoolExecutor    
from terminator.data.mpnn_data import worker_init_fn, get_pdbs, loader_pdb, build_training_clusters, PDB_dataset, StructureDataset, StructureLoader, StructureSampler
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from terminator.data.data import (WDSBatchSamplerWrapper, TERMLazyDataset, TERMBatchSamplerWrapper, TERMDataset, TERMLazyBatchSampler)
from terminator.models.TERMinator import TERMinator
from terminator.utils.model.loop_utils import run_epoch
from terminator.utils.model.loss_fn import construct_loss_fn
import webdataset as wds
# for autosummary import purposes
# pylint: disable=wrong-import-order,wrong-import-position
sys.path.insert(0, os.path.dirname(__file__))
from terminator.utils.model.default_hparams import DEFAULT_MODEL_HPARAMS, DEFAULT_TRAIN_HPARAMS
from terminator.utils.model.optim import get_std_opt
from train import _setup_dataloaders_wds, _setup_hparams, _setup_model, _load_checkpoint

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

def main(args):
    dev = args.dev
    if torch.cuda.device_count() == 0:
        dev = "cpu"
    run_dir = args.run_dir
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)

    # setup dataloaders
    data_path = args.dataset
    args.rescut = 3.5
    args.max_protein_length = 4000
    args.num_examples_per_epoch = 500000
    args.debug = False
    args.batch_size = 4000
    params = {
        "LIST"    : f"{data_path}/list.csv", 
        "VAL"     : f"{data_path}/valid_clusters.txt",
        "TEST"    : f"{data_path}/test_clusters.txt",
        "DIR"     : f"{data_path}",
        "DATCUT"  : "2030-Jan-01",
        "RESCUT"  : args.rescut, #resolution cutoff for PDBs
        "HOMO"    : 0.70 #min seq.id. to detect homo chains
    }


    LOAD_PARAM = {'batch_size': 1,
                  'shuffle': True,
                  'pin_memory':False,
                  'num_workers': 4}

    print('loaded args')
    train, valid, test = build_training_clusters(params, args.debug)
    print('build clusters')
     
    train_set = PDB_dataset(list(train.keys()), loader_pdb, train, params)
    train_loader = DataLoader(train_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)
    valid_set = PDB_dataset(list(valid.keys()), loader_pdb, valid, params)
    valid_loader = DataLoader(valid_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)

    kwargs = {}
    kwargs['num_workers'] = 32
    print('loaded loaders')

    # load checkpoint
    model_hparams, run_hparams = _setup_hparams(args)
    esm_options = {'use_esm_attns': run_hparams['use_esm_attns'], 'esm_embed_layer': model_hparams['esm_embed_layer'], 'from_rla': model_hparams['rla_feats'],
                   'use_reps': run_hparams['use_reps'], 'connect_chains': run_hparams['connect_chains'], 'one_hot': model_hparams['one_hot']
                   }
    checkpoint_dict = _load_checkpoint(run_dir, dev, run_hparams['finetune'])
    best_validation = checkpoint_dict["best_validation"]
    best_checkpoint = checkpoint_dict["best_checkpoint"]
    start_epoch = checkpoint_dict["start_epoch"]
    last_optim_state = checkpoint_dict["last_optim_state"]
    writer = checkpoint_dict["writer"]
    training_curves = checkpoint_dict["training_curves"]

    isDataParallel = True if torch.cuda.device_count() > 1 and dev != "cpu" else False
    finetune = run_hparams["finetune"]

    # construct terminator, loss fn, and optimizer
    use_esm = model_hparams['nodes_to_probs'] or model_hparams['edges_to_seq']
    load_esm = False
    for loss_key in run_hparams['loss_config'].keys():
        if loss_key.find('esm_cmap') > -1 or loss_key.find('esm_loss') > -1:
            use_esm = not model_hparams['mpnn_decoder']
            load_esm = model_hparams['mpnn_decoder']
            break
    terminator, terminator_module = _setup_model(model_hparams, run_hparams, best_checkpoint, dev, use_esm, model_hparams['edges_to_seq'])
    loss_fn = construct_loss_fn(run_hparams)
    optimizer = get_std_opt(terminator.parameters(),
                            d_model=model_hparams['energies_hidden_dim'],
                            regularization=run_hparams['regularization'],
                            state=last_optim_state,
                            finetune=finetune,
                            finetune_lr=run_hparams["finetune_lr"],
                            warmup=run_hparams['warmup'],
                            lr_multiplier=run_hparams['lr_multiplier'])
    best_val_loss = np.Inf
    iters_since_best_loss = 0
    finetune_loss = 'evcouplings_loss_corr'
    if finetune_loss not in run_hparams['loss_config'].keys():
        finetune_loss = 'rocklin_loss'
    print('set up model and loss')
    # if loss involves esm, load model
    use_esm = use_esm or model_hparams['esm_feats']
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
            esm = esm.to('cpu')
        for p in esm.parameters():
            p.requires_grad = False
        for p in converter.parameters():
            p.requires_grad = False
        esm.eval()
        converter.eval()
    else:
        esm = None
        batch_converter = None
        converter = None
    if run_hparams['num_recycles'] > 0:
        rec_esm = copy.deepcopy(esm)
        rec_esm = rec_esm.to(dev)
    print('Set up ESM (if needed): ', use_esm)

    # with ProcessPoolExecutor(max_workers=12) as executor:
    #     print('entering process pool')
    #     q = queue.Queue(maxsize=3)
    #     p = queue.Queue(maxsize=3)
    #     for i in range(3):
    #         print('it, ', i)
    #         q.put_nowait(executor.submit(get_pdbs, train_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
    #         p.put_nowait(executor.submit(get_pdbs, valid_loader, 1, args.max_protein_length, args.num_examples_per_epoch))

    #     pdb_dict_train = q.get().result()
    #     pdb_dict_valid = p.get().result()

    pdb_dict_train = get_pdbs(train_loader, 1, args.max_protein_length, args.num_examples_per_epoch)
    pdb_dict_valid = get_pdbs(valid_loader, 1, args.max_protein_length, args.num_examples_per_epoch)

    dataset_train = StructureDataset(pdb_dict_train, truncate=None, max_length=args.max_protein_length) 
    dataset_valid = StructureDataset(pdb_dict_valid, truncate=None, max_length=args.max_protein_length)

    print('Got datasets!')
    
    # loader_train = StructureLoader(dataset_train, batch_size=args.batch_size)
    mask_options = {
        'sc_mask_schedule': run_hparams['sc_mask_schedule'],
        'base_sc_mask': run_hparams['base_sc_mask'],
        'sc_mask_rate': run_hparams['sc_mask_rate'],
        'mask_neighbors': run_hparams['mask_neighbors'],
        'mask_interface': run_hparams['mask_interface'],
        'half_interface': run_hparams['half_interface'],
        'inter_cutoff': run_hparams['inter_cutoff'],
    }
    ldev = 'cpu'
    train_batch_sampler = StructureSampler(dataset_train, batch_size=args.batch_size, device=ldev,
                                           flex_type=run_hparams['flex_type'], augment_eps=run_hparams['noise_level'], replicate=run_hparams['replicate'], mask_options=mask_options, esm=esm, batch_converter=batch_converter, noise_lim=run_hparams['noise_lim'], use_sc=run_hparams['use_sc'], convert_to_esm=model_hparams['convert_to_esm'], use_esm_attns=run_hparams['use_esm_attns'], esm_embed_layer=model_hparams['esm_embed_layer'], from_rla=model_hparams['rla_feats'], use_reps=run_hparams['use_reps'], connect_chains=run_hparams['connect_chains'], one_hot=model_hparams['one_hot'], post_esm_mask=run_hparams['post_esm_mask'])
    loader_train = DataLoader(dataset_train, batch_sampler=train_batch_sampler, collate_fn=train_batch_sampler.package, pin_memory=True, **kwargs)
    # loader_valid = StructureLoader(dataset_valid, batch_size=args.batch_size)
    valid_batch_sampler = StructureSampler(dataset_valid, batch_size=args.batch_size, device=ldev,
                                           flex_type=run_hparams['flex_type'], augment_eps=run_hparams['noise_level'], replicate=run_hparams['replicate'], mask_options=mask_options, esm=esm, batch_converter=batch_converter, noise_lim=run_hparams['noise_lim'], use_sc=run_hparams['use_sc'], convert_to_esm=model_hparams['convert_to_esm'], use_esm_attns=run_hparams['use_esm_attns'], esm_embed_layer=model_hparams['esm_embed_layer'], from_rla=model_hparams['rla_feats'], use_reps=run_hparams['use_reps'], connect_chains=run_hparams['connect_chains'], one_hot=model_hparams['one_hot'], post_esm_mask=run_hparams['post_esm_mask'])
    loader_valid = DataLoader(dataset_valid, batch_sampler=valid_batch_sampler, collate_fn=valid_batch_sampler.package, pin_memory=True, **kwargs)
    
    reload_c = 0 
    print('got dataloaders!')
    reload_c = 0
    for epoch in range(start_epoch, args.epochs):
        print('epoch', epoch)
        if epoch % run_hparams['reload_data_every_n_epochs'] == 0:
            if reload_c != 0:
                # pdb_dict_train = q.get().result()
                # dataset_train = StructureDataset(pdb_dict_train, truncate=None, max_length=args.max_protein_length)
                # train_batch_sampler = StructureSampler(dataset_train, batch_size=args.batch_size, device=dev, flex_type=args.noise_type, augment_eps=args.backbone_noise, replicate=args.replicate)
                # loader_train = DataLoader(dataset_train, batch_sampler=train_batch_sampler, collate_fn=train_batch_sampler.package, pin_memory=True, **kwargs)
                # pdb_dict_valid = p.get().result()
                # dataset_valid = StructureDataset(pdb_dict_valid, truncate=None, max_length=args.max_protein_length)
                # valid_batch_sampler = StructureSampler(dataset_valid, batch_size=args.batch_size, device=dev)
                # loader_valid = DataLoader(dataset_valid, batch_sampler=valid_batch_sampler, collate_fn=valid_batch_sampler.package, pin_memory=True, **kwargs)
                # q.put_nowait(executor.submit(get_pdbs, train_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
                # p.put_nowait(executor.submit(get_pdbs, valid_loader, 1, args.max_protein_length, args.num_examples_per_epoch))


                pdb_dict_train = get_pdbs(train_loader, 1, args.max_protein_length, args.num_examples_per_epoch)
                pdb_dict_valid = get_pdbs(valid_loader, 1, args.max_protein_length, args.num_examples_per_epoch)
                dataset_train = StructureDataset(pdb_dict_train, truncate=None, max_length=args.max_protein_length) 
                dataset_valid = StructureDataset(pdb_dict_valid, truncate=None, max_length=args.max_protein_length)
                
                # loader_train = StructureLoader(dataset_train, batch_size=args.batch_size)
                train_batch_sampler = StructureSampler(dataset_train, batch_size=args.batch_size, 
                                                       device=ldev, flex_type=run_hparams['flex_type'], augment_eps=run_hparams['noise_level'], replicate=run_hparams['replicate'], mask_options=mask_options, esm=esm, batch_converter=batch_converter,  noise_lim=run_hparams['noise_lim'], use_sc=run_hparams['use_sc'], convert_to_esm=model_hparams['convert_to_esm'], use_esm_attns=run_hparams['use_esm_attns'], esm_embed_layer=model_hparams['esm_embed_layer'], from_rla=model_hparams['rla_feats'], use_reps=run_hparams['use_reps'], connect_chains=run_hparams['connect_chains'], one_hot=model_hparams['one_hot'], post_esm_mask=run_hparams['post_esm_mask'])
                loader_train = DataLoader(dataset_train, batch_sampler=train_batch_sampler, collate_fn=train_batch_sampler.package, pin_memory=True, **kwargs)
                # loader_valid = StructureLoader(dataset_valid, batch_size=args.batch_size)
                valid_batch_sampler = StructureSampler(dataset_valid, batch_size=args.batch_size, 
                                                       device=ldev, flex_type=run_hparams['flex_type'], augment_eps=run_hparams['noise_level'], replicate=run_hparams['replicate'], mask_options=mask_options,esm=esm, batch_converter=batch_converter, noise_lim=run_hparams['noise_lim'], use_sc=run_hparams['use_sc'], convert_to_esm=model_hparams['convert_to_esm'], use_esm_attns=run_hparams['use_esm_attns'], esm_embed_layer=model_hparams['esm_embed_layer'], from_rla=model_hparams['rla_feats'], use_reps=run_hparams['use_reps'], connect_chains=run_hparams['connect_chains'], one_hot=model_hparams['one_hot'], post_esm_mask=run_hparams['post_esm_mask'])
                loader_valid = DataLoader(dataset_valid, batch_sampler=valid_batch_sampler, collate_fn=valid_batch_sampler.package, pin_memory=True, **kwargs)
            reload_c += 1
        train_batch_sampler._set_epoch(epoch)
        valid_batch_sampler._set_epoch(epoch)

        epoch_loss, epoch_ld, _ = run_epoch(terminator, loader_train, loss_fn, run_hparams, epoch=epoch, optimizer=optimizer, grad=True, dev=dev, finetune=finetune, isDataParallel=isDataParallel, esm=rec_esm, batch_converter=batch_converter, converter=converter, num_recycles=run_hparams['num_recycles'], keep_seq=run_hparams['keep_seq_recycle'], esm_options=esm_options, only_loss_recycle=run_hparams['only_loss_recycle'])
        print('epoch loss', epoch_loss, 'epoch_ld', epoch_ld)
        writer.add_scalar('training loss', epoch_loss, epoch)

        # validate
        val_loss, val_ld, _ = run_epoch(terminator, loader_valid, loss_fn, epoch=epoch, grad=False, dev=dev, esm=rec_esm, batch_converter=batch_converter, converter=converter, num_recycles=run_hparams['num_recycles'], keep_seq=run_hparams['keep_seq_recycle'], esm_options=esm_options, only_loss_recycle=run_hparams['only_loss_recycle'])
        print('val loss', val_loss, 'val ld', val_ld)
        writer.add_scalar('val loss', val_loss, epoch)

        training_curves["train_loss"].append((epoch_loss, epoch_ld))
        training_curves["val_loss"].append((val_loss, val_ld))

        comp = (val_ld[finetune_loss]['loss'] < best_validation) if finetune else (val_loss < best_validation)

        # save a state checkpoint
        checkpoint_state = {
            'epoch': epoch,
            'state_dict': terminator_module.state_dict(),
            'best_model': comp, # (val_loss < best_validation)
            'val_loss': best_validation,
            'optimizer_state': optimizer.state_dict(),
            'training_curves': training_curves
        }
        torch.save(checkpoint_state, os.path.join(run_dir, 'net_last_checkpoint.pt'))
        if comp: # if (val_loss < best_validation)
            if finetune:
                best_validation = val_ld[finetune_loss]['loss']
            else:
                best_validation = val_loss
            best_checkpoint = copy.deepcopy(terminator_module.state_dict())
            torch.save(checkpoint_state, os.path.join(run_dir, 'net_best_checkpoint.pt'))

        if val_loss < best_val_loss:
            val_loss = best_val_loss
            iters_since_best_loss = 0
        else:
            iters_since_best_loss += 1
        if iters_since_best_loss > run_hparams['patience']:
            print("Exiting due to early stopping")
            break
    # save model params
    print(training_curves)
    torch.save(terminator_module.state_dict(), os.path.join(run_dir, 'net_last.pt'))
    torch.save(best_checkpoint, os.path.join(run_dir, 'net_best.pt'))

    # test
    writer.flush()
    writer.close()


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
    parser.add_argument('--msa_dataset', help='input msa folder .features files in proper directory structure.', default=None)
    parser.add_argument('--n_trials', help="number of trials for optuna optimization", default=0, type=int)
    parser.add_argument('--converter_path', help='Path to model converting coord to esm embeddings', default='/home/gridsan/fbirnbaum/TERMinator/analysis/converter_model_coord.pt')
    parsed_args = parser.parse_args()
    print('args: ')
    print(parsed_args)

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

    main(parsed_args)