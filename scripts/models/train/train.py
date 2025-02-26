"""Train TERMinator model.

Usage:
    .. code-block::

        python train.py \\
            --dataset <dataset_dir> \\
            --model_hparams <model_hparams_file_path> \\
            --run_hparams <run_hparams_file_path> \\
            --run_dir <run_dir> \\
            [--train <train_split_file>] \\
            [--validation <val_split_file>] \\
            [--test <test_split_file>] \\
            [--out_dir <out_dir>] \\
            [--dev <device>] \\
            [--epochs <num_epochs>]
            [--lazy]

    If :code:`--out_dir <out_dir>` is not set, :code:`net.out` will be dumped
    into :code:`<run_dir>`.

    For any of the split files, if the option is not provided, :code:`train.py` will
    look for them within :code:`<dataset_dir>`.

See :code:`python train.py --help` for more info.
"""

import argparse
import copy
import json
import os
import pickle
import sys
import datetime
import numpy as np
import esm as esmlib
from transformers import AutoModelForMaskedLM
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from terminator.data.data import (WDSBatchSamplerWrapper, TERMLazyDataset, TERMBatchSamplerWrapper, TERMDataset, TERMLazyBatchSampler)
from terminator.models.TERMinator import TERMinator
from terminator.utils.model.loop_utils import run_epoch
from terminator.utils.model.loss_fn import construct_loss_fn
from terminator.utils.common import backwards_compat, Converter
import webdataset as wds
# for autosummary import purposes
# pylint: disable=wrong-import-order,wrong-import-position
sys.path.insert(0, os.path.dirname(__file__))
from terminator.utils.model.default_hparams import DEFAULT_MODEL_HPARAMS, DEFAULT_TRAIN_HPARAMS
from terminator.utils.model.optim import get_std_opt

# pylint: disable=unspecified-encoding


torch.set_printoptions(threshold=10000)
torch.set_printoptions(linewidth=1000)
torch.set_printoptions(precision=2)
torch.set_printoptions(sci_mode=False)
torch.set_printoptions(profile="full")

def _load_hparams(hparam_path, default_hparams, output_name, run_dir=None):
    # load hparams
    hparams = json.load(open(hparam_path, 'r'))
    for key, default_val in default_hparams.items():
        if key not in hparams:
            hparams[key] = default_val

    if run_dir is None:
        return hparams
    
    hparams_path = os.path.join(run_dir, output_name)
    if os.path.isfile(hparams_path):
        previous_hparams = json.load(open(hparams_path, 'r'))
        for key, default_val in default_hparams.items():
            if key not in previous_hparams and key in hparams:
                previous_hparams[key] = default_val
        if previous_hparams != hparams:
            raise Exception('Given hyperparameters do not agree with previous hyperparameters.')
    else:
        json.dump(hparams, open(hparams_path, 'w'))

    return hparams

def _setup_hparams(args):
    """ Setup the hparams dictionary using defaults and return it

    Args
    ----
    args : argparse.Namespace
        Parsed arguments

    Returns
    -------
    model_hparams : dict
        Fully configured model hparams dictionary (see :code:`terminator/utils/model/default_hparams.py`)
    run_hparams : dict
        Fully configured training run hparams dictionary (see :code:`terminator/utils/model/default_hparams.py`)
    """

    model_hparams = _load_hparams(args.model_hparams, DEFAULT_MODEL_HPARAMS, 'model_hparams.json', run_dir=args.run_dir)
    run_hparams = _load_hparams(args.run_hparams, DEFAULT_TRAIN_HPARAMS, 'run_hparams.json', run_dir=args.run_dir)

    model_hparams, run_hparams = backwards_compat(model_hparams, run_hparams)
    return model_hparams, run_hparams

def _setup_dataloaders(args, run_hparams, model_hparams):
    """ Setup dataloaders needed for training

    Args
    ----
    args : argparse.Namespace
        Parsed arguments
    run_hparams : dict
        Fully configured hparams dictionary (see :code:`terminator/utils/model/default_hparams.py`)

    Returns
    -------
    train_dataloader, val_dataloader, test_dataloader : torch.utils.data.DataLoader
        DataLoaders for the train, validation, and test datasets
    """
    kwargs = {}
    kwargs['num_workers'] = 2

    # set up dataloaders
    train_ids = []
    with open(args.train, 'r') as f:
        for line in f:
            train_ids += [line.strip()]
    print('msa dataset: ', args.msa_dataset)
    if args.msa_dataset is not None:
        train_dataset = args.msa_dataset
        valid_dataset = args.msa_dataset
    else:
        train_dataset = args.dataset
        valid_dataset = args.dataset
    print('train dataset: ', train_dataset)
    print('all datasets: ', train_dataset, valid_dataset, args.dataset)
    validation_ids = []
    with open(args.validation, 'r') as f:
        for line in f:
            validation_ids += [line.strip()]
    test_ids = []
    with open(args.test, 'r') as f:
        for line in f:
            test_ids += [line.strip()]

    train_dataset = TERMDataset(train_dataset, pdb_ids=train_ids)
    val_dataset = TERMDataset(valid_dataset, pdb_ids=validation_ids)
    test_dataset = TERMDataset(args.dataset, pdb_ids=test_ids)

    train_batch_sampler = TERMBatchSamplerWrapper(ddp=dist.is_initialized())
    train_batch_sampler = train_batch_sampler.sampler(train_batch_sampler.ddp, train_dataset, args.dev,
                                            replicate=run_hparams['replicate'],
                                            batch_size=run_hparams['train_batch_size'],
                                            shuffle=run_hparams['shuffle'],
                                            semi_shuffle=run_hparams['semi_shuffle'],
                                            sort_data=run_hparams['sort_data'],
                                            max_term_res=run_hparams['max_term_res'],
                                            max_seq_tokens=run_hparams['max_seq_tokens'],
                                            flex_type=run_hparams['flex_type'],
                                            msa_type=run_hparams['msa_type'],
                                            msa_id_cutoff=run_hparams['msa_id_cutoff'],
                                            noise_level=run_hparams['noise_level'],
                                            bond_length_noise_level=run_hparams['bond_length_noise_level'],
                                            noise_lim=run_hparams['noise_lim'])
    val_batch_sampler = TERMBatchSamplerWrapper(ddp=dist.is_initialized())
    val_batch_sampler = val_batch_sampler.sampler(val_batch_sampler.ddp, val_dataset, args.dev, 
                                        replicate=run_hparams['replicate'], batch_size=1, 
                                        shuffle=False, msa_type=run_hparams['msa_type'], msa_id_cutoff=run_hparams['msa_id_cutoff'])
    test_batch_sampler = TERMBatchSamplerWrapper(ddp=dist.is_initialized())
    test_batch_sampler = test_batch_sampler.sampler(test_batch_sampler.ddp, test_dataset, args.dev, 
                                        replicate=run_hparams['replicate'], batch_size=1,
                                        shuffle=False)
    return train_dataset, train_batch_sampler, val_dataset, val_batch_sampler, test_dataset, test_batch_sampler

def _setup_dataloaders_wds(args, run_hparams, model_hparams, dataset_name, suffix=None, aux_data=False):
    """ Setup dataloaders needed for training

    Args
    ----
    args : argparse.Namespace
        Parsed arguments
    run_hparams : dict
        Fully configured hparams dictionary (see :code:`terminator/utils/model/default_hparams.py`)

    Returns
    -------
    train_dataloader, val_dataloader, test_dataloader : torch.utils.data.DataLoader
        DataLoaders for the train, validation, and test datasets
    """

    # set up dataloaders
    # train_ids = []
    # with open(args.train, 'r') as f:
    #     for line in f:
    #         train_ids += [line[:-1]]
    # validation_ids = []
    # with open(args.validation, 'r') as f:
    #     for line in f:
    #         validation_ids += [line[:-1]]
    # test_ids = []
    # with open(args.test, 'r') as f:
    #     for line in f:
    #         test_ids += [line[:-1]]
    dataset = args.dataset
    if run_hparams['msa_type']:
        dataset = args.msa_dataset

    model_hparams['struct_predict'] = run_hparams['use_struct_predict']

    # train_dataset = TERMDataset(dataset, pdb_ids=train_ids)
    # val_dataset = TERMDataset(dataset, pdb_ids=validation_ids)
    # test_dataset = TERMDataset(args.dataset, pdb_ids=test_ids)
    # print(len(train_dataset))
    # print(len(val_dataset))
    # print(len(test_dataset))
    pair_etab_dir = ''
    if 'edge_loss_msa' in run_hparams['loss_config'].keys() or 'msa_weighted_nlcpl' in run_hparams['loss_config'].keys():
        pair_etab_dir = dataset
    cols = ['inp.pyd']
    if model_hparams['esm_feats']:
        if model_hparams['esm_model'] == '650':
            esm, alphabet = esmlib.pretrained.esm2_t33_650M_UR50D()
        elif model_hparams['esm_model'] == '3B':
            esm, alphabet = esmlib.pretrained.esm2_t36_3B_UR50D()
        else:
            esm, alphabet = esmlib.pretrained.esm2_t30_150M_UR50D()
        if model_hparams['use_pepmlm']:
            esm = AutoModelForMaskedLM.from_pretrained("ChatterjeeLab/PepMLM-650M")
        batch_converter = alphabet.get_batch_converter()
        esm = esm.eval()
        if run_hparams['data_dev'] == 'cuda:0':
            esm = esm.cuda()
    elif model_hparams['rla_feats']:
        _, alphabet = esmlib.pretrained.esm2_t30_150M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        sys_keys = copy.deepcopy(list(sys.modules.keys()))
        for mod in sys_keys:
            if mod.find('terminator') > -1:
                del sys.modules[mod]
        from load_rla import load_rla
        rla = load_rla('cpu')
        sys_keys = copy.deepcopy(list(sys.modules.keys()))
        for mod in sys_keys:
            if mod.find('terminator') > -1:
                del sys.modules[mod]
        new_path = []
        for dir_ in sys.path:
            if dir_.find('joint-protein-embs') == -1:
                new_path.append(dir_)
        sys.path = new_path
        esm = rla.text_model
    else:
        esm = None
        batch_converter = None

    from terminator.data.data import (WDSBatchSamplerWrapper, TERMLazyDataset, TERMBatchSamplerWrapper, TERMDataset, TERMLazyBatchSampler)
    from terminator.models.TERMinator import TERMinator
    from terminator.utils.model.loop_utils import run_epoch
    from terminator.utils.model.loss_fn import construct_loss_fn
    if run_hparams['all_batch']:
        run_hparams['train_batch_size'] = len(train_dataset)
    if aux_data:
        train_path = os.path.join(args.dataset, f'{dataset_name}_{suffix}.wds')
    elif suffix is not None:
        train_path = os.path.join(os.path.dirname(args.dataset), f'{dataset_name}_train{suffix}.wds')
        val_path = os.path.join(os.path.dirname(args.dataset), f'{dataset_name}_val{suffix}.wds')
        test_path = os.path.join(os.path.dirname(args.dataset), f'{dataset_name}_test{suffix}.wds')
    else:
        train_path = os.path.join(os.path.dirname(args.dataset), f'{dataset_name}_train.wds')
        val_path = os.path.join(os.path.dirname(args.dataset), f'{dataset_name}_val.wds')
        test_path = os.path.join(os.path.dirname(args.dataset), f'{dataset_name}_test.wds')
    if not aux_data and not os.path.exists(test_path):
        test_path = val_path
    
    train_dataset = wds.WebDataset(train_path).decode().to_tuple(*cols)
    print('train dataset: ', train_path)
    print('aux data: ', (run_hparams['name_cluster']) or (aux_data and run_hparams['aux_name_cluster']))
    if run_hparams['max_seq_tokens']:
        max_len = run_hparams['max_seq_tokens']
    else:
        max_len = 2000
    train_dataset = [data for data in train_dataset if (data[0]['coords'].shape[0] >= model_hparams['k_neighbors'])] #(data[0]['coords'].shape[0] == data[0]['sc_coords'].shape[0])
    train_dataset = [data for data in train_dataset if (data[0]['coords'].shape[0] < max_len)]
    train_dataset = [data for data in train_dataset if (data[0]['coords'].shape[0] == len(data[0]['sequence']))]
    train_dataset = [data for data in train_dataset if (data[0]['coords'].shape[0] == data[0]['sc_coords'].shape[0])]

    train_batch_sampler = WDSBatchSamplerWrapper(dist.is_initialized())
    train_batch_sampler = train_batch_sampler.sampler(train_batch_sampler.ddp, train_dataset,
                                            batch_size=run_hparams['train_batch_size'],
                                            shuffle=run_hparams['shuffle'],
                                            semi_shuffle=run_hparams['semi_shuffle'],
                                            sort_data=run_hparams['sort_data'],
                                            max_term_res=run_hparams['max_term_res'],
                                            max_seq_tokens=run_hparams['max_seq_tokens'],
                                            msa_type=run_hparams['msa_type'],
                                            msa_id_cutoff=run_hparams['msa_id_cutoff'],
                                            flex_type=run_hparams['flex_type'],
                                            replicate=run_hparams['replicate'],
                                            noise_level=run_hparams['noise_level'],
                                            bond_length_noise_level=run_hparams['bond_length_noise_level'],
                                            bond_angle_noise_level=run_hparams['bond_angle_noise_level'],
                                            noise_lim=run_hparams['noise_lim'],
                                            pair_etab_dir=pair_etab_dir,
                                            use_sc=run_hparams["use_sc"], 
                                            mpnn_dihedrals=model_hparams["mpnn_dihedrals"],
                                            sc_mask_rate=run_hparams['sc_mask_rate'],
                                            base_sc_mask=run_hparams['base_sc_mask'],
                                            sc_mask=run_hparams['sc_mask'],
                                            sc_mask_schedule=run_hparams['sc_mask_schedule'],
                                            sc_info=run_hparams['sc_info'],
                                            sc_noise=run_hparams['sc_noise'],
                                            mask_neighbors=run_hparams['mask_neighbors'],
                                            mask_interface=run_hparams['mask_interface'],
                                            half_interface=run_hparams['half_interface'],
                                            interface_pep=run_hparams['interface_pep'],
                                            inter_cutoff=run_hparams['inter_cutoff'],
                                            dev=run_hparams['data_dev'],
                                            esm=esm,
                                            batch_converter=batch_converter,
                                            use_esm_attns=run_hparams['use_esm_attns'],
                                            use_reps=run_hparams['use_reps'],
                                            post_esm_mask=run_hparams['post_esm_mask'],
                                            from_rla=model_hparams['rla_feats'],
                                            esm_embed_layer=model_hparams['esm_embed_layer'],
                                            connect_chains=run_hparams['connect_chains'],
                                            convert_to_esm=model_hparams['convert_to_esm'],
                                            one_hot=model_hparams['one_hot'],
                                            all_batch=run_hparams['all_batch'],
                                            chain_handle=model_hparams['chain_handle'],
                                            name_cluster=(run_hparams['name_cluster']) or (aux_data and run_hparams['aux_name_cluster']),
                                            openfold_backbone=model_hparams['struct_predict'],
                                            random_interface=run_hparams['random_interface'],
                                            no_mask=run_hparams['no_mask'],
                                            from_pepmlm=model_hparams['from_pepmlm'],
                                            nrg_noise=run_hparams['nrg_noise'],
                                            fix_multi_rate=run_hparams['fix_multi_rate'],
                                            load_etab_dir=model_hparams['load_etab_dir'])
    if aux_data:
        return train_dataset, train_batch_sampler
    print('val dataset: ', val_path)
    print('test dataset: ', test_path)
    val_dataset = wds.WebDataset(val_path).decode().to_tuple(*cols)
    
    test_dataset = wds.WebDataset(test_path).decode().to_tuple(*cols)
    val_dataset = [data for data in val_dataset if (data[0]['coords'].shape[0] >= model_hparams['k_neighbors'])]
    val_dataset = [data for data in val_dataset if (data[0]['coords'].shape[0] == len(data[0]['sequence']))]
    val_dataset = [data for data in val_dataset if (data[0]['coords'].shape[0] == data[0]['sc_coords'].shape[0])]
    test_dataset = [data for data in test_dataset if (data[0]['coords'].shape[0] >= model_hparams['k_neighbors'])]
    test_dataset = [data for data in test_dataset if (data[0]['coords'].shape[0] == len(data[0]['sequence']))]
    test_dataset = [data for data in test_dataset if (data[0]['coords'].shape[0] == data[0]['sc_coords'].shape[0])]

    limit_size = True
    for loss_key in run_hparams['loss_config'].keys():
        if loss_key.find('esm_cmap') > -1 or loss_key.find('esm_loss') > -1:
            limit_size = True
    if (model_hparams['esm_feats'] and run_hparams['use_esm_attns']) or (limit_size):
        val_dataset = [data for data in val_dataset if (data[0]['coords'].shape[0] == len(data[0]['sequence'])) and (data[0]['coords'].shape[0] < max_len)]
        test_dataset = [data for data in test_dataset if (data[0]['coords'].shape[0] == len(data[0]['sequence'])) and (data[0]['coords'].shape[0] < max_len)]
    print('train len: ', len(train_dataset))
    print('val len: ', len(val_dataset))
    print('test len: ', len(test_dataset))
    
    val_batch_sampler = WDSBatchSamplerWrapper(dist.is_initialized())
    if run_hparams['all_batch']:
        run_hparams['train_batch_size'] = len(val_dataset)
    else:
        run_hparams['train_batch_size'] = 1
    val_batch_sampler = val_batch_sampler.sampler(val_batch_sampler.ddp, val_dataset, batch_size=run_hparams['train_batch_size'], shuffle=False, msa_type=run_hparams['msa_type'], msa_id_cutoff=run_hparams['msa_id_cutoff'], pair_etab_dir=pair_etab_dir, use_sc=run_hparams["use_sc"], mpnn_dihedrals=model_hparams["mpnn_dihedrals"], sc_mask_rate=run_hparams['sc_mask_rate'], base_sc_mask=run_hparams['base_sc_mask'], sc_mask=run_hparams['sc_mask'], sc_mask_schedule=run_hparams['sc_mask_schedule'], sc_info=run_hparams['sc_info'], mask_neighbors=run_hparams['mask_neighbors'], mask_interface=run_hparams['mask_interface'], half_interface=run_hparams['half_interface'], interface_pep=run_hparams['interface_pep'], inter_cutoff=run_hparams['inter_cutoff'], dev=run_hparams['data_dev'], esm=esm, batch_converter=batch_converter, use_esm_attns=run_hparams['use_esm_attns'], use_reps=run_hparams['use_reps'], post_esm_mask=run_hparams['post_esm_mask'], from_rla=model_hparams['rla_feats'], esm_embed_layer=model_hparams['esm_embed_layer'], connect_chains=run_hparams['connect_chains'], convert_to_esm=model_hparams['convert_to_esm'], one_hot=model_hparams['one_hot'], all_batch=run_hparams['all_batch'], chain_handle=model_hparams['chain_handle'], name_cluster=run_hparams['name_cluster'], openfold_backbone=model_hparams['struct_predict'], random_interface=run_hparams['random_interface'], no_mask=run_hparams['no_mask'], from_pepmlm=model_hparams['from_pepmlm'], fix_multi_rate=run_hparams['fix_multi_rate'], load_etab_dir=model_hparams['load_etab_dir'])

    test_batch_sampler = WDSBatchSamplerWrapper(dist.is_initialized())
    if run_hparams['all_batch']:
        run_hparams['train_batch_size'] = len(test_dataset)
    else:
        run_hparams['train_batch_size'] = 1
    test_batch_sampler = test_batch_sampler.sampler(test_batch_sampler.ddp, test_dataset, batch_size=run_hparams['train_batch_size'], shuffle=False, use_sc=run_hparams["use_sc"], mpnn_dihedrals=model_hparams["mpnn_dihedrals"], sc_mask_rate=run_hparams['sc_mask_rate'], base_sc_mask=run_hparams['base_sc_mask'], sc_mask=run_hparams['sc_mask'], sc_mask_schedule=run_hparams['sc_mask_schedule'], sc_info=run_hparams['sc_info'], mask_neighbors=run_hparams['mask_neighbors'], mask_interface=run_hparams['mask_interface'], half_interface=run_hparams['half_interface'], interface_pep=run_hparams['interface_pep'], inter_cutoff=run_hparams['inter_cutoff'], dev=run_hparams['data_dev'], esm=esm, batch_converter=batch_converter, use_esm_attns=run_hparams['use_esm_attns'], use_reps=run_hparams['use_reps'], post_esm_mask=run_hparams['post_esm_mask'], from_rla=model_hparams['rla_feats'], esm_embed_layer=model_hparams['esm_embed_layer'], connect_chains=run_hparams['connect_chains'], convert_to_esm=model_hparams['convert_to_esm'], one_hot=model_hparams['one_hot'],all_batch=run_hparams['all_batch'], chain_handle=model_hparams['chain_handle'], name_cluster=run_hparams['name_cluster'], openfold_backbone=model_hparams['struct_predict'], random_interface=run_hparams['random_interface'], no_mask=run_hparams['no_mask'], from_pepmlm=model_hparams['from_pepmlm'], fix_multi_rate=run_hparams['fix_multi_rate'], load_etab_dir=model_hparams['load_etab_dir'])
    
    return train_dataset, train_batch_sampler, val_dataset, val_batch_sampler, test_dataset, test_batch_sampler


def _load_checkpoint(run_dir, dev, finetune=False):
    """ If a training checkpoint exists, load the checkpoint. Otherwise, setup checkpointing initial values.

    Args
    ----
    run_dir : str
        Path to directory containing the training run checkpoint, as well the tensorboard output.

    Returns
    -------
    dict
        Dictionary containing
        - "best_checkpoint_state": the best checkpoint state during the run
        - "last_checkpoint_state": the most recent checkpoint state during the run
        - "best_checkpoint": the best model parameter set during the run
        - "best_validation": the best validation loss during the run
        - "last_optim_state": the most recent state of the optimizer
        - "start_epoch": what epoch to resume training from
        - "writer": SummaryWriter for tensorboard
        - "training_curves": pairs of (train_loss, val_loss) representing the training and validation curves
    """

    if os.path.isfile(os.path.join(run_dir, 'net_best_checkpoint.pt')):
        best_checkpoint_state = torch.load(os.path.join(run_dir, 'net_best_checkpoint.pt'), map_location=torch.device(dev))
        last_checkpoint_state = torch.load(os.path.join(run_dir, 'net_last_checkpoint.pt'), map_location=torch.device(dev))
        best_checkpoint = best_checkpoint_state['state_dict']
        best_validation = best_checkpoint_state['val_loss']
        last_optim_state = last_checkpoint_state["optimizer_state"]
        start_epoch = last_checkpoint_state['epoch'] + 1
        writer = SummaryWriter(log_dir=os.path.join(run_dir, 'tensorboard'), purge_step=start_epoch + 1)
        training_curves = last_checkpoint_state["training_curves"]
    else:
        best_checkpoint_state, last_checkpoint_state = None, None
        best_checkpoint = None
        best_validation = 10e8
        last_optim_state = None
        start_epoch = 0
        writer = SummaryWriter(log_dir=os.path.join(run_dir, 'tensorboard'))
        training_curves = {"train_loss": [], "val_loss": []}
        if finetune: # load existing model for finetuning
            best_checkpoint_state = torch.load(os.path.join(run_dir, 'net_original.pt'), map_location=torch.device(dev))
            best_checkpoint = best_checkpoint_state['state_dict']
            if 'top.V_out.bias' not in best_checkpoint.keys():
                best_checkpoint['top.V_out.bias'] = torch.rand(20)
                best_checkpoint['top.V_out.weight'] = torch.rand((20, 128))

    return {"best_checkpoint_state": best_checkpoint_state,
            "last_checkpoint_state": last_checkpoint_state,
            "best_checkpoint": best_checkpoint,
            "best_validation": best_validation,
            "last_optim_state": last_optim_state,
            "start_epoch": start_epoch,
            "writer": writer,
            "training_curves": training_curves}


def _setup_model(model_hparams, run_hparams, checkpoint, dev, use_esm, edges_to_seq, parallel=False, from_finetune=False):
    """ Setup a TERMinator model using hparams, a checkpoint if provided, and a computation device.

    Args
    ----
    model_hparams : dict
        Fully configured model hparams dictionary (see :code:`terminator/utils/model/default_hparams.py`)
    run_hparams : dict
        Fully configured training run hparams dictionary (see :code:`terminator/utils/model/default_hparams.py`)
    checkpoint : OrderedDict or None
        Model parameters
    dev : str
        Computation device to use

    Returns
    -------
    terminator : TERMinator or nn.DataParallel(TERMinator)
        Potentially parallelized TERMinator to use for training
    terminator_module : TERMinator
        Inner TERMinator, unparallelized
    """
    if checkpoint is not None and 'top.E_outs.0.bias' in checkpoint:
        edges_to_seq = True
    if from_finetune:
        transfer_save = model_hparams['use_transfer_model']
        model_hparams['use_transfer_model'] = False
    if not run_hparams["use_struct_predict"] and checkpoint is not None:
        model_hparams["struct_predict"] = False
        key_list = copy.deepcopy(list(checkpoint.keys()))
        for key in key_list:
            if 'struct_module' in key:
                del checkpoint[key]
    terminator = TERMinator(hparams=model_hparams, device=dev, use_esm=use_esm, edges_to_seq=edges_to_seq)
    if model_hparams['distill_checkpoint'] != '':
        distill_model_hparams = _load_hparams(model_hparams['distill_model_hparams'], DEFAULT_MODEL_HPARAMS, 'model_hparams.json')
        distill_run_hparams = _load_hparams(run_hparams['distill_run_hparams'], DEFAULT_TRAIN_HPARAMS, 'run_hparams.json')

        distill_model_hparams, distill_run_hparams = backwards_compat(distill_model_hparams, distill_run_hparams)
        distill_terminator = TERMinator(hparams=distill_model_hparams, device=dev, use_esm=use_esm, edges_to_seq=edges_to_seq)
        distill_terminator.load_state_dict(model_hparams['distill_checkpoint'])
    else:
        distill_terminator = None
        
    if checkpoint is not None:
        terminator.load_state_dict(checkpoint)
    print(terminator)
    print("terminator hparams", terminator.hparams)

    if parallel and torch.cuda.device_count() > 1 and dev != "cpu":
        terminator = nn.DataParallel(terminator)
        terminator_module = terminator.module
    else:
        terminator_module = terminator
    terminator.to(dev)

    if from_finetune:
        model_hparams['use_transfer_model'] = transfer_save

    return terminator, terminator_module, distill_terminator

def main(args):
    """ Train TERMinator """
    dev = args.dev
    run_dir = args.run_dir
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)

    # setup dataloaders
    kwargs = {}
    # kwargs['num_workers'] = 20
    model_hparams, run_hparams = _setup_hparams(args)
    
    esm_options = {'use_esm_attns': run_hparams['use_esm_attns'], 'esm_embed_layer': model_hparams['esm_embed_layer'], 'from_rla': model_hparams['rla_feats'],
                   'use_reps': run_hparams['use_reps'], 'connect_chains': run_hparams['connect_chains'], 'one_hot': model_hparams['one_hot']
                   }
    if run_hparams['from_wds']:
        train_dataset, train_batch_sampler, val_dataset, val_batch_sampler, test_dataset, test_batch_sampler = _setup_dataloaders_wds(args, run_hparams, model_hparams, dataset_name=run_hparams['dataset_name'], suffix=run_hparams['suffix'])
    else:
        train_dataset, train_batch_sampler, val_dataset, val_batch_sampler, test_dataset, test_batch_sampler = _setup_dataloaders(args, run_hparams, model_hparams)
    train_dataloader = DataLoader(train_dataset,
                                batch_sampler=train_batch_sampler,
                                collate_fn=train_batch_sampler.package,
                                pin_memory=(run_hparams['data_dev'] == 'cpu'))
    val_dataloader = DataLoader(val_dataset,
                                batch_sampler=val_batch_sampler,
                                collate_fn=val_batch_sampler.package,
                                pin_memory=(run_hparams['data_dev'] == 'cpu'))
    test_dataloader = DataLoader(test_dataset,
                                batch_sampler=test_batch_sampler,
                                collate_fn=test_batch_sampler.package,
                                pin_memory=(run_hparams['data_dev'] == 'cpu'))

    # load checkpoint
    checkpoint_dict = _load_checkpoint(run_dir, dev, run_hparams['finetune'])
    best_validation = checkpoint_dict["best_validation"]
    if "best_aux" in checkpoint_dict:
        best_aux = checkpoint_dict["best_aux"]
    else:
        best_aux = 10e8
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
            use_esm = (not model_hparams['mpnn_decoder']) and (not model_hparams['pifold_decoder'])
            load_esm = model_hparams['mpnn_decoder'] or model_hparams['pifold_decoder']
            break
    terminator, terminator_module, distill_terminator = _setup_model(model_hparams, run_hparams, best_checkpoint, dev, use_esm, model_hparams['edges_to_seq'])
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

    # if loss involves esm, load model
    if use_esm or run_hparams['num_recycles'] > 0 or load_esm:
        esm, alphabet = esmlib.pretrained.esm2_t30_150M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        
        converter_state = torch.load(args.converter_path, map_location=torch.device(dev))
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
        for p in esm.parameters():
            p.requires_grad = False
        for p in converter.parameters():
            p.requires_grad = False
        esm = esm.eval()
        converter = converter.eval()
    else:
        esm = None
        batch_converter = None
        converter = None

    # If tracking aux loss, load data and create loss_fn
    if run_hparams['aux_loss'] is not None:
        base_dataset = copy.deepcopy(args.dataset)
        args.dataset = run_hparams['aux_dataset']
        aux_dataset, aux_batch_sampler = _setup_dataloaders_wds(args, run_hparams, model_hparams, dataset_name=run_hparams['aux_dataset_name'], suffix=run_hparams['aux_suffix'], aux_data=True)
        aux_dataloader = DataLoader(aux_dataset,
                                batch_sampler=aux_batch_sampler,
                                collate_fn=aux_batch_sampler.package,
                                pin_memory=(run_hparams['data_dev'] == 'cpu'))
        aux_loss_fn = construct_loss_fn(run_hparams, aux_loss=True)
        args.dataset = base_dataset
        if "aux_loss" not in training_curves.keys(): training_curves["aux_loss"] = []
    # esm = esm.cpu()
    # converter = converter.cpu()
    best_aux = False
    try:
        for epoch in range(start_epoch, args.epochs):
            print('epoch', epoch)
            train_batch_sampler._set_epoch(epoch)
            val_batch_sampler._set_epoch(epoch)
            test_batch_sampler._set_epoch(epoch)

            epoch_loss, epoch_ld, _ = run_epoch(terminator, train_dataloader, loss_fn, run_hparams, epoch=epoch, optimizer=optimizer, grad=True, dev=dev, finetune=finetune, isDataParallel=isDataParallel, esm=esm, batch_converter=batch_converter, converter=converter, num_recycles=run_hparams['num_recycles'], keep_seq=run_hparams['keep_seq_recycle'], keep_sc_mask_recycle=run_hparams['keep_sc_mask_recycle'], esm_options=esm_options, only_loss_recycle=run_hparams['only_loss_recycle'], recycle_confidence=run_hparams['recycle_confidence'], recycle_teacher_forcing=run_hparams['recycle_teacher_forcing'], keep_sc_mask_loss=run_hparams['keep_sc_mask_loss'], loss_weight_schedule=run_hparams['loss_weight_schedule'], distill_terminator=distill_terminator)
            print('epoch loss', epoch_loss, 'epoch_ld', epoch_ld)
            writer.add_scalar('training loss', epoch_loss, epoch)

            # validate
            val_loss, val_ld, _ = run_epoch(terminator, val_dataloader, loss_fn, epoch=epoch, grad=False, dev=dev, esm=esm, batch_converter=batch_converter, converter=converter, num_recycles=run_hparams['num_recycles'], keep_seq=run_hparams['keep_seq_recycle'], keep_sc_mask_recycle=run_hparams['keep_sc_mask_recycle'], esm_options=esm_options, only_loss_recycle=run_hparams['only_loss_recycle'], recycle_confidence=run_hparams['recycle_confidence'], keep_sc_mask_loss=run_hparams['keep_sc_mask_loss'], loss_weight_schedule=run_hparams['loss_weight_schedule'], distill_terminator=distill_terminator)
            print('val loss', val_loss, 'val ld', val_ld, 'best val loss ', best_validation)
            writer.add_scalar('val loss', val_loss, epoch)

            # Aux loss if specified
            if run_hparams['aux_loss'] is not None:
                try:
                    if run_hparams['aux_dev'] == 'cpu':
                        terminator = terminator.cpu()
                    aux_loss, aux_ld, _ = run_epoch(terminator, aux_dataloader, aux_loss_fn, epoch=epoch, grad=run_hparams['aux_grad'], dev=run_hparams['aux_dev'], esm=esm, batch_converter=batch_converter, converter=converter, num_recycles=run_hparams['num_recycles'], keep_seq=run_hparams['keep_seq_recycle'], keep_sc_mask_recycle=run_hparams['keep_sc_mask_recycle'], esm_options=esm_options, only_loss_recycle=run_hparams['only_loss_recycle'], recycle_confidence=run_hparams['recycle_confidence'], keep_sc_mask_loss=run_hparams['keep_sc_mask_loss'], silenced=True, loss_weight_schedule=run_hparams['loss_weight_schedule'])
                    print('auxiliary loss', aux_loss, 'val ld', aux_ld, 'best aux loss ', best_aux)
                    aux_comp = (not best_aux) or aux_loss < best_aux
                    training_curves["aux_loss"].append((aux_loss, aux_ld))
                    writer.add_scalar('aux loss', aux_loss, epoch)
                    if run_hparams['aux_dev'] == 'cpu' and dev == 'cuda:0':
                        terminator = terminator.cuda()
                except:
                    terminator = terminator.cuda()
                    aux_comp = False
                    aux_loss = 0
                    best_aux = 0
            else:
                aux_comp = False
                aux_loss = 0
                best_aux = 0

            training_curves["train_loss"].append((epoch_loss, epoch_ld))
            training_curves["val_loss"].append((val_loss, val_ld))

            comp = (val_ld[finetune_loss]['loss'] < best_validation) if finetune else (val_loss < best_validation)
            

            # save a state checkpoint
            checkpoint_state = {
                'epoch': epoch,
                'state_dict': terminator_module.state_dict(),
                'best_model': comp, # (val_loss < best_validation)
                'best_aux_model': aux_comp,
                'val_loss': val_loss,
                'aux_loss': aux_loss,
                'best_val_loss': min(val_loss, best_validation),
                'best_aux_loss': min(aux_loss, best_aux),
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
                iters_since_best_loss = 0
            if aux_comp:
                best_aux = aux_loss
                best_aux_checkpoint = copy.deepcopy(terminator_module.state_dict())
                torch.save(checkpoint_state, os.path.join(run_dir, 'net_aux_best_checkpoint.pt'))
                iters_since_best_loss = 0
            if not comp and not aux_comp:
                iters_since_best_loss += 1

            if iters_since_best_loss > run_hparams['patience']:
                print("Exiting due to early stopping")
                break

    except KeyboardInterrupt:
        pass

    # save model params
    print(training_curves)
    torch.save(terminator_module.state_dict(), os.path.join(run_dir, 'net_last.pt'))
    torch.save(best_checkpoint, os.path.join(run_dir, 'net_best.pt'))
    if run_hparams['aux_loss'] is not None: torch.save(best_aux_checkpoint, os.path.join(run_dir, 'net_aux_best.pt'))

    # test
    terminator_module.load_state_dict(best_checkpoint)
    if 'edge_loss_msa' in run_hparams['loss_config'].keys() or 'msa_weighted_nlcpl' in run_hparams['loss_config'].keys():
        run_hparams['loss_config'] = {'edge_loss': 1}
        loss_fn = construct_loss_fn(run_hparams)
    test_loss, test_ld, dump = run_epoch(terminator, test_dataloader, loss_fn, epoch=epoch, grad=False, test=True, dev=dev, num_recycles=run_hparams['num_recycles'], keep_seq=run_hparams['keep_seq_recycle'], keep_sc_mask_recycle=run_hparams['keep_sc_mask_recycle'],esm_options=esm_options, loss_weight_schedule=run_hparams['loss_weight_schedule'], distill_terminator=distill_terminator)
    print(f"test loss {test_loss} test loss dict {test_ld}")
    # dump outputs
    # if args.out_dir:
    #     if not os.path.isdir(args.out_dir):
    #         os.mkdir(args.out_dir)
    #     net_out_path = os.path.join(args.out_dir, "net.out")
    # else:
    #     net_out_path = os.path.join(run_dir, "net.out")
    # # save etab outputs for dTERMen runs
    # with open(net_out_path, 'wb') as fp:
    #     pickle.dump(dump, fp)
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