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
import shutil
import datetime
import numpy as np
import esm as esmlib
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from terminator.utils.model.loop_utils import run_epoch
from terminator.utils.model.loss_fn import construct_loss_fn
import webdataset as wds
# for autosummary import purposes
# pylint: disable=wrong-import-order,wrong-import-position
sys.path.insert(0, os.path.dirname(__file__))
from terminator.utils.model.optim import get_std_opt
from terminator.utils.model.transfer_model import TransferModel, PottsTransferModel
from train import _setup_dataloaders_wds, _setup_hparams, _setup_model

# pylint: disable=unspecified-encoding


torch.set_printoptions(threshold=10000)
torch.set_printoptions(linewidth=1000)
torch.set_printoptions(precision=2)
torch.set_printoptions(sci_mode=False)
torch.set_printoptions(profile="full")

def _load_checkpoint(run_dir, dev, model_hparams, finetune=False, top_nodes=False, use_esm=False, predict_confidence=False):
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
        last_checkpoint_state = torch.load(os.path.join(run_dir, 'net_best_checkpoint.pt'), map_location=torch.device(dev))
        best_checkpoint = best_checkpoint_state['state_dict']
        if use_esm:
            if 'top.V_out.bias' not in best_checkpoint.keys():
                best_checkpoint['top.V_out.bias'] = torch.rand(20)
                best_checkpoint['top.V_out.weight'] = torch.rand((20, 128))
            if 'top.E_outs.0.bias' not in best_checkpoint.keys():
                best_checkpoint['top.E_outs.0.bias'] = torch.rand(256)
                best_checkpoint['top.E_outs.0.weight'] = torch.rand((256, 600))
                best_checkpoint['top.E_outs.1.bias'] = torch.rand(64)
                best_checkpoint['top.E_outs.1.weight'] = torch.rand((64, 256))
                best_checkpoint['top.E_outs.2.bias'] = torch.rand(1)
                best_checkpoint['top.E_outs.2.weight'] = torch.rand((1, 64))
        if predict_confidence:
            if 'top.confidence_norm.weight' not in best_checkpoint.keys():
                best_checkpoint['top.confidence_norm.weight'] = torch.rand((128))
                best_checkpoint['top.confidence_norm.bias'] = torch.rand((128))
                best_checkpoint['top.confidence_module.layers.0.weight'] = torch.rand((128, 128))
                best_checkpoint['top.confidence_module.layers.0.bias'] = torch.rand((128))
                best_checkpoint['top.confidence_module.layers.1.weight'] = torch.rand((64, 128))
                best_checkpoint['top.confidence_module.layers.1.bias'] = torch.rand((64))
                best_checkpoint['top.confidence_module.layers.2.weight'] = torch.rand((1, 64))
                best_checkpoint['top.confidence_module.layers.2.bias'] = torch.rand((1))
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
            if top_nodes:
                if 'top.V_out.bias' not in best_checkpoint.keys():
                    best_checkpoint['top.V_out.bias'] = torch.rand(20)
                    best_checkpoint['top.V_out.weight'] = torch.rand((20, 128))
                if 'top.E_outs.0.bias' not in best_checkpoint.keys():
                    best_checkpoint['top.E_outs.0.bias'] = torch.rand(256)
                    best_checkpoint['top.E_outs.0.weight'] = torch.rand((256, 600))
                    best_checkpoint['top.E_outs.1.bias'] = torch.rand(64)
                    best_checkpoint['top.E_outs.1.weight'] = torch.rand((64, 256))
                    best_checkpoint['top.E_outs.2.bias'] = torch.rand(1)
                    best_checkpoint['top.E_outs.2.weight'] = torch.rand((1, 64))
            if predict_confidence:
                if 'top.confidence_norm.weight' not in best_checkpoint.keys():
                    best_checkpoint['top.confidence_norm.weight'] = torch.rand((128))
                    best_checkpoint['top.confidence_norm.bias'] = torch.rand((128))
                    best_checkpoint['top.confidence_module.layers.0.weight'] = torch.rand((128, 128))
                    best_checkpoint['top.confidence_module.layers.0.bias'] = torch.rand((128))
                    best_checkpoint['top.confidence_module.layers.1.weight'] = torch.rand((64, 128))
                    best_checkpoint['top.confidence_module.layers.1.bias'] = torch.rand((64))
                    best_checkpoint['top.confidence_module.layers.2.weight'] = torch.rand((1, 64))
                    best_checkpoint['top.confidence_module.layers.2.bias'] = torch.rand((1))
    if (model_hparams['ener_finetune_mlp']) and ('top.ener_mlp.layers.0.bias' not in best_checkpoint.keys()):
        best_checkpoint['top.ener_mlp.layers.0.bias'] = torch.ones(400).to(dtype=torch.float32)
        best_checkpoint['top.ener_mlp.layers.0.weight'] = torch.ones((400, 400)).to(dtype=torch.float32)
        best_checkpoint['top.ener_mlp.layers.1.bias'] = torch.ones(400).to(dtype=torch.float32)
        best_checkpoint['top.ener_mlp.layers.1.weight'] = torch.ones((400, 400)).to(dtype=torch.float32)
        best_checkpoint['top.ener_mlp.layers.2.bias'] = torch.ones(400).to(dtype=torch.float32)
        best_checkpoint['top.ener_mlp.layers.2.weight'] = torch.ones((400, 400)).to(dtype=torch.float32)
    if not model_hparams['nodes_to_probs']:
        if 'top.V_out.bias' in best_checkpoint.keys():
            del best_checkpoint['top.V_out.bias']
            del best_checkpoint['top.V_out.weight']
    if not model_hparams['edges_to_seq']:
        if 'top.E_outs.0.bias' in best_checkpoint.keys():
            del best_checkpoint['top.E_outs.0.bias']
            del best_checkpoint['top.E_outs.0.weight']
            del best_checkpoint['top.E_outs.1.bias']
            del best_checkpoint['top.E_outs.1.weight']
            del best_checkpoint['top.E_outs.2.bias']
            del best_checkpoint['top.E_outs.2.weight']
    return {"best_checkpoint_state": best_checkpoint_state,
            "last_checkpoint_state": last_checkpoint_state,
            "best_checkpoint": best_checkpoint,
            "best_validation": best_validation,
            "last_optim_state": last_optim_state,
            "start_epoch": start_epoch,
            "writer": writer,
            "training_curves": training_curves}

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
    
def finetune(args):
    """ Train TERMinator """
    dev = args.dev
    # dev = 'cpu'
    run_dir = args.run_dir
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)
    # setup dataloaders
    kwargs = {}
    kwargs['num_workers'] = 2
    model_hparams, run_hparams = _setup_hparams(args)
    model_hparams['ft_dropout'] = run_hparams['ft_dropout']
    print('devices: ', run_hparams['data_dev'], run_hparams['aux_dev'])
    print('name cluster: ', run_hparams['name_cluster'])
    esm_options = {'use_esm_attns': run_hparams['use_esm_attns'], 'esm_embed_layer': model_hparams['esm_embed_layer'], 'from_rla': model_hparams['rla_feats'],
                   'use_reps': run_hparams['use_reps'], 'connect_chains': run_hparams['connect_chains'], 'one_hot': model_hparams['one_hot']
                   }

    train_dataset, train_batch_sampler, val_dataset, val_batch_sampler, test_dataset, test_batch_sampler = _setup_dataloaders_wds(args, run_hparams, model_hparams, dataset_name=run_hparams['dataset_name'], suffix=run_hparams['suffix'])
    train_dataloader = DataLoader(train_dataset,
                                  batch_sampler=train_batch_sampler,
                                  collate_fn=train_batch_sampler.package,
                                  pin_memory=(run_hparams['data_dev'] == 'cpu'))
    val_dataloader = DataLoader(val_dataset,
                                batch_sampler=val_batch_sampler,
                                collate_fn=val_batch_sampler.package,
                                pin_memory=(run_hparams['data_dev'] == 'cpu'))

    # load checkpoint
    use_esm = model_hparams['nodes_to_probs'] or model_hparams['edges_to_seq']
    load_esm = False
    for loss_key in run_hparams['loss_config'].keys():
        if False and loss_key.find('esm_loss') > -1: #loss_key.find('esm_cmap') > -1 or 
            use_esm = not model_hparams['mpnn_decoder']
            load_esm = model_hparams['mpnn_decoder']
            break
    print('use esm: ', use_esm)
    checkpoint_dict = _load_checkpoint(run_dir, dev, model_hparams, True, top_nodes=use_esm, use_esm=use_esm, predict_confidence=model_hparams['predict_confidence'])
    from_base = True
    if os.path.isfile(os.path.join(run_dir, 'net_best_checkpoint.pt')):
        from_base=False
    best_validation = checkpoint_dict["best_validation"]
    best_checkpoint = checkpoint_dict["best_checkpoint"]
    if 'top.features.node_layers.0.weight' not in best_checkpoint.keys():
        model_hparams['old'] = True
    else:
        model_hparams['old'] = False
    start_epoch = checkpoint_dict["start_epoch"]
    last_optim_state = checkpoint_dict["last_optim_state"]
    writer = checkpoint_dict["writer"]
    training_curves = checkpoint_dict["training_curves"]

    isDataParallel = True if torch.cuda.device_count() > 1 and dev != "cpu" else False
    finetune = run_hparams["finetune"]

    # construct terminator, loss fn, and optimizer
    
    terminator, terminator_module, _ = _setup_model(model_hparams, run_hparams, best_checkpoint, dev, use_esm, model_hparams['edges_to_seq'], from_finetune=from_base)
    if finetune:
        if run_hparams['use_pretrained_out']:
            tune_W_out = terminator.top.W_out
        else:
            tune_W_out = torch.nn.Linear(in_features=terminator.top.W_out.in_features, out_features=terminator.top.W_out.out_features, bias=True).to(device=dev)
            torch.nn.init.xavier_uniform_(tune_W_out.weight)
            torch.nn.init.zeros_(tune_W_out.bias)
        terminator.top.W_dropout = torch.nn.Dropout(p=run_hparams['ft_dropout']).to(device=dev)
        terminator.top.W_out = tune_W_out
        if run_hparams['node_finetune']:
            feats = 20 if (model_hparams['node_self_sub'] and model_hparams['restrict_pifold_output']) else 22
            tune_V_out = torch.nn.Linear(in_features=feats, out_features=feats, bias=True).to(device=dev)
            torch.nn.init.xavier_uniform_(tune_V_out.weight)
            torch.nn.init.zeros_(tune_V_out.bias)
            terminator.top.V_dropout = torch.nn.Dropout(p=run_hparams['ft_dropout']).to(device=dev)
            terminator.top.V_out = tune_V_out
        term_params = tune_W_out.parameters()
    else:
        tune_W_out = terminator.top.W_out
    if model_hparams['use_transfer_model']:
        if model_hparams['transfer_model'] == 'nodes': transfer_model = TransferModel(embeddings_dim=model_hparams['energies_hidden_dim'], out_dim=None)
        else: 
            transfer_model = PottsTransferModel(embeddings_dim=model_hparams['transfer_hidden_dim'], out_dim=model_hparams['transfer_hidden_dim'], mult_etabs=model_hparams['mult_etabs'], single_etab_dense=model_hparams['single_etab_dense'], use_light_attn=model_hparams['transfer_use_light_attn'], num_linear=model_hparams['transfer_num_linear'], use_out=model_hparams['transfer_use_out'], use_esm=model_hparams['transfer_use_esm'])
            print('Potts transfer model')
        transfer_model = transfer_model.to(device=dev)
        if finetune or run_hparams['train_transfer_only']: term_params = transfer_model.parameters()
    else:
        transfer_model = None
    terminator.transfer_model = transfer_model

    if finetune: # and run_hparams['unfreeze_esm']: # freeze all but the last output layer
        for (name, module) in terminator.named_children():
            if name == "top":
                for (n, m) in module.named_children():
                    if 'node_combine' in n or 'edge_combine' in n or 'encoder' in n:
                        for layer in m[:-1]:
                            for param in layer.parameters():
                                param.requires_grad = False

                        for param in m[-1].parameters():
                            param.requires_grad = True
                        # print(n)
                        # for layer in m:
                        #     for param in layer.parameters():
                        #         print('\t', param.requires_grad)
                    elif 'dec_out' in n or 'W_out' in n:
                        m.requires_grad = True
                        # print(n, m.requires_grad)
                    else:
                        m.requires_grad = False
            elif name == "transfer_model":
                module.requires_grad = True
            else:
                module.requires_grad = False
    elif run_hparams['train_transfer_only']:
        for (name, module) in terminator.named_children():
            if name == "transfer_model":
                module.requires_grad = True
            else:
                module.requires_grad = False

    if finetune or run_hparams['train_transfer_only']: term_params = (param for param in terminator.parameters() if param.requires_grad)
    else: term_params = terminator.parameters()
    loss_fn = construct_loss_fn(run_hparams)
    optimizer = get_std_opt(term_params,
                            d_model=model_hparams['energies_hidden_dim'],
                            regularization=run_hparams['regularization'],
                            state=None,
                            finetune=finetune,
                            finetune_lr=run_hparams["finetune_lr"],
                            warmup=run_hparams['warmup'],
                            lr_multiplier=run_hparams['lr_multiplier'])
    best_val_loss = np.Inf
    iters_since_best_loss = 0
    finetune_loss = 'evcouplings_loss_corr'
    if finetune_loss not in run_hparams['loss_config'].keys():
        finetune_loss = 'rocklin_loss'
    if finetune_loss not in run_hparams['loss_config'].keys():
        finetune_loss = 'esm_cmap_loss_tp'
    if finetune_loss not in run_hparams['loss_config'].keys():
        finetune_loss = 'nlll_loss'
    if finetune_loss not in run_hparams['loss_config'].keys():
        finetune_loss = 'sortcery_loss'
    if finetune_loss not in run_hparams['loss_config'].keys():
        finetune_loss = 'sortcery_loss_mult_bb'
    if finetune_loss not in run_hparams['loss_config'].keys():
        finetune_loss = 'stability_loss'
    if finetune_loss not in run_hparams['loss_config'].keys():
        finetune_loss = 'stability_loss_loop'
    if finetune_loss not in run_hparams['loss_config'].keys():
        finetune_loss = 'cluster_stability_loss'
    if finetune_loss not in run_hparams['loss_config'].keys():
        finetune_loss = 'cluster_stability_loss_mse'
    if finetune_loss not in run_hparams['loss_config'].keys():
        finetune_loss = 'stability_loss_mse'
    if finetune_loss not in run_hparams['loss_config'].keys():
        finetune_loss = 'stability_loss_mse_loop'
    if finetune_loss not in run_hparams['loss_config'].keys():
        finetune_loss = 'stability_loss_mse_raw'
    if finetune_loss not in run_hparams['loss_config'].keys():
        finetune_loss = 'nlcpl'
    if finetune_loss not in run_hparams['loss_config'].keys():
        finetune_loss = 'loss_smoothed'
    if finetune_loss not in run_hparams['loss_config'].keys():
        finetune_loss = 'thermo_loss_mse'
    if finetune_loss not in run_hparams['loss_config'].keys():
        finetune_loss = 'thermo_loss_corr'
    if finetune_loss not in run_hparams['loss_config'].keys():
        finetune_loss = 'cluster_stability_loss_diff'
    if finetune_loss not in run_hparams['loss_config'].keys():
        finetune_loss = 'cluster_stability_loss_diff_mse'
    if finetune_loss not in run_hparams['loss_config'].keys():
        finetune_loss = 'global_monogram_loss'
    if finetune_loss not in run_hparams['loss_config'].keys():
        finetune_loss = 'likelihood_ener_loss'
    
    
    assert finetune_loss in run_hparams['loss_config'].keys()
    # if loss involves esm, load model
    if use_esm or run_hparams['num_recycles'] > 0 or load_esm:
        esm, alphabet = esmlib.pretrained.esm2_t30_150M_UR50D()
        esm = esm.requires_grad_(False)
        
        batch_converter = alphabet.get_batch_converter()
        converter_state = torch.load(args.converter_path, map_location=torch.device(args.dev))
        converter = Converter(22, 640)
        converter.load_state_dict(converter_state)
        converter = converter.requires_grad_(False)
        if finetune_loss == 'esm_cmap_loss_tp':
            esm = esm.requires_grad_(True)
            converter = converter.requires_grad_(True)
        if dev == 'cpu':
            esm = esm.float()  # convert to fp32 as ESM-2 in fp16 is not supported on CPU
            esm = esm.cpu()
            converter = converter.cpu()
        else:
            esm = esm.float()
            esm = esm.to(device='cuda:0')
            converter = converter.to(device='cuda:0')
            # esm = esm.to('cpu')
        if finetune_loss != 'esm_cmap_loss_tp':
            esm.eval()
            converter.eval()
    else:
        esm = None
        batch_converter = None
        converter = None
    if esm is not None:
        esm = esm.cpu()
        converter = converter.cpu()
        for p in esm.parameters():
            p.requires_grad = False
        for p in converter.parameters():
            p.requires_grad = False
    start_epoch = 0

    # If tracking aux loss, load data and create loss_fn
    if run_hparams['aux_loss'] is not None:
        base_dataset = copy.deepcopy(args.dataset)
        args.dataset = run_hparams['aux_dataset']
        aux_dataset, aux_batch_sampler = _setup_dataloaders_wds(args, run_hparams, model_hparams, dataset_name=run_hparams['aux_dataset_name'], suffix=run_hparams['aux_suffix'], aux_data=True)
        aux_dataloader = DataLoader(aux_dataset,
                                batch_sampler=aux_batch_sampler,
                                collate_fn=aux_batch_sampler.package,
                                pin_memory=True)
        aux_loss_fn = construct_loss_fn(run_hparams, aux_loss=True)
        args.dataset = base_dataset
        if "aux_loss" not in training_curves.keys(): training_curves["aux_loss"] = []
    best_aux = False
    print(f'num epochs: {args.epochs}')
    try:
        for epoch in range(start_epoch, args.epochs):
            print('epoch: ', epoch)
            train_batch_sampler._set_epoch(epoch)
            val_batch_sampler._set_epoch(epoch)
            test_batch_sampler._set_epoch(epoch)

            epoch_loss, epoch_ld, _ = run_epoch(terminator, train_dataloader, loss_fn, run_hparams, epoch=epoch, optimizer=optimizer, grad=True, dev=dev, finetune=finetune, isDataParallel=isDataParallel, esm=esm, batch_converter=batch_converter, converter=converter, num_recycles=run_hparams['num_recycles'], keep_seq=run_hparams['keep_seq_recycle'], keep_sc_mask_recycle=run_hparams['keep_sc_mask_recycle'], esm_options=esm_options, only_loss_recycle=run_hparams['only_loss_recycle'], recycle_confidence=run_hparams['recycle_confidence'], recycle_teacher_forcing=run_hparams['recycle_teacher_forcing'], keep_sc_mask_loss=run_hparams['keep_sc_mask_loss'], tune_W_out=tune_W_out, energy_merge_fn=model_hparams['energy_merge_fn'], use_transfer_model=model_hparams['use_transfer_model'],train_transfer_only=run_hparams['train_transfer_only'], from_train=False, max_loop_tokens=run_hparams['max_loop_tokens'], node_finetune=run_hparams['node_finetune'], node_self_sub=model_hparams['node_self_sub'])
            print('epoch loss', epoch_loss, 'epoch_ld', epoch_ld)
            writer.add_scalar('training loss', epoch_loss, epoch)

            # validate
            val_loss, val_ld, _ = run_epoch(terminator, val_dataloader, loss_fn, run_hparams, epoch=epoch, grad=False, dev=dev, finetune=finetune, isDataParallel=isDataParallel, esm=esm, batch_converter=batch_converter, converter=converter, num_recycles=run_hparams['num_recycles'], keep_seq=run_hparams['keep_seq_recycle'], keep_sc_mask_recycle=run_hparams['keep_sc_mask_recycle'], esm_options=esm_options, only_loss_recycle=run_hparams['only_loss_recycle'], recycle_confidence=run_hparams['recycle_confidence'], recycle_teacher_forcing=run_hparams['recycle_teacher_forcing'], keep_sc_mask_loss=run_hparams['keep_sc_mask_loss'], tune_W_out=tune_W_out, energy_merge_fn=model_hparams['energy_merge_fn'], use_transfer_model=model_hparams['use_transfer_model'],train_transfer_only=run_hparams['train_transfer_only'], from_train=False, max_loop_tokens=run_hparams['max_loop_tokens'], node_finetune=run_hparams['node_finetune'], node_self_sub=model_hparams['node_self_sub'])
            print('epoch loss', epoch_loss, 'epoch_ld', epoch_ld)

            print('val loss', val_loss, 'val ld', val_ld, 'best val loss ', best_validation)
            writer.add_scalar('val loss', val_loss, epoch)

            if False and run_hparams['aux_loss'] is not None:
                if run_hparams['aux_dev'] == 'cpu':
                    terminator = terminator.cpu()
                aux_loss, aux_ld, _ = run_epoch(terminator, aux_dataloader, aux_loss_fn, epoch=epoch, grad=run_hparams['aux_grad'], dev=run_hparams['aux_dev'], esm=esm, batch_converter=batch_converter, converter=converter, num_recycles=run_hparams['num_recycles'], keep_seq=run_hparams['keep_seq_recycle'], keep_sc_mask_recycle=run_hparams['keep_sc_mask_recycle'], esm_options=esm_options, only_loss_recycle=run_hparams['only_loss_recycle'], recycle_confidence=run_hparams['recycle_confidence'], keep_sc_mask_loss=run_hparams['keep_sc_mask_loss'], silenced=True, tune_W_out=tune_W_out, energy_merge_fn=model_hparams['energy_merge_fn'], use_transfer_model=model_hparams['use_transfer_model'], from_train=False, max_loop_tokens=run_hparams['max_loop_tokens'])
                print('auxiliary loss', aux_loss, 'val ld', aux_ld, 'best aux loss ', best_aux)
                aux_comp = (not best_aux) or aux_loss < best_aux
                training_curves["aux_loss"].append((aux_loss, aux_ld))
                writer.add_scalar('aux loss', aux_loss, epoch)
                if run_hparams['aux_dev'] == 'cpu' and dev == 'cuda:0':
                    terminator = terminator.cuda()
            else:
                aux_comp = False

            training_curves["train_loss"].append((epoch_loss, epoch_ld))
            training_curves["val_loss"].append((val_loss, val_ld))
            
            comp = (val_ld[finetune_loss]['loss'] < best_validation) if finetune else (val_loss < best_validation)

            # save a state checkpoint
            terminator.top.W_out = tune_W_out
            checkpoint_state = {
                'epoch': epoch,
                'state_dict': terminator.state_dict(),
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
                best_checkpoint = copy.deepcopy(terminator.state_dict())
                torch.save(checkpoint_state, os.path.join(run_dir, 'net_best_checkpoint.pt'))
            elif aux_comp:
                best_aux = aux_loss
                best_aux_checkpoint = copy.deepcopy(terminator.state_dict())
                torch.save(checkpoint_state, os.path.join(run_dir, 'net_aux_best_checkpoint.pt'))
                iters_since_best_loss = 0
            else:
                iters_since_best_loss += 1
            if iters_since_best_loss > run_hparams['patience']:
                print("Exiting due to early stopping")
                break

    except KeyboardInterrupt:
        pass

    # save model params
    print(training_curves)
    torch.save(terminator.state_dict(), os.path.join(run_dir, 'net_last.pt'))
    torch.save(best_checkpoint, os.path.join(run_dir, 'net_best.pt'))

    # test
    terminator.load_state_dict(best_checkpoint)
    if 'edge_loss_msa' in run_hparams['loss_config'].keys() or 'msa_weighted_nlcpl' in run_hparams['loss_config'].keys():
        run_hparams['loss_config'] = {'edge_loss': 1}
        loss_fn = construct_loss_fn(run_hparams)

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
    parser.add_argument('--orig_model', help='Baseline model to finetune', default=None)
    parser.add_argument('--n_trials', help="number of trials for optuna optimization", default=0, type=int)
    parser.add_argument('--converter_path', help='Path to model converting coord to esm embeddings', default='/home/gridsan/fbirnbaum/TERMinator/analysis/converter_model_coord.pt')
    parsed_args = parser.parse_args()
    # parsed_args.dev = 'cpu'
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

    finetune(parsed_args)