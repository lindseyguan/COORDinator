import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from terminator.data.data import (WDSBatchSamplerWrapper, TERMLazyDataset, TERMBatchSamplerWrapper, TERMDataset, TERMLazyBatchSampler)
from terminator.models.TERMinator import TERMinator
from terminator.utils.model.loop_utils import run_epoch
from terminator.utils.model.loss_fn import construct_loss_fn
from terminator.utils.common import backwards_compat
from terminator.data.load_data import load_data
from terminator.utils.model.transfer_model import TransferModel, PottsTransferModel, LightAttention
import sys
import json
import esm as esmlib
import copy
import webdataset as wds

class energy_transfer(nn.Module):
    def __init__(self, model_hparams):
        super().__init__()

        light_attention = LightAttention(embeddings_dim=model_hparams['energies_hidden_dim'] * model_hparams['num_edge_transfer'] )
        energy_out = nn.Sequential()
        for sz1, sz2 in zip(model_hparams['edge_transfer_in'], model_hparams['edge_transfer_out']):
            energy_out.append(nn.ReLU())
            energy_out.append(nn.Linear(sz1, sz2))
        self.light_attention = light_attention
        self.energy_out = energy_out

    def forward(self, h_E_combine):
        h_E_combine = self.light_attention(h_E_combine)
        h_E = self.energy_out(h_E_combine)
        return h_E

def parse_pdbs(pdb_dir, pdb_list):
    dataset = []
    return dataset

def load_model(args, model_hparams=None, run_hparams=None):
    if args.verbose: print('loading model')
    if torch.cuda.device_count() == 0:
        dev = "cpu"
    else:
        dev = args.dev

    if args.subset:
        test_ids = []
        with open(os.path.join(args.subset), 'r') as f:
            for line in f:
                test_ids += [line.strip()]
    elif len(args.subset_list) > 0:
        test_ids = args.subset_list
    else:
        test_ids = []
    if model_hparams is None:
        with open(os.path.join(args.model_dir, "model_hparams.json")) as fp:
            model_hparams = json.load(fp)
    if run_hparams is None:
        with open(os.path.join(args.model_dir, "run_hparams.json")) as fp:
            run_hparams = json.load(fp)

    # backwards compatability
    model_hparams, run_hparams = backwards_compat(model_hparams, run_hparams)
    model_hparams['center_node_ablation'] = args.center_node_ablation
    model_hparams['center_node'] = args.center_node
    if args.verbose:
        print('run_hparams: ', run_hparams)
        print('model_hparams: ', model_hparams)
    use_esm = model_hparams['nodes_to_probs'] or model_hparams['edges_to_seq']
    load_esm = False
    for loss_key in run_hparams['loss_config'].keys():
        if loss_key.find('esm_cmap') > -1 or loss_key.find('esm_loss') > -1:
            use_esm = not model_hparams['mpnn_decoder']
            load_esm = model_hparams['mpnn_decoder']
            break
            
    # run_hparams["loss_config"] = {"nlcpl": 1}
    if args.return_esm:
        try:
            esm = args.esm
            assert esm is not None
            batch_converter = args.batch_converter
        except:
            if model_hparams['esm_feats'] or run_hparams['num_recycles'] > 0:
                if model_hparams['esm_model'] == '650':
                    esm, alphabet = esmlib.pretrained.esm2_t33_650M_UR50D()
                elif model_hparams['esm_model'] == '3B':
                    esm, alphabet = esmlib.pretrained.esm2_t36_3B_UR50D()
                else:
                    esm, alphabet = esmlib.pretrained.esm2_t30_150M_UR50D()
                batch_converter = alphabet.get_batch_converter()
                esm = esm.eval().cpu()
            elif model_hparams['rla_feats']:
                _, alphabet = esmlib.pretrained.esm2_t30_150M_UR50D()
                batch_converter = alphabet.get_batch_converter()
                sys_keys = copy.deepcopy(list(sys.modules.keys()))
                for mod in sys_keys:
                    if mod.find('terminator') > -1:
                        del sys.modules[mod]
                sys.path.insert(0,'/home/gridsan/fbirnbaum/TERMinator-public_mirror/scripts/models/train')
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
            args.esm = esm
            args.batch_converter = batch_converter

    from terminator.data.data import (WDSBatchSamplerWrapper, TERMLazyDataset, TERMBatchSamplerWrapper, TERMDataset, TERMLazyBatchSampler)
    from terminator.models.TERMinator import TERMinator
    from terminator.utils.model.loop_utils import run_epoch
    from terminator.utils.model.loss_fn import construct_loss_fn
    cols = ['inp.pyd']
    if args.from_wds:
        test_dataset = wds.WebDataset(args.dataset).decode().to_tuple(*cols)
    else:
        print('seq df: ', args.seq_df)
        test_dataset, test_dataset_target, test_dataset_binder = load_data(args.dataset, args.pdb_list, args.add_gap, args.ener_col, args.sep_complex, args.ener_data, args.adj_index, args.seq_df, args.seq_col)
    if len(test_ids) > 0:
        test_dataset = [data for data in test_dataset if data[0]['pdb'] in test_ids]
    test_dataset = [data for data in test_dataset if (data[0]['coords'].shape[0] > model_hparams['k_neighbors']) and (data[0]['coords'].shape[0] < run_hparams['max_seq_tokens'])]
    # test_batch_sampler = TERMBatchSampler(False, test_dataset, batch_size=1, shuffle=False, msa_type='single', dev=args.dev)
    test_batch_sampler = WDSBatchSamplerWrapper(ddp=False)
    test_batch_sampler.sampler.monogram_stats = args.monogram_stats
 
    if 'sc_screen_range' in dir(args):
        sc_screen_range = args.sc_screen_range
    else:
        sc_screen_range = []
    if 'no_mask' in dir(args):
        no_mask = args.no_mask
    else:
        no_mask = False
    if 'mpnn_dihedrals' in dir(args):
        mpnn_dihedrals = args.mpnn_dihedrals
    else:
        mpnn_dihedrals = False
    if 'mask_interface' in dir(args):
        run_hparams['mask_interface'] = args.mask_interface
    if 'half_interface' in dir(args):
        run_hparams['half_interface'] = args.half_interface
    if 'chain_handle' in dir(args):
        model_hparams['chain_handle'] = args.chain_handle
    if 'use_sc' in dir(args):
        run_hparams['use_sc'] = args.use_sc
    esm_options = {'use_esm_attns': run_hparams['use_esm_attns'], 'esm_embed_layer': model_hparams['esm_embed_layer'], 'from_rla': model_hparams['rla_feats'],
                   'use_reps': run_hparams['use_reps'], 'connect_chains': run_hparams['connect_chains'], 'one_hot': model_hparams['one_hot']
                   }
    if 'mask_neighbors' in dir(args):
        run_hparams['mask_neighbors'] = args.mask_neighbors
    if 'random_interface' in dir(args):
        run_hparams['random_interface'] = args.random_interface
    if 'inter_cutoff' in dir(args):
        run_hparams['inter_cutoff'] = args.inter_cutoff
    if 'batch_size' in dir(args):
        batch_size = args.batch_size
    else:
        batch_size = 1

    if args.verbose:
        print('use sc: ', run_hparams['use_sc'])
        print('mask neighbors: ', run_hparams['mask_neighbors'])
        print('name cluster: ', args.name_cluster)
        print('no mask: ', no_mask)
        print('pepmlm: ', model_hparams['from_pepmlm'])
        print('dev: ', dev)
        print(run_hparams["use_struct_predict"])
        print(("use_struct_predict" in dir(args) and not args.use_struct_predict))

    if not run_hparams["use_struct_predict"] or ("use_struct_predict" in dir(args) and not args.use_struct_predict):
        model_hparams["struct_predict"] = False

    if args.verbose:
        print('struct predict: ', model_hparams['struct_predict'])

    test_batch_sampler = test_batch_sampler.sampler(test_batch_sampler.ddp, test_dataset, batch_size=batch_size, shuffle=False, use_sc=run_hparams["use_sc"], mpnn_dihedrals=mpnn_dihedrals, no_mask=no_mask, sc_mask=args.sc_mask, sc_mask_rate=args.sc_mask_rate, base_sc_mask=args.base_sc_mask, sc_mask_schedule=run_hparams['sc_mask_schedule'], sc_info=run_hparams['sc_info'], mask_neighbors=run_hparams['mask_neighbors'], mask_interface=run_hparams['mask_interface'], half_interface=run_hparams['half_interface'], inter_cutoff=run_hparams['inter_cutoff'], dev=dev, esm=args.esm, batch_converter=args.batch_converter, use_esm_attns=run_hparams['use_esm_attns'], use_reps=run_hparams['use_reps'], post_esm_mask=run_hparams['post_esm_mask'], from_rla=model_hparams['rla_feats'], esm_embed_layer=model_hparams['esm_embed_layer'], connect_chains=run_hparams['connect_chains'], convert_to_esm=model_hparams['convert_to_esm'], one_hot=model_hparams['one_hot'], sc_screen=args.sc_screen, sc_screen_range=sc_screen_range, name_cluster=args.name_cluster, chain_handle=model_hparams['chain_handle'], random_interface=run_hparams['random_interface'], from_pepmlm=model_hparams['from_pepmlm'], openfold_backbone=model_hparams['struct_predict'], alphabetize_data=args.alphabetize_data)

    # test_batch_sampler = test_batch_sampler.sampler(test_batch_sampler.ddp, test_dataset, batch_size=1, shuffle=False, use_sc=run_hparams["use_sc"], mpnn_dihedrals=mpnn_dihedrals, no_mask=no_mask, sc_mask=args.sc_mask, sc_mask_rate=args.sc_mask_rate, base_sc_mask=args.base_sc_mask, sc_mask_schedule=run_hparams['sc_mask_schedule'], sc_info=run_hparams['sc_info'], mask_neighbors=run_hparams['mask_neighbors'], mask_interface=run_hparams['mask_interface'], half_interface=run_hparams['half_interface'], interface_pep=run_hparams['interface_pep'], inter_cutoff=run_hparams['inter_cutoff'], dev='cpu', esm=esm, batch_converter=batch_converter, use_esm_attns=run_hparams['use_esm_attns'], use_reps=run_hparams['use_reps'], post_esm_mask=run_hparams['post_esm_mask'], from_rla=model_hparams['rla_feats'], esm_embed_layer=model_hparams['esm_embed_layer'], connect_chains=run_hparams['connect_chains'], convert_to_esm=model_hparams['convert_to_esm'], one_hot=model_hparams['one_hot'],all_batch=run_hparams['all_batch'], sc_screen=args.sc_screen, sc_screen_range=sc_screen_range)

    test_dataloader = DataLoader(test_dataset,
                                 batch_sampler=test_batch_sampler,
                                 collate_fn=test_batch_sampler.package) 
    

    if args.sep_complex:
        test_batch_sampler_target = WDSBatchSamplerWrapper(ddp=False)
        test_batch_sampler_binder = WDSBatchSamplerWrapper(ddp=False)

        test_batch_sampler_target = test_batch_sampler_target.sampler(test_batch_sampler_target.ddp, test_dataset_target, batch_size=batch_size, shuffle=False, use_sc=run_hparams["use_sc"], mpnn_dihedrals=mpnn_dihedrals, no_mask=no_mask, sc_mask=args.sc_mask, sc_mask_rate=args.sc_mask_rate, base_sc_mask=args.base_sc_mask, sc_mask_schedule=run_hparams['sc_mask_schedule'], sc_info=run_hparams['sc_info'], mask_neighbors=run_hparams['mask_neighbors'], mask_interface=run_hparams['mask_interface'], half_interface=run_hparams['half_interface'], inter_cutoff=run_hparams['inter_cutoff'], dev=dev, esm=args.esm, batch_converter=args.batch_converter, use_esm_attns=run_hparams['use_esm_attns'], use_reps=run_hparams['use_reps'], post_esm_mask=run_hparams['post_esm_mask'], from_rla=model_hparams['rla_feats'], esm_embed_layer=model_hparams['esm_embed_layer'], connect_chains=run_hparams['connect_chains'], convert_to_esm=model_hparams['convert_to_esm'], one_hot=model_hparams['one_hot'], sc_screen=args.sc_screen, sc_screen_range=sc_screen_range, name_cluster=args.name_cluster, chain_handle=model_hparams['chain_handle'], random_interface=run_hparams['random_interface'], from_pepmlm=model_hparams['from_pepmlm'], openfold_backbone=model_hparams['struct_predict'], alphabetize_data=args.alphabetize_data)

        test_dataloader_target = DataLoader(test_dataset_target,
                                    batch_sampler=test_batch_sampler_target,
                                    collate_fn=test_batch_sampler_target.package) 
        
        test_batch_sampler_binder = test_batch_sampler_binder.sampler(test_batch_sampler_binder.ddp, test_dataset_binder, batch_size=batch_size, shuffle=False, use_sc=run_hparams["use_sc"], mpnn_dihedrals=mpnn_dihedrals, no_mask=no_mask, sc_mask=args.sc_mask, sc_mask_rate=args.sc_mask_rate, base_sc_mask=args.base_sc_mask, sc_mask_schedule=run_hparams['sc_mask_schedule'], sc_info=run_hparams['sc_info'], mask_neighbors=run_hparams['mask_neighbors'], mask_interface=run_hparams['mask_interface'], half_interface=run_hparams['half_interface'], inter_cutoff=run_hparams['inter_cutoff'], dev=dev, esm=args.esm, batch_converter=args.batch_converter, use_esm_attns=run_hparams['use_esm_attns'], use_reps=run_hparams['use_reps'], post_esm_mask=run_hparams['post_esm_mask'], from_rla=model_hparams['rla_feats'], esm_embed_layer=model_hparams['esm_embed_layer'], connect_chains=run_hparams['connect_chains'], convert_to_esm=model_hparams['convert_to_esm'], one_hot=model_hparams['one_hot'], sc_screen=args.sc_screen, sc_screen_range=sc_screen_range, name_cluster=args.name_cluster, chain_handle=model_hparams['chain_handle'], random_interface=run_hparams['random_interface'], from_pepmlm=model_hparams['from_pepmlm'], openfold_backbone=model_hparams['struct_predict'], alphabetize_data=args.alphabetize_data)

        test_dataloader_binder = DataLoader(test_dataset_binder,
                                    batch_sampler=test_batch_sampler_binder,
                                    collate_fn=test_batch_sampler_binder.package) 
    else:
        test_dataloader_target, test_dataloader_binder = None, None

    if args.data_only:
        return test_dataloader
    
    if args.verbose:
        print('Loading from: ', os.path.join(args.model_dir, args.checkpoint))
    best_checkpoint_state = torch.load(os.path.join(args.model_dir, args.checkpoint), map_location=dev, weights_only=False)
    best_checkpoint = best_checkpoint_state['state_dict']
    if use_esm or run_hparams['num_recycles'] > 0 or load_esm:
        if 'top.V_out.bias' not in best_checkpoint.keys() and model_hparams['nodes_to_probs']:
            best_checkpoint['top.V_out.bias'] = torch.rand(20)
            best_checkpoint['top.V_out.weight'] = torch.rand((20, 128))
        if 'top.E_outs.0.bias' not in best_checkpoint.keys() and model_hparams['edges_to_seq']:
            best_checkpoint['top.E_outs.0.bias'] = torch.rand(256)
            best_checkpoint['top.E_outs.0.weight'] = torch.rand((256, 600))
            best_checkpoint['top.E_outs.1.bias'] = torch.rand(64)
            best_checkpoint['top.E_outs.1.weight'] = torch.rand((64, 256))
            best_checkpoint['top.E_outs.2.bias'] = torch.rand(1)
            best_checkpoint['top.E_outs.2.weight'] = torch.rand((1, 64))


    if not model_hparams['nodes_to_probs']:
        if not args.load_V_out and 'top.V_out.bias' in best_checkpoint.keys():
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
    if 'top.features.node_layers.0.weight' not in best_checkpoint.keys():
        model_hparams['old'] = True
    else:
        model_hparams['old'] = False

    if not model_hparams['struct_predict']:
        key_list = copy.deepcopy(list(best_checkpoint.keys()))
        for key in key_list:
            if 'struct_module' in key:
                del best_checkpoint[key]


    if args.verbose:
        print(model_hparams['old'])
        print('old: ', model_hparams['old'])
    if 'zero_node_features' in dir(args):
        model_hparams['zero_node_features'] = args.zero_node_features
    if 'zero_edge_features' in dir(args):
        model_hparams['zero_edge_features'] = args.zero_edge_features

    terminator = TERMinator(hparams=model_hparams, device=dev, use_esm=use_esm, edges_to_seq=model_hparams['edges_to_seq'])
    if args.load_V_out:
            feats = 20 if (model_hparams['node_self_sub'] and model_hparams['restrict_pifold_output']) else 22
            tune_V_out = torch.nn.Linear(in_features=feats, out_features=feats, bias=True).to(device=dev)
            terminator.top.V_out = tune_V_out
    if model_hparams['edge_transfer']:
        terminator.energy_transfer = energy_transfer(model_hparams).to(device=dev)
        terminator.top.W_out = None

    terminator.load_state_dict(best_checkpoint)
    terminator.to(dev)
    terminator.eval()
    if args.return_esm:
        return terminator, test_dataloader, test_dataloader_target, test_dataloader_binder, test_batch_sampler, esm, batch_converter, esm_options, run_hparams, model_hparams

    return terminator, test_dataloader, test_dataloader_target, test_dataloader_binder, test_batch_sampler, run_hparams, model_hparams