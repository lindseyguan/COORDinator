import os
import numpy as np
import sys
sys.path.insert(0, '/home/gridsan/fbirnbaum/joint-protein-embs')
import src.models_and_optimizers as model_utils
from types import SimpleNamespace
import src.data_utils as data_utils
from clip_main import get_wds_loaders
import torch
import webdataset as wds
import esm as esmlib
from transformers import EsmTokenizer
from torch.nn.utils.rnn import pad_sequence

def load_rla(dev, data_root="/data1/groups/keatinglab/fosterb/ingraham_data", train_wds_path="ingraham_clip_test.wds", val_wds_path="ingraham_clip_test.wds"):
    cwd = os.getcwd()
    os.chdir('/home/gridsan/fbirnbaum/joint-protein-embs')
    model_dir = "new_blacklist/version_0"
    CLIP_MODE = False
    ROOT = "/data1/groups/keating_madry/runs/"
    root_path = os.path.join(ROOT, model_dir)
    path = os.path.join(root_path, "checkpoints/checkpoint_latest.pt")
    args_path = os.path.join(ROOT, model_dir, 
                            [u for u in os.listdir(os.path.join(ROOT, model_dir)) if u.endswith('.pt')][0])
    backwards_compat = {
        'masked_rate': -1,
        'masked_mode': 'MASK',
        'lm_only_text': 1,
        'lm_weight': 1,
        'resid_weight': 1,
        'language_head': False,
        'language_head_type': 'MLP',
        'zip_enabled': False,
        'num_mutations': False,
    }
    hparams = torch.load(args_path)
    args_dict = hparams['args']
    args_dict['batch_size'] = 1
    args_dict['data_root'] = data_root
    args_dict['train_wds_path'] = train_wds_path
    args_dict['val_wds_path'] = val_wds_path
    args_dict['distributed'] = 0
    args_dict['blacklist_file'] = ''
    for k in backwards_compat.keys():
        if k not in args_dict:
            args_dict[k] = backwards_compat[k]
    args = SimpleNamespace(**args_dict)

    print('rla args: ', vars(args))

    coordinator_params = data_utils.get_coordinator_params(os.path.join('/home/gridsan/fbirnbaum/joint-protein-embs', args.coordinator_hparams))
    coordinator_params['num_positional_embeddings'] = args.gnn_num_pos_embs
    coordinator_params['zero_out_pos_embs']= args.gnn_zero_out_pos_embs
    coordinator_params['clip_mode'] = True
    args_dict['arch']= '/data1/groups/keating_madry/huggingface/esm2_t30_150M_UR50D'

    model = model_utils.load_model(path, args_dict['arch'], dev)
    os.chdir(cwd)
    return model
