import sys
import wandb
import argparse
import copy
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import MeanSquaredError, R2Score, SpearmanCorrCoef, PearsonCorrCoef
from omegaconf import OmegaConf
import datetime
import esm as esmlib
from terminator.utils.model.transfer_model_real import TransferModel
from train import _setup_dataloaders_wds, _setup_hparams, _setup_model

def get_metrics():
    return {
        "r2": R2Score(),
        "mse": MeanSquaredError(squared=True),
        "rmse": MeanSquaredError(squared=False),
        "spearman": SpearmanCorrCoef(),
    }

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
        elif key == 'gvp_data':
            data_dict['gvp_data'] = [data.to(dev) for data in data_dict['gvp_data']]
        elif key == 'edge_update_inds':
            data_dict['edge_update_inds'] = [data.to(dev) for data in data_dict['edge_update_inds']]
        elif key == 'chain_ends_info' and isinstance(data_dict['chain_ends_info'], dict):
            data_dict['chain_ends_info'] = {k: v.to(dev) for k, v in data_dict[key].items()}


class TransferModelPL(pl.LightningModule):
    """Class managing training loop with pytorch lightning"""
    def __init__(self, coordinator, learn_rate = 0.001, mpnn_learn_rate = None, lr_schedule = True, device='cuda:0'):
        super().__init__()
        self.model = TransferModel()
        self.coordinator = coordinator

        self.learn_rate = learn_rate
        self.mpnn_learn_rate = mpnn_learn_rate
        self.lr_schedule = lr_schedule

        # set up metrics dictionary
        self.metrics = nn.ModuleDict()
        for split in ("train_metrics", "val_metrics"):
            self.metrics[split] = nn.ModuleDict()
            out = "ddG"
            self.metrics[split][out] = nn.ModuleDict()
            for name, metric in get_metrics().items():
                self.metrics[split][out][name] = metric

        self.dev = device

    def forward(self, *args):
        return self.model(*args)

    def shared_eval(self, batch, batch_idx, prefix):

        _to_dev(batch, self.dev)
        max_seq_len = max(batch['seq_lens'].tolist())
        _, h_V, _, _, _, _ = self.coordinator(batch, max_seq_len, use_transfer_model=True)
        h_V = h_V[0]
        preds, reals = self(h_V, batch)

        ddg_mses = []
        for pred, real in zip(preds, reals):
            ddg_mses.append(F.mse_loss(pred, real))
            for metric in self.metrics[f"{prefix}_metrics"]["ddG"].values():
                metric.update(pred, real)

        loss = 0.0 if len(ddg_mses) == 0 else torch.stack(ddg_mses).mean()
        on_step = False
        on_epoch = not on_step

        output = "ddG"
        for name, metric in self.metrics[f"{prefix}_metrics"][output].items():
            try:
                metric.compute()
            except ValueError:
                continue
            self.log(f"{prefix}_{output}_{name}", metric, prog_bar=True, on_step=on_step, on_epoch=on_epoch,
                        batch_size=len(batch))
        if loss == 0.0:
            return None
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'test')

    def configure_optimizers(self):
        if self.stage == 2: # for second stage, drop LR by factor of 10
            self.learn_rate /= 10.
            print('New second-stage learning rate: ', self.learn_rate)

        param_list = []

        if self.model.lightattn:  # adding light attention parameters
            if self.stage == 2:
                param_list.append({"params": self.model.light_attention.parameters(), "lr": 0.})
            else:
                param_list.append({"params": self.model.light_attention.parameters()})
        if self.model.condense_esm:
            if self.stage == 2:
                param_list.append({"params": self.model.esm_reps_embedding.parameters(), "lr": 0.})
            else:
                param_list.append({"params": self.model.esm_reps_embedding.parameters()})


        mlp_params = [
            {"params": self.model.both_out.parameters()},
            {"params": self.model.ddg_out.parameters()}
            ]

        param_list = param_list + mlp_params
        opt = torch.optim.AdamW(param_list, lr=self.learn_rate)

        if self.lr_schedule: # enable additional lr scheduler conditioned on val ddG mse
            lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=opt, verbose=True, mode='min', factor=0.5)
            return {
                'optimizer': opt,
                'lr_scheduler': lr_sched,
                'monitor': 'val_ddG_mse'
            }
        else:
            return opt

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
        training_curves = last_checkpoint_state["training_curves"]
    else:
        best_checkpoint_state, last_checkpoint_state = None, None
        best_checkpoint = None
        best_validation = 10e8
        last_optim_state = None
        start_epoch = 0
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
    print('use esm: ', use_esm)
    checkpoint_dict = _load_checkpoint(run_dir, dev, model_hparams, True, top_nodes=use_esm, use_esm=use_esm, predict_confidence=model_hparams['predict_confidence'])
    from_base = True
    if os.path.isfile(os.path.join(run_dir, 'net_best_checkpoint.pt')):
        from_base=False
    best_checkpoint = checkpoint_dict["best_checkpoint"]
    if 'top.features.node_layers.0.weight' not in best_checkpoint.keys():
        model_hparams['old'] = True
    else:
        model_hparams['old'] = False
    finetune = run_hparams["finetune"]

    # construct terminator, loss fn, and optimizer
    
    terminator, terminator_module, _ = _setup_model(model_hparams, run_hparams, best_checkpoint, dev, use_esm, model_hparams['edges_to_seq'], from_finetune=from_base)

    transfer_model = None
    terminator.transfer_model = transfer_model

    if finetune or run_hparams['train_transfer_only']: term_params = (param for param in terminator.parameters() if param.requires_grad)
    else: term_params = terminator.parameters()

    # if loss involves esm, load model
    if use_esm or run_hparams['num_recycles'] > 0 or load_esm:
        esm, alphabet = esmlib.pretrained.esm2_t30_150M_UR50D()
        esm = esm.requires_grad_(False)
        
        converter_state = torch.load(args.converter_path, map_location=torch.device(args.dev))
        converter = Converter(22, 640)
        converter.load_state_dict(converter_state)
        converter = converter.requires_grad_(False)
        if dev == 'cpu':
            esm = esm.float()  # convert to fp32 as ESM-2 in fp16 is not supported on CPU
            esm = esm.cpu()
            converter = converter.cpu()
        else:
            esm = esm.float()
            esm = esm.to(device='cuda:0')
            converter = converter.to(device='cuda:0')
            # esm = esm.to('cpu')
    else:
        esm = None
        converter = None
    if esm is not None:
        esm = esm.cpu()
        converter = converter.cpu()
        for p in esm.parameters():
            p.requires_grad = False
        for p in converter.parameters():
            p.requires_grad = False

    model_pl = TransferModelPL(coordinator=terminator, device=dev)
    model_pl.stage = 1

    name = os.path.splitext(os.path.basename(args.run_dir))[0]
    filename = name + '_{epoch:02d}_{val_ddG_spearman:.02}'
    monitor = 'val_ddG_spearman'
    checkpoint_callback = ModelCheckpoint(monitor=monitor, mode='max', dirpath=args.run_dir, filename=filename)
    max_ep = args.epochs

    if dev == 'cpu':
        accel = 'cpu'
    else:
        accel = 'gpu'

    trainer = pl.Trainer(callbacks=[checkpoint_callback], logger=None, log_every_n_steps=1, max_epochs=max_ep,
                         accelerator=accel, devices=1)
    trainer.fit(model_pl, train_dataloader, val_dataloader)

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
    # wandb.init(project=os.path.splitext(os.path.basename(parsed_args.run_dir))[0], name=os.path.splitext(os.path.basename(parsed_args.run_dir))[0])
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