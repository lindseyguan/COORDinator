""" Utilities for running training and evaluation loops """
import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import time
import numpy as np
import esm as esmlib
import copy
import random
# pylint: disable=no-member
from terminator.data.noise import generate_noise
from terminator.models.layers.utils import _esm_featurize, gather_edges
from terminator.utils.common import ints_to_seq_torch
from terminator.utils.model.ener_finetune_utils import get_out, get_esm_embs, expand_etab, condense_etab_expanded, score_sequence, CHAIN_LENGTH
from terminator.utils.model.loss_fn import construct_loss_fn
from terminator.models.layers.utils import merge_duplicate_pairE, merge_duplicate_edges, get_merge_dups_mask
from terminator.models.layers.energies.s2s import merge_dups
torch.autograd.set_detect_anomaly(True)
import GPUtil

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



def _ld_item_values(ld):
    """ Convert all 0-dim tensors in a loss dictionary into python native types.

    Args
    ----
    ld : dict
        Dictionary with keys :code:`loss_fn_name` and values of dictionaries with
        - :code:`loss` corresponding to the loss value
        - :code:`count` corresponding to the normalizing factor
        - :code:`scaling_factor` corresponding to the scaling coefficient in the loss function

    Returns
    -------
    ld_copy
        Updated loss dictionary with all 0-dim tensors converted to python native types.
    """

    ld_copy = ld.copy()
    for sub_ld in ld_copy.values():
        for key, val in sub_ld.items():
            if isinstance(val, torch.Tensor):
                if len(val.shape) == 0:
                    sub_ld[key] = val.item()
                elif key != 'seqs' and key != 'nsr':
                    raise RuntimeError(f"loss dictionary contains non-0-dim tensor value: {key}, {val}")
                elif key == 'seqs':
                    sub_ld['seqs'] = []
                elif key == 'nsr':
                    sub_ld['nsr'] = []
    return ld_copy


def _compute_loss(loss_dict, epoch, loss_weight_schedule={}, stall=False):
    """ Compute the total loss given a loss dictionary

    Args
    ----
    loss_dict : dict
        Dictionary with keys :code:`loss_fn_name` and values of dictionaries with
        - :code:`loss` corresponding to the loss value
        - :code:`count` corresponding to the normalizing factor
        - :code:`scaling_factor` corresponding to the scaling coefficient in the loss function

    Returns
    -------
    loss : torch.Tensor of size 1
        Computed loss
    """
    loss = 0
    for loss_name, subloss_dict in loss_dict.items():
        if stall and epoch < 20 and "cmap" in loss_dict.values():
            continue
        if loss_name in loss_weight_schedule.keys():
            if loss_weight_schedule[loss_name] == 'linear':
                if torch.is_tensor(loss_dict["structure_loss"]["loss"]):
                    scale = (1 - loss_dict["structure_loss"]["loss"].cpu().item()) * subloss_dict["scaling_factor"]
                else:
                    scale = (1 - loss_dict["structure_loss"]["loss"]) * subloss_dict["scaling_factor"]
                loss += subloss_dict["loss"] * scale
        else:
            loss += subloss_dict["loss"] * subloss_dict["scaling_factor"]
    return loss

def _extract_seq(loss_dict):
    for loss_name, subloss_dict in loss_dict.items():
        if loss_name == 'nlll_loss' or loss_name == 'loss_smoothed' or loss_name == 'pssm_loss':
            return subloss_dict['seqs']

def _extract_nsr(loss_dict):
    for loss_name, subloss_dict in loss_dict.items():
        if loss_name == 'nlll_loss' or loss_name == 'loss_smoothed' or loss_name == 'pssm_loss':
            return subloss_dict['nsr']
    return torch.Tensor([0])

def _sum_loss_dicts(total_ld, batch_ld):
    """ Add all values in :code:`batch_ld` into the corresponding values in :code:`total_ld`

    Args
    ----
    total_ld, batch_ld : dict
        Dictionary with keys :code:`loss_fn_name` and values of dictionaries with
        - :code:`loss` corresponding to the loss value
        - :code:`count` corresponding to the normalizing factor
        - :code:`scaling_factor` corresponding to the scaling coefficient in the loss function

    Returns
    -------
    combined_ld : dict
        Combined loss dictionary with the same structure as the input dictionaries
    """
    def _weighted_loss_sum(ld1, ld2):
        """ Compute the weighted loss between two loss dictionaries """
        c1, c2 = ld1["count"], ld2["count"]
        if (c1 + c2) == 0:
            return 0*ld1["loss"]
        return (ld1["loss"] * c1 + ld2["loss"] * c2)/(c1 + c2)
    def _weighted_nsr_sum(ld1, ld2):
        """ Compute the weighted nsr between two loss dictionaries """
        c1, c2 = ld1["count"], ld2["count"]
        if (c1 + c2) == 0:
            return 0*ld1["nsr"]
        return (ld1["nsr"] * c1 + ld2["nsr"] * c2)/(c1 + c2)
    # combine the two loss dictionaries
    combined_ld = total_ld.copy()
    for loss_fn_name, batch_subld in batch_ld.items():
        if loss_fn_name not in combined_ld.keys():
            combined_ld[loss_fn_name] = batch_subld
        else:
            combined_subld = combined_ld[loss_fn_name]
            assert combined_subld["scaling_factor"] == batch_subld["scaling_factor"]
            combined_subld["loss"] = _weighted_loss_sum(combined_subld, batch_subld)
            if loss_fn_name == 'nlll_loss' or loss_fn_name == 'loss_smoothed' or loss_fn_name == 'pssm_loss':
                combined_subld["nsr"] = _weighted_nsr_sum(combined_subld, batch_subld)
            combined_subld["count"] += batch_subld["count"]
    return combined_ld

def _sum_all_loss_dicts(batch_ld_list):
    """ Add all values in :code:`batch_ld` into the corresponding values in :code:`total_ld`

    Args
    ----
    total_ld, batch_ld : dict
        Dictionary with keys :code:`loss_fn_name` and values of dictionaries with
        - :code:`loss` corresponding to the loss value
        - :code:`count` corresponding to the normalizing factor
        - :code:`scaling_factor` corresponding to the scaling coefficient in the loss function

    Returns
    -------
    combined_ld : dict
        Combined loss dictionary with the same structure as the input dictionaries
    """
    def _loss_sum(ld_list):
        """ Compute the average loss between all loss dictionaries """
        losses = []
        for ld in ld_list:
            if ld["count"] > 0:
                losses.append(ld["loss"].cpu().detach().numpy().item())
        if len(losses) > 0:
            return np.mean(losses)
        else:
            return 0
    def _nsr_sum(ld_list):
        """ Compute the average nsr between all loss dictionaries """
        nsrs = []
        for ld in ld_list:
            if ld["count"] > 0:
                nsrs.append(ld["nsr"].cpu().detach().numpy().item())
        if len(nsrs) > 0:
            return np.mean(nsrs)
        else:
            return 0
    def _count_sum(ld_list):
        """ Sum counts for loss dictionaries"""
        sum_count = 0
        for ld in ld_list:
            sum_count += ld["count"]
        return sum_count
    # combine the two loss dictionaries
    combined_ld = {}
    for loss_fn_name in batch_ld_list[0].keys():
        if loss_fn_name not in combined_ld.keys():
            combined_ld[loss_fn_name] = {}
        ld_list = [batch_ld[loss_fn_name] for batch_ld in batch_ld_list]
        combined_ld[loss_fn_name]["loss"] = _loss_sum(ld_list)
        if loss_fn_name == 'nlll_loss' or loss_fn_name == 'loss_smoothed' or loss_fn_name == 'pssm_loss':
            combined_ld[loss_fn_name]["nsr"] = _nsr_sum(ld_list)
        combined_ld[loss_fn_name]["count"] = _count_sum(ld_list)
        combined_ld[loss_fn_name]["scaling_factor"] = batch_ld_list[0][loss_fn_name]["scaling_factor"]
    return combined_ld

def _log_rank_0(message):
    """ Logs input message if local process rank is 0 or ddp not enabled

    Args
    ----
    message : str
        String containing message to log

    Returns
    -------
    """
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message)
    else:
        print(message)

def apply_finetune_layers(h_E, h_V, h_C, E_idx, x_mask, model, energy_merge_fn, node_finetune=False, node_self_sub=True, restrict_output=True):
    if node_finetune:
        h_V = model.top.V_dropout(h_V)
        h_V = model.top.V_out(h_V)
    else:
        h_E = model.top.W_dropout(h_E)
        h_E = model.top.W_out(h_E)
        n_batch, n_res, k, out_dim = h_E.shape
        h_E = h_E * x_mask.view(n_batch, n_res, 1, 1) # ensure output etab is masked properly
        if energy_merge_fn == 'default':
            h_E = h_E.unsqueeze(-1).view(n_batch, n_res, k, 20, 20)
            h_E[:, :, 0] = h_E[:, :, 0] * torch.eye(20).to(h_E.device) # zero off-diagonal energies
            h_E = merge_duplicate_pairE(h_E, E_idx)
        elif energy_merge_fn == 'identical':
            inv_mapping, _ = get_merge_dups_mask(E_idx)
            h_E = merge_dups(h_E, E_idx, inv_mapping)
            h_E = h_E.unsqueeze(-1).view(n_batch, n_res, k, 20, 20)
            h_E[:, :, 0] = h_E[:, :, 0] * torch.eye(20).to(h_E.device) # zero off-diagonal energies
        # reshape to fit kNN output format
        h_E = h_E.view(n_batch, n_res, k, out_dim)
    # if specified, generate self energies from node embeddings
    if node_self_sub:
        # h_V = self.W_proj(h_V)
        # h_E[..., 0, :] = torch.diag_embed(h_V, dim1=-2, dim2=-1).flatten(start_dim=-2, end_dim=-1)
        if restrict_output:
            pad = (0, 2)
            h_V = torch.nn.functional.pad(h_V, pad, "constant", 0)

    return h_E, h_V, h_C, E_idx


def run_iter(iter, model, optimizer, loss_fn, data, teacher_forcing, num_recycles, keep_seq, keep_sc_mask_recycle, running_loss_dict, all_batch_loss_dicts, dump, esm, batch_converter, tokenizer, esm_type, esm_options, converter, grad, epoch, test, stall, dev, progress=None, reload_esm=False, from_ener=False, only_loss_recycle=False, recycle_teacher_forcing=False, recycle_confidence=False, keep_sc_mask_loss=True, skip_backprop=False, confidence_threshold=0.5, silenced=False, save_seq=False, finetune=False, tune_W_out=None, energy_merge_fn=None, use_transfer_model=False, unfreeze_esm=True, train_transfer_only=False, run_discriminator=False, loss_weight_schedule={}, distill_terminator=None, node_finetune=False, node_self_sub=True):
    if from_ener:
        loss_fn = construct_loss_fn({'loss_config': {'sortcery_loss': 1}})
    for num_recycle in range(num_recycles+1):
        # Set up sequence embeddings if needed
        # a small hack for DataParallel to know which device got which proteins
        data['scatter_idx'] = torch.arange(len(data['seq_lens']))
        _to_dev(data, dev)
        max_seq_len = max(data['seq_lens'].tolist())
        ids = data['ids']
        try:
            if distill_terminator is not None:
                with torch.no_grad():
                    distill_etab, _, _, _, _, _ = distill_terminator(data, max_seq_len)
            elif data['distill_etab'].numel() > 0:
                distill_etab = data['distill_etab']
            else:
                distill_etab = None
            if not grad:
                with torch.no_grad():
                    etab, h_V, h_C, E_idx, frames, pos = model(data, max_seq_len, finetune=finetune, use_transfer_model=use_transfer_model)
            else:
                etab, h_V, h_C, E_idx, frames, pos = model(data, max_seq_len, finetune=finetune, use_transfer_model=use_transfer_model)
            ## Check gradients enabled
            # for (name, module) in model.named_children():
            #     try:
            #         print(name, module.requires_grad)
            #     except:
            #         print(name)
            #     try:
            #         for (n, m) in module.named_children():
            #             try:
            #                 print('\t', n, m.requires_grad)
            #             except:
            #                 print('\t', n, m)
            #     except:
            #         pass
            # Iterate through all named parameters
            # print("Layer Name | Gradients Enabled")
            # print("--------------------------------")
            # for name, param in model.named_parameters():
            #     print(f"{name:<20} {param.requires_grad}")

            # raise ValueError
                
            if finetune: # and not use_transfer_model:
                etab, h_V, h_C, E_idx = apply_finetune_layers(etab, h_V, h_C, E_idx, data['x_mask'], model, energy_merge_fn, node_finetune=node_finetune, node_self_sub=node_self_sub)
            if use_transfer_model:
                pred_ener = model.transfer_model(data, h_V, etab, E_idx)
            else:
                pred_ener = None
            if num_recycle > 0:
                data['seqs'] = orig_seqs
                if (not grad) or keep_sc_mask_loss:
                    data['x_mask_sc'] = orig_x_mask_sc
            batch_loss_dict = loss_fn(etab, h_V, h_C, pred_ener, E_idx, frames, data, esm, batch_converter, tokenizer, esm_type, converter, not grad, distill_etab)
            if (len(batch_loss_dict.keys()) == 0) or (('nlcpl' in batch_loss_dict.keys()) and (batch_loss_dict['nlcpl']["count"] == -1)) or (('stability' in batch_loss_dict.keys()) and (batch_loss_dict['stability']["count"] == -1)):
                return running_loss_dict, all_batch_loss_dicts, dump, progress
            loss = _compute_loss(batch_loss_dict, epoch, loss_weight_schedule)
            if isinstance(loss, int):
                # _log_rank_0(ids)
                return running_loss_dict, all_batch_loss_dicts, dump, progress
        except Exception as e:
            _log_rank_0(ids)
            raise e
        if grad and not from_ener and ((only_loss_recycle and num_recycle == num_recycles) or not only_loss_recycle) and not skip_backprop:
            # torch.autograd.set_detect_anomaly(True)
            import psutil        

            # print("disks")
            # print(psutil.disk_partitions())
            # print(psutil.disk_usage('/'))

            # print("memory")
            # print(f"CPU utilization: {psutil.cpu_percent()}%") 
            # print(f"Memory utilization: {psutil.virtual_memory().percent}%") 
            # print(psutil.virtual_memory())
            # print(psutil.swap_memory())
            # print('etab shape: ', etab.shape, data['sortcery_nrgs'].shape, data['sortcery_seqs'].shape)
            # print('--------')
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if num_recycle > 0:
            new_seqs = _extract_seq(batch_loss_dict)
        all_batch_loss_dicts.append(batch_loss_dict.copy())
        # running_loss_dict = _sum_loss_dicts(running_loss_dict,
            # _ld_item_values(batch_loss_dict))
        # Reset sequences for potential recycles
        if num_recycle == num_recycles:
            continue
        esm_embs = []
        esm_attns = []
        orig_seqs = copy.deepcopy(data['seqs'])
        orig_x_mask_sc = copy.deepcopy(data['x_mask_sc'])
        for i in range(len(data['chain_lens'])):
            new_seq = "".join(ints_to_seq_torch(new_seqs[i][:sum(data['chain_lens'][i])].to(dtype=torch.int32)))
            new_seq = torch.tensor([ord(c) for c in new_seq]).to(device=dev)
            if keep_seq and data['x_mask_sc'].numel() > 0:
                mask = data['x_mask_sc'][i][:sum(data['chain_lens'][i])].to(dtype=torch.bool)
                base_seq = "".join(ints_to_seq_torch(data['seqs'][i][:sum(data['chain_lens'][i])].to(dtype=torch.int32)))
                base_seq = torch.tensor([ord(c) for c in base_seq]).to(device=dev)
                seq = torch.where(mask, base_seq, new_seq)
            elif grad and recycle_teacher_forcing:
                mask = data['seqs'][i][:sum(data['chain_lens'][i])] != new_seqs[i][:sum(data['chain_lens'][i])]
                base_seq = "X"*sum(data['chain_lens'][i])
                base_seq = torch.tensor([ord(c) for c in base_seq]).to(device=dev)
                seq = torch.where(mask, base_seq, new_seq)
            elif recycle_confidence: #not grad and 
                mask = h_C[i][:sum(data['chain_lens'][i])] < confidence_threshold
                base_seq = "X"*sum(data['chain_lens'][i])
                base_seq = torch.tensor([ord(c) for c in base_seq]).to(device=dev)
                seq = torch.where(mask, base_seq, new_seq)
            else:
                seq = new_seq
            seq = ''.join([chr(int(c)) for c in seq])
            if dev == 'cpu':
                esm_emb, esm_attn = _esm_featurize(data['chain_lens'][i], seq, esm.cpu(), batch_converter, esm_options['use_esm_attns'], esm_options['esm_embed_layer'], esm_options['from_rla'], esm_options['use_reps'], esm_options['connect_chains'], esm_options['one_hot'], dev=dev)
            else:
                esm_emb, esm_attn = _esm_featurize(data['chain_lens'][i], seq, esm.cuda(), batch_converter, esm_options['use_esm_attns'], esm_options['esm_embed_layer'], esm_options['from_rla'], esm_options['use_reps'], esm_options['connect_chains'], esm_options['one_hot'], dev=dev)
            if esm_options['use_esm_attns']:
                esm_attn = esm_attn.unsqueeze(0)
                esm_attn = gather_edges(esm_attn, E_idx[i,:sum(data['chain_lens'][i])].unsqueeze(0)).squeeze(0)
                esm_attns.append(esm_attn)
            esm_embs.append(esm_emb)
        esm_embs = pad_sequence(esm_embs, batch_first=True, padding_value = 0)
        data['esm_embs'] = esm_embs 
        if esm_options['use_esm_attns']:
            esm_attns = pad_sequence(esm_attns, batch_first=True, padding_value = 0)
            data['esm_attns'] = esm_attns

        if keep_seq and data['x_mask_sc'].numel() > 0:
            new_seqs = new_seqs.to(dtype = data['seqs'].dtype)
            new_seqs = torch.where(data['x_mask_sc'] == 1, data['seqs'], new_seqs)
        elif not keep_seq or grad:
            model.top.teacher_forcing = True
        # if not keep_sc_mask_recycle:
        #     data['x_mask_sc'] = torch.ones_like(data['x_mask_sc']).to(device=data['x_mask_sc'].device)
        if grad and recycle_teacher_forcing:
            data['x_mask_sc'] = (data['seqs'] == new_seqs).to(dtype=data['x_mask_sc'].dtype, device=data['x_mask_sc'].device)
        elif recycle_confidence: #not grad and
            data['x_mask_sc'] = (h_C >= confidence_threshold).to(dtype=data['x_mask_sc'].dtype, device=data['x_mask_sc'].device)
        data['seqs'] = (new_seqs * data['x_mask']).to(dtype=data['seqs'].dtype, device=data['seqs'].device)
    model.top.teacher_forcing = teacher_forcing
    res_mask_eff = int(data['x_mask'].sum().item() / data['x_mask'].numel() * 100)
    # compactify what's printed to stdout\
    print_update = not dist.is_initialized() and num_recycle == num_recycles and not silenced
    if silenced:
        running_loss_dict = _sum_all_loss_dicts(all_batch_loss_dicts)

    if print_update and (progress is not None):
        loss_breakdown = {}
        running_loss_dict = _sum_all_loss_dicts(all_batch_loss_dicts)
        for loss_fn_name, subloss_dict in running_loss_dict.items():
            if stall and epoch < 20 and 'cmap' in loss_fn_name:
                continue
            loss_breakdown[loss_fn_name] = round(subloss_dict["loss"], 2)
        avg_loss = round(_compute_loss(running_loss_dict, epoch, loss_weight_schedule), 2)
        avg_nsr = _extract_nsr(running_loss_dict)
        if iter % 10 == 0:
            progress.update(10)
            progress.refresh()
            progress.set_description_str(f'avg loss {avg_loss} {loss_breakdown} | eff 0.{res_mask_eff}| nsr {avg_nsr}')
    if test:
        if grad and skip_backprop:
            loss = loss.detach()
            etab = etab.detach()
            h_V = h_V.detach()
            h_C = h_C.detach()
            data['x_mask_sc'] = data['x_mask_sc'].detach()
        n_batch, l, n = etab.shape[:3]
        if from_ener:
            dump.append({
            'loss': loss,
            'out': etab,
            'nodes': h_V,
            'confidence': h_C,
            'idx': E_idx,
            'ids': ids,
            'mask': data['x_mask_sc']
        })
        elif save_seq:
            dump.append({
                'loss': loss,
                'out': etab.view(n_batch, l, n, 20, 20).cpu().numpy(),
                'nodes': h_V,
                'confidence': h_C,
                'idx': E_idx.cpu().numpy(),
                'ids': ids,
                'mask': data['x_mask_sc'].cpu().numpy(),
                'sequence': ints_to_seq_torch(data['seqs'][0])
            })
        else:
            dump.append({
                'loss': loss,
                'out': etab.view(n_batch, l, n, 20, 20).cpu().numpy(),
                'nodes': h_V,
                'confidence': h_C,
                'idx': E_idx.cpu().numpy(),
                'ids': ids,
                'mask': data['x_mask_sc'].cpu().numpy()
            })
    return running_loss_dict, all_batch_loss_dicts, dump, progress

def run_iter_finetune(iter, model, optimizer, loss_fn, data, teacher_forcing, num_recycles, keep_seq, keep_sc_mask_recycle, running_loss_dict, all_batch_loss_dicts, dump, esm, batch_converter, tokenizer, esm_type, esm_options, converter, grad, epoch, test, stall, dev, progress=None, reload_esm=False, from_ener=False, only_loss_recycle=False, recycle_teacher_forcing=False, recycle_confidence=False, keep_sc_mask_loss=True, skip_backprop=False, confidence_threshold=0.5, silenced=False, save_seq=False, finetune=False, tune_W_out=None, energy_merge_fn=None, use_transfer_model=False, unfreeze_esm=True, train_transfer_only=False, run_discriminator=False, loss_weight_schedule={}, distill_terminator=None, max_loop_tokens=None, node_finetune=False, node_self_sub=True):
    if from_ener:
        loss_fn = construct_loss_fn({'loss_config': {'sortcery_loss': 1}})
    for num_recycle in range(num_recycles+1):
        # Set up sequence embeddings if needed
        # a small hack for DataParallel to know which device got which proteins
        data['scatter_idx'] = torch.arange(len(data['seq_lens']))
        _to_dev(data, dev)
        max_seq_len = max(data['seq_lens'].tolist())
        ids = data['ids']
        all_sort_seqs = copy.deepcopy(data['sortcery_seqs'])
        all_sort_nrgs = copy.deepcopy(data['sortcery_nrgs'])
        n = data['X'].shape[1]
        if (max_loop_tokens is not None) and (n*all_sort_nrgs.shape[1] > max_loop_tokens):
            batch_size = int(max_loop_tokens / n)
        else:
            batch_size = all_sort_nrgs.shape[1]
        for batch in range(0, all_sort_nrgs.shape[1], batch_size):
            try:
                if distill_terminator is not None:
                    with torch.no_grad():
                        distill_etab, _, _, _, _, _ = distill_terminator(data, max_seq_len)
                elif data['distill_etab'].numel() > 0:
                    distill_etab = data['distill_etab']
                else:
                    distill_etab = None
                if not grad:
                    with torch.no_grad():
                        etab, h_V, h_C, E_idx, frames, pos = model(data, max_seq_len, finetune=finetune, use_transfer_model=use_transfer_model)
                else:
                    etab, h_V, h_C, E_idx, frames, pos = model(data, max_seq_len, finetune=finetune, use_transfer_model=use_transfer_model)
                    
                if finetune: # and not use_transfer_model:
                    if not grad:
                        with torch.no_grad():
                            etab, h_V, h_C, E_idx = apply_finetune_layers(etab, h_V, h_C, E_idx, data['x_mask'], model, energy_merge_fn, node_finetune=node_finetune, node_self_sub=node_self_sub)
                    else:
                        etab, h_V, h_C, E_idx = apply_finetune_layers(etab, h_V, h_C, E_idx, data['x_mask'], model, energy_merge_fn, node_finetune=node_finetune, node_self_sub=node_self_sub)
                if use_transfer_model:
                    pred_ener = model.transfer_model(data, h_V, etab, E_idx)
                else:
                    pred_ener = None
                if num_recycle > 0:
                    data['seqs'] = orig_seqs
                    if (not grad) or keep_sc_mask_loss:
                        data['x_mask_sc'] = orig_x_mask_sc
                
                data['sortcery_seqs'] = all_sort_seqs[:,batch:batch+batch_size]
                data['sortcery_nrgs'] = all_sort_nrgs[:,batch:batch+batch_size]
                batch_loss_dict = loss_fn(etab, h_V, h_C, pred_ener, E_idx, frames, data, esm, batch_converter, tokenizer, esm_type, converter, not grad, distill_etab)
                if (len(batch_loss_dict.keys()) == 0) or (('nlcpl' in batch_loss_dict.keys()) and (batch_loss_dict['nlcpl']["count"] == -1)) or (('stability' in batch_loss_dict.keys()) and (batch_loss_dict['stability']["count"] == -1)):
                    return running_loss_dict, all_batch_loss_dicts, dump, progress
                loss = _compute_loss(batch_loss_dict, epoch, loss_weight_schedule)
                if isinstance(loss, int):
                    continue
    
                if grad and not from_ener and ((only_loss_recycle and num_recycle == num_recycles) or not only_loss_recycle) and not skip_backprop:
                    # torch.autograd.set_detect_anomaly(True)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
            except Exception as e:
                _log_rank_0(ids)
                raise e
            all_batch_loss_dicts.append(batch_loss_dict.copy())
        if len(all_batch_loss_dicts) == 0:
            return running_loss_dict, all_batch_loss_dicts, dump, progress
        if num_recycle > 0:
            new_seqs = _extract_seq(batch_loss_dict)
        
        # running_loss_dict = _sum_loss_dicts(running_loss_dict,
            # _ld_item_values(batch_loss_dict))
        # Reset sequences for potential recycles
        if num_recycle == num_recycles:
            continue
        esm_embs = []
        esm_attns = []
        orig_seqs = copy.deepcopy(data['seqs'])
        orig_x_mask_sc = copy.deepcopy(data['x_mask_sc'])
        for i in range(len(data['chain_lens'])):
            new_seq = "".join(ints_to_seq_torch(new_seqs[i][:sum(data['chain_lens'][i])].to(dtype=torch.int32)))
            new_seq = torch.tensor([ord(c) for c in new_seq]).to(device=dev)
            if keep_seq and data['x_mask_sc'].numel() > 0:
                mask = data['x_mask_sc'][i][:sum(data['chain_lens'][i])].to(dtype=torch.bool)
                base_seq = "".join(ints_to_seq_torch(data['seqs'][i][:sum(data['chain_lens'][i])].to(dtype=torch.int32)))
                base_seq = torch.tensor([ord(c) for c in base_seq]).to(device=dev)
                seq = torch.where(mask, base_seq, new_seq)
            elif grad and recycle_teacher_forcing:
                mask = data['seqs'][i][:sum(data['chain_lens'][i])] != new_seqs[i][:sum(data['chain_lens'][i])]
                base_seq = "X"*sum(data['chain_lens'][i])
                base_seq = torch.tensor([ord(c) for c in base_seq]).to(device=dev)
                seq = torch.where(mask, base_seq, new_seq)
            elif recycle_confidence: #not grad and 
                mask = h_C[i][:sum(data['chain_lens'][i])] < confidence_threshold
                base_seq = "X"*sum(data['chain_lens'][i])
                base_seq = torch.tensor([ord(c) for c in base_seq]).to(device=dev)
                seq = torch.where(mask, base_seq, new_seq)
            else:
                seq = new_seq
            seq = ''.join([chr(int(c)) for c in seq])
            if dev == 'cpu':
                esm_emb, esm_attn = _esm_featurize(data['chain_lens'][i], seq, esm.cpu(), batch_converter, esm_options['use_esm_attns'], esm_options['esm_embed_layer'], esm_options['from_rla'], esm_options['use_reps'], esm_options['connect_chains'], esm_options['one_hot'], dev=dev)
            else:
                esm_emb, esm_attn = _esm_featurize(data['chain_lens'][i], seq, esm.cuda(), batch_converter, esm_options['use_esm_attns'], esm_options['esm_embed_layer'], esm_options['from_rla'], esm_options['use_reps'], esm_options['connect_chains'], esm_options['one_hot'], dev=dev)
            if esm_options['use_esm_attns']:
                esm_attn = esm_attn.unsqueeze(0)
                esm_attn = gather_edges(esm_attn, E_idx[i,:sum(data['chain_lens'][i])].unsqueeze(0)).squeeze(0)
                esm_attns.append(esm_attn)
            esm_embs.append(esm_emb)
        esm_embs = pad_sequence(esm_embs, batch_first=True, padding_value = 0)
        data['esm_embs'] = esm_embs 
        if esm_options['use_esm_attns']:
            esm_attns = pad_sequence(esm_attns, batch_first=True, padding_value = 0)
            data['esm_attns'] = esm_attns

        if keep_seq and data['x_mask_sc'].numel() > 0:
            new_seqs = new_seqs.to(dtype = data['seqs'].dtype)
            new_seqs = torch.where(data['x_mask_sc'] == 1, data['seqs'], new_seqs)
        elif not keep_seq or grad:
            model.top.teacher_forcing = True
        # if not keep_sc_mask_recycle:
        #     data['x_mask_sc'] = torch.ones_like(data['x_mask_sc']).to(device=data['x_mask_sc'].device)
        if grad and recycle_teacher_forcing:
            data['x_mask_sc'] = (data['seqs'] == new_seqs).to(dtype=data['x_mask_sc'].dtype, device=data['x_mask_sc'].device)
        elif recycle_confidence: #not grad and
            data['x_mask_sc'] = (h_C >= confidence_threshold).to(dtype=data['x_mask_sc'].dtype, device=data['x_mask_sc'].device)
        data['seqs'] = (new_seqs * data['x_mask']).to(dtype=data['seqs'].dtype, device=data['seqs'].device)
    model.top.teacher_forcing = teacher_forcing
    res_mask_eff = int(data['x_mask'].sum().item() / data['x_mask'].numel() * 100)
    # compactify what's printed to stdout\
    print_update = not dist.is_initialized() and num_recycle == num_recycles and not silenced
    if silenced:
        running_loss_dict = _sum_all_loss_dicts(all_batch_loss_dicts)

    if print_update and (progress is not None):
        loss_breakdown = {}
        running_loss_dict = _sum_all_loss_dicts(all_batch_loss_dicts)
        for loss_fn_name, subloss_dict in running_loss_dict.items():
            if stall and epoch < 20 and 'cmap' in loss_fn_name:
                continue
            loss_breakdown[loss_fn_name] = round(subloss_dict["loss"], 2)
        avg_loss = round(_compute_loss(running_loss_dict, epoch, loss_weight_schedule), 2)
        avg_nsr = _extract_nsr(running_loss_dict)
        if iter % 10 == 0:
            progress.update(10)
            progress.refresh()
            progress.set_description_str(f'avg loss {avg_loss} {loss_breakdown} | eff 0.{res_mask_eff}| nsr {avg_nsr}')
    if test:
        if grad and skip_backprop:
            loss = loss.detach()
            etab = etab.detach()
            h_V = h_V.detach()
            h_C = h_C.detach()
            data['x_mask_sc'] = data['x_mask_sc'].detach()
        n_batch, l, n = etab.shape[:3]
        if from_ener:
            dump.append({
            'loss': loss,
            'out': etab,
            'nodes': h_V,
            'confidence': h_C,
            'idx': E_idx,
            'ids': ids,
            'mask': data['x_mask_sc']
        })
        elif save_seq:
            dump.append({
                'loss': loss,
                'out': etab.view(n_batch, l, n, 20, 20).cpu().numpy(),
                'nodes': h_V,
                'confidence': h_C,
                'idx': E_idx.cpu().numpy(),
                'ids': ids,
                'mask': data['x_mask_sc'].cpu().numpy(),
                'sequence': ints_to_seq_torch(data['seqs'][0])
            })
        else:
            dump.append({
                'loss': loss,
                'out': etab.view(n_batch, l, n, 20, 20).cpu().numpy(),
                'nodes': h_V,
                'confidence': h_C,
                'idx': E_idx.cpu().numpy(),
                'ids': ids,
                'mask': data['x_mask_sc'].cpu().numpy()
            })
    return running_loss_dict, all_batch_loss_dicts, dump, progress


def run_epoch(model, dataloader, loss_fn, hparams=None, epoch=0, optimizer=None, scheduler=None, grad=False, test=False, dev="cuda:0", isDataParallel=False, finetune=False, esm=None, batch_converter=None, tokenizer=None, esm_type='esm', stall=False, converter=None, num_recycles=0, keep_seq=True, keep_sc_mask_recycle=False, esm_options=None, ener_finetune_mlp=False, only_loss_recycle=False, recycle_teacher_forcing=False, recycle_confidence=False, keep_sc_mask_loss=True, silenced=False, save_seq=False, tune_W_out=None, energy_merge_fn=None, use_transfer_model=False, train_transfer_only=False, loss_weight_schedule={}, distill_terminator=None, from_train=True, max_loop_tokens=None, node_finetune=False, node_self_sub=True):
    """ Run :code:`model` on one epoch of :code:`dataloader`

    Args
    ----
    model : terminator.model.TERMinator.TERMinator
        An instance of TERMinator
    dataloader : torch.utils.data.DataLoader
        A torch DataLoader that wraps either terminator.data.data.TERMDataLoader or terminator.data.data.TERMLazyDataLoader
    loss_fn : function
        Loss function with signature :code:`loss_fn(etab, E_idx, data)` and returns :code`loss, batch_count`,
        where
        - :code:`etab, E_idx` is the outputted Potts Model
        - :code:`data` is the input data produced by :code:`dataloader`
        - :code:`hparams` is the model hyperparameters
        - :code:`loss` is the loss value
        - :code:`batch_count` is the averaging factor
    optimizer : torch optimizer or None
        An optimizer for :code:`model`. Used when :code:`grad=True, test=False`
    scheduler : torch scheduler or None
        The associted scheduler for the given optimizer
    grad : bool
        Whether or not to compute gradients. :code:`True` to train the model, :code:`False` to use model in evaluation mode.
    test : bool
        Whether or not to save the outputs of the model. Requires :code:`grad=False`.
    dev : str, default="cuda:0"
        What device to compute on
    esm : esm.model
        An instance of ESM

    Returns
    -------
    epoch_loss : float
        Loss on the run epoch
    running_loss_dict : dict
        Loss breakdown into component sublosses and scaling factors of epoch_loss
    dump : list of dicts, conditionally present
        Outputs of the model. Present when :code:`test=True`
    """
    # arg checking
    if test:
        assert not grad, "grad should not be on for test set"
    if grad:
        assert optimizer is not None, "require an optimizer if using grads"
    if scheduler is not None:
        assert optimizer is not None, "using a scheduler requires an optimizer"
    torch.set_grad_enabled(True)
    running_loss_dict = {}
    all_batch_loss_dicts = []
    teacher_forcing = copy.deepcopy(model.top.teacher_forcing)
    # set grads properly
    if grad:
        model = model.train()

    else:
        model = model.eval()
        model.top.teacher_forcing = False

    # record inference outputs if necessary
    dump = []
    progress = tqdm(total=len(dataloader))
    reload_esm = (hparams is not None and "esm_cmap_loss_tp" in hparams['loss_config'].keys())
    import tracemalloc
    tracemalloc.start()

    for iter, data in enumerate(dataloader):
        if from_train:
            running_loss_dict, all_batch_loss_dicts, dump, progress = run_iter(iter, model, optimizer, loss_fn, data, model.top.teacher_forcing, num_recycles, keep_seq, keep_sc_mask_recycle, running_loss_dict, all_batch_loss_dicts, dump, esm, batch_converter, tokenizer, esm_type, esm_options, converter, grad, epoch, test, stall, dev, progress, reload_esm=reload_esm, only_loss_recycle=only_loss_recycle, recycle_teacher_forcing=recycle_teacher_forcing, recycle_confidence=recycle_confidence, keep_sc_mask_loss=keep_sc_mask_loss, silenced=silenced, save_seq=save_seq, finetune=finetune, tune_W_out=tune_W_out, energy_merge_fn=energy_merge_fn, use_transfer_model=use_transfer_model, train_transfer_only=train_transfer_only, loss_weight_schedule=loss_weight_schedule, distill_terminator=None, node_finetune=node_finetune, node_self_sub=node_self_sub)
        else:
            running_loss_dict, all_batch_loss_dicts, dump, progress = run_iter_finetune(iter, model, optimizer, loss_fn, data, model.top.teacher_forcing, num_recycles, keep_seq, keep_sc_mask_recycle, running_loss_dict, all_batch_loss_dicts, dump, esm, batch_converter, tokenizer, esm_type, esm_options, converter, grad, epoch, test, stall, dev, progress, reload_esm=reload_esm, only_loss_recycle=only_loss_recycle, recycle_teacher_forcing=recycle_teacher_forcing, recycle_confidence=recycle_confidence, keep_sc_mask_loss=keep_sc_mask_loss, silenced=silenced, save_seq=save_seq, finetune=finetune, tune_W_out=tune_W_out, energy_merge_fn=energy_merge_fn, use_transfer_model=use_transfer_model, train_transfer_only=train_transfer_only, loss_weight_schedule=loss_weight_schedule, distill_terminator=None, max_loop_tokens=max_loop_tokens, node_finetune=node_finetune, node_self_sub=node_self_sub)
        torch.cuda.empty_cache()


        # import psutil
        

        # print("disks")
        # print(psutil.disk_partitions())
        # print(psutil.disk_usage('/'))

        # print("memory")
        # print(f"CPU utilization: {psutil.cpu_percent()}%") 
        # print(f"Memory utilization: {psutil.virtual_memory().percent}%") 
        # print(psutil.virtual_memory())
        # print(psutil.swap_memory())

        # print("pytorch stats")
        # # print(torch.cuda.memory_summary('cuda:0'))
        # snapshot = tracemalloc.take_snapshot()

        # # Display top memory usage by lines of code
        # top_stats = snapshot.statistics('lineno')
        # for stat in top_stats[:10]:  # Top 10 memory usage
        #     print(stat)

    progress.close()
    epoch_loss = _compute_loss(running_loss_dict, epoch, loss_weight_schedule)

    if dist.is_initialized():
        epoch_loss = epoch_loss / dist.get_world_size()
        epoch_loss_tensor = torch.tensor([epoch_loss], dtype=torch.float).to(dev)
        dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM)
        epoch_loss = float(epoch_loss_tensor[0])
    if scheduler is not None:
        scheduler.step(epoch_loss)
    model.top.teacher_forcing = teacher_forcing
    if test:
        return epoch_loss, running_loss_dict, dump
    return epoch_loss, running_loss_dict, None

def run_epoch_ener_ft(model, optimizer, dataloader, pep_bind_ener, args, run_hparams, model_hparams, loss_fn, dev, grad, finetune, load_model, epoch=0, test=False, num_recycles=0, keep_seq=True, keep_sc_mask_recycle=False, data_source='bcl2', esm=None, batch_converter=None, tokenizer=None, esm_type="esm", stall=None, converter=None, esm_options=None, ener_finetune_mlp=False, run_quick=False):

    # set grads properly
    if grad:
        torch.set_grad_enabled(True)
        model.train()
        if finetune: # freeze all but the last output layer
            for (name, module) in model.named_children():
                if name == "top":
                    for (n, m) in module.named_children():
                        if n == "W_out" or "V_out" in n:
                            m.requires_grad = True
                        else:
                            m.requires_grad = False
                else:
                    module.requires_grad = False
        else:
            torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)
    progress = tqdm(total=len(dataloader))
    avg_loss = 0
    i_item = 0
    torch.autograd.set_detect_anomaly(True)
    teacher_forcing = copy.deepcopy(model.top.teacher_forcing)
    for data in dataloader:
        avg_loss, i_item, progress = run_iter_ener(model, optimizer, loss_fn, data, load_model, data_source, pep_bind_ener, args, model_hparams, run_hparams, teacher_forcing, num_recycles, keep_seq, keep_sc_mask_recycle, esm, batch_converter, esm_type, esm_options, converter, grad, dev, progress, i_item, avg_loss, run_quick=run_quick)  
        if run_quick:
            break      
    torch.set_grad_enabled(True)
    if i_item == 0:
        i_item = 1
    return avg_loss / i_item