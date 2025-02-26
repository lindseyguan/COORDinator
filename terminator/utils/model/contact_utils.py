import torch
from torch.cuda.amp import GradScaler, autocast
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from transformers import EsmTokenizer


def get_formatted_inp(seqs, tokenizer, dev):
    text_inp = tokenizer(seqs, return_tensors='pt', padding=True, 
                              truncation=True, max_length=1024+2)
    text_inp = {k: v.to(device=dev) for k, v in text_inp.items()}
    return text_inp

def postprocess_attention_features(attention_features, inp_dict, tokenizer, placeholder_mask):
    # remove class token, eos token
    attention_features, text_mask = _adjust_text_features(attention_features, inp_dict, tokenizer)
    # remove placeholders
    attention_features, text_mask = _remove_placeholders(attention_features, text_mask, placeholder_mask)
    return attention_features, text_mask


def _adjust_text_features(attention_features, inp_dict, tokenizer):
    mask = inp_dict['attention_mask'].clone()
    toks = inp_dict['input_ids']
    eos_token = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
    mask[toks == eos_token] = 0
    # ignore class token and last eos token
    mask = mask[:, 1:-1]
    attention_features = attention_features[:, :, 1:-1, 1:-1]
    return attention_features, mask

def _remove_placeholders(attention_features, text_mask, placeholder_mask):
    B = placeholder_mask.shape[0]
    filtered = []
    new_masks = []
    for b in range(B):
        p_m = placeholder_mask[b]
        new_text_feat = attention_features[b][:, p_m]
        new_text_feat = new_text_feat[:, :, p_m]
        filtered.append(new_text_feat)
        new_masks.append(text_mask[b][p_m])
    new_masks = pad_sequence(new_masks, batch_first=True)
    dest_shape = 1024
    padded_filtered = []
    for f in filtered:
        amt_to_pad = dest_shape - f.shape[-1]
        padded = torch.nn.functional.pad(f, (0, amt_to_pad, 0, amt_to_pad))
        padded_filtered.append(padded)
    return torch.stack(padded_filtered), new_masks

def get_attentions(model, text_inp_):
    with torch.no_grad():
        out = model.text_model(**text_inp_, output_attentions=True)
        all_attns = torch.cat(out['attentions'], 1)
        num_heads = len(out['attentions'])
    return all_attns, num_heads

def APC_correction(attn_):
    F_i = attn_.sum(keepdims=True, dim=2)
    F_j = attn_.sum(keepdims=True, dim=3)
    F = attn_.sum(dim=2, keepdims=True).sum(dim=3, keepdims=True)
    apc_corr = (F_i * F_j)/F
    attn_corr = attn_ - apc_corr
    return attn_corr

def calc_attns(model, text_inp, tokenizer, x_mask, dev='cuda:0'):
    text_inp = get_formatted_inp(text_inp, tokenizer, dev)
    attns, _ = get_attentions(model, text_inp)
    attns, _ = postprocess_attention_features(attns, text_inp, tokenizer, x_mask.to(dtype=torch.bool))
    attns = APC_correction(attns)
    attns = attns.permute(0, 2, 3, 1)
    return attns


def get_esm_contacts(esm, nodes_esm, eos_mask):
    # esm_data = []
    # for i_s, seq in enumerate(seqs):
    #     esm_data.append((f"protein_{i_s}", seq))
    # _, _, batch_tokens = batch_converter(esm_data)
    # print('esm data: ', esm_data.requires_grad, ' ', seqs.requires_grad)
    # batch_tokens = batch_tokens.to(device='cuda:0')
    eos_mask = torch.cat([torch.zeros_like(eos_mask[:,0]).unsqueeze(1), eos_mask], dim=1)
    eos_mask = eos_mask != -1
    results = esm(nodes_esm, repr_layers=[30], return_contacts=True, embed_model=True, custom_eos_mask=eos_mask)
    preds = results['contacts']
    return preds