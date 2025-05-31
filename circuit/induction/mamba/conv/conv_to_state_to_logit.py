import os
import sys
from typing import Any

from einops import rearrange
from fancy_einsum import einsum
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
from mamba_lens import HookedMamba
from easy_transformer.ioi_dataset import IOIDataset, get_end_idxs
from tqdm import tqdm

from path_patching import mamba_utils

torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = HookedMamba.from_pretrained("state-spaces/mamba-130m", device=device)

tokenizer = model.tokenizer
vocab = tokenizer.vocab

up_layer = 12
bias = 1

N = 2048
batch_size = 32
seq_len = 64

names_filter = []
for layer in range(model.cfg.n_layers):
    names_filter.append(f'blocks.{layer}.hook_in_proj')
    names_filter.append(f'blocks.{layer}.hook_conv')
    names_filter.append(f'blocks.{layer}.hook_h.{seq_len - 1}')
    
sum_logit_new = torch.zeros([bias + 1]).to(device)
sum_p_new = torch.zeros([bias + 1]).to(device)
sum_logit_orig = torch.zeros([bias + 1]).to(device)
sum_p_orig = torch.zeros([bias + 1]).to(device)

for batch in tqdm(range(N)):
    random_input = torch.randint(0, len(vocab), size=(batch_size, seq_len)).to(device)
    source_index = torch.randint(0, int(seq_len / 2), size=(batch_size,)).to(device)
    object_index = torch.ones_like(source_index)
    object_index *= (seq_len - 1)

    source_values = random_input.gather(1, source_index.unsqueeze(1))
    induction_input = random_input.clone()
    induction_input.scatter_(1, object_index.unsqueeze(1), source_values)
    patch_input = random_input.clone()
    patch_input.scatter_(1, source_index.unsqueeze(1), torch.randint(0, len(vocab), size=(batch_size, seq_len)).to(device))
    patch_input.scatter_(1, (source_index + 1).unsqueeze(1), torch.randint(0, len(vocab), size=(batch_size, seq_len)).to(device))
    source_1_values = random_input.gather(1, (source_index + 1).unsqueeze(1)).squeeze(dim=1)
    
    logits, cache = model.run_with_cache(induction_input, names_filter=names_filter, fast_ssm=False, fast_conv=False, warn_disabled_hooks=False)
    _, patch_cache = model.run_with_cache(patch_input, names_filter=names_filter, fast_ssm=False, fast_conv=False, warn_disabled_hooks=False)
    logit_orig = logits[torch.arange(logits.shape[0]), object_index, source_1_values]
    p_orig = F.softmax(logits, dim=-1)[torch.arange(logits.shape[0]), object_index, source_1_values]
    sum_logit_orig += logit_orig.sum()
    sum_p_orig += p_orig.sum()
    def generate_replacement_hook(layer):
        def replacement_hook(activations: torch.Tensor, hook: Any):
            activations = cache[f'blocks.{layer}.hook_h.{seq_len - 1}']
            return activations
        return replacement_hook
    
    for channel in range(bias + 1):
        def conv_out_replacement_hook(activations: torch.Tensor, hook: Any):
            x_in = cache[f'blocks.{up_layer}.hook_in_proj']
            x_in[torch.arange(x_in.shape[0]), source_index + bias - channel] = patch_cache[f'blocks.{up_layer}.hook_in_proj'][torch.arange(x_in.shape[0]), source_index + bias - channel] 
            # x_in[torch.arange(x_in.shape[0]), source_index + bias] = patch_cache[f'blocks.{up_layer}.hook_in_proj'][torch.arange(x_in.shape[0]), source_index + bias] 
            # x_in[torch.arange(x_in.shape[0]), source_index + bias - 1] = patch_cache[f'blocks.{up_layer}.hook_in_proj'][torch.arange(x_in.shape[0]), source_index + bias - 1]
            # x_in[torch.arange(x_in.shape[0]), source_index + bias - 2] = patch_cache[f'blocks.{up_layer}.hook_in_proj'][torch.arange(x_in.shape[0]), source_index + bias - 2]      
            x_conv = rearrange(x_in, 'B L E -> B E L')
            x_conv_out = model.blocks[up_layer].conv1d(x_conv)
            x_conv_out = rearrange(x_conv_out, 'B E L -> B L E')
            x_conv_out_cutoff = x_conv_out[:,:seq_len,:]
            activations[torch.arange(activations.shape[0]), source_index + bias] = x_conv_out_cutoff[torch.arange(activations.shape[0]), source_index + bias]
            '''
            activations[torch.arange(batch_size), source_index + bias] = patch_cache[f'blocks.{up_layer}.hook_conv'][torch.arange(batch_size), source_index + bias]
            '''
            return activations
        resid = model.run_with_hooks(induction_input, fwd_hooks=[(f'blocks.{up_layer}.hook_conv', conv_out_replacement_hook)], stop_at_layer=up_layer + 1)
        def resid_replacement_hook(activations: torch.Tensor, hook: Any):
            return resid
        fwd_hooks = [(f'blocks.{up_layer}.hook_resid_post', resid_replacement_hook)]
        for l in range(up_layer + 1, model.cfg.n_layers, 1):
            hook_name = f'blocks.{l}.hook_h.{seq_len - 1}'
            fwd_hooks.append((hook_name, generate_replacement_hook(l)))
        logits = model.run_with_hooks(induction_input, fwd_hooks=fwd_hooks, fast_ssm=False, fast_conv=False)
        logit_new = logits[torch.arange(logits.shape[0]), object_index, source_1_values]
        p_new = F.softmax(logits, dim=-1)[torch.arange(logits.shape[0]), object_index, source_1_values]
        sum_logit_new[channel] += logit_new.sum()
        sum_p_new[channel] += p_new.sum()

mean_logit_new = sum_logit_new / (N * batch_size)
mean_p_new = sum_p_new / (N * batch_size)
mean_logit_orig = sum_logit_orig / (N * batch_size)
mean_p_orig = sum_p_orig / (N * batch_size)

result_dir = f'circuit/induction/mamba/result/L{up_layer}_bias{bias}'
os.makedirs(result_dir, exist_ok=True)
torch.save(mean_logit_new, os.path.join(result_dir, 'mean_logit_new.pt'))
torch.save(mean_p_new, os.path.join(result_dir, 'mean_p_new.pt'))
torch.save(mean_logit_orig, os.path.join(result_dir, 'mean_logit_orig.pt'))
torch.save(mean_p_orig, os.path.join(result_dir, 'mean_p_orig.pt'))