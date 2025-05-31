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
from tqdm import tqdm

from path_patching import mamba_utils

torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = HookedMamba.from_pretrained("state-spaces/mamba-130m", device=device)

tokenizer = model.tokenizer
vocab = tokenizer.vocab

N = 128
batch_size = 16
seq_len = 64

names_filter = []
for layer in range(model.cfg.n_layers):
    names_filter.append(f'blocks.{layer}.hook_h.{seq_len - 1}')
    names_filter.append(f'blocks.{layer}.hook_h.{seq_len - 2}')
    names_filter.append(f'blocks.{layer}.hook_A_bar')
    names_filter.append(f'blocks.{layer}.hook_B_bar')
    names_filter.append(f'blocks.{layer}.hook_ssm_input')

sum_logit_new = torch.zeros([model.cfg.n_layers]).to(device)
sum_p_new = torch.zeros([model.cfg.n_layers]).to(device)
sum_logit_orig = torch.zeros([model.cfg.n_layers]).to(device)
sum_p_orig = torch.zeros([model.cfg.n_layers]).to(device)

for batch in tqdm(range(N)):
    random_input = torch.randint(0, len(vocab), size=(batch_size, seq_len)).to(device)
    source_index = torch.randint(0, int(seq_len / 2), size=(batch_size,)).to(device)
    object_index = torch.ones_like(source_index)
    object_index *= (seq_len - 1)

    source_values = random_input.gather(1, source_index.unsqueeze(1))
    induction_input = random_input.clone()
    induction_input.scatter_(1, object_index.unsqueeze(1), source_values)
    patch_input = induction_input.clone()
    # patch_input.scatter_(1, source_index.unsqueeze(1), torch.randint(0, len(vocab), size=(batch_size, seq_len)).to(device))
    patch_input.scatter_(1, (source_index + 1).unsqueeze(1), torch.randint(0, len(vocab), size=(batch_size, seq_len)).to(device))
    source_1_values = random_input.gather(1, (source_index + 1).unsqueeze(1)).squeeze(dim=1)
    
    logits, cache = model.run_with_cache(induction_input, names_filter=names_filter, fast_ssm=False, fast_conv=True, warn_disabled_hooks=False)
    _, patch_cache = model.run_with_cache(patch_input, names_filter=names_filter, fast_ssm=False, fast_conv=True, warn_disabled_hooks=False)
    logit_orig = logits[torch.arange(logits.shape[0]), object_index, source_1_values]
    p_orig = F.softmax(logits, dim=-1)[torch.arange(logits.shape[0]), object_index, source_1_values]
    sum_logit_orig += logit_orig.sum()
    sum_p_orig += p_orig.sum()
    def generate_replacement_hook(layer, patch_layer, cache, patch_cache):
        if layer == patch_layer:
            def replacement_hook(activations: torch.Tensor, hook: Any):
                activations = patch_cache[f'blocks.{layer}.hook_h.{seq_len - 2}'] * cache[f'blocks.{layer}.hook_A_bar'][:,-1,:,:] + cache[f'blocks.{layer}.hook_B_bar'][:,-1,:,:] * cache[f'blocks.{layer}.hook_ssm_input'][:,-1].view(batch_size, model.cfg.d_inner, 1)
                return activations
        else:
            def replacement_hook(activations: torch.Tensor, hook: Any):
                activations = cache[f'blocks.{layer}.hook_h.{seq_len - 1}']
                return activations
        return replacement_hook
    
    for layer in range(model.cfg.n_layers):
        fwd_hooks = []
        for l in range(model.cfg.n_layers):
            hook_name = f'blocks.{l}.hook_h.{seq_len - 1}'
            fwd_hooks.append((hook_name, generate_replacement_hook(l, layer, cache, patch_cache)))
        logits = model.run_with_hooks(induction_input, fwd_hooks=fwd_hooks, fast_ssm=False, fast_conv=True, warn_disabled_hooks=False)
        logit_new = logits[torch.arange(logits.shape[0]), object_index, source_1_values]
        p_new = F.softmax(logits, dim=-1)[torch.arange(logits.shape[0]), object_index, source_1_values]
        sum_logit_new[layer] += logit_new.sum()
        sum_p_new[layer] += p_new.sum()

mean_logit_new = sum_logit_new / (N * batch_size)
mean_p_new = sum_p_new / (N * batch_size)
mean_logit_orig = sum_logit_orig / (N * batch_size)
mean_p_orig = sum_p_orig / (N * batch_size)

result_dir = 'circuit/induction/mamba/result/state_to_logit'
os.makedirs(result_dir, exist_ok=True)
torch.save(mean_logit_new, os.path.join(result_dir, 'mean_logit_new.pt'))
torch.save(mean_p_new, os.path.join(result_dir, 'mean_p_new.pt'))
torch.save(mean_logit_orig, os.path.join(result_dir, 'mean_logit_orig.pt'))
torch.save(mean_p_orig, os.path.join(result_dir, 'mean_p_orig.pt'))