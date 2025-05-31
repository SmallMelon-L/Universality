import os
import sys
from typing import Any

from einops import rearrange
from fancy_einsum import einsum
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
from easy_transformer.ioi_dataset import IOIDataset, get_end_idxs
from tqdm import tqdm

from path_patching import pythia_utils

torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

hf_model = AutoModelForCausalLM.from_pretrained(
    'EleutherAI/pythia-160m',
)
hf_tokenizer = AutoTokenizer.from_pretrained(
    'EleutherAI/pythia-160m',
	trust_remote_code=True,
	use_fast=True,
	add_bos_token=True,
)
tokenizer = hf_tokenizer
model = HookedTransformer.from_pretrained(
    'EleutherAI/pythia-160m',
	device=device,
	hf_model=hf_model,
	tokenizer=hf_tokenizer,
)

vocab = tokenizer.vocab

N = 2048
batch_size = 32
seq_len = 64

sum_logit_new = torch.zeros([model.cfg.n_layers, model.cfg.n_heads]).to(device)
sum_p_new = torch.zeros([model.cfg.n_layers, model.cfg.n_heads]).to(device)
sum_logit_orig = torch.zeros([model.cfg.n_layers, model.cfg.n_heads]).to(device)
sum_p_orig = torch.zeros([model.cfg.n_layers, model.cfg.n_heads]).to(device)

for batch in tqdm(range(N), desc='Batch'):
    random_input = torch.randint(0, len(vocab), size=(batch_size, seq_len)).to(device)
    source_index = torch.randint(0, int(seq_len / 2), size=(batch_size,)).to(device)
    object_index = torch.ones_like(source_index).to(device)
    object_index *= (seq_len - 1)

    source_values = random_input.gather(1, source_index.unsqueeze(1))
    induction_input = random_input.clone()
    induction_input.scatter_(1, object_index.unsqueeze(1), source_values)
    patch_input = torch.randint(0, len(vocab), size=(batch_size, seq_len)).to(device)
    # patch_input = induction_input.clone()
    # patch_input.scatter_(1, (source_index + 1).unsqueeze(1), torch.randint(0, len(vocab), size=(batch_size, seq_len)).to(device))
    source_1_values = random_input.gather(1, (source_index + 1).unsqueeze(1)).squeeze(dim=1)
    
    logits, cache = model.run_with_cache(induction_input)
    _, patch_cache = model.run_with_cache(patch_input)
    logit_orig = logits[torch.arange(logits.shape[0]), object_index, source_1_values]
    p_orig = F.softmax(logits, dim=-1)[torch.arange(logits.shape[0]), object_index, source_1_values]
    sum_logit_orig += logit_orig.sum()
    sum_p_orig += p_orig.sum()
    def generate_replacement_hook(layer, head, l):
        if l == layer:
            def replacement_hook(activations: torch.Tensor, hook: Any):
                activations[:, -1] = cache[f'blocks.{l}.attn.hook_z'][:, -1]
                activations[:, -1, head, :] = patch_cache[f'blocks.{l}.attn.hook_z'][:, -1, head, :]
                return activations
        else:
            def replacement_hook(activations: torch.Tensor, hook: Any):
                activations[:, -1] = cache[f'blocks.{l}.attn.hook_z'][:, -1]
                return activations
        return replacement_hook
    
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            fwd_hooks = []
            for l in range(model.cfg.n_layers):
                hook_name = f'blocks.{l}.attn.hook_z'
                fwd_hooks.append((hook_name, generate_replacement_hook(layer, head, l)))
            logits = model.run_with_hooks(induction_input, fwd_hooks=fwd_hooks)
            logit_new = logits[torch.arange(logits.shape[0]), object_index, source_1_values]
            p_new = F.softmax(logits, dim=-1)[torch.arange(logits.shape[0]), object_index, source_1_values]
            sum_logit_new[layer, head] += logit_new.sum()
            sum_p_new[layer, head] += p_new.sum()
            
                        
                        

mean_logit_new = sum_logit_new / (N * batch_size)
mean_p_new = sum_p_new / (N * batch_size)
mean_logit_orig = sum_logit_orig / (N * batch_size)
mean_p_orig = sum_p_orig / (N * batch_size)

result_dir = 'circuit/induction/pythia/result/head_to_logit'
os.makedirs(result_dir, exist_ok=True)
torch.save(mean_logit_new, os.path.join(result_dir, 'mean_logit_new.pt'))
torch.save(mean_p_new, os.path.join(result_dir, 'mean_p_new.pt'))
torch.save(mean_logit_orig, os.path.join(result_dir, 'mean_logit_orig.pt'))
torch.save(mean_p_orig, os.path.join(result_dir, 'mean_p_orig.pt'))