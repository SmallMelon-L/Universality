import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer

def patch_attn_out_to_logits(model, resid_orig, attn_out_orig, attn_out_new):
    resid_new = resid_orig - attn_out_orig + attn_out_new
    resid_new_normed = model.ln_final(resid_new)
    logits = model.unembed(resid_new_normed)
    return logits

def patch_mlp_out_to_logits(model, resid_orig, mlp_out_orig, mlp_out_new):
    resid_new = resid_orig - mlp_out_orig + mlp_out_new
    resid_new_normed = model.ln_final(resid_new)
    logits = model.unembed(resid_new_normed)
    return logits