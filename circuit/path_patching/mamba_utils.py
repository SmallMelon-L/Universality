import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
from mamba_lens import HookedMamba
from fancy_einsum import einsum

def patch_out_proj_to_logits(model, resid_orig, out_proj_orig, out_proj_new):
    resid_new = resid_orig - out_proj_orig + out_proj_new
    resid_new_normed = model.norm(resid_new)
    logits = model.lm_head(resid_new_normed)
    return logits

def patch_ssm_output_to_out_proj(model, ssm_output_new, skip_orig, layer):
    out_proj_new = model.blocks[layer].out_proj(ssm_output_new * F.silu(skip_orig))
    return out_proj_new

def patch_skip_to_out_proj(model, ssm_output_orig, skip_new, layer):
    out_proj_new = model.blocks[layer].out_proj(ssm_output_orig * F.silu(skip_new))
    return out_proj_new

def patch_y_to_out_proj(model, y_new, ssm_input_orig, skip_orig, layer):
    out_proj_new = model.blocks[layer].out_proj((y_new + ssm_input_orig * model.blocks[layer].W_D) * F.silu(skip_orig))
    return out_proj_new

def patch_shortcut_to_out_proj(model, y_orig, ssm_input_new, skip_orig, layer):
    out_proj_new = model.blocks[layer].out_proj((y_orig + ssm_input_new * model.blocks[layer].W_D) * F.silu(skip_orig))
    return out_proj_new

def patch_C_to_out_proj(model, h_orig, C_new, ssm_input_orig, skip_orig, layer):
    out_proj_new = model.blocks[layer].out_proj((h_orig @ C_new + ssm_input_orig * model.blocks[layer].W_D) * F.silu(skip_orig))
    return out_proj_new 

def patch_h_to_out_proj(model, h_new, C_orig, ssm_input_orig, skip_orig, layer):
    out_proj_new = model.blocks[layer].out_proj((h_new @ C_orig + ssm_input_orig * model.blocks[layer].W_D) * F.silu(skip_orig))
    return out_proj_new

def patch_state_to_out_proj(model, out_proj_orig, h_orig, h_new, C_orig, C_new, skip_orig, layer, state):
    state_to_out_proj_orig = model.blocks[layer].out_proj(h_orig[..., state] * C_orig[..., state] * F.silu(skip_orig))
    state_to_out_proj_new = model.blocks[layer].out_proj(h_new[..., state] * C_new[..., state] * F.silu(skip_orig))
    state_to_out_proj_diff = state_to_out_proj_new - state_to_out_proj_orig
    out_proj_new = out_proj_orig + state_to_out_proj_diff
    return out_proj_new

def patch_state_to_y(y_orig, h_orig, h_new, C_orig, C_new, state):
    state_to_y_orig = einsum('b d, b -> b d', h_orig[..., state], C_orig[..., state])
    state_to_y_new = einsum('b d, b -> b d', h_new[..., state], C_new[..., state])
    state_to_y_diff = state_to_y_new - state_to_y_orig
    y_new = y_orig + state_to_y_diff
    return y_new