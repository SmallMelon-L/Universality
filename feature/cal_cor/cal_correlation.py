import os
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm
import matplotlib.pyplot as plt
import time  # 导入time模块

from lm_saes.config import SAEConfig
from lm_saes.sae import SparseAutoEncoder
torch.set_grad_enabled(False)

# 获取命令行输入的参数
if len(sys.argv) != 12:
    print("Usage: python script_name.py <layer_1> <layer_2> <d_model> <exp_factor> <sae_path_1> <sae_path_2> <model_name_1> <model_name_2> <sae_type_1> <sae_type_2> <dataset_name>")
    sys.exit(1)

layer_1 = int(sys.argv[1])
layer_2 = int(sys.argv[2])
d_model = int(sys.argv[3])
exp_factor = int(sys.argv[4])
sae_path_1 = str(sys.argv[5])
sae_path_2 = str(sys.argv[6])
model_name_1 = str(sys.argv[7])
model_name_2 = str(sys.argv[8])
sae_type_1 = str(sys.argv[9])
sae_type_2 = str(sys.argv[10])
dataset_name = str(sys.argv[11])

total_tokens = 1_000_000
device = 'cuda:0'

# mamba-sae
sae_name_1 = f"L{layer_1}R"
sae_cfg_1 = SAEConfig.from_pretrained(os.path.join(sae_path_1, sae_name_1), device=device)
sae_1 = SparseAutoEncoder.from_config(sae_cfg_1).to(torch.bfloat16)

act_dir_1 = "path/to/act_dir_1"
act_dir_1 = os.path.join(act_dir_1, dataset_name)
act_dir_1 = os.path.join(act_dir_1, model_name_1)
act_dir_1 = os.path.join(act_dir_1, f'layer_{layer_1}')

# pythia-sae
sae_name_2 = f"L{layer_2}R"
sae_cfg_2 = SAEConfig.from_pretrained(os.path.join(sae_path_2, sae_name_2), device=device)
sae_2 = SparseAutoEncoder.from_config(sae_cfg_2).to(torch.bfloat16)

act_dir_2 = "path/to/act_dir_2"
act_dir_2 = os.path.join(act_dir_2, dataset_name)
act_dir_2 = os.path.join(act_dir_2, model_name_2)
act_dir_2 = os.path.join(act_dir_2, f'layer_{layer_2}')

m = torch.zeros([d_model * exp_factor], dtype=torch.float32).to(device)
p = torch.zeros([d_model * exp_factor], dtype=torch.float32).to(device)
mm = torch.zeros([d_model * exp_factor], dtype=torch.float32).to(device)
pp = torch.zeros([d_model * exp_factor], dtype=torch.float32).to(device)
mp = torch.zeros([d_model * exp_factor, d_model * exp_factor], dtype=torch.float32).to(device)

current_tokens = 0

with tqdm(total=total_tokens, desc=f'sampled tokens') as pbar:
    while current_tokens < total_tokens:
        
        act_1 = torch.load(os.path.join(act_dir_1, f'activations_{current_tokens}.pt'))
        act_2 = torch.load(os.path.join(act_dir_2, f'activations_{current_tokens}.pt'))

        features_1 = sae_1.encode(act_1.to(torch.bfloat16))
        features_2 = sae_2.encode(act_2.to(torch.bfloat16))
        
        process_tokens = features_1.shape[0]
        m += torch.sum(features_1, dim=0)
        p += torch.sum(features_2, dim=0)
        mm += torch.sum(features_1 ** 2, dim=0)
        pp += torch.sum(features_2 ** 2, dim=0)
        mp += torch.matmul(features_1.T, features_2)

        current_tokens += process_tokens
        pbar.update(process_tokens)

del act_1, act_2, features_1, features_2, sae_1, sae_2
torch.cuda.empty_cache()

n = current_tokens

block_size = 1024
num_blocks = (d_model * exp_factor + block_size - 1) // block_size

correlation_matrix = torch.zeros([d_model * exp_factor, d_model * exp_factor], dtype=torch.float32).to(device)

for i in range(num_blocks):
    start_i = i * block_size
    end_i = min((i + 1) * block_size, d_model * exp_factor)
    
    m_block = m[start_i:end_i]
    mp_block = mp[start_i:end_i, :]
    
    for j in range(num_blocks):
        start_j = j * block_size
        end_j = min((j + 1) * block_size, d_model * exp_factor)
        
        p_block = p[start_j:end_j]
        mp_subblock = mp_block[:, start_j:end_j]
        
        denominator = torch.sqrt(
            (n * mm[start_i:end_i] - m_block ** 2).unsqueeze(1) @
            (n * pp[start_j:end_j] - p_block ** 2).unsqueeze(0)
        )
        denominator = torch.clamp(denominator, min=1e-12)
        
        correlation_subblock = (n * mp_subblock - m_block.unsqueeze(1) @ p_block.unsqueeze(0)) / denominator
        
        correlation_matrix[start_i:end_i, start_j:end_j] = correlation_subblock


# correlation_matrix = (n * mp.to(torch.float32) - m.to(torch.float32).unsqueeze(1) @ p.to(torch.float32).unsqueeze(0)) / torch.clamp(torch.sqrt((n * mm.to(torch.float32) - m.to(torch.float32) ** 2).unsqueeze(1) @ (n * pp.to(torch.float32) - p.to(torch.float32) ** 2).unsqueeze(0)), min=1e-12)

result_dir = 'path/to/result_dir'

result_dir_1 = os.path.join(result_dir, f'{model_name_1}->{model_name_2}')
result_dir_1 = os.path.join(result_dir_1, f'{sae_type_1}->{sae_type_2}')
result_dir_1 = os.path.join(result_dir_1, f'layer2layer')
os.makedirs(result_dir_1, exist_ok=True)
torch.save(correlation_matrix.max(dim=1), os.path.join(result_dir_1, f'L{layer_1}->L{layer_2}.pt'))

result_dir_2 = os.path.join(result_dir, f'{model_name_2}->{model_name_1}')
result_dir_2 = os.path.join(result_dir_2, f'{sae_type_2}->{sae_type_1}')
result_dir_2 = os.path.join(result_dir_2, f'layer2layer')
os.makedirs(result_dir_2, exist_ok=True)
torch.save(correlation_matrix.max(dim=0), os.path.join(result_dir_2, f'L{layer_2}->L{layer_1}.pt'))
