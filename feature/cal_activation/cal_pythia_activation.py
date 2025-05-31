import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformer_lens import HookedTransformer

# 固定随机种子
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 禁用梯度计算
torch.set_grad_enabled(False)

# 参数设置
device = 'cuda:0'
batch_size = 8
max_length = 1024
total_tokens = 10_000_000

# 模型加载
hf_model = AutoModelForCausalLM.from_pretrained(
    'path/to/model'
)
model = HookedTransformer.from_pretrained(
    'path/to/model', 
    device=device, 
    hf_model=hf_model
)
tokenizer = model.tokenizer

# 设置评估模式，避免非确定性行为
hf_model.eval()
model.eval()

# 数据加载
dataset = load_from_disk(
    "path/to/dataset"
)

# 设置 DataLoader 的生成器
data_gen = torch.Generator()
data_gen.manual_seed(seed)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=data_gen)
data_iter = iter(dataloader)

# 激活保存路径
output_dir = "path/to/output_dir"
os.makedirs(output_dir, exist_ok=True)

# 初始化层激活保存的文件夹
start_layer_num = 0
end_layer_num = 12
layer_dirs = [os.path.join(output_dir, f'layer_{l}') for l in range(start_layer_num, end_layer_num)]
for layer_dir in layer_dirs:
    os.makedirs(layer_dir, exist_ok=True)

# 当前已采样 token 数
current_tokens = 0

# 主循环
with tqdm(total=total_tokens, desc='Sampled tokens') as pbar:
    while current_tokens < total_tokens:
        try:
            # 获取一个批次
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        # 文本到 token 转换
        sentences = batch['text']
        tokens = model.to_tokens(sentences, prepend_bos=True).to(device)
        tokens = tokens[:, :max_length]

        # 过滤特殊 token
        filter_mask = torch.logical_and(tokens.ne(tokenizer.eos_token_id), tokens.ne(tokenizer.pad_token_id))
        filter_mask = torch.logical_and(filter_mask, tokens.ne(tokenizer.bos_token_id))

        # 计算激活值
        _, cache = model.run_with_cache(tokens, names_filter=[f'blocks.{l}.hook_resid_post' for l in range(start_layer_num, end_layer_num)])

        # 保存激活
        for l in range(start_layer_num, end_layer_num):
            # 获取过滤后的激活值
            activations = cache[f'blocks.{l}.hook_resid_post'][filter_mask].to(torch.bfloat16)

            # 构建存储路径
            save_path = os.path.join(layer_dirs[l], f'activations_{current_tokens}.pt')

            # 保存激活值
            torch.save(activations, save_path)

        current_tokens += activations.size(0)
        pbar.update(activations.size(0))
