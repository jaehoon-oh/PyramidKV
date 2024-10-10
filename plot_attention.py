import argparse
from transformers import AutoConfig, AutoTokenizer
from custom_models.modeling_llama import LlamaForCausalLM, LlamaRotaryEmbedding
from custom_models.modeling_llama import apply_rotary_pos_emb, repeat_kv

import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
# from matplotlib.colors import LogNorm
# import os


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",
                        type=str,
                        default="meta-llama/Meta-Llama-3-8B")
    
    args = parser.parse_args()

    model_path = args.model_path
    model_name = model_path.split('/')[-1]

    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=config.torch_dtype, device_map='cuda:0')
    model.eval()

    seq = "Summer is warm. Winter is cold."
    input_ids = tokenizer(seq, return_tensors="pt", add_special_tokens=False)
    input_ids, attention_mask = input_ids.input_ids.to(model.device), input_ids.attention_mask.to(model.device)

    results = model(input_ids, output_hidden_states=True)
    hidden_states_list = results["hidden_states"]

    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads
    hidden_size = config.hidden_size
    head_dim = hidden_size // num_heads
    num_key_value_groups = num_heads // num_key_value_heads

    attentions = []

    for layer_idx, hidden_states in enumerate(hidden_states_list):
        if layer_idx == num_layers:
            break

        bsz, q_len, _ = hidden_states.size()

        hidden_states = model.model.layers[layer_idx].input_layernorm(hidden_states)

        query_states = model.model.layers[layer_idx].self_attn.q_proj(hidden_states)
        key_states = model.model.layers[layer_idx].self_attn.k_proj(hidden_states)
        value_states = model.model.layers[layer_idx].self_attn.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)

        rotary_emb = LlamaRotaryEmbedding(
            head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            )
        position_ids = torch.arange(q_len).unsqueeze(0).to(model.device)
        cos, sin = rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # if past_key_value is not None:
        #     # sin and cos are specific to RoPE models; cache_position needed for the static cache
        #     cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        #     key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, num_key_value_groups)
        value_states = repeat_kv(value_states, num_key_value_groups)
        
        ##### Added Part Start ##### (should match with L346-L352 in custom_models/modeling_llama.py)
        query_states_avg = torch.mean(query_states, dim=1, keepdim=True).expand(-1, num_heads, -1, -1)
        key_states_avg = torch.mean(key_states, dim=1, keepdim=True).expand(-1, num_heads, -1, -1)

        query_states -= query_states_avg
        key_states -= key_states_avg
        ##### Added Part End #####

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)

        causal_mask = torch.full(
            (q_len, q_len), fill_value=torch.finfo(torch.bfloat16).min, dtype=config.torch_dtype, device=model.device
        )
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask = causal_mask[None, None, :, :]
        attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=0.0, training=False)
        attentions.append(attn_weights)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, num_heads, q_len, head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, num_heads, q_len, head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, hidden_size)
        attn_output = model.model.layers[layer_idx].self_attn.o_proj(attn_output)

    for layer_idx, attention in enumerate(attentions):
        fig, axes = plt.subplots(1, 8, figsize=(14, 2))
        for head_idx, ax in enumerate(axes):
            sns.heatmap(attention[0, head_idx].detach().float().cpu().numpy(), ax=ax, cmap='viridis', cbar=False, vmin=0, vmax=1)
            ax.set_title(f'head {head_idx}')
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        plt.savefig(f'src/{model_name}_layer{layer_idx:02d}.png')
        plt.close()