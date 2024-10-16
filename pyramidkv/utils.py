
import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import math

# perform qk calculation and get indices
# this version will not update in inference mode

# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def get_attn_weights(query_states, key_states, window_size, head_dim, q='obs_window'):
    assert q == 'obs_window' or q == 'full', "q should be obs_window or full"

    if q=='obs_window':
        attn_weights = torch.matmul(query_states[..., -window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
        mask = torch.full((window_size, window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    else:
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
        mask = torch.full((attn_weights.shape[2], attn_weights.shape[3]), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(attn_weights.device)
    attention_mask = mask[None, None, :, :]
    if q=='obs_window':
        attn_weights[:, :, -window_size:, -window_size:] += attention_mask
    else:
        attn_weights += attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    return attn_weights

def pool_kv(attn_weights_sum, pooling, kernel_size):
    if pooling == 'avgpool':
        attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = kernel_size, padding=kernel_size//2, stride=1)
    elif pooling == 'maxpool':
        attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = kernel_size, padding=kernel_size//2, stride=1)
    else:
        raise ValueError('Pooling method not supported')
    return attn_cache

def gather_kv(key_states, value_states, window_size, indices):
    k_past_compress = key_states[:, :, :-window_size, :].gather(dim=2, index=indices)
    v_past_compress = value_states[:, :, :-window_size, :].gather(dim=2, index=indices)
    k_cur = key_states[:, :, -window_size:, :]
    v_cur = value_states[:, :, -window_size:, :]
    key_states = torch.cat([k_past_compress, k_cur], dim=2)
    value_states = torch.cat([v_past_compress, v_cur], dim=2)
    return key_states, value_states

class PyramidKVCluster():
    def __init__(self, kv_compression_method, num_hidden_layers = 32, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', beta = 20, num_layers = 80, layer_idx=None):
        self.kv_compression_method = kv_compression_method
        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers
        
        self.steps = -1
        self.beta = beta
        
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        
        # TODO
        # window_sizes = 32
        min_num = (self.max_capacity_prompt - self.window_size) // self.beta
        max_num = (self.max_capacity_prompt - self.window_size) * 2 - min_num
        
        if max_num >= q_len - self.window_size:
            max_num = q_len - self.window_size
            min_num = (self.max_capacity_prompt - self.window_size) * 2 - max_num
        
        steps = (max_num - min_num) // (self.num_hidden_layers - 1)
        max_capacity_prompt = max_num - self.layer_idx * steps
        # print(f"PyramidKV max_capacity_prompt {max_capacity_prompt}")
        
        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            attn_weights = get_attn_weights(query_states, key_states, self.window_size, head_dim, q='obs_window')
            attn_weights_sum = attn_weights[:, :, -self.window_size:, :-self.window_size].sum(dim = -2)

            if "avg" in self.kv_compression_method:
                key_states_avg = torch.mean(key_states, dim=1, keepdim=True)
                key_states_avg = key_states_avg.expand(-1, num_heads, -1, -1)
                avg_attn_weights = get_attn_weights(query_states, key_states_avg, self.window_size, head_dim, q='obs_window')
                avg_attn_weights_sum = avg_attn_weights[:, :, -self.window_size:, :-self.window_size].sum(dim = -2)
                attn_weights_sum += avg_attn_weights_sum

            attn_cache = pool_kv(attn_weights_sum, self.pooling, self.kernel_size)

            if q_len < (self.max_capacity_prompt - self.window_size) * 2:
                indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
            else:
                indices = attn_cache.topk(max_capacity_prompt, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            key_states, value_states = gather_kv(key_states, value_states, self.window_size, indices)
            return key_states, value_states

class SnapKVCluster():
    def __init__(self, kv_compression_method, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.kv_compression_method = kv_compression_method
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        # print(f"SnapKV max_capacity_prompt {self.max_capacity_prompt}")
        
        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            attn_weights = get_attn_weights(query_states, key_states, self.window_size, head_dim, q='obs_window')
            attn_weights_sum = attn_weights[:, :, -self.window_size:, :-self.window_size].sum(dim = -2)

            if "avg" in self.kv_compression_method:
                key_states_avg = torch.mean(key_states, dim=1, keepdim=True)
                key_states_avg = key_states_avg.expand(-1, num_heads, -1, -1)
                avg_attn_weights = get_attn_weights(query_states, key_states_avg, self.window_size, head_dim, q='obs_window')
                avg_attn_weights_sum = avg_attn_weights[:, :, -self.window_size:, :-self.window_size].sum(dim = -2)
                attn_weights_sum += avg_attn_weights_sum

            attn_cache = pool_kv(attn_weights_sum, self.pooling, self.kernel_size)

            indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            key_states, value_states = gather_kv(key_states, value_states, self.window_size, indices)
            return key_states, value_states


class H2OKVCluster():
    def __init__(self, kv_compression_method, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.kv_compression_method = kv_compression_method
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        # print(f"H2O max_capacity_prompt {self.max_capacity_prompt}")
        
        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            attn_weights = get_attn_weights(query_states, key_states, self.window_size, head_dim, q='full')
            attn_weights_sum = attn_weights[:, :, :, :-self.window_size].sum(dim = -2)
            divisors = torch.arange(7500, 7500-attn_weights_sum.shape[2], -1).float() # 7500 is max length for llama3-8b-instruct
            divisors = divisors.unsqueeze(0).unsqueeze(0).expand(bsz, num_heads, -1).to(attn_weights.device)
            attn_weights_sum = attn_weights_sum / divisors

            if "avg" in self.kv_compression_method:
                key_states_avg = torch.mean(key_states, dim=1, keepdim=True)
                key_states_avg = key_states_avg.expand(-1, num_heads, -1, -1)
                avg_attn_weights = get_attn_weights(query_states, key_states_avg, self.window_size, head_dim, q='full')
                avg_attn_weights_sum = avg_attn_weights[:, :, :, :-self.window_size].sum(dim = -2)
                avg_attn_weights_sum = avg_attn_weights_sum / divisors
                attn_weights_sum += avg_attn_weights_sum

            attn_cache = attn_weights_sum

            indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            key_states, value_states = gather_kv(key_states, value_states, self.window_size, indices)
            return key_states, value_states


class StreamingLLMKVCluster():
    def __init__(self, kv_compression_method, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.kv_compression_method = kv_compression_method
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        # print(f"StreamingLLM max_capacity_prompt {self.max_capacity_prompt}")

        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            indices = torch.tensor(range(self.max_capacity_prompt - self.window_size), dtype=torch.int64).to(key_states.device)
            indices = indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(bsz, num_heads, 1, head_dim)
            key_states, value_states = gather_kv(key_states, value_states, self.window_size, indices)
            return key_states, value_states


def init_pyramidkv(self, kv_compression_method, num_hidden_layers):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
    
    self.kv_cluster = PyramidKVCluster( 
        kv_compression_method = kv_compression_method,
        num_hidden_layers = num_hidden_layers,
        layer_idx = self.layer_idx,
        window_size = self.config.window_size, 
        max_capacity_prompt = self.config.max_capacity_prompt, 
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling
        )
 
def init_snapkv(self, kv_compression_method):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 4096
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
    
    self.kv_cluster = SnapKVCluster( 
        kv_compression_method = kv_compression_method,
        window_size = self.config.window_size, 
        max_capacity_prompt = self.config.max_capacity_prompt, 
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling
        )

def init_H2O(self, kv_compression_method):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
    
    self.kv_cluster = H2OKVCluster(
        kv_compression_method = kv_compression_method,
        window_size = self.config.window_size, 
        max_capacity_prompt = self.config.max_capacity_prompt, 
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling
        )

def init_StreamingLLM(self, kv_compression_method):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
    
    self.kv_cluster = StreamingLLMKVCluster(
        kv_compression_method = kv_compression_method,
        window_size = self.config.window_size, 
        max_capacity_prompt = self.config.max_capacity_prompt, 
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling
        )
