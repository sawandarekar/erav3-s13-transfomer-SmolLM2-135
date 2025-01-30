import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import yaml



class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, config):
        super().__init__()
        max_position_embeddings = config['model']['model_config'].get('max_position_embeddings', 2048)
        base = config['model']['model_config'].get('rope_theta', 10000.0)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_position_embeddings = max_position_embeddings
        self.dim = dim

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]
        position = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        sincos = torch.einsum("i,j->ij", position, self.inv_freq)
        emb = torch.cat((sincos.sin(), sincos.cos()), dim=-1)
        return emb[None, :, None, :]

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps or 1e-5

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

class LlamaSdpaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config['model']['model_config']['hidden_size']
        self.num_heads = config['model']['model_config']['num_attention_heads']
        self.num_key_value_heads = config['model']['model_config'].get('num_key_value_heads', self.num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, config)

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim)
        v = v.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim)

        rotary_emb = self.rotary_emb(q, seq_length)
        q = self._apply_rotary_pos_emb(q, rotary_emb)
        k = self._apply_rotary_pos_emb(k, rotary_emb)

        q = q.transpose(1, 2)  # [batch_size, num_heads, seq_length, head_dim]
        k = k.transpose(1, 2)  # [batch_size, num_kv_heads, seq_length, head_dim]
        v = v.transpose(1, 2)  # [batch_size, num_kv_heads, seq_length, head_dim]

        # Repeat k,v heads if num_heads > num_key_value_heads
        if self.num_key_value_heads != self.num_heads:
            k = k.repeat_interleave(self.num_heads // self.num_key_value_heads, dim=1)
            v = v.repeat_interleave(self.num_heads // self.num_key_value_heads, dim=1)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_length, self.hidden_size)
        
        return self.o_proj(attn_output)

    def _apply_rotary_pos_emb(self, x, rope_embed):
        x_rope = x.float()
        cos = rope_embed[..., 1::2]
        sin = rope_embed[..., ::2]
        x_rope = torch.cat((x_rope[..., ::2] * cos - x_rope[..., 1::2] * sin,
                          x_rope[..., ::2] * sin + x_rope[..., 1::2] * cos), dim=-1)
        return x_rope.type_as(x)

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config['model']['model_config']['hidden_size']
        intermediate_size = config['model']['model_config']['intermediate_size']
        
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = LlamaSdpaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(
            config['model']['model_config']['hidden_size'],
            config['model']['model_config'].get('rms_norm_eps', 1e-5)
        )
        self.post_attention_layernorm = LlamaRMSNorm(
            config['model']['model_config']['hidden_size'],
            config['model']['model_config'].get('rms_norm_eps', 1e-5)
        )

    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

class LlamaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config['model']['model_config']
        
        self.embed_tokens = nn.Embedding(
            self.config['vocab_size'],
            self.config['hidden_size']
        )
        
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config) 
            for _ in range(self.config['num_hidden_layers'])
        ])
        
        self.norm = LlamaRMSNorm(
            self.config['hidden_size'],
            self.config.get('rms_norm_eps', 1e-5)
        )

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.embed_tokens(input_ids)
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
            
        hidden_states = self.norm(hidden_states)
        return hidden_states

class LlamaForCausalLM(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = load_config()
            
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(
            config['model']['model_config']['hidden_size'],
            config['model']['model_config']['vocab_size'],
            bias=False
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        std = self.config['model'].get('init_method', {}).get('std', 0.02)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.model(input_ids, attention_mask)
        logits = self.lm_head(hidden_states)
        return logits

    def generate(self, input_ids, max_length=50):
        self.eval()
        with torch.no_grad():
            for _ in range(max_length - input_ids.shape[1]):
                outputs = self(input_ids)
                next_token_logits = outputs[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1)
                input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
        return input_ids
