import torch
import torch.nn as nn

class LlamaForCausalLM(nn.Module):
    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__()
        self.embed_tokens = nn.Embedding(49152, config['hidden_size'])
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(
                self_attn=LlamaSdpaAttention(
                    q_proj=nn.Linear(config['hidden_size'], config['hidden_size'], bias=False),
                    k_proj=nn.Linear(config['hidden_size'], config['hidden_size'], bias=False),
                    v_proj=nn.Linear(config['hidden_size'], config['hidden_size'], bias=False),
                    o_proj=nn.Linear(config['hidden_size'], config['hidden_size'], bias=False),
                    rotary_emb=LlamaRotaryEmbedding(dim=config['hidden_size'])
                ),
                mlp=LlamaMLP(
                    gate_proj=nn.Linear(config['hidden_size'], config['intermediate_size'], bias=False),
                    up_proj=nn.Linear(config['hidden_size'], config['intermediate_size'], bias=False),
                    down_proj=nn.Linear(config['intermediate_size'], config['hidden_size'], bias=False),
                    act_fn=nn.SiLU()
                ),
                input_layernorm=LlamaRMSNorm((config['hidden_size'],), eps=1e-05),
                post_attention_layernorm=LlamaRMSNorm((config['hidden_size'],), eps=1e-05)
            ) for _ in range(config['num_layers'])
        ])
        self.norm = LlamaRMSNorm((config['hidden_size'],), eps=1e-05)
        self.lm_head = nn.Linear(config['hidden_size'], 49152, bias=False)

    def forward(self, input_ids):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

class LlamaDecoderLayer(nn.Module):
    def __init__(self, self_attn, mlp, input_layernorm, post_attention_layernorm):
        super(LlamaDecoderLayer, self).__init__()
        self.self_attn = self_attn
        self.mlp = mlp
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm

    def forward(self, hidden_states):
        hidden_states = self.input_layernorm(hidden_states)
        attention_output = self.self_attn(hidden_states)
        hidden_states = hidden_states + attention_output
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = hidden_states + mlp_output
        return hidden_states

class LlamaSdpaAttention(nn.Module):
    def __init__(self, q_proj, k_proj, v_proj, o_proj, rotary_emb):
        super(LlamaSdpaAttention, self).__init__()
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.o_proj = o_proj
        self.rotary_emb = rotary_emb

    def forward(self, hidden_states):
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        # Apply rotary embedding to query and key
        query, key = self.rotary_emb(query, key)
        # Scaled dot-product attention
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / query.size(-1)**0.5
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        context_layer = torch.matmul(attention_probs, value)
        attention_output = self.o_proj(context_layer)
        return attention_output

class LlamaMLP(nn.Module):
    def __init__(self, gate_proj, up_proj, down_proj, act_fn):
        super(LlamaMLP, self).__init__()
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj
        self.act_fn = act_fn

    def forward(self, hidden_states):
        gate_output = self.gate_proj(hidden_states)
        up_output = self.up_proj(hidden_states)
        hidden_states = gate_output * self.act_fn(up_output)
        hidden_states = self.down_proj(hidden_states)
        return hidden_states

class LlamaRMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-05):
        super(LlamaRMSNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states

class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super(LlamaRotaryEmbedding, self).__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, query, key):
        seq_len = query.size(1)
        t = torch.arange(seq_len, device=query.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        cos_emb = emb.cos().expand_as(query)
        sin_emb = emb.sin().expand_as(query)

        query = (query * cos_emb) + (self.rotate_half(query) * sin_emb)
        key = (key * cos_emb) + (self.rotate_half(key) * sin_emb)
        return query, key

    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
