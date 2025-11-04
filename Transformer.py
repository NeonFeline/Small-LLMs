import torch
import torch.nn as nn
import math


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Cache for cos and sin
        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_cache(self, seq_len, device):
        if seq_len != self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()[None, :, None, :]
            self._sin_cached = emb.sin()[None, :, None, :]

    def forward(self, x):
        seq_len = x.shape[1]
        self._update_cos_sin_cache(seq_len, x.device)
        return self._cos_cached[:, :seq_len, :, :], self._sin_cached[:, :seq_len, :, :]


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embedding to queries and keys."""

    # Split into two halves for rotation
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    # Apply rotation
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadAttentionWithRoPE(nn.Module):
    def __init__(self, hidden_size, n_head, dropout=0.1, max_seq_len=2048):
        super().__init__()
        assert hidden_size % n_head == 0

        self.n_head = n_head
        self.head_dim = hidden_size // n_head
        self.hidden_size = hidden_size

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # Project and reshape
        q = self.q_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim)

        # Apply RoPE
        cos, sin = self.rope(x)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Transpose for attention: (batch, n_head, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Attend to values
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        out = self.o_proj(out)

        return out


class TransformerBlockWithRoPE(nn.Module):
    def __init__(self, hidden_size, n_head, dropout=0.1, max_seq_len=2048):
        super().__init__()
        self.ln1 = nn.RMSNorm(hidden_size)
        self.attn = MultiHeadAttentionWithRoPE(hidden_size, n_head, dropout, max_seq_len)
        self.ln2 = nn.RMSNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.SiLU(),
            nn.Linear(4 * hidden_size, hidden_size),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # Pre-norm architecture (like GPT-2, LLaMA)
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerDecoderWithRoPE(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, n_head=8, dropout=0.1, max_seq_len=2048):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        self.blocks = nn.ModuleList([
            TransformerBlockWithRoPE(hidden_size, n_head, dropout, max_seq_len)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.RMSNorm(hidden_size)
        self.output_proj = nn.Linear(hidden_size, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        batch_size, seq_len = x.shape

        # Create causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).view(1, 1, seq_len, seq_len)

        # Embedding (no positional encoding added!)
        x = self.embedding(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        # Output
        x = self.ln_f(x)
        logits = self.output_proj(x)

        return logits