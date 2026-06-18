import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout, bias):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, channels = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(batch_size, seq_len, self.n_head, channels // self.n_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, channels // self.n_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, channels // self.n_head).transpose(1, 2)
        if hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))
            mask = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool).tril()
            att = att.masked_fill(~mask, float("-inf"))
            att = self.attn_dropout(F.softmax(att, dim=-1))
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, n_embd, dropout, bias, act_layer):
        super().__init__()
        hidden_dim = 4 * n_embd
        self.c_fc = nn.Linear(n_embd, hidden_dim, bias=bias)
        self.act = act_layer(hidden_dim)
        self.c_proj = nn.Linear(hidden_dim, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.act(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout, bias, attn_norm_layer, mlp_norm_layer, act_layer):
        super().__init__()
        self.ln_1 = attn_norm_layer(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, bias)
        self.ln_2 = mlp_norm_layer(n_embd)
        self.mlp = MLP(n_embd, dropout, bias, act_layer)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size,
        n_layer,
        n_head,
        n_embd,
        dropout=0.0,
        bias=False,
        norm_layer=nn.LayerNorm,
        attn_norm_layer=None,
        mlp_norm_layer=None,
        final_norm_layer=None,
        act_layer=nn.GELU,
    ):
        super().__init__()
        self.block_size = block_size
        attn_norm_layer = attn_norm_layer or norm_layer
        mlp_norm_layer = mlp_norm_layer or norm_layer
        final_norm_layer = final_norm_layer or norm_layer
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(vocab_size, n_embd),
                wpe=nn.Embedding(block_size, n_embd),
                drop=nn.Dropout(dropout),
                h=nn.ModuleList(
                    [
                        Block(n_embd, n_head, dropout, bias, attn_norm_layer, mlp_norm_layer, act_layer)
                        for _ in range(n_layer)
                    ]
                ),
                ln_f=final_norm_layer(n_embd),
            )
        )
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self):
        return sum(param.numel() for param in self.parameters())

    def forward(self, idx, targets=None):
        batch_size, seq_len = idx.size()
        if seq_len > self.block_size:
            raise ValueError(f"Input sequence length {seq_len} exceeds block_size {self.block_size}.")
        pos = torch.arange(0, seq_len, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        logits = self.lm_head(self.transformer.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(float(temperature), 1e-5)
            if top_k is not None:
                values, _ = torch.topk(logits, min(int(top_k), logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
