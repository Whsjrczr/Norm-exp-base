import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenPositionEmbedding(nn.Module):
    def __init__(self, vocab_size, max_len, n_embd, pad_token_id, dropout):
        super().__init__()
        self.token = nn.Embedding(vocab_size, n_embd, padding_idx=pad_token_id)
        self.position = nn.Embedding(max_len, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens):
        seq_len = tokens.size(1)
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0)
        return self.dropout(self.token(tokens) + self.position(positions))


class FeedForward(nn.Module):
    def __init__(self, n_embd, hidden_dim, dropout, act_layer):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, hidden_dim),
            act_layer(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, n_embd, n_head, hidden_dim, dropout, norm_layer, act_layer):
        super().__init__()
        self.norm_1 = norm_layer(n_embd)
        self.self_attn = nn.MultiheadAttention(n_embd, n_head, dropout=dropout, batch_first=True)
        self.norm_2 = norm_layer(n_embd)
        self.ffn = FeedForward(n_embd, hidden_dim, dropout, act_layer)

    def forward(self, x, key_padding_mask=None):
        normed = self.norm_1(x)
        attn, _ = self.self_attn(normed, normed, normed, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + attn
        x = x + self.ffn(self.norm_2(x))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, n_embd, n_head, hidden_dim, dropout, norm_layer, act_layer):
        super().__init__()
        self.norm_1 = norm_layer(n_embd)
        self.self_attn = nn.MultiheadAttention(n_embd, n_head, dropout=dropout, batch_first=True)
        self.norm_2 = norm_layer(n_embd)
        self.cross_attn = nn.MultiheadAttention(n_embd, n_head, dropout=dropout, batch_first=True)
        self.norm_3 = norm_layer(n_embd)
        self.ffn = FeedForward(n_embd, hidden_dim, dropout, act_layer)

    def forward(self, x, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        normed = self.norm_1(x)
        attn, _ = self.self_attn(
            normed,
            normed,
            normed,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
        )
        x = x + attn
        normed = self.norm_2(x)
        attn, _ = self.cross_attn(
            normed,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        x = x + attn
        x = x + self.ffn(self.norm_3(x))
        return x


class BertTranslationModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_src_len,
        max_tgt_len,
        pad_token_id,
        bos_token_id,
        eos_token_id,
        n_layer=4,
        n_head=4,
        n_embd=256,
        ffn_mult=4,
        dropout=0.1,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
    ):
        super().__init__()
        self.max_src_len = int(max_src_len)
        self.max_tgt_len = int(max_tgt_len)
        self.pad_token_id = int(pad_token_id)
        self.bos_token_id = int(bos_token_id)
        self.eos_token_id = int(eos_token_id)
        hidden_dim = int(ffn_mult * n_embd)
        max_len = max(self.max_src_len, self.max_tgt_len + 1)
        self.src_embedding = TokenPositionEmbedding(vocab_size, max_len, n_embd, pad_token_id, dropout)
        self.tgt_embedding = TokenPositionEmbedding(vocab_size, max_len, n_embd, pad_token_id, dropout)
        self.encoder = nn.ModuleList(
            [EncoderLayer(n_embd, n_head, hidden_dim, dropout, norm_layer, act_layer) for _ in range(n_layer)]
        )
        self.decoder = nn.ModuleList(
            [DecoderLayer(n_embd, n_head, hidden_dim, dropout, norm_layer, act_layer) for _ in range(n_layer)]
        )
        self.final_norm = norm_layer(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.tgt_embedding.token.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self):
        return sum(param.numel() for param in self.parameters())

    def _causal_mask(self, seq_len, device):
        return torch.ones(seq_len, seq_len, device=device, dtype=torch.bool).triu(1)

    def encode(self, src):
        if src.size(1) > self.max_src_len:
            raise ValueError(f"Source length {src.size(1)} exceeds max_src_len {self.max_src_len}.")
        src_padding = src.eq(self.pad_token_id)
        x = self.src_embedding(src) * math.sqrt(self.src_embedding.token.embedding_dim)
        for layer in self.encoder:
            x = layer(x, key_padding_mask=src_padding)
        return x, src_padding

    def decode(self, tgt_input, memory, src_padding):
        if tgt_input.size(1) > self.max_tgt_len:
            raise ValueError(f"Target length {tgt_input.size(1)} exceeds max_tgt_len {self.max_tgt_len}.")
        tgt_padding = tgt_input.eq(self.pad_token_id)
        tgt_mask = self._causal_mask(tgt_input.size(1), tgt_input.device)
        x = self.tgt_embedding(tgt_input) * math.sqrt(self.tgt_embedding.token.embedding_dim)
        for layer in self.decoder:
            x = layer(
                x,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_padding,
                memory_key_padding_mask=src_padding,
            )
        return self.lm_head(self.final_norm(x))

    def forward(self, src, tgt_input, tgt_labels=None):
        memory, src_padding = self.encode(src)
        logits = self.decode(tgt_input, memory, src_padding)
        loss = None
        if tgt_labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt_labels.reshape(-1),
                ignore_index=self.pad_token_id,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, src, max_new_tokens=None):
        was_1d = src.dim() == 1
        if was_1d:
            src = src.unsqueeze(0)
        if src.size(1) < self.max_src_len:
            pad_len = self.max_src_len - src.size(1)
            padding = torch.full(
                (src.size(0), pad_len),
                self.pad_token_id,
                dtype=torch.long,
                device=src.device,
            )
            src = torch.cat([src, padding], dim=1)
        memory, src_padding = self.encode(src)
        max_new_tokens = int(max_new_tokens or self.max_tgt_len)
        generated = torch.full(
            (src.size(0), 1),
            self.bos_token_id,
            dtype=torch.long,
            device=src.device,
        )
        done = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
        for _ in range(max_new_tokens):
            decode_input = generated
            logit_pos = generated.size(1) - 1
            if generated.size(1) < self.max_tgt_len:
                pad_len = self.max_tgt_len - generated.size(1)
                padding = torch.full(
                    (generated.size(0), pad_len),
                    self.pad_token_id,
                    dtype=torch.long,
                    device=generated.device,
                )
                decode_input = torch.cat([generated, padding], dim=1)
            logits = self.decode(decode_input, memory, src_padding)
            next_token = logits[:, logit_pos].argmax(dim=-1, keepdim=True)
            next_token = torch.where(done.unsqueeze(1), torch.full_like(next_token, self.pad_token_id), next_token)
            generated = torch.cat([generated, next_token], dim=1)
            done = done | next_token.squeeze(1).eq(self.eos_token_id)
            if bool(done.all()):
                break
        return generated[0] if was_1d else generated
