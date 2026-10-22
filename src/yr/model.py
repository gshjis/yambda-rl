"""GPT-2 decoder over item-id tokens with an in-batch sampled softmax."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d: int, n_head: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_head, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d), nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + h
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab: int, n_layer: int = 4, n_head: int = 4, n_embd: int = 128,
                 ctx: int = 200, dropout: float = 0.1):
        super().__init__()
        self.vocab, self.ctx, self.n_embd = vocab, ctx, n_embd
        self.tok = nn.Embedding(vocab, n_embd)
        self.pos = nn.Embedding(ctx, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.config = dict(vocab=vocab, n_layer=n_layer, n_head=n_head, n_embd=n_embd, ctx=ctx, dropout=dropout)
        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith("attn.out_proj.weight") or name.endswith("mlp.2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.shape
        assert t <= self.ctx, f"sequence {t} > ctx {self.ctx}"
        pos = torch.arange(t, device=idx.device)
        x = self.drop(self.tok(idx) + self.pos(pos))
        # causal + padding mask (right padding)
        causal = torch.triu(torch.ones(t, t, device=idx.device, dtype=torch.bool), diagonal=1)
        for block in self.blocks:
            x = block(x, attn_mask=causal)
        return self.ln_f(x)


def sampled_softmax_loss(h: torch.Tensor, targets: torch.Tensor, item_emb: torch.Tensor,
                         mask: torch.Tensor | None = None, uniform_neg: int = 1024,
                         generator: torch.Generator | None = None) -> torch.Tensor:
    """In-batch + uniform negatives. h: (B,T,d), targets: (B,T), item_emb: (V,d)."""
    b, t, d = h.shape
    h = h.reshape(-1, d)
    targets = targets.reshape(-1)
    if mask is not None:
        valid = mask.reshape(-1)
        h, targets = h[valid], targets[valid]
    if h.numel() == 0:
        return h.sum() * 0.0
    uniq, inv = torch.unique(targets, return_inverse=True)
    pos_logits = h @ item_emb[uniq].T
    if uniform_neg > 0:
        neg = torch.randint(0, item_emb.shape[0], (uniform_neg,), device=h.device, generator=generator)
        neg_logits = h @ item_emb[neg].T
        logits = torch.cat([pos_logits, neg_logits], dim=1)
    else:
        logits = pos_logits
    return F.cross_entropy(logits, inv)


@torch.no_grad()
def rank_all(h: torch.Tensor, item_emb: torch.Tensor, topk: int = 100, chunk: int = 100_000):
    """Exact top-k over the full catalog for a batch of final hidden states."""
    best_s = torch.full((h.shape[0], topk), -torch.inf, device=h.device)
    best_i = torch.zeros((h.shape[0], topk), dtype=torch.long, device=h.device)
    for start in range(0, item_emb.shape[0], chunk):
        s = h @ item_emb[start : start + chunk].T
        k = min(topk, s.shape[1])
        cs, ci = s.topk(k, dim=1)
        all_s = torch.cat([best_s, cs], dim=1)
        all_i = torch.cat([best_i, ci + start], dim=1)
        best_s, sel = all_s.topk(topk, dim=1)
        best_i = torch.gather(all_i, 1, sel)
    return best_i
