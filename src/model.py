"""Tiny character transformer designed to be copied exactly into spreadsheet formulas.

Every op here has a one-line spreadsheet equivalent:
  embedding lookup  -> INDEX(table, token+1, col)
  layer norm        -> AVERAGE, SUMPRODUCT((x-mu)^2)/d, SQRT
  q/k/v, proj, MLP  -> MMULT(row, W) + b   (legacy array formula)
  attention score   -> SUMPRODUCT(q_t, k_j) * scale        (only for j <= t: the causal mask)
  softmax           -> EXP(s - MAX(s)) / SUM(...)
  attention mix     -> MMULT(weights_row, V_block)
  relu              -> MAX(0, z)
  argmax            -> MATCH(MAX(logits), logits, 0) - 1
"""
import math
import torch
import torch.nn as nn

class Config:
    def __init__(self, vocab_size, block_size=16, n_layer=2, n_head=2, d_model=32, d_mlp=64, dropout=0.0):
        self.vocab_size, self.block_size = vocab_size, block_size
        self.n_layer, self.n_head, self.d_model, self.d_mlp = n_layer, n_head, d_model, d_mlp
        self.dropout = dropout  # training only; inactive in eval, so the sheet never sees it

    def to_dict(self):
        return dict(vocab_size=self.vocab_size, block_size=self.block_size, n_layer=self.n_layer,
                    n_head=self.n_head, d_model=self.d_model, d_mlp=self.d_mlp)

EPS = 1e-5

def layer_norm(x, g, b):
    # Written out by hand so the maths matches the sheet line for line (biased variance).
    mu = x.mean(-1, keepdim=True)
    var = ((x - mu) ** 2).mean(-1, keepdim=True)
    return (x - mu) / torch.sqrt(var + EPS) * g + b

class LN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d)); self.b = nn.Parameter(torch.zeros(d))
    def forward(self, x):
        return layer_norm(x, self.g, self.b)

class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        d = c.d_model
        self.c = c
        self.ln1 = LN(d)
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d); self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.ln2 = LN(d)
        self.fc1 = nn.Linear(d, c.d_mlp); self.fc2 = nn.Linear(c.d_mlp, d)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x, trace=None):
        B, T, D = x.shape
        H = self.c.n_head; hd = D // H
        h = self.ln1(x)
        q, k, v = self.q(h), self.k(h), self.v(h)
        q = q.view(B, T, H, hd).transpose(1, 2); k = k.view(B, T, H, hd).transpose(1, 2)
        v = v.view(B, T, H, hd).transpose(1, 2)
        s = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hd))
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        s_raw = s
        s = s.masked_fill(~mask, float('-inf'))
        s = s - s.max(-1, keepdim=True).values
        e = torch.exp(s)
        w = e / e.sum(-1, keepdim=True)
        a = (w @ v).transpose(1, 2).reshape(B, T, D)
        proj = self.o(a)
        x = x + self.drop(proj)
        mid = x
        h2 = self.ln2(x)
        z = self.fc1(h2)
        r = torch.relu(z)
        x = x + self.drop(self.fc2(r))
        if trace is not None:
            trace.append(dict(ln1=h, q=q, k=k, v=v, scores=s_raw, attn=w, mix=a, proj=proj, mid=mid,
                              ln2=h2, pre=z, relu=r, mlp=x - mid, out=x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.wte = nn.Embedding(c.vocab_size, c.d_model)
        self.wpe = nn.Embedding(c.block_size, c.d_model)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layer)])
        self.lnf = LN(c.d_model)
        self.head = nn.Linear(c.d_model, c.vocab_size)
        self.drop0 = nn.Dropout(c.dropout)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02 * 4); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.3)

    def forward(self, idx, trace=None):
        B, T = idx.shape
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        x = self.drop0(x)
        if trace is not None:
            trace.append(dict(x0=x))
        for blk in self.blocks:
            x = blk(x, trace)
        h = self.lnf(x)
        logits = self.head(h)
        if trace is not None:
            trace.append(dict(lnf=h, logits=logits))
        return logits

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
