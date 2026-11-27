import argparse, json, math, random, time, sys, os
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from model import Config, TinyGPT
from vocab import Vocab, greedy

PROMPTS = ['', 'sid', 'excel', 'pizza', 'taylor', 'elon', 'mom', 'abc', 'tiktok', 'chatgpt', 'banana',
           'linkedin', 'monday', 'coffee', 'zz', 'q', 'brooklyn', 'tofu', 'grandma', 'wifi', 'netflix',
           'spreadsheet', 'boston', 'x', 'kevin', 'lol', 'crypto', 'pickle', 'doge', 'yoga']

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--heads', type=int, default=2)
    ap.add_argument('--d', type=int, default=32)
    ap.add_argument('--mlp', type=int, default=64)
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--wd', type=float, default=0.05)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--maxlen', type=int, default=15)
    ap.add_argument('--dropout', type=float, default=0.0)
    ap.add_argument('--block', type=int, default=0, help='positions the model sees (default maxlen+1)')
    a = ap.parse_args()
    torch.set_num_threads(2); torch.manual_seed(a.seed); random.seed(a.seed)
    names = [w.strip() for w in open(a.corpus) if w.strip()]
    names = [w for w in names if len(w) <= a.maxlen]
    chars = sorted(set(''.join(names)))
    vocab = Vocab(chars)
    random.shuffle(names)
    nval = max(50, len(names) // 10)
    val, tr = names[:nval], names[nval:]
    T = a.block or (a.maxlen + 1)
    def enc(ws):
        X = torch.zeros(len(ws), T, dtype=torch.long); Y = torch.full((len(ws), T), -100, dtype=torch.long)
        for i, w in enumerate(ws):
            s = ([0] + [vocab.stoi[c] for c in w] + [0])[:T + 1]
            X[i, :len(s) - 1] = torch.tensor(s[:-1]); Y[i, :len(s) - 1] = torch.tensor(s[1:])
        return X, Y
    Xtr, Ytr = enc(tr); Xva, Yva = enc(val)
    c = Config(len(vocab), T, a.layers, a.heads, a.d, a.mlp, a.dropout)
    best = (1e9, None, -1)
    m = TinyGPT(c)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.98))
    t0 = time.time()
    for step in range(a.steps + 1):
        lr = a.lr * 0.5 * (1 + math.cos(math.pi * step / a.steps)) if step > 200 else a.lr * step / 200
        for g in opt.param_groups: g['lr'] = lr
        ix = torch.randint(0, len(Xtr), (a.bs,))
        loss = F.cross_entropy(m(Xtr[ix]).reshape(-1, c.vocab_size), Ytr[ix].reshape(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if step % 250 == 0:
            m.eval()
            with torch.no_grad():
                vl = F.cross_entropy(m(Xva).reshape(-1, c.vocab_size), Yva.reshape(-1), ignore_index=-100).item()
                tl = F.cross_entropy(m(Xtr[:2000]).reshape(-1, c.vocab_size), Ytr[:2000].reshape(-1), ignore_index=-100).item()
            m.train()
            if vl < best[0]:
                best = (vl, {k: v.clone() for k, v in m.state_dict().items()}, step)
            if step % 1000 == 0:
                print(f'step {step} train {tl:.3f} val {vl:.3f} ({time.time()-t0:.0f}s)', flush=True)
    m.load_state_dict(best[1]); vl = best[0]; m.eval()
    with torch.no_grad():
        tl = F.cross_entropy(m(Xtr[:2000]).reshape(-1, c.vocab_size), Ytr[:2000].reshape(-1), ignore_index=-100).item()
    print(f'best val {vl:.3f} at step {best[2]} (train {tl:.3f})')
    os.makedirs(a.out, exist_ok=True)
    torch.save(m.state_dict(), f'{a.out}/model.pt')
    json.dump(dict(config=c.to_dict(), best_step=best[2], chars=''.join(chars), val_loss=vl, train_loss=tl, args=vars(a),
                   n_params=m.n_params()), open(f'{a.out}/meta.json', 'w'), indent=1)
    md = m.double()
    def fn(toks):
        with torch.no_grad():
            return md(torch.tensor([toks]))[0]
    print('params', m.n_params(), 'vocab', len(vocab))
    trset = set(names)
    lines = []
    for p in PROMPTS:
        _, txt = greedy(fn, vocab, p, T)
        lines.append(f'{p!r:16} -> {txt}{"  (real)" if txt in trset else ""}')
    open(f'{a.out}/samples.txt', 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))

if __name__ == '__main__':
    main()
