"""PyTorch reference in float64: greedy text plus every intermediate the sheet shows."""
import argparse, json, os, sys
import torch
sys.path.insert(0, os.path.dirname(__file__))
from model import Config, TinyGPT
from vocab import Vocab, greedy

def load(run):
    meta = json.load(open(f'{run}/meta.json'))
    c = Config(**meta['config'])
    m = TinyGPT(c)
    m.load_state_dict(torch.load(f'{run}/model.pt', map_location='cpu'))
    m.eval()
    return m.double(), Vocab(meta['chars']), meta

def trace_prompt(m, vocab, prompt):
    P = m.c.block_size          # the sheet computes every position the model has
    fn = lambda toks: m(torch.tensor([toks]))[0]
    with torch.no_grad():
        toks, text = greedy(fn, vocab, prompt, P)
        tr = []
        logits = m(torch.tensor([toks[:P]]), trace=tr)[0]
    out = {'prompt': prompt, 'tokens': toks, 'text': text,
           'n_prompt': len(vocab.clean_prompt(prompt, P))}
    T = {}
    T['x0'] = tr[0]['x0'][0]
    H = m.c.n_head
    for l in range(m.c.n_layer):
        b = tr[1 + l]; n = l + 1
        T[f'L{n}.ln1'] = b['ln1'][0]
        for k in 'qkv':
            T[f'L{n}.{k}'] = b[k][0].transpose(0, 1).reshape(P, -1)
        for h in range(H):
            s = b['scores'][0, h].clone(); w = b['attn'][0, h].clone()
            mask = torch.tril(torch.ones(P, P, dtype=torch.bool))
            s[~mask] = float('nan'); w[~mask] = float('nan')
            T[f'L{n}.scores.h{h}'] = s; T[f'L{n}.attn.h{h}'] = w
        for k in ('mix', 'proj', 'mid', 'ln2', 'pre', 'relu', 'mlp', 'out'):
            T[f'L{n}.{k}'] = b[k][0]
    T['lnf'] = tr[-1]['lnf'][0]
    T['logits'] = logits
    srt = logits.sort(-1, descending=True).values
    T['margin'] = (srt[:, 0] - srt[:, 1]).unsqueeze(1)
    T['argmax'] = logits.argmax(-1).double().unsqueeze(1)
    T['probs'] = torch.softmax(logits, -1)
    out['tensors'] = {k: [[None if v != v else float(v) for v in row] for row in t.tolist()] for k, t in T.items()}
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--prompts', required=True, help='json list')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    torch.set_num_threads(2)
    m, vocab, meta = load(a.run)
    prompts = json.load(open(a.prompts))
    res = [trace_prompt(m, vocab, p) for p in prompts]
    json.dump(res, open(a.out, 'w'))
    for r in res:
        print(f"{r['prompt']!r:24} -> {r['text']}")
