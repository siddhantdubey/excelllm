"""Run the same model in PyTorch: python src/generate.py excel sidcodes chatgpt"""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(__file__))
from reference import load
from vocab import greedy

m, vocab, meta = load(os.path.join(os.path.dirname(__file__), '..', 'model'))
fn = lambda toks: m(torch.tensor([toks]))[0]
for prompt in sys.argv[1:] or ['excel']:
    with torch.no_grad():
        _, text = greedy(fn, vocab, prompt, m.c.block_size)
    print(f'{prompt!r:12} -> {text.capitalize()}')
