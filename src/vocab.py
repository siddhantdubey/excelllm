"""Token 0 is '.', which marks both the start and the end of a name."""
import json, re

class Vocab:
    def __init__(self, chars):
        self.itos = ['.'] + list(chars)
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def clean_prompt(self, text, max_len):
        """Same rules as the sheet: lowercase, TRIM spaces, drop anything not in the vocab."""
        t = re.sub(' +', ' ', (text or '').lower()).strip(' ')   # Excel TRIM: spaces only
        return [self.stoi[c] for c in t if c in self.stoi and c != '.'][:max_len]

def greedy(model_fn, vocab, prompt, n_pos):
    """Unrolled exactly like the sheet. The sheet computes positions 0..n_pos-1 and shows
    tokens 1..n_pos. Position 0 is '.', positions 1..len(prompt) are the prompt, every later
    token is the argmax of the previous position's logits. Returns (tokens 0..n_pos, text)."""
    p = vocab.clean_prompt(prompt, n_pos)
    toks = [0] + p
    while len(toks) < n_pos + 1:
        logits = model_fn(toks)
        toks.append(int(logits[len(toks) - 1].argmax()))
    text = ''
    for t in toks[1:]:
        if t == 0:
            break
        text += vocab.itos[t]
    return toks, text
