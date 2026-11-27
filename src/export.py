"""Write the trained model into an .xlsx where every activation is a spreadsheet formula.

Layout rule: one row per position (0..P-1), one column per feature. Position t only reads
rows <= t, so the whole forward pass is unrolled by position and the token at position t is
either the typed letter or the argmax of row t-1 of the logits. No VBA, no iteration.

Also writes cellmap.json: where every tensor lives, so the verifiers can read it back.
"""
import argparse, json, math, os, sys
import torch
from openpyxl import Workbook
from openpyxl.formatting.rule import ColorScaleRule, FormulaRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter as L
from openpyxl.workbook.defined_name import DefinedName
from openpyxl.worksheet.formula import ArrayFormula

sys.path.insert(0, os.path.dirname(__file__))
from model import EPS

C0 = 3            # first data column (C); column B holds live row labels
RAW = 40          # how many characters of A1 are scanned
FONT = 'Calibri'

# --- styles ---------------------------------------------------------------
INK, GREY, FAINT = '1F2328', '6E7781', 'AFB8C1'
F_DATA = Font(name=FONT, size=8, color='3B4148')
F_HDR = Font(name=FONT, size=7, color=FAINT)
F_LABEL = Font(name=FONT, size=8, color=GREY)
F_BLOCK = Font(name=FONT, size=10, bold=True, color=INK)
F_NOTE = Font(name=FONT, size=9, italic=True, color=GREY)
F_TITLE = Font(name=FONT, size=18, bold=True, color='FFFFFF')
F_SUB = Font(name=FONT, size=10, color='D0D7DE')
F_SECTION = Font(name=FONT, size=11, bold=True, color='FFFFFF')
FILL_TITLE = PatternFill('solid', fgColor='161B22')
FILL_ACT = PatternFill('solid', fgColor='2F81F7')     # activations section bar
FILL_PAR = PatternFill('solid', fgColor='8250DF')     # parameters section bar
FILL_MASK = PatternFill('solid', fgColor='F0F2F4')
NUM = '0.00;-0.00;0.00'
CENTER = Alignment(horizontal='center', vertical='center')

def diverging(ws, ref):
    ws.conditional_formatting.add(ref, ColorScaleRule(
        start_type='min', start_color='5B8DEF', mid_type='num', mid_value=0, mid_color='FFFFFF',
        end_type='max', end_color='EF6B5B'))

def warm(ws, ref, color='F59E0B'):
    ws.conditional_formatting.add(ref, ColorScaleRule(
        start_type='num', start_value=0, start_color='FFFFFF', end_type='num', end_value=1, end_color=color))

def A(r, c, absr=False, absc=False):
    return f"{'$' if absc else ''}{L(c)}{'$' if absr else ''}{r}"

def rng(r1, c1, r2, c2, absr=False, absc=False):
    return f'{A(r1, c1, absr, absc)}:{A(r2, c2, absr, absc)}'

def q(sheet):
    return f"'{sheet}'" if (' ' in sheet or '-' in sheet) else sheet


class Book:
    def __init__(self, sd, meta):
        self.sd = {k: v.double() for k, v in sd.items()}
        self.meta = meta
        c = meta['config']
        self.V, self.D, self.H = c['vocab_size'], c['d_model'], c['n_head']
        self.M, self.NL = c['d_mlp'], c['n_layer']
        self.P = c['block_size']          # positions computed in the sheet (0..P-1); shows tokens 1..P
        self.hd = self.D // self.H
        self.chars = '.' + meta['chars']
        self.wb = Workbook()
        self.map = {'P': self.P, 'tensors': {}, 'names': {}}
        self.param_names = []

    # -- helpers -------------------------------------------------------------
    def sheet(self, title, color):
        ws = self.wb.create_sheet(title)
        ws.sheet_properties.tabColor = color
        ws.sheet_view.showGridLines = False
        ws.sheet_view.zoomScale = 100
        ws.column_dimensions['A'].width = 2
        ws.column_dimensions['B'].width = 9
        return ws

    def banner(self, ws, title, sub, width):
        for r, h in ((1, 30), (2, 18)):
            ws.row_dimensions[r].height = h
            for c in range(1, width + 1):
                ws.cell(r, c).fill = FILL_TITLE
        ws.cell(1, 2, title).font = F_TITLE
        ws.cell(2, 2, sub).font = F_SUB
        return 4

    def section(self, ws, r, text, fill, width):
        for c in range(2, width + 1):
            ws.cell(r, c).fill = fill
        ws.cell(r, 2, text).font = F_SECTION
        return r + 2

    def name(self, nm, sheet, r1, c1, r2, c2):
        ref = f"{q(sheet)}!{rng(r1, c1, r2, c2, True, True)}"
        self.wb.defined_names[nm] = DefinedName(nm, attr_text=ref)
        self.map['names'][nm] = [sheet, r1, c1, r2, c2]

    def record(self, key, sheet, r, c, n):
        self.map['tensors'][key] = [sheet, r, c, n]

    def labels(self, ws, r0):
        """Column B: live label for each position, e.g. '3  r'."""
        for p in range(self.P):
            cell = ws.cell(r0 + p, 2, f"=Tokens!$I${self.tok_row + p}")
            cell.font = F_LABEL
            cell.alignment = Alignment(horizontal='left')

    def header(self, ws, r, n, title, note=None, names=None, c0=C0):
        ws.cell(r, 2, title).font = F_BLOCK
        if note:
            ws.cell(r, c0 + 6 if n > 12 else c0 + n + 1, note).font = F_NOTE
        for j in range(n):
            cell = ws.cell(r + 1, c0 + j, names[j] if names else j + 1)
            cell.font = F_HDR
            cell.alignment = CENTER
        return r + 2

    def weights(self, ws, r, mat, title, nm, note=None, row_names=None, col_names=None):
        """Write a parameter matrix (rows x cols) as constants and give it a defined name."""
        mat = mat.reshape(mat.shape[0], -1) if mat.dim() > 1 else mat.reshape(1, -1)
        nr, nc = mat.shape
        r = self.header(ws, r, nc, title, note, col_names)
        for i in range(nr):
            if row_names:
                ws.cell(r + i, 2, row_names[i]).font = F_LABEL
            for j in range(nc):
                cell = ws.cell(r + i, C0 + j, float(mat[i, j]))
                cell.number_format = NUM
                cell.font = F_DATA
        diverging(ws, rng(r, C0, r + nr - 1, C0 + nc - 1))
        self.name(nm, ws.title, r, C0, r + nr - 1, C0 + nc - 1)
        self.param_names.append(nm)
        return r + nr + 1

    def block(self, ws, r, n, title, note, fn, key=None, color='div', names=None, array=False):
        """A P x n block of formulas. fn(p) returns either a list of n formulas or (array formula)."""
        r = self.header(ws, r, n, title, note, names)
        self.labels(ws, r)
        for p in range(self.P):
            rr = r + p
            if array:
                f = fn(p, rr)
                ws.cell(rr, C0).value = ArrayFormula(rng(rr, C0, rr, C0 + n - 1), f)
            else:
                for j, f in enumerate(fn(p, rr)):
                    ws.cell(rr, C0 + j).value = f
            for j in range(n):
                cell = ws.cell(rr, C0 + j)
                cell.number_format = NUM
                cell.font = F_DATA
        ref = rng(r, C0, r + self.P - 1, C0 + n - 1)
        if color == 'div':
            diverging(ws, ref)
        elif color == 'pos':
            ws.conditional_formatting.add(ref, ColorScaleRule(
                start_type='num', start_value=0, start_color='FFFFFF', end_type='max', end_color='EF6B5B'))
        if key:
            self.record(key, ws.title, r, C0, n)
        return r, r + self.P + 1

    def widths(self, ws, ncols, w=5.6):
        for c in range(C0, C0 + ncols + 4):
            ws.column_dimensions[L(c)].width = w

    # -- sheets ---------------------------------------------------------------
    def build(self, default_prompt):
        wb = self.wb
        front = wb.active
        front.title = 'Prompt'
        self.tok_row = 9                       # first row of the per-position token table (Tokens sheet)
        self.sheet_tokens()
        self.sheet_embeddings()
        x_ref = ('Embeddings', self.map['tensors']['x0'])
        for l in range(self.NL):
            x_ref = self.sheet_attention(l, x_ref)
            x_ref = self.sheet_mlp(l, x_ref)
        self.sheet_logits(x_ref)
        self.sheet_front(front, default_prompt)
        self.sheet_about()
        wb.calculation.fullCalcOnLoad = True
        wb.calculation.calcMode = 'auto'
        wb.active = 0
        return wb

    def sheet_tokens(self):
        ws = self.sheet('Tokens', '57606A')
        V, P = self.V, self.P
        self.banner(ws, 'Tokens', 'Letters in, numbers out. Typed letters fill the first rows; '
                    'after that each letter is the model\'s own previous answer.', 20)
        ws.column_dimensions['B'].width = 16
        for c, w in zip('CDEFGHIJKLMNOPQRS', (9, 9, 8, 8, 9, 9, 9, 4, 5, 6, 6, 6, 6, 4, 5, 5, 6)):
            ws.column_dimensions[c].width = w
        t0 = self.tok_row          # per-position table B..I, rows t0..t0+P
        r0 = t0                    # scan of A1 in K..O, rows r0..r0+RAW-1
        # cleaned prompt
        ws.cell(4, 2, 'Cleaned prompt').font = F_BLOCK
        ws['C4'] = '=LOWER(TRIM(Prompt!A1))'
        ws['C4'].font = Font(name=FONT, size=11, bold=True, color='2F81F7')
        ws.cell(5, 2, 'letters used').font = F_LABEL
        # vocab table at Q..S
        ws.cell(t0 - 2, 17, 'Vocabulary').font = F_BLOCK
        for j, h in enumerate(('id', 'char', 'code')):
            ws.cell(t0 - 1, 17 + j, h).font = F_HDR
        for i, ch in enumerate(self.chars):
            ws.cell(t0 + i, 17, i).font = F_DATA
            ws.cell(t0 + i, 18, ch).font = F_DATA
            ws.cell(t0 + i, 19, 0 if i == 0 else ord(ch)).font = F_DATA
        self.name('VocabChar', 'Tokens', t0, 18, t0 + V - 1, 18)
        self.name('VocabCode', 'Tokens', t0 + 1, 19, t0 + V - 1, 19)   # ids 1..V-1 ('.' cannot be typed)
        # scan of A1, one character per row
        ws.cell(r0 - 2, 11, 'Scan of A1').font = F_BLOCK
        for j, h in enumerate(('#', 'char', 'code', 'token', 'kept')):
            ws.cell(r0 - 1, 11 + j, h).font = F_HDR
        for i in range(RAW):
            r = r0 + i
            ws.cell(r, 11, i + 1).font = F_LABEL
            ws.cell(r, 12, f'=MID($C$4,{i + 1},1)')
            ws.cell(r, 13, f'=IF(L{r}="",0,CODE(L{r}))')
            ws.cell(r, 14, f'=IFERROR(MATCH(M{r},VocabCode,0),0)')
            # IF(...,1,0), not (N>0): Excel's MATCH does not treat TRUE as 1 (LibreOffice does)
            ws.cell(r, 15, f'=IF(N{r}>0,1,0)' if i == 0 else f'=O{r - 1}+IF(N{r}>0,1,0)')
            for c in range(12, 16):
                ws.cell(r, c).font = F_DATA
                ws.cell(r, c).alignment = CENTER
        rl = r0 + RAW - 1
        ws['C5'] = f'=MIN({P},O{rl})'
        ws['C5'].font = F_DATA
        ws['C5'].alignment = Alignment(horizontal='left')
        # per-position table
        ws.cell(t0 - 2, 2, 'Sequence').font = F_BLOCK
        ws.cell(t0 - 2, 3, '0 is "." (start). A predicted "." ends the name.').font = F_NOTE
        for j, h in enumerate(('typed', 'predicted', 'token', 'char', 'finished', 'source', 'label')):
            ws.cell(t0 - 1, 3 + j, h).font = F_HDR
        for p in range(P + 1):
            r = t0 + p
            ws.cell(r, 2, f'position {p}').font = F_LABEL
            if p == 0:
                ws.cell(r, 3, '')
                ws.cell(r, 4, '')
                ws.cell(r, 5, 0)
                ws.cell(r, 7, False)
                ws.cell(r, 8, 'start')
            else:
                ws.cell(r, 3, f'=IF({p}<=$C$5,INDEX($N${r0}:$N${rl},MATCH({p},$O${r0}:$O${rl},0)),"")')
                ws.cell(r, 4, f'=Logits!{A(self.argmax_row(p - 1), self.argmax_col(), absc=True)}')
                ws.cell(r, 5, f'=IF({p}<=$C$5,C{r},D{r})')
                ws.cell(r, 7, f'=OR(G{r - 1},E{r}=0)')
                ws.cell(r, 8, f'=IF(G{r - 1},"",IF({p}<=$C$5,"you",IF(E{r}=0,"end","model")))')
            ws.cell(r, 6, f'=INDEX(VocabChar,E{r}+1)')
            ws.cell(r, 9, f'={p}&"  "&F{r}')
            for c in range(3, 10):
                ws.cell(r, c).font = F_DATA
                ws.cell(r, c).alignment = CENTER
            ws.cell(r, 6).font = Font(name='Consolas', size=12, bold=True, color=INK)
        ws.conditional_formatting.add(rng(t0, 6, t0 + P, 6), FormulaRule(
            formula=[f'$H{t0}="model"'], font=Font(color='1A7F37', bold=True)))
        self.record('tokens', 'Tokens', t0, 5, 1)
        ws.freeze_panes = 'A3'

    def argmax_row(self, p):
        return self.logit_r0 + p

    def argmax_col(self):
        return C0 + self.V + 1

    def sheet_embeddings(self):
        ws = self.sheet('Embeddings', '2DA44E')
        D, V, P = self.D, self.V, self.P
        self.widths(ws, D)
        r = self.banner(ws, 'Embeddings', 'Each letter becomes 32 numbers, plus 32 more for where it sits. '
                        'INDEX looks them up.', C0 + D)
        r = self.section(ws, r, 'ACTIVATIONS  ·  formulas, change with the prompt', FILL_ACT, C0 + D)
        # x0 block: token embedding + position embedding
        self.emb_r = None
        start = r + 2
        def fn(p, rr):
            return [f"=INDEX(TokenEmbedding,Tokens!$E${self.tok_row + p}+1,{j + 1})"
                    f"+INDEX(PositionEmbedding,{p + 1},{j + 1})" for j in range(D)]
        r0, r = self.block(ws, r, D, 'x = token embedding + position embedding',
                           None, fn, key='x0')
        r += 1
        r = self.section(ws, r, 'PARAMETERS  ·  learned in PyTorch, plain numbers', FILL_PAR, C0 + D)
        r = self.weights(ws, r, self.sd['wte.weight'], 'Token embedding  (one row per letter)',
                         'TokenEmbedding', row_names=[c if c != '.' else '. (start/end)' for c in self.chars])
        r = self.weights(ws, r, self.sd['wpe.weight'][:P], 'Position embedding  (one row per position)',
                         'PositionEmbedding', row_names=[f'pos {i}' for i in range(P)])
        ws.freeze_panes = 'C3'

    def ln_block(self, ws, r, x_ref, prefix, title, gname, bname, key):
        """mean, 1/sd and the normalised block. x_ref = (sheet, [sheet, r, c, n])."""
        D, P = self.D, self.P
        xs, (_, xr, xc, _) = x_ref[0], x_ref[1]
        mc, sc = C0 + D + 1, C0 + D + 2
        X = lambda p, absc=False: f"{q(xs)}!{rng(xr + p, xc, xr + p, xc + D - 1, absc=True)}"
        def fn(p, rr):
            return [f"=({q(xs)}!{A(xr + p, xc + j)}-${L(mc)}{rr})*${L(sc)}{rr}*INDEX({gname},1,{j + 1})"
                    f"+INDEX({bname},1,{j + 1})" for j in range(D)]
        r0, rn = self.block(ws, r, D, title, None, fn, key=key)
        ws.cell(r0 - 1, mc, 'mean').font = F_HDR
        ws.cell(r0 - 1, sc, '1/sd').font = F_HDR
        for p in range(P):
            rr = r0 + p
            ws.cell(rr, mc, f"=AVERAGE({X(p)})")
            ws.cell(rr, sc, f"=1/SQRT(SUMPRODUCT(({X(p)}-{L(mc)}{rr})^2)/{D}+0.00001)")
            for c in (mc, sc):
                ws.cell(rr, c).number_format = NUM
                ws.cell(rr, c).font = F_LABEL
        return r0, rn

    def sheet_attention(self, l, x_ref):
        D, P, H, hd = self.D, self.P, self.H, self.hd
        n = l + 1
        title = f'Layer {n} Attention'
        ws = self.sheet(title, 'BF8700')
        self.widths(ws, D)
        pre = f'Layer{n}_'
        sd = lambda k: self.sd[f'blocks.{l}.{k}']
        r = self.banner(ws, f'Layer {n} · Attention', 'Each letter asks a question (query), every earlier letter '
                        'offers an answer (key), and the best matches pass their values forward.', C0 + D + 2)
        r = self.section(ws, r, 'ACTIVATIONS  ·  formulas, change with the prompt', FILL_ACT, C0 + D + 2)
        xs, (_, xr, xc, _) = x_ref
        # residual stream in (a copy so the sheet reads top to bottom)
        def fx(p, rr):
            return [f"={q(xs)}!{A(xr + p, xc + j)}" for j in range(D)]
        xin, r = self.block(ws, r, D, 'Residual stream in', 'from the previous tab', fx)
        x_here = (title, [title, xin, C0, D])
        ln, r = self.ln_block(ws, r, x_here, pre, 'LayerNorm', pre + 'ln1_gain', pre + 'ln1_bias', f'L{n}.ln1')
        LN = lambda p: rng(ln + p, C0, ln + p, C0 + D - 1)
        rows = {}
        for k, nm, desc in (('q', 'Wq', 'Queries  Q = MMULT(LayerNorm, Wq) + bq'),
                            ('k', 'Wk', 'Keys  K = MMULT(LayerNorm, Wk) + bk'),
                            ('v', 'Wv', 'Values  V = MMULT(LayerNorm, Wv) + bv')):
            hdrs = [f'h{h + 1}·{i + 1}' for h in range(H) for i in range(hd)]
            def fa(p, rr, nm=nm):
                return f"=MMULT({LN(p)},{pre}{nm})+{pre}b{nm[1]}"
            rows[k], r = self.block(ws, r, D, desc, None, fa, key=f'L{n}.{k}', names=hdrs, array=True)
        # scores and weights: P x P triangles per head, side by side
        scale = 1 / math.sqrt(hd)
        scale_s = f'{scale:g}' if scale == 0.25 else f'(1/SQRT({hd}))'
        tri_w = P + 1
        lettr = [f'=Tokens!$F${self.tok_row + j}' for j in range(P)]
        for kind in ('scores', 'attn'):
            ttl = ('Attention scores  =SUMPRODUCT(query, key) × ' + scale_s +
                   '    (blank = the future, masked)') if kind == 'scores' else \
                  'Attention weights  softmax: EXP(score − max) / sum of EXP    (each row adds to 1)'
            ws.cell(r, 2, ttl).font = F_BLOCK
            for h in range(H):
                c0 = C0 + h * tri_w
                ws.cell(r + 1, c0, f'head {h + 1}').font = Font(name=FONT, size=8, bold=True, color=GREY)
                for j in range(P):
                    cell = ws.cell(r + 2, c0 + j, lettr[j])
                    cell.font = F_HDR
                    cell.alignment = CENTER
            r0 = r + 3
            self.labels(ws, r0)
            for h in range(H):
                c0 = C0 + h * tri_w
                qc = C0 + h * hd
                for p in range(P):
                    rr = r0 + p
                    for j in range(P):
                        cell = ws.cell(rr, c0 + j)
                        if j > p:
                            cell.fill = FILL_MASK
                            continue
                        if kind == 'scores':
                            qr = rng(rows['q'] + p, qc, rows['q'] + p, qc + hd - 1, absc=True)
                            kr = rng(rows['k'] + j, qc, rows['k'] + j, qc + hd - 1, absc=True)
                            cell.value = f"=SUMPRODUCT({qr},{kr})*{scale_s}"
                        else:
                            sr = rows['scores'] + p
                            srow = f"{A(sr, c0, absc=True)}:{A(sr, c0 + p, absc=True)}"
                            cell.value = (f"=EXP({A(sr, c0 + j)}-MAX({srow}))"
                                          f"/SUMPRODUCT(EXP({srow}-MAX({srow})))")
                        cell.number_format = NUM
                        cell.font = F_DATA
                ref = rng(r0, c0, r0 + P - 1, c0 + P - 1)
                if kind == 'attn':
                    warm(ws, ref)
                else:
                    diverging(ws, ref)
                self.record(f'L{n}.{kind}.h{h}', title, r0, c0, P)
            rows[kind] = r0
            r = r0 + P + 1
        # mix: per head MMULT(weights row, V block)
        ws.cell(r, 2, 'Mix  =MMULT(attention weights, V)    (what each letter takes from the earlier ones)').font = F_BLOCK
        for j in range(D):
            cell = ws.cell(r + 1, C0 + j, f'h{j // hd + 1}·{j % hd + 1}')
            cell.font = F_HDR
            cell.alignment = CENTER
        r0 = r + 2
        self.labels(ws, r0)
        for p in range(P):
            rr = r0 + p
            for h in range(H):
                c0 = C0 + h * tri_w
                w = rng(rows['attn'] + p, c0, rows['attn'] + p, c0 + p, absc=True)
                vb = rng(rows['v'], C0 + h * hd, rows['v'] + p, C0 + h * hd + hd - 1, absr=True, absc=True)
                ws.cell(rr, C0 + h * hd).value = ArrayFormula(
                    rng(rr, C0 + h * hd, rr, C0 + h * hd + hd - 1), f"=MMULT({w},{vb})")
            for j in range(D):
                ws.cell(rr, C0 + j).number_format = NUM
                ws.cell(rr, C0 + j).font = F_DATA
        diverging(ws, rng(r0, C0, r0 + P - 1, C0 + D - 1))
        self.record(f'L{n}.mix', title, r0, C0, D)
        rows['mix'] = r0
        r = r0 + P + 1
        MIX = lambda p: rng(rows['mix'] + p, C0, rows['mix'] + p, C0 + D - 1)
        rows['proj'], r = self.block(ws, r, D, 'Output  =MMULT(Mix, Wo) + bo', None,
                                     lambda p, rr: f"=MMULT({MIX(p)},{pre}Wo)+{pre}bo",
                                     key=f'L{n}.proj', array=True)
        rows['mid'], r = self.block(ws, r, D, 'Residual stream out  = in + Output', 'goes to the MLP tab',
                                    lambda p, rr: [f"={A(xin + p, C0 + j)}+{A(rows['proj'] + p, C0 + j)}"
                                                   for j in range(D)], key=f'L{n}.mid')
        r += 1
        r = self.section(ws, r, 'PARAMETERS  ·  learned in PyTorch, plain numbers', FILL_PAR, C0 + D + 2)
        r = self.weights(ws, r, sd('ln1.g'), 'LayerNorm gain', pre + 'ln1_gain')
        r = self.weights(ws, r, sd('ln1.b'), 'LayerNorm bias', pre + 'ln1_bias')
        for k, nm in (('q', 'Wq'), ('k', 'Wk'), ('v', 'Wv'), ('o', 'Wo')):
            r = self.weights(ws, r, sd(f'{k}.weight').T, f'{nm}  (32 × 32)', pre + nm)
            r = self.weights(ws, r, sd(f'{k}.bias'), f'b{nm[1]}', pre + f'b{nm[1]}')
        ws.freeze_panes = 'C3'
        return (title, [title, rows['mid'], C0, D])

    def sheet_mlp(self, l, x_ref):
        D, P, M = self.D, self.P, self.M
        n = l + 1
        title = f'Layer {n} MLP'
        ws = self.sheet(title, 'CF222E')
        self.widths(ws, M)
        pre = f'Layer{n}_'
        sd = lambda k: self.sd[f'blocks.{l}.{k}']
        r = self.banner(ws, f'Layer {n} · MLP', f'Each letter on its own: {M} feature detectors, '
                        'ReLU keeps the positive ones, then back to 32 numbers.', C0 + M)
        r = self.section(ws, r, 'ACTIVATIONS  ·  formulas, change with the prompt', FILL_ACT, C0 + M)
        xs, (_, xr, xc, _) = x_ref
        ln, r = self.ln_block(ws, r, x_ref, pre, 'LayerNorm  (reads the attention tab)',
                              pre + 'ln2_gain', pre + 'ln2_bias', f'L{n}.ln2')
        LN = lambda p: rng(ln + p, C0, ln + p, C0 + D - 1)
        pre_r, r = self.block(ws, r, M, f'Hidden  =MMULT(LayerNorm, W1) + b1', None,
                              lambda p, rr: f"=MMULT({LN(p)},{pre}W1)+{pre}b1", key=f'L{n}.pre', array=True)
        relu_r, r = self.block(ws, r, M, 'ReLU  =MAX(0, Hidden)', None,
                               lambda p, rr: [f"=MAX(0,{A(pre_r + p, C0 + j)})" for j in range(M)],
                               key=f'L{n}.relu', color='pos')
        RL = lambda p: rng(relu_r + p, C0, relu_r + p, C0 + M - 1)
        out_r, r = self.block(ws, r, D, 'Output  =MMULT(ReLU, W2) + b2', None,
                              lambda p, rr: f"=MMULT({RL(p)},{pre}W2)+{pre}b2", key=f'L{n}.mlp', array=True)
        res_r, r = self.block(ws, r, D, 'Residual stream out  = in + Output', 'goes to the next tab',
                              lambda p, rr: [f"={q(xs)}!{A(xr + p, xc + j)}+{A(out_r + p, C0 + j)}"
                                             for j in range(D)], key=f'L{n}.out')
        r += 1
        r = self.section(ws, r, 'PARAMETERS  ·  learned in PyTorch, plain numbers', FILL_PAR, C0 + M)
        r = self.weights(ws, r, sd('ln2.g'), 'LayerNorm gain', pre + 'ln2_gain')
        r = self.weights(ws, r, sd('ln2.b'), 'LayerNorm bias', pre + 'ln2_bias')
        r = self.weights(ws, r, sd('fc1.weight').T, f'W1  (32 × {M})', pre + 'W1')
        r = self.weights(ws, r, sd('fc1.bias'), 'b1', pre + 'b1')
        r = self.weights(ws, r, sd('fc2.weight').T, f'W2  ({M} × 32)', pre + 'W2')
        r = self.weights(ws, r, sd('fc2.bias'), 'b2', pre + 'b2')
        ws.freeze_panes = 'C3'
        return (title, [title, res_r, C0, D])

    def sheet_logits(self, x_ref):
        D, P, V = self.D, self.P, self.V
        title = 'Logits'
        ws = self.sheet(title, '8250DF')
        self.widths(ws, max(D, V + 6))
        r = self.banner(ws, 'Logits', 'A score for every possible next letter. The biggest one wins '
                        '(MATCH(MAX(...))), so the same prompt always gives the same name.', C0 + D + 2)
        r = self.section(ws, r, 'ACTIVATIONS  ·  formulas, change with the prompt', FILL_ACT, C0 + D + 2)
        ln, r = self.ln_block(ws, r, x_ref, 'Final_', 'Final LayerNorm', 'Final_ln_gain', 'Final_ln_bias', 'lnf')
        LN = lambda p: rng(ln + p, C0, ln + p, C0 + D - 1)
        letters = list(self.chars)
        # logits block is placed so argmax_row() is known before Tokens refers to it
        lg = r + 2
        assert lg == self.logit_r0, (lg, self.logit_r0)
        lr, r = self.block(ws, r, V, 'Logits  =MMULT(LayerNorm, Unembed) + bias', None,
                           lambda p, rr: f"=MMULT({LN(p)},Unembed)+Unembed_bias", key='logits',
                           names=letters, array=True)
        ac, cc, mc = C0 + V + 1, C0 + V + 2, C0 + V + 3
        for c, h in ((ac, 'argmax'), (cc, 'next'), (mc, 'margin')):
            ws.cell(lr - 1, c, h).font = Font(name=FONT, size=7, bold=True, color=GREY)
        for p in range(P):
            rr = lr + p
            row = rng(rr, C0, rr, C0 + V - 1, absc=True)
            ws.cell(rr, ac, f"=MATCH(MAX({row}),{row},0)-1")
            ws.cell(rr, cc, f"=INDEX(VocabChar,{L(ac)}{rr}+1)")
            ws.cell(rr, mc, f"=MAX({row})-LARGE({row},2)")
            ws.cell(rr, ac).font = F_DATA
            ws.cell(rr, cc).font = Font(name=FONT, size=10, bold=True, color='8250DF')
            ws.cell(rr, mc).font = F_LABEL
            ws.cell(rr, mc).number_format = NUM
            for c in (ac, cc, mc):
                ws.cell(rr, c).alignment = CENTER
        self.record('argmax', title, lr, ac, 1)
        self.record('margin', title, lr, mc, 1)
        def fp(p, rr):
            row = rng(lr + p, C0, lr + p, C0 + V - 1, absc=True)
            return [f"=EXP({A(lr + p, C0 + j)}-MAX({row}))/SUMPRODUCT(EXP({row}-MAX({row})))" for j in range(V)]
        pr, r = self.block(ws, r, V, 'Probabilities  (softmax, for display only)', None, fp,
                           key='probs', color=None, names=letters)
        warm(ws, rng(pr, C0, pr + P - 1, C0 + V - 1), color='8250DF')
        for p in range(P):
            for j in range(V):
                ws.cell(pr + p, C0 + j).number_format = '[<0.005]"";0%'
        tc = C0 + V + 1
        ws.cell(pr - 1, tc, 'top').font = Font(name=FONT, size=7, bold=True, color=GREY)
        for p in range(P):
            rr = pr + p
            ws.cell(rr, tc, f"=MAX({rng(rr, C0, rr, C0 + V - 1)})").number_format = '0%'
            ws.cell(rr, tc).font = F_LABEL
        self.top_prob = (pr, tc)
        r += 1
        r = self.section(ws, r, 'PARAMETERS  ·  learned in PyTorch, plain numbers', FILL_PAR, C0 + D + 2)
        r = self.weights(ws, r, self.sd['lnf.g'], 'Final LayerNorm gain', 'Final_ln_gain')
        r = self.weights(ws, r, self.sd['lnf.b'], 'Final LayerNorm bias', 'Final_ln_bias')
        r = self.weights(ws, r, self.sd['head.weight'].T, f'Unembed  (32 × {V}, one column per letter)',
                         'Unembed', col_names=letters)
        r = self.weights(ws, r, self.sd['head.bias'], 'Unembed bias', 'Unembed_bias', col_names=letters)
        ws.freeze_panes = 'C3'

    def sheet_front(self, ws, default_prompt):
        P = self.P
        ws.sheet_view.showGridLines = False
        ws.sheet_properties.tabColor = '1F2328'
        ws.column_dimensions['A'].width = 30
        for c in range(2, 2 + P):
            ws.column_dimensions[L(c)].width = 5.2
        ws.column_dimensions[L(2 + P)].width = 4
        ws.row_dimensions[1].height = 42
        ws.row_dimensions[3].height = 46
        a1 = ws['A1']
        a1.value = default_prompt
        a1.font = Font(name=FONT, size=26, bold=True, color=INK)
        a1.fill = PatternFill('solid', fgColor='FFF8C5')
        a1.alignment = Alignment(vertical='center', indent=1)
        thin = Side(style='medium', color='D4A72C')
        a1.border = Border(left=thin, right=thin, top=thin, bottom=thin)
        ws['B1'] = '← type a few letters here and press Enter'
        ws['B1'].font = Font(name=FONT, size=11, italic=True, color=GREY)
        ws['B1'].alignment = Alignment(vertical='center')
        ws['A3'] = 'the spreadsheet continues →'
        ws['A3'].font = Font(name=FONT, size=11, color=GREY)
        ws['A3'].alignment = Alignment(horizontal='right', vertical='center')
        ws['A4'] = 'who wrote it'
        ws['A5'] = 'how sure (top probability)'
        for a in ('A4', 'A5'):
            ws[a].font = Font(name=FONT, size=8, color=FAINT)
            ws[a].alignment = Alignment(horizontal='right')
        box = Side(style='thin', color='D0D7DE')
        t0 = self.tok_row
        pr, tc = self.top_prob
        for p in range(1, P + 1):
            c = 1 + p
            col = L(c)
            cell = ws.cell(3, c, f'=IF(Tokens!$G${t0 + p},"",INDEX(VocabChar,Tokens!$E${t0 + p}+1))')
            cell.font = Font(name='Consolas', size=26, bold=True, color=INK)
            cell.alignment = CENTER
            cell.border = Border(left=box, right=box, top=box, bottom=box)
            src = ws.cell(4, c, f'=Tokens!$H${t0 + p}')
            src.font = Font(name=FONT, size=7, color=FAINT)
            src.alignment = CENTER
            conf = ws.cell(5, c, f'=IF({col}$4="model",{q("Logits")}!${L(tc)}${pr + p - 1},"")')
            conf.font = Font(name=FONT, size=7, color=FAINT)
            conf.number_format = '0%'
            conf.alignment = CENTER
        out = rng(3, 2, 3, 1 + P)
        ws.conditional_formatting.add(out, FormulaRule(formula=['B$4="model"'], stopIfTrue=True,
                                      font=Font(color='1A7F37', bold=True), fill=PatternFill('solid', fgColor='DAFBE1')))
        ws.conditional_formatting.add(out, FormulaRule(formula=['B$4="you"'], stopIfTrue=True,
                                      font=Font(color='57606A', bold=True), fill=PatternFill('solid', fgColor='F6F8FA')))
        ws.row_dimensions[7].height = 34
        word = '&'.join(L(1 + p) + '3' for p in range(1, P + 1))
        ws['A7'] = f'=PROPER({word})'
        ws['A7'].font = Font(name=FONT, size=22, bold=True, color='1A7F37')
        ws['A7'].alignment = Alignment(horizontal='right', vertical='center')
        ws['B7'] = '← a brand new dinosaur, generated by formulas'
        ws['B7'].font = Font(name=FONT, size=10, italic=True, color=GREY)
        ws['B7'].alignment = Alignment(vertical='center')
        counts = ','.join(self.param_names)
        ws['A10'] = f'=COUNT({counts})'
        ws['A10'].number_format = '#,##0" parameters"'
        ws['A10'].font = Font(name=FONT, size=12, bold=True, color=INK)
        ws['A10'].alignment = Alignment(horizontal='right')
        ws['B10'] = (f'{self.NL} layers · {self.H} attention heads · {self.D} numbers per letter · '
                     f'{self.P}-letter context')
        ws['B10'].font = Font(name=FONT, size=10, color=GREY)
        ws['A11'] = 'no macros, no VBA'
        ws['A11'].font = Font(name=FONT, size=12, bold=True, color=INK)
        ws['A11'].alignment = Alignment(horizontal='right')
        ws['B11'] = 'every number on the other tabs is a formula: INDEX, MMULT, SUMPRODUCT, EXP, MAX, MATCH'
        ws['B11'].font = Font(name=FONT, size=10, color=GREY)
        ws['A13'] = 'inspired by'
        ws['A13'].font = Font(name=FONT, size=9, color=FAINT)
        ws['A13'].alignment = Alignment(horizontal='right')
        ws['B13'] = 'Ishan Anand, "Spreadsheets are all you need" (GPT-2 small running in Excel)'
        ws['B13'].font = Font(name=FONT, size=9, color=GREY)
        ws.sheet_view.selection[0].activeCell = 'A1'
        ws.sheet_view.selection[0].sqref = 'A1'

    def sheet_about(self):
        ws = self.sheet('How it works', '57606A')
        ws.column_dimensions['B'].width = 120
        r = self.banner(ws, 'How it works', 'A tiny GPT-style transformer, trained on dinosaur names, '
                        'copied into a spreadsheet.', 2)
        lines = [
            ('Read the tabs left to right. That is the forward pass.', True),
            ('Tokens: A1 is lowercased and cleaned. Position 0 is "." (start). Typed letters fill the next '
             'positions; every later position takes the argmax from the row above it on the Logits tab.', False),
            ('Embeddings: INDEX pulls one row of 32 numbers per letter and adds one row per position.', False),
            ('Layer N Attention: LayerNorm, then Q, K and V with MMULT. Scores use SUMPRODUCT, only for '
             'earlier positions (that is the causal mask: the grey triangle). Softmax is EXP divided by a sum.',
             False),
            ('Layer N MLP: LayerNorm, MMULT up to the hidden layer, ReLU as MAX(0, x), MMULT back down.', False),
            ('Logits: final LayerNorm, MMULT against the unembedding, then MATCH(MAX(...)) picks the letter.',
             False),
            ('No loops and no iterative calculation: position t only reads positions before it, so the '
             'sheet is unrolled one row per position, and generation is a chain of ordinary formulas.', False),
            ('No VBA, no macros, no add-ins. Grey numbers in the PARAMETERS sections are the trained '
             'weights; everything in ACTIVATIONS is a formula.', False),
            ('', False),
            ('Training data: genus names of non-bird dinosaurs from the Paleobiology Database '
             '(paleobiodb.org, CC BY 4.0).', False),
            ('Inspired by Ishan Anand, "Spreadsheets are all you need": GPT-2 small implemented in Excel.',
             False),
        ]
        for text, bold in lines:
            c = ws.cell(r, 2, text)
            c.font = Font(name=FONT, size=11, bold=bold, color=INK)
            c.alignment = Alignment(wrap_text=True, vertical='top')
            r += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--prompt', default='excel')
    a = ap.parse_args()
    meta = json.load(open(f'{a.run}/meta.json'))
    sd = torch.load(f'{a.run}/model.pt', map_location='cpu')
    b = Book(sd, meta)
    # The Logits block position must be known before the Tokens sheet refers to it.
    # It is deterministic: compute it with a dry run.
    b.logit_r0 = 0
    try:
        b.build(a.prompt)
    except AssertionError as e:
        b2 = Book(sd, meta)
        b2.logit_r0 = e.args[0][0]
        b = b2
        b.build(a.prompt)
    b.wb.save(a.out)
    b.map['chars'] = b.chars
    b.map['param_names'] = b.param_names
    json.dump(b.map, open(os.path.splitext(a.out)[0] + '.cellmap.json', 'w'))
    print('wrote', a.out, os.path.getsize(a.out), 'bytes')


if __name__ == '__main__':
    main()
