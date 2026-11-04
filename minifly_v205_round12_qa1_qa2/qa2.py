"""Round-12/13 bAbI QA2 learner: object permanence via event-time binding + learned operation arbiter.
No backprop, no attention. Byte n-gram hashed stores, local delta-rule writes, single-layer logistic arbiter.

Built-in (architecture, NOT learned):
  * event = sentence (segmented at '.', '?'); words = space-separated tokens; positional roles p = 0..P-1
  * location store M (byte-level, step-tagged values, '\\n' end marker): written for words in the learned answer
    vocabulary (VOCAB, from teach cues), keyed by the whole sentence with that word masked (order-free binding)
  * at EVERY event and for EVERY position p, two candidate bindings keyed by the (unmasked) sentence bag, tagged (op,p):
        H_p : value = the word at position p                       ("who/what was involved")
        L_p : value = location recalled NOW for the word at p      ("where it was at event time")
  * E store: value = hashed bag-of-words code of the event        (lets a later query reinstate *what kind* of event it was)
  * at a question: each channel proposes a candidate; H candidates are re-queried in M ("where is that word now")
  * read weighting by question-stream n-gram IDF (QIDF, round 12)
Learned:
  * VOCAB (answer words), M contents (episodic), IDF statistics,
  * arbiter: per-channel logistic readout W[c]·phi_hat + b[c] predicting "this channel's candidate is correct",
    trained with a local delta rule on revealed answers (frozen at test when FREEZE=1).
Knobs (env): P, D, ETA, FREEZE, VALID, TRAIN/TEST, ARB (1=learned arbiter; 0=fixed best single channel picked by
             training accuracy), PYTHONHASHSEED.
"""
import numpy as np, os, json, re, sys
E = os.environ.get
N = 1 << 16; NC = 1 << 16; V = int(E('V', '512')); LEXADD = E('LEXADD', '0') == '1'; MASKV = E('MASKV', '0') == '1'; D = int(E('D', '1024')); FH = int(E('FH', '2'))
P = int(E('P', '6')); ETA = float(E('ETA', '0.2')); FREEZE = E('FREEZE', '1') == '1'; VALID = E('VALID', '0') == '1'
ARB = E('ARB', '1'); EPOCHS = int(E('EPOCHS', '1')); THR = float(E('THR', '0.3')); MAXW = 16
TRAIN = E('TRAIN', '/tmp/babi/qa2_two-supporting-facts_train.txt'); TEST = E('TEST', '/tmp/babi/qa2_two-supporting-facts_test.txt'); VALFN = E('VALFN', TRAIN)
MODE = E('MODE', 'full')   # full | direct (QA1-style: recall the question directly from the location store; no channels)
QIDF = E('QIDF', '1') == '1'; DUMP = E('DUMP', '')

def parse(fn):
    out = []
    for line in open(fn):
        line = line.rstrip('\n'); i = line.index(' '); n = int(line[:i]); txt = line[i + 1:]
        if n == 1: out.append(('story', None))
        if '\t' in txt:
            q, a, sup = txt.split('\t'); out.append(('q', (q.strip(), a)))
        else: out.append(('s', txt))
    return out

DFQ = {}; NQ = [0]; FMAP = {}
def grams(h):
    return [h[i:i + k] for k in (2, 3, 4) for i in range(len(h) - k + 1)]
def idf(g): return np.log((1 + NQ[0]) / (1 + DFQ.get(g, 0))) if QIDF else 1.0
def keyunits(h, tag, n, per=4):
    d = {}
    for g in grams(bytes(h)):
        w = idf(g); base = hash((g, tag))
        for j in range(per):
            u = (base * 2654435761 + j * 40503) % n; d[u] = max(d.get(u, 0), w)
    return np.fromiter(d.keys(), np.int64), np.fromiter(d.values(), float)
def keyunits_w(h, tag, n, per=4):   # write keys: unweighted
    d = {}
    for g in grams(bytes(h)):
        base = hash((g, tag))
        for j in range(per): d[(base * 2654435761 + j * 40503) % n] = 1.0
    return np.fromiter(d.keys(), np.int64), np.fromiter(d.values(), float)

class Lex:
    def __init__(s): s.id = {}; s.w = []
    def get(s, w):
        if w not in s.id:
            if len(s.w) >= V: return None
            s.id[w] = len(s.w); s.w.append(w)
        return s.id[w]

DROPV = re.compile(r' (dropped|discarded|left|put down) the (\w+)'); GETV = re.compile(r' (got|grabbed|picked up|took) the (\w+)')
def heur(ctx, q, a):   # symbolic baselines over the story so far (evaluation only)
    obj = q.split()[-1].rstrip('?'); loc = {}; hold = {}; objloc = {}; lastloc = None
    for s in ctx:
        if s.startswith('Where'): continue
        m = re.match(r'(\w+) .* the (\w+)\.$', s); who = s.split()[0]
        g = GETV.search(s); d = DROPV.search(s)
        if g: hold[g.group(2)] = who; objloc.pop(g.group(2), None)
        elif d: objloc[d.group(2)] = loc.get(who); hold.pop(d.group(2), None)
        elif m: loc[who] = m.group(2); lastloc = m.group(2)
    nodrop = None   # chain that ignores drops: last getter's current location
    for s in ctx:
        g = GETV.search(s)
        if g and g.group(2) == obj: nodrop = s.split()[0]
    full = loc.get(hold[obj]) if obj in hold else objloc.get(obj)
    return (lastloc == a, loc.get(nodrop) == a, full == a)
def run():
    M = np.zeros((N, 256), np.float32); MC = np.zeros((NC, V), np.float32); ME = np.zeros((NC, D), np.float32)
    lex = Lex(); VOCAB = set(); FROZ = [False]
    chans = [(op, p) for op in 'HL' for p in range(P)]
    W = np.zeros((len(chans), D)); B = np.zeros(len(chans)); TTB = np.zeros((len(chans), D, 2)); CHACC = np.zeros((len(chans), 2))
    def wr(store, ix, y):
        T = np.zeros(store.shape[1], np.float32); T[y] = 1; store[ix] += (T[None, :] - store[ix])
    def wrvec(store, ix, vec):
        store[ix] += (vec[None, :].astype(np.float32) - store[ix])
    def recall(cue):   # step-tagged byte recall from M; returns bytes or None
        out = bytearray()
        for t in range(MAXW):
            ix, v = keyunits(cue, ('M', t), N); s = v @ M[ix]
            if s.max() <= 1e-6: return None
            y = int(np.argmax(s))
            if y == 10: return bytes(out)
            out.append(y)
        return None
    def phi(words):
        f = np.zeros(D)
        for w in words:
            for j in range(FH): f[hash((b'F', w, j)) % D] += 1; FMAP.setdefault(hash((b'F', w, j)) % D, set()).add(w)
        n = np.linalg.norm(f); return f / n if n > 0 else f
    def event(sent):
        words = sent.split(b' ')
        # 1) location binding for answer-vocabulary words (order-free, masked key)
        for wi, w in enumerate(words):
            if w in VOCAB:
                key = b' '.join(words[:wi] + [b'\x01'] + words[wi + 1:])
                for t, y in enumerate(w + b'\n'):
                    ix, _ = keyunits_w(key, ('M', t), N); wr(M, ix, y)
        # 2) candidate bindings for every position (keys: sentence with answer-vocabulary words masked if MASKV=1)
        ksent = b' '.join(b'\x01' if w in VOCAB else w for w in words) if MASKV else sent
        for p, w in enumerate(words[:P]):
            wid = lex.get(w)
            if wid is not None:
                ix, _ = keyunits_w(ksent, ('H', p), NC, per=2); wr(MC, ix, wid)
            loc = recall(w)
            if loc and (loc in lex.id or LEXADD):   # lexical decision: only bind recalls that match an observed word form
                lid = lex.get(loc)
                if lid is not None:
                    ix, _ = keyunits_w(ksent, ('L', p), NC, per=2); wr(MC, ix, lid)
        # 3) event-type code
        ix, _ = keyunits_w(ksent, ('E',), NC, per=2); wrvec(ME, ix, phi(words))
    def answer(q):
        if MODE == 'direct':
            out = recall(q); return out, {}, np.zeros(D)
        cands = {}
        for ci, (op, p) in enumerate(chans):
            ix, v = keyunits(q, (op, p), NC, per=2); s = v @ MC[ix]
            if s.max() <= 0: continue
            w = lex.w[int(np.argmax(s))]
            loc = recall(w) if op == 'H' else w
            if loc: cands[ci] = loc
        ix, v = keyunits(q, ('E',), NC, per=2); ph = v @ ME[ix]; n = np.linalg.norm(ph); ph = ph / n if n > 0 else ph
        if not cands: return None, cands, ph
        if ARB in ('1', 'soft'):
            sc = {ci: W[ci] @ ph + B[ci] for ci in cands}
        elif ARB == 'ttb':   # take-the-best: channel reliability conditioned on the single most diagnostic feature present
            act = np.nonzero(ph > 0.5 * ph.max())[0] if ph.max() > 0 else []
            sc = {ci: max([(TTB[ci, f, 0] + 1) / (TTB[ci, f, 1] + 2) for f in act] or [0.5]) for ci in cands}
        else:   # ablation: fixed channel ranking by training accuracy (no event conditioning)
            sc = {ci: CHACC[ci, 0] / max(1, CHACC[ci, 1]) for ci in cands}
        best = max(sc, key=sc.get)
        return cands[best], cands, ph
    def split(fn):
        it = parse(fn); idx = [i for i, (k, _) in enumerate(it) if k == 'story']; c = idx[int(len(idx) * 0.8)]
        return it[:c], it[c:]
    if E('TRAINSEQ'):   # sequential curriculum: train on first 80% of each file in order, then evaluate frozen on each last 20%
        seq = E('TRAINSEQ').split(','); evs = E('EVALS', E('TRAINSEQ')).split(',')
        if E('FULLSEQ') == '1':   # final protocol: full training files in order, full evaluation files (e.g. official test sets)
            phases = tuple((parse(f), False) for f in seq for _ in range(EPOCHS)) + tuple((parse(f), True) for f in evs)
        else:
            phases = tuple((split(f)[0], False) for f in seq for _ in range(EPOCHS)) + tuple((split(f)[1], True) for f in evs)
        PHASE_NAMES = [os.path.basename(f) for f in evs]
    else:
        tr = parse(TRAIN); PHASE_NAMES = None
        if VALID:
            idx = [i for i, (k, _) in enumerate(tr) if k == 'story']; cut = idx[int(len(idx) * 0.8)]
            va = parse(VALFN); idv = [i for i, (k, _) in enumerate(va) if k == 'story']; cutv = idv[int(len(idv) * 0.8)]
            phases = tuple((tr[:cut], False) for _ in range(EPOCHS)) + ((va[cutv:], True),)
        else: phases = tuple((tr, False) for _ in range(EPOCHS)) + ((parse(TEST), True),)
    res = []; dump = []; heur_hit = []; CTX = []; HEUR = []
    PER = []
    for items, test in phases:
        FROZ[0] = FREEZE and test; n0 = len(res)
        for kind, x in items:
            if kind == 'story': CTX.clear(); continue
            CTX.append(x if kind == 's' else x[0])
            if kind == 's':
                event(x.rstrip('.').encode())
            else:
                q, a = x; qb = q.rstrip('?').encode(); ab = a.encode()
                out, cands, ph = answer(qb)
                if test:
                    res.append(out == ab); HEUR.append(heur(CTX[:-1], q, a))
                    if DUMP: dump.append({'q': q, 'a': a, 'out': out.decode('latin1') if out else None,
                                          'cands': {f'{chans[c][0]}{chans[c][1]}': v.decode('latin1') for c, v in cands.items()},
                                          'scores': {f'{chans[c][0]}{chans[c][1]}': round(float(W[c] @ ph + B[c]),3) for c in cands},
                                          'phi_top': [(b'/'.join(sorted(FMAP.get(int(i), {b'?'}))).decode('latin1'), round(float(ph[i]),2)) for i in np.argsort(-ph)[:6]],
                                          'ctx': CTX[-8:]})
                if not FROZ[0]:
                    if ARB == 'soft' and cands:   # multinomial delta rule over the available channels (single layer, local)
                        cs = list(cands); tt = np.array([1.0 if cands[c] == ab else 0.0 for c in cs])
                        if tt.sum() > 0:
                            z = np.array([W[c] @ ph + B[c] for c in cs]); pz = np.exp(z - z.max()); pz /= pz.sum(); tt /= tt.sum()
                            for c, tc, pc in zip(cs, tt, pz): W[c] += ETA * (tc - pc) * ph; B[c] += ETA * (tc - pc)
                    for ci, loc in cands.items():
                        t = 1.0 if loc == ab else 0.0
                        if ARB == '1':
                            z = W[ci] @ ph + B[ci]; pz = 1 / (1 + np.exp(-z))
                            W[ci] += ETA * (t - pz) * ph; B[ci] += ETA * (t - pz)
                        CHACC[ci, 0] += t; CHACC[ci, 1] += 1
                        if ARB == 'ttb' and ph.max() > 0:
                            act = np.nonzero(ph > 0.5 * ph.max())[0]; TTB[ci, act, 0] += t; TTB[ci, act, 1] += 1
                    for g in set(grams(qb)): DFQ[g] = DFQ.get(g, 0) + 1
                    NQ[0] += 1
                    VOCAB.add(ab); lex.get(ab)
        if test: PER.append(round(float(np.mean(res[n0:])), 4))
    if E('WDUMP'):
        words=[b'got',b'grabbed',b'picked',b'up',b'took',b'dropped',b'discarded',b'left',b'put',b'down',b'there',b'the',b'Mary',b'John',b'milk',b'apple',b'football']
        h0=chans.index(('H',0)); l0=chans.index(('L',0))
        for w in words:
            ix=[hash((b'F', w, j)) % D for j in range(FH)]
            print(w.decode(), 'H0 %.2f L0 %.2f' % (W[h0][ix].sum(), W[l0][ix].sum()), file=sys.stderr)
        print('bias H0 %.2f L0 %.2f' % (B[h0], B[l0]), file=sys.stderr)
    chan_acc = {f'{op}{p}': round(CHACC[ci, 0] / max(1, CHACC[ci, 1]), 3) for ci, (op, p) in enumerate(chans)}
    if DUMP: json.dump(dump, open(DUMP, 'w'), indent=0)
    H = np.array(HEUR, float).mean(0) if HEUR else [0, 0, 0]
    extra = {'per_eval': dict(zip(PHASE_NAMES, PER))} if PHASE_NAMES else {}
    return {'acc': round(float(np.mean(res)), 4), 'n': len(res), **extra, 'heur_last_location': round(H[0], 3), 'heur_chain_ignoring_drops': round(H[1], 3), 'heur_oracle_parse_with_drops': round(H[2], 3), 'train_channel_acc': chan_acc}

cfg = {k: E(k) for k in ('TRAINSEQ', 'EVALS', 'FULLSEQ', 'MASKV', 'MODE', 'P', 'D', 'FH', 'ETA', 'EPOCHS', 'FREEZE', 'VALID', 'ARB', 'QIDF', 'PYTHONHASHSEED') if E(k) is not None}
cfg['train'] = os.path.basename(TRAIN); cfg['eval'] = ('valid(last20%):' + os.path.basename(VALFN)) if VALID else os.path.basename(TEST)
print(json.dumps({**cfg, **run()}))
