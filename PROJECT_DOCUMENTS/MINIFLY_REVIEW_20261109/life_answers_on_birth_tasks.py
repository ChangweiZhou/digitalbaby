"""What does a network answer, at the end of a life, on a birth task the life never rehearsed?  (15 November 2026)

Read-only: forward passes on saved weights with numpy, greedy two-digit answers on every operand pair without letters.
For add and sub mod 89 it reports the share of answers that equal the result mod 89, mod 97 and mod 73 (the life's two
moduli), and exact answers separately for pairs that need no wrap (a + b < 89, or a - b >= 0) and pairs that do.

Run from the repository root:
  python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/life_answers_on_birth_tasks.py <file> [<file> ...]
Files are paths under MOD97_LOCAL_RULE_20261106/exp45/ on the default branch, e.g. e26/e26_life_S_73129.pt.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from np_gru import load_weights, sigmoid, DEV, LIFE, BOS, EOS, P, apply, SYM
def answers(W, task):
    op, m, code = task
    ab = np.array([(a, b) for a in range(m) for b in range(m)])
    z1 = np.tanh(W['book.weight'] @ W['l1.weight'].T + W['l1.bias']); gi_tok = z1 @ W['rnn.weight_ih_l0'].T + W['rnn.bias_ih_l0']; l2_tok = z1 @ W['l2.weight'].T + W['l2.bias']
    Whh, bhh = W['rnn.weight_hh_l0'], W['rnn.bias_hh_l0']; U = Whh.shape[1]
    x = np.array([[BOS] + list(code) + list(f"{a:02d}".encode()) + [ord(SYM[op])] + list(f"{b:02d}".encode()) + [ord("=")] for a, b in ab], dtype=np.int64)
    h = np.zeros((len(x), U), dtype=np.float32)
    def step(h, tok):
        gi = gi_tok[tok]; gh = h @ Whh.T + bhh
        r = sigmoid(gi[:, :U] + gh[:, :U]); z = sigmoid(gi[:, U:2 * U] + gh[:, U:2 * U]); n = np.tanh(gi[:, 2 * U:] + r * gh[:, 2 * U:])
        return (1 - z) * n + z * h
    out = lambda h, tok: (np.tanh(l2_tok[tok] + h @ W['lm.weight'].T + W['lm.bias']) @ W['l3.weight'].T + W['l3.bias']).argmax(1)
    for t in range(x.shape[1]): h = step(h, x[:, t])
    d1 = out(h, x[:, -1]); h = step(h, d1); d2 = out(h, d1); h = step(h, d2); end = out(h, d2)
    return ab, d1, d2, end
for f in sys.argv[1:]:
    W = load_weights(P + f)[0]
    for tname, task in (("add89", DEV[0]), ("sub89", DEV[1])):
        op, m, code = task
        ab, d1, d2, end = answers(W, task); a, b = ab[:, 0], ab[:, 1]
        isnum = (d1 >= 48) & (d1 <= 57) & (d2 >= 48) & (d2 <= 57)
        val = np.where(isnum, (d1 - 48) * 10 + (d2 - 48), -1)
        s = a + b if op == "add" else a - b
        c89 = s % 89; c97 = s % 97; c73 = s % 73
        wrap = (s >= 89) if op == "add" else (s < 0)
        print(f"{f.split('/')[-1]:26s} {tname}: two digits {isnum.mean()*100:5.1f}% | = mod 89 {np.mean(val==c89)*100:5.1f} | = mod 97 {np.mean(val==c97)*100:5.1f} | = mod 73 {np.mean(val==c73)*100:5.1f} | right, no wrap {np.mean((val==c89)[~wrap])*100:5.1f} | right, wrap {np.mean((val==c89)[wrap])*100:5.1f} | units digit right {np.mean((d2-48)==c89%10)*100:5.1f} | tens right {np.mean((d1-48)==c89//10)*100:5.1f} | off by at most 2 (circular) {np.mean(np.minimum((val-c89)%89,(c89-val)%89)<=2)*100:5.1f}")
