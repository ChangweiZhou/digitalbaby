"""Where does a life break an unrehearsed birth skill?  Weight grafts between a birth and the life that started from it.  (15 November 2026)

Read-only: forward passes on saved weights with numpy; nothing trains.  A hybrid network takes every weight from the
end of the life except one named group, which is put back to its value at birth.  Scored: the network's own exact
answers (all three target bytes) on every operand pair without distractor letters, for a birth task and a life task.

Groups: input = byte embeddings and the first layer (book, l1); recurrent = the GRU's weights (rnn.*);
        readout = the path from the hidden state to the output bytes (lm, l2, l3).

Run from the repository root:
  python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/graft_birth_into_life.py add89,sub89,add97,sub97 <birth file> <life file> [<life file> ...]
Files are paths under MOD97_LOCAL_RULE_20261106/exp45/ on the default branch.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from np_gru import load_weights, sigmoid, DEV, LIFE, BOS, EOS, P, apply, SYM

TASKS = {"add89": DEV[0], "sub89": DEV[1], "max89": DEV[2], "min89": DEV[3], "add97": LIFE[0], "sub97": LIFE[1], "max97": LIFE[2], "min97": LIFE[3]}
GROUPS = {"input": ("book.", "l1."), "recurrent": ("rnn.",), "readout": ("lm.", "l2.", "l3.")}

def own_exact(W, task):
    """Same arithmetic as np_gru.forward, with the per-byte parts looked up once per byte value."""
    op, m, code = task
    ab = np.array([(a, b) for a in range(m) for b in range(m)])
    c = np.array([apply(op, m, int(u), int(v)) for u, v in ab])
    x = np.array([[BOS] + list(code) + list(f"{a:02d}".encode()) + [ord(SYM[op])] + list(f"{b:02d}".encode()) + [ord("=")] + list(f"{r:02d}".encode()) for (a, b), r in zip(ab, c)], dtype=np.int64)
    tgt = np.array([list(f"{r:02d}".encode()) + [EOS] for r in c], dtype=np.int64)
    z1 = np.tanh(W['book.weight'] @ W['l1.weight'].T + W['l1.bias'])                       # one row per byte value
    gi_tok = z1 @ W['rnn.weight_ih_l0'].T + W['rnn.bias_ih_l0']
    l2_tok = z1 @ W['l2.weight'].T + W['l2.bias']
    Whh, bhh = W['rnn.weight_hh_l0'], W['rnn.bias_hh_l0']; U = Whh.shape[1]
    h = np.zeros((len(x), U), dtype=np.float32); ok = np.ones(len(x), dtype=bool)
    L = x.shape[1]
    for t in range(L):
        gi = gi_tok[x[:, t]]; gh = h @ Whh.T + bhh
        r = sigmoid(gi[:, :U] + gh[:, :U]); z = sigmoid(gi[:, U:2 * U] + gh[:, U:2 * U])
        n = np.tanh(gi[:, 2 * U:] + r * gh[:, 2 * U:]); h = (1 - z) * n + z * h
        if t >= L - 3:
            z2 = np.tanh(l2_tok[x[:, t]] + h @ W['lm.weight'].T + W['lm.bias'])
            ok &= (z2 @ W['l3.weight'].T + W['l3.bias']).argmax(1) == tgt[:, t - (L - 3)]
    return ok.mean() * 100

def graft(life, birth, group):
    return {k: (birth[k] if k.startswith(GROUPS[group]) else v) for k, v in life.items()}

if __name__ == "__main__":
    want = sys.argv[1].split(",")
    birth = load_weights(P + sys.argv[2])[0]
    row = lambda name, W: print(f"{name:52s} " + " | ".join(f"{t} {own_exact(W, TASKS[t]):5.1f}" for t in want), flush=True)
    row("birth  " + sys.argv[2], birth)
    for f in sys.argv[3:]:
        life = load_weights(P + f)[0]
        assert set(life) == set(birth)
        row("life   " + f, life)
        for g in GROUPS: row(f"   life with the birth's {g}", graft(life, birth, g))
        moved = {g: float(np.sqrt(sum(((life[k] - birth[k]) ** 2).sum() for k in life if k.startswith(GROUPS[g])) / sum((birth[k] ** 2).sum() for k in life if k.startswith(GROUPS[g])))) for g in GROUPS}
        print("   relative change of each group over the life (norm of the difference / norm at birth): " + ", ".join(f"{g} {v:.2f}" for g, v in moved.items()), flush=True)
