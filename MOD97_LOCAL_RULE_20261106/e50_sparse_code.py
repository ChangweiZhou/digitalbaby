"""E50: a fixed input code in which a value stands for one byte. Coordinator's specification file, 16 Nov 2026 (the job the audit of E49
names as next). The byte table and l1 are replaced by a fixed code: each of the 46 bytes the first family uses turns on 11 of 512 values
of its own, and nothing else; the code is not trained. On it, E49's two masks. A cell that reads 16 values then receives a non-zero
input for only a few bytes.
    python3 -I e50_sparse_code_spec.py        self-checks, no training"""
import sys
from pathlib import Path
from random import Random

HERE = Path(__file__).resolve()
for _cand in (HERE.parent, HERE.parents[2] / "MOD97_LOCAL_RULE_20261106"):
    if (_cand / "exp26_candidate.py").exists():
        sys.path.insert(0, str(_cand)); break
sys.path.insert(0, str(HERE.parent))
import exp26_candidate as C26        # noqa: E402
import arch26                        # noqa: E402
import torch as T                    # noqa: E402
from exp3_state import lin           # noqa: E402
import e42_cells as E42              # noqa: E402
import e49_interface as E49          # noqa: E402

U, D, ON = 256, 512, 11
BOS = 257
USED = sorted(set(b"0123456789+-><=ABCDEFGHIJKLMNOPQRSTUVWXYZpqvw")) + [BOS]      # the 46 symbols that occur as input in the first family
NEEDED = sorted(set(b"0123456789+-><=pqvw")) + [BOS]                               # the 20 the task must tell apart (letters are fillers)


def code_table(seed, amplitude):
    """(258, 512): row b is the code of byte b. Used byte number k (in the order of USED) turns on values vals[11 k : 11 k + 11]."""
    vals = list(range(D)); Random(f"e50|code|{seed}").shuffle(vals)
    table = T.zeros(258, D); owner = [-1] * D
    for k, b in enumerate(USED):
        for v in vals[ON * k: ON * k + ON]:
            table[b, v] = amplitude; owner[v] = b
    return table, owner


def amplitude_for(m, seed):
    """The value a turned-on entry takes: chosen once so that the code of a used byte has the mean squared length the present network's
    code (tanh(l1(book))) has over the used bytes at the start."""
    with T.no_grad():
        z1 = T.tanh(m.l1(m.book(T.tensor(USED))))
        return float((z1.pow(2).sum(1).mean() / ON).sqrt())


def forward(m, layer, iface, table, x, local, silence=False):
    """E49's forward with the byte's code read from the fixed table. The byte table and l1 take no part."""
    assert layer.K == 1
    z1 = table[x]                                                             # (B, L, 512), fixed
    B, L = x.shape
    Wi, bi = m.rnn.weight_ih_l0, m.rnn.bias_ih_l0
    gi = lin(z1, arch26.Mat(Wi, bi), local) if iface.k_in == D else E49.masked(z1, Wi, bi, iface.mask_in.repeat(3, 1).float(), local)
    Wh, bh = m.rnn.weight_hh_l0, m.rnn.bias_hh_l0
    scale = layer.sign_scale(m) if (local and not layer.full) else None
    h = T.zeros(B, U); hs = []
    for t in range(L):
        gh = E42.rec(layer, m, h, Wh, bh, local, scale)
        i_r, i_z, i_n = gi[:, t].chunk(3, 1); h_r, h_z, h_n = gh.chunk(3, 1)
        r = T.sigmoid(i_r + h_r); z = T.sigmoid(i_z + h_z); n = T.tanh(i_n + r * h_n)
        h = (1 - z) * n + z * h
        hs.append(h)
    Hs = T.stack(hs, 1)
    if silence:
        Hs = T.zeros_like(Hs)
    out = lin(Hs, m.lm, local) if iface.k_out == U else E49.masked(Hs, m.lm.weight, m.lm.bias, iface.mask_out.float(), local)
    z2 = T.tanh(lin(z1, m.l2, local) + out)
    return lin(z2, m.l3, local), hs


def what_cells_see(iface, owner):
    """For each cell, the set of bytes for which it receives a non-zero input."""
    return [sorted({owner[v] for v in iface.mask_in[i].nonzero().flatten().tolist() if owner[v] >= 0}) for i in range(U)]


if __name__ == "__main__":
    import hashlib, json
    T.set_num_threads(1); arch26.install(); E, X = C26.E, C26.X
    seed = 73129
    tp = [(t, E.split(t, .5, seed)[0]) for t in E.DEV]
    items = next(E.batches(tp, seed, E.KMAX["M"])); x, y, mask = X.batch_tensor(items)
    seen_in_data = sorted(set(x.unique().tolist()))
    m0 = arch26.new_net(seed); amp = amplitude_for(m0, seed); table, owner = code_table(seed, amp)
    print(f"1. used symbols {len(USED)}; every input symbol of the first batch is among them or is the end symbol, which stands only after an answer and has the zero code: {set(seen_in_data) <= set(USED) | {256}}; values owned {sum(o >= 0 for o in owner)} of {D};"
          f" each used byte turns on {int((table[USED] != 0).sum(1).min())} to {int((table[USED] != 0).sum(1).max())} values; no value has two owners: {bool(((table != 0).sum(0) <= 1).all())};"
          f" amplitude {amp:.4f}; digest of the code {hashlib.sha256(json.dumps(owner).encode()).hexdigest()[:16]}")
    for k_in, k_out in ((D, U), (64, 32), (16, 8)):
        m = arch26.new_net(seed); lay = E42.Layer(m, seed, N=10); f = E49.Interface(m, seed, k_in, k_out)
        see = what_cells_see(f, owner)
        n_all = sorted(len(s) for s in see); n_need = sorted(len(set(s) & set(NEEDED)) for s in see); n_dig = sorted(len(set(s) & set(b"0123456789")) for s in see)
        cells_per_byte = sorted(sum(b in s for s in see) for b in NEEDED)
        res = {}
        for local in (False, True):
            for q in m.parameters(): q.grad = None
            lg = forward(m, lay, f, table, x, local)[0]; loss = E.cross_entropy(lg[mask], y[mask]); loss.backward()
            unused = m.book.weight.grad is None and m.l1.weight.grad is None
            outside = bool((m.rnn.weight_ih_l0.grad[~f.mask_in.repeat(3, 1)] == 0).all()) and bool((m.lm.weight.grad[~f.mask_out] == 0).all())
            res[local] = (float(loss), unused, outside, bool(T.isfinite(lg).all()))
        active = (3 * U * k_in * ON) // D if k_in < D else 3 * U * ON
        print(f"2. K_IN {k_in}, K_OUT {k_out}: a cell receives input for {n_all[0]} to {n_all[-1]} of the 46 used bytes (median {n_all[U // 2]}), for {n_need[0]} to {n_need[-1]} of the 20 it must tell apart"
              f" (median {n_need[U // 2]}), for {n_dig[0]} to {n_dig[-1]} of the ten digits (median {n_dig[U // 2]}); cells with no input at all {sum(1 for s in see if not s)};"
              f" each needed byte reaches {cells_per_byte[0]} to {cells_per_byte[-1]} cells; first-batch loss B {res[False][0]:.3f} S {res[True][0]:.3f};"
              f" byte table and l1 without gradient {res[False][1] and res[True][1]}; gradients zero outside the masks {res[False][2] and res[True][2]}; finite {res[False][3] and res[True][3]};"
              f" input products that are not zero at a position, about {active:,}")
