"""Coordinator check for the E35 briefs (15 Nov 2026). No training. The fixed operations of inventory A (E10's ring operations at 89
positions) and inventory B (phase multiplication and a scalar comparator), exactly as the briefs write them, tested exhaustively on exact
codes, and their scrambled controls.   python3 -I innate_ops_check.py"""
import math
import torch as T

MOD = 89
IDX = T.arange(MOD)

# ---------------- inventory A: distributions over 89 positions (exp10_ring.py lines 43-61 with MOD = 89)
def a_add(a, b): return T.einsum("bi,bki->bk", a, b[:, (IDX[:, None] - IDX[None, :]) % MOD])
def a_sub(a, b): return T.einsum("bi,bki->bk", a, b[:, (IDX[None, :] - IDX[:, None]) % MOD])
def a_max(a, b):
    F = a.cumsum(1) * b.cumsum(1); return T.cat([F[:, :1], F[:, 1:] - F[:, :-1]], 1)
def a_min(a, b):
    S = a.flip(1).cumsum(1).flip(1) * b.flip(1).cumsum(1).flip(1); return T.cat([S[:, :-1] - S[:, 1:], S[:, -1:]], 1)

def a_table(tab):                      # scrambled control: a fixed table tab[i, j] -> k in place of an operation
    flat = tab.reshape(-1)
    def f(a, b):
        outer = (a[:, :, None] * b[:, None, :]).reshape(len(a), -1)      # B x 7921
        return T.zeros(len(a), MOD, dtype=a.dtype).index_add(1, flat, outer)
    return f

# ---------------- inventory B: K unit phasors (re, im) and one scalar
def b_mul(a, b):                       # (B, K, 2) x (B, K, 2): complex product, angles add
    return T.stack([a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1], a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0]], -1)
def b_div(a, b):                       # product with the conjugate, angles subtract
    return T.stack([a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1], a[..., 1] * b[..., 0] - a[..., 0] * b[..., 1]], -1)
def b_max(a, b, sa, sb):                # comparator routes a whole code: the code whose scalar is larger passes
    g = T.sigmoid(sa - sb)[:, None, None]; return g * a + (1 - g) * b
def b_min(a, b, sa, sb):
    g = T.sigmoid(sa - sb)[:, None, None]; return (1 - g) * a + g * b

pa, pb = T.meshgrid(IDX, IDX, indexing="ij"); pa, pb = pa.reshape(-1), pb.reshape(-1)
A, B = T.eye(MOD)[pa], T.eye(MOD)[pb]
want = {"add": (pa + pb) % MOD, "sub": (pa - pb) % MOD, "max": T.maximum(pa, pb), "min": T.minimum(pa, pb)}
for n, f in (("add", a_add), ("sub", a_sub), ("max", a_max), ("min", a_min)):
    out = f(A, B)
    print("A", n, "exact on all 7921 pairs:", bool((out.argmax(1) == want[n]).all()), "| mass on the answer min %.6f" % float(out.gather(1, want[n][:, None]).min()))
g = T.Generator().manual_seed(35)
tab = T.randint(0, MOD, (MOD, MOD), generator=g)
out = a_table(tab)(A, B)
print("A scrambled table: output is one-hot:", bool((out.max(1).values == 1).all()), "| agrees with add on %.1f%% of pairs" % float(100 * (out.argmax(1) == want["add"]).float().mean()))
ok = a_table((IDX[:, None] + IDX[None, :]) % MOD)(A, B)
print("A table form with the true add table equals a_add:", bool(T.allclose(ok, a_add(A, B))))

K = 4; freqs = T.tensor([1., 2., 3., 5.])
def enc(v): ang = 2 * math.pi * v[:, None].float() * freqs[None] / MOD; return T.stack([ang.cos(), ang.sin()], -1)
def dec(z):                            # nearest code among the 89 exact codes
    codes = enc(IDX).reshape(MOD, -1); return ((z.reshape(len(z), -1)[:, None] - codes[None]) ** 2).sum(-1).argmin(1)
print("B mul decodes to add on all pairs:", bool((dec(b_mul(enc(pa), enc(pb))) == want["add"]).all()))
print("B div decodes to sub on all pairs:", bool((dec(b_div(enc(pa), enc(pb))) == want["sub"]).all()))
s, t = 10. * pa.float(), 10. * pb.float()
print("B max/min by comparator routing (scalar = 10 x the number) decode exactly:", bool((dec(b_max(enc(pa), enc(pb), s, t)) == want["max"]).all() and (dec(b_min(enc(pa), enc(pb), s, t)) == want["min"]).all()))
W = T.randn(2 * K, 2 * K, 2 * K, generator=g) / (2 * K)
sc = T.einsum("bi,bj,ijk->bk", enc(pa).reshape(-1, 2 * K), enc(pb).reshape(-1, 2 * K), W)
print("B scrambled bilinear map: output norm mean %.3f (unit phasors have norm %.3f)" % (float(sc.norm(dim=1).mean()), math.sqrt(K)))
