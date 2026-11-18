"""Checks for the audit of the two E35 briefs (commit 907c9fc4). numpy only; nothing is trained.

a. Inventory jia: the four fixed operations, written from the brief's formulas, on all 7,921 pairs of one-hot codes.
b. Ways the learned interfaces could use each module as a plain conduit for an answer the controller already holds
   (so that "uses the module" by ablation would be met without using the structure).
c. Inventory yi: the two primitives with an exact phase code, and with a code that is NOT a homomorphism (random angles):
   how many distinct result codes the read-out must then tell apart.
d. Whether every operand value appears in both slots among the taught pairs at fractions 0.10 and 0.25 (exp11_cell.split, re-derived).
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from np_gru import split, DEV
N = 89; I = np.arange(N); eye = np.eye(N)
def op_add(a, b): return np.einsum('bi,bki->bk', a, b[:, (I[:, None] - I[None, :]) % N])
def op_sub(a, b): return np.einsum('bi,bki->bk', a, b[:, (I[None, :] - I[:, None]) % N])
def op_max(a, b): F = a.cumsum(1) * b.cumsum(1); return np.concatenate([F[:, :1], F[:, 1:] - F[:, :-1]], 1)
def op_min(a, b): S = a[:, ::-1].cumsum(1)[:, ::-1] * b[:, ::-1].cumsum(1)[:, ::-1]; return np.concatenate([S[:, :-1] - S[:, 1:], S[:, -1:]], 1)
A, Bv = np.meshgrid(I, I, indexing='ij'); A, Bv = A.ravel(), Bv.ravel(); a, b = eye[A], eye[Bv]
print('a. jia, exact on all', len(A), 'pairs:', {n: bool((f(a, b).argmax(1) == t).all() and np.allclose(f(a, b).max(1), 1)) for n, f, t in
      (('add', op_add, (A + Bv) % N), ('sub', op_sub, (A - Bv) % N), ('max', op_max, np.maximum(A, Bv)), ('min', op_min, np.minimum(A, Bv)))})
print('b. conduits:')
print('   jia: max(A, code of 0) returns A exactly:', bool(np.allclose(op_max(eye, np.repeat(eye[:1], N, 0)), eye)), '| add(A, neutral) is uniform whatever A is:', bool(np.allclose(op_add(eye, np.full((N, N), 1 / N)), 1 / N)))
rng = np.random.default_rng(35); tab = rng.integers(0, N, (4, N, N))
print('   jia scrambled: columns of a random table that are one-to-one in the other operand:', int(sum(len(set(tab[j][:, k])) == N for j in range(4) for k in range(N))), 'of', 4 * N,
      '| mean distinct outputs per column:', round(float(np.mean([len(set(tab[j][:, k])) for j in range(4) for k in range(N)])), 1), 'of 89 | agreement with addition:', f"{100 * np.mean(tab[0] == (I[:, None] + I[None, :]) % N):.1f}%")
def mul(p, q): return np.stack([p[..., 0] * q[..., 0] - p[..., 1] * q[..., 1], p[..., 0] * q[..., 1] + p[..., 1] * q[..., 0]], -1)
def code(ang): return np.stack([np.cos(ang), np.sin(ang)], -1)
neutral = code(np.zeros((1, 8)))
x = code(rng.uniform(0, 6.28, (50, 8)))
print('   yi: mul(A, neutral) returns A exactly (register B never written):', bool(np.allclose(mul(x, neutral), x)))
f = np.array([1, 2, 3, 5, 7, 11, 13, 17]); exact = code(2 * np.pi * f[None, :] * I[:, None] / N)
r = mul(exact[A], exact[Bv]); d = ((r[:, None] - exact[None]) ** 2).sum((2, 3)).argmin(1)
print('c. yi, exact phase code: mul decodes to add on all pairs:', bool((d == (A + Bv) % N).all()), '| distinct result codes:', len(np.unique(np.round(r.reshape(len(A), -1), 6), axis=0)))
rnd = code(rng.uniform(0, 2 * np.pi, (N, 8))); r2 = mul(rnd[A], rnd[Bv])
print('   yi, random angles (not a homomorphism): distinct result codes the read-out must separate:', len(np.unique(np.round(r2.reshape(len(A), -1), 6), axis=0)), '(ordered pairs; 4,005 if unordered pairs coincide)')
print('d. taught pairs per task and operand values missing from a slot (seed 73129):')
for frac in (0.10, 0.25):
    for t in DEV[:2]:
        taught = split(t, frac, 73129)[0]
        print(f"   f = {frac}: {t[0]}89: {len(taught)} taught pairs; values never first: {N - len({p[0] for p in taught})}, never second: {N - len({p[1] for p in taught})}; fewest appearances of a value in either slot: {min(min(sum(p[s] == v for p in taught) for v in range(N)) for s in (0, 1))}")
