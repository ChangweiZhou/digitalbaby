"""Sizing sweep, stage 1: the prefix-only solver, parameterized (a copy of ../target_solver.py; checks 1, 3 and 9).

Changes from target_solver.py: the modulus and the counts come from one explicit configuration passed to the constructor (it reads
cfg["modulus"], cfg["contexts"], cfg["cues_per_family"]); feed() splits its buffer once per call instead of once per line (same tokens,
linear rather than quadratic in the bytes fed, needed at 256 presentations). Nothing else changed.

Step 0 docstring:

It is given bytes and, as evaluator-only metadata, the change point (a byte offset). It is never given the hidden values, and it imports
nothing from the generator. What it knows is the stream's public format and the rule's form: family 1 (cues a-l) outcome = v1(X) + u1(c),
family 2 (cues m-x) outcome = v2(X) - u2(c), modulo 97, and that at the change point cue values may change and context values do not.
It does not know which cues changed, so it keeps the old and the current value of every cue apart (more cautious than the brief needs).

Each revealed episode and each disclosure line is a linear equation over Z_97 in the unknowns; an answer is determined when its equation
lies in the span of what the prefix holds (Gaussian elimination, exact). With shared_context_values=True (a fixture world only) v2 = v1.
"""
import hashlib
import json

FILL = b"!#%&*+,-"


class PrefixSolver:
    def __init__(self, cfg, change_at=None, shared_context_values=False):
        for k in ("modulus", "contexts", "cues_per_family"):
            if k not in cfg:
                raise ValueError(f"sz_solver: configuration lacks {k}")
        self.cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
        self.M, self.NCTX, self.NCUE = int(cfg["modulus"]), int(cfg["contexts"]), int(cfg["cues_per_family"])
        self.NV = 2 * self.NCTX                  # v[f][x]
        self.N = self.NV + 2 * 2 * self.NCUE     # u[era][f][c]
        self.change_at, self.shared = change_at, shared_context_values
        self.rows = []                 # (pivot column, row with N + 1 entries, pivot entry 1)
        self.seen = set()
        self.pos = 0                   # bytes consumed
        self.buf = b""
        self.cue_last_context = {}     # (f, c) -> the context most recently seen with that cue (for the one-answer-per-suffix guess)
        self.inconsistent = 0

    # -- unknowns
    def _v(self, f, x):
        return (0 if self.shared else f) * self.NCTX + x

    def _u(self, era, f, c):
        return self.NV + era * 2 * self.NCUE + f * self.NCUE + c

    def era_at(self, offset):
        return 1 if (self.change_at is not None and offset >= self.change_at) else 0

    # -- linear algebra over Z_97
    def _reduce(self, row):
        for p, r in self.rows:
            if row[p]:
                k = row[p]
                row = [(a - k * b) % self.M for a, b in zip(row, r)]
        return row

    def _add(self, coeffs, rhs):
        key = (tuple(sorted(coeffs.items())), rhs)
        if key in self.seen:
            return
        self.seen.add(key)
        M, N = self.M, self.N
        row = [0] * (N + 1)
        for j, a in coeffs.items():
            row[j] = (row[j] + a) % M
        row[N] = rhs % M
        row = self._reduce(row)
        p = next((j for j in range(N) if row[j]), None)
        if p is None:
            if row[N]:
                self.inconsistent += 1
            return
        inv = pow(row[p], M - 2, M)
        self.rows.append((p, [a * inv % M for a in row]))

    # -- reading bytes
    def feed(self, data):
        self.buf += data
        if b"\n" not in self.buf:
            return
        *toks, self.buf = self.buf.split(b"\n")
        for tok in toks:
            start = self.pos; self.pos += len(tok) + 1
            k = 0
            while k < len(tok) and tok[k] in FILL:         # extra ordinary bytes before an episode
                k += 1
            self._token(tok[k:], start + k)

    def _token(self, t, offset):
        if not t:
            return
        era = self.era_at(offset)
        x = t[0] - 65
        assert 0 <= x < self.NCTX, t
        if len(t) >= 2 and t[1:2] == b":":                 # disclosure line  X:vv c:uu
            vv, cb, uu = int(t[2:4]), t[5], int(t[7:9])
            f, c = (cb - 97) // self.NCUE, (cb - 97) % self.NCUE
            self._add({self._v(f, x): 1}, vv); self._add({self._u(era, f, c): 1}, uu)
            return
        q = t.index(b"=")                                  # episode  X filler c = dd
        cb = t[q - 1]; f, c = (cb - 97) // self.NCUE, (cb - 97) % self.NCUE; o = int(t[q + 1:q + 3])
        self._add({self._v(f, x): 1, self._u(era, f, c): 1 if f == 0 else self.M - 1}, o)
        self.cue_last_context[(f, c)] = x

    # -- answers
    def query(self, f, x, c, era=None):
        """The outcome of (x, c) in family f at the current position (or the given era), or None if the prefix does not fix it."""
        era = self.era_at(self.pos + len(self.buf)) if era is None else era      # every byte given so far, parsed or not
        M, N = self.M, self.N
        row = [0] * (N + 1)
        row[self._v(f, x)] = 1; row[self._u(era, f, c)] = 1 if f == 0 else M - 1
        row = self._reduce(row)
        if any(row[:N]):
            return None
        return (-row[N]) % M

    def answer_prompt(self, prompt):
        """Answer a probe from the bytes of its prompt (ending at '='). With a context byte in the prompt, that context is used. Without one
        (only the shared ten-byte suffix), the solver can only make one guess per suffix: the context it last saw with this cue."""
        q = prompt.rindex(b"=")
        cb = prompt[q - 1]; f, c = (cb - 97) // self.NCUE, (cb - 97) % self.NCUE
        caps = [b for b in prompt[:q] if 65 <= b < 65 + self.NCTX]
        x = caps[-1] - 65 if caps else self.cue_last_context.get((f, c))
        if x is None:
            return None
        return self.query(f, x, c)
