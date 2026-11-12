"""Sizing sweep, stage 1: the oracle, parameterized (a copy of ../target_oracle.py).

It is given one explicit configuration (it reads only cfg["modulus"]), the hidden values, the change point, and for each probe its family,
context, cue and position. It reads nothing from any stream and imports nothing from the generator or from any other module of the sweep.
"""
import hashlib
import json


class Oracle:
    def __init__(self, cfg, v, u_old, u_new, change_at):
        if "modulus" not in cfg:
            raise ValueError("sz_oracle: configuration lacks the modulus")
        self.cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
        self.M = int(cfg["modulus"])
        self.v, self.u_old, self.u_new, self.change_at = v, u_old, dict(u_new), change_at

    def answer(self, f, x, c, at):
        after = at >= self.change_at
        u = self.u_new[c] if (f == 0 and after and c in self.u_new) else self.u_old[f][c]
        return (self.v[f][x] + u) % self.M if f == 0 else (self.v[f][x] - u) % self.M
