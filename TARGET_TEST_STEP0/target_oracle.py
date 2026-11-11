"""Target test, step 0: the oracle (check 2).

It is given the hidden values (context values, cue values before and after the change), the change point, and for each probe its family,
context, cue and position. It reads nothing from any stream and imports nothing from the generator.
"""
M = 97


class Oracle:
    def __init__(self, v, u_old, u_new, change_at):
        """v[f][x]; u_old[f][c]; u_new = {family-1 cue: value after the change}; change_at = the first episode index after the change."""
        self.v, self.u_old, self.u_new, self.change_at = v, u_old, dict(u_new), change_at

    def answer(self, f, x, c, at):
        """The outcome of (x, c) in family f (0 = the first family) for a probe standing at episode index `at` (before that episode is
        revealed); the episode at the change index and every later one belong to the new era."""
        after = at >= self.change_at
        u = self.u_new[c] if (f == 0 and after and c in self.u_new) else self.u_old[f][c]
        return (self.v[f][x] + u) % M if f == 0 else (self.v[f][x] - u) % M
