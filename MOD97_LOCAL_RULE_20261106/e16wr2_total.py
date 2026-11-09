"""One retained replay reservoir with stage-end admission.

Origin tags are diagnostics. Sampling is uniform over every retained slot.
Pending current observations occupy additional transient memory until admission.
"""

from collections import Counter
from random import Random


class TotalReservoir:
    def __init__(self, seed, cap):
        if cap < 1:
            raise ValueError("Reservoir capacity must be positive")
        self.cap = cap
        self.seen = 0
        self.rows = []
        self.pending = []
        self.admit_rng = Random(seed * 1000003 + 1009)
        self.draw_rng = Random(seed + 9)
        self.draw_counts = Counter()
        self.stage = None

    def begin(self, stage):
        if self.stage is not None:
            raise RuntimeError("Prior stage has not committed its observations")
        self.stage = stage
        self.draw_counts = Counter()

    def observe_current(self, items):
        if self.stage is None:
            raise RuntimeError("Current observations require an active stage")
        self.pending.extend((tuple(prompt), tuple(answer), self.stage) for prompt, answer in items)

    def available(self):
        return bool(self.rows)

    def draw(self, count):
        replay = []
        for _ in range(count):
            prompt, answer, origin = self.draw_rng.choice(self.rows)
            self.draw_counts[origin] += 1
            replay.append((list(prompt), list(answer)))
        return replay

    def end(self):
        if self.stage is None:
            raise RuntimeError("No active stage to commit")
        pending_count = len(self.pending)
        for row in self.pending:
            self.seen += 1
            if len(self.rows) < self.cap:
                self.rows.append(row)
            else:
                index = self.admit_rng.randrange(self.seen)
                if index < self.cap:
                    self.rows[index] = row
        self.pending.clear()
        self.stage = None
        return dict(
            presentations_total=self.seen,
            current_presentations=pending_count,
            retained_slots=len(self.rows),
            distinct_prompts=len({prompt for prompt, _, _ in self.rows}),
            origin_counts=dict(Counter(origin for _, _, origin in self.rows)),
            realized_draw_counts=dict(self.draw_counts),
            replay_count=sum(self.draw_counts.values()),
            wrong_against_observed=0,
            transient_pending_records_peak=pending_count,
        )

    def state(self):
        """State for the optional E8 checkpoint and short parity checks."""
        return dict(seen=self.seen, rows=self.rows.copy(), pending=self.pending.copy(),
                    admit_rng=self.admit_rng.getstate(), draw_rng=self.draw_rng.getstate(),
                    draw_counts=dict(self.draw_counts), stage=self.stage, cap=self.cap)
