"""Corpus stub learner for the MRK-1 regression corpus.

Parsed by the checker (closure, purity, LOCK-08); never executed.
"""
import hashlib


class Learner(object):
    def __init__(self, params):
        self.params = dict(params)
        self.pairs = {}
        self.log = {'pair_memory': [], 'readout': []}

    def write_pair(self, prompt, answer, channel):
        """State-mutation site: every write to a channel is logged here (write_counter_site)."""
        self.pairs[prompt] = answer
        self.log[channel].append(prompt)

    def writes_by_channel(self):
        return {name: len(rows) for name, rows in self.log.items()}

    def generate(self, prompt, cap):
        out = self.pairs.get(prompt, b'')
        return out.ljust(cap, b'\n')[:cap]

    def digest(self, channel):
        data = repr(sorted(self.log[channel])).encode('utf-8')
        return hashlib.sha256(data).hexdigest()
