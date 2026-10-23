"""MiniFly-CLS: the native mushroom body as fast memory plus a slow learned core, coupled by sleep replay.

wake(record)   the fast memory stores the record at once (native error-corrective write); the record enters a bounded
               episode cache (`replay="reservoir"`: uniform reservoir over everything experienced, so old tasks stay
               represented; `replay="recent"`: only the most recent `cache_size` records, the no-replay control).
sleep(steps)   the slow core is trained on batches replayed from the cache, with decoupled weight decay as the
               downscaling pressure; `mem_replay` cached records are also re-presented to the fast memory (its own
               error-corrective write, so already-correct facts are barely touched); then the fast memory rests for
               `sleep_seconds` of brain time.
answer(prompt) both systems answer; each answer's confidence (mean top-2 margin) is divided by that system's median
               confidence on cached (experienced) prompts; the higher relative confidence wins. Only experienced
               records enter the calibration; never-taught prompts are never used for it.
"""
import random

import numpy as np

import fastmem
import slowcore


class CLSFly:
    def __init__(self, seed, mem_kwargs=None, slow_kwargs=None, cache_size=160, replay="reservoir",
                 replay_batch=80, sleep_seconds=8 * 3600., mem_replay=0):
        if replay not in ("reservoir", "recent"):
            raise ValueError("replay in {reservoir, recent}")
        self.mem = fastmem.FastMemory(**(mem_kwargs or {}))
        self.slow = slowcore.SlowCore(seed, **(slow_kwargs or {}))
        self.cache, self.cache_size, self.replay = [], cache_size, replay
        self.replay_batch, self.sleep_seconds, self.mem_replay = replay_batch, sleep_seconds, mem_replay
        self.rng = random.Random(seed)
        self.seen = 0
        self.calibration = None

    def wake(self, record):
        self.mem.train(*record)
        self.seen += 1
        if self.replay == "recent":
            self.cache.append(record)
            del self.cache[:-self.cache_size]
        elif len(self.cache) < self.cache_size:
            self.cache.append(record)
        else:
            j = self.rng.randrange(self.seen)
            if j < self.cache_size:
                self.cache[j] = record
        self.calibration = None

    def sleep(self, steps):
        loss = None
        for _ in range(steps):
            k = min(self.replay_batch, len(self.cache))
            loss = self.slow.step(self.rng.sample(self.cache, k))
        for record in self.rng.sample(self.cache, min(self.mem_replay, len(self.cache))):
            self.mem.train(*record)
        self.mem.rest(self.sleep_seconds)
        self.calibration = None
        return loss

    def _calibrate(self):
        prompts = list({p: None for p, _ in self.cache})[:40]
        self.calibration = (float(np.median([self.mem.generate(p)[1] for p in prompts])) or 1.,
                            float(np.median([self.slow.generate(p)[1] for p in prompts])) or 1.)

    def answer(self, prompt):
        if self.calibration is None:
            self._calibrate()
        (m_ans, m_conf), (s_ans, s_conf) = self.mem.generate(prompt), self.slow.generate(prompt)
        use_mem = m_conf / self.calibration[0] >= s_conf / self.calibration[1]
        return dict(combined=m_ans if use_mem else s_ans, mem=m_ans, slow=s_ans, used_mem=bool(use_mem))

    def score(self, records):
        out = [self.answer(p) for p, _ in records]
        acc = {k: float(np.mean([o[k] == bytes(a) for o, (_, a) in zip(out, records)])) for k in ("combined", "mem", "slow")}
        acc["used_mem"] = float(np.mean([o["used_mem"] for o in out]))
        return acc
