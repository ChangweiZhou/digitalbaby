import hashlib
import numpy as np

VERSION = "F151-BYTE-V9.2"
HASH_SEED = b"V9.2-context-pilot-20260925"
ROWS = 2816
OUTPUTS = 256
RATES = (0.03, 0.03, 0.10, 0.10)
BIAS_RATE = 0.001


def _hash_pair(context: bytes, depth: int, buckets: int) -> tuple[int, int]:
    raw = hashlib.blake2s(HASH_SEED + bytes((depth,)) + context,
                          digest_size=8).digest()
    return (int.from_bytes(raw[:4], "little") % buckets,
            int.from_bytes(raw[4:], "little") % buckets)


class F151ByteBrainV92:
    def _features(self) -> tuple[tuple[int, float], ...]:
        n = len(self.history)
        out: list[tuple[int, float]] = []
        if n >= 1:
            out.append((int(self.history[-1]), RATES[0]))
        for depth, first, buckets, rate in (
                (2, 256, 256, RATES[1]),
                (3, 768, 512, RATES[2]),
                (4, 1792, 512, RATES[3])):
            if n >= depth:
                a, b = _hash_pair(bytes(self.history[-depth:]), depth, buckets)
                out.extend(((first + a, rate), (first + buckets + b, rate)))
        return tuple(out)

    def _distribution(self, *, lesion_temporal: bool = False) -> np.ndarray:
        logits = self.bias.astype(np.float64)
        if not lesion_temporal:
            for row, _ in self._features():
                logits += self.w[row].astype(np.float64)
        logits -= float(logits.max())
        p = np.exp(logits)
        p /= float(p.sum())
        return p

    def predict(self):
        raise NotImplementedError
