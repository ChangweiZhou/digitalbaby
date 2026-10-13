"""Engineering transfer of three declared mechanisms. No arithmetic evaluator.

ByteCore owns payload admission, framing, ordered hashing and autoregression.
This subclass changes representation/routing/writing only. Targets enter AFTER
prediction, and guesses never call observe. Fixed arrays share birth seeds,
not object identities. This is not an unchanged native Full151 implementation.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'generic_byte_io_v1_20261012'))
from generic_byte_io import ByteCore, CoreConfig, Prediction, EOS

CONFIGS = tuple(f'{g}-{r}-{c}' for g in ('LINEAR', 'COMP')
                for r in ('FIXED', 'LOCAL') for c in ('DIAG', 'FULL')) + ('HASH',)
BASELINE = 'LINEAR-FIXED-FULL'


def unit(x):
    norm = float(np.linalg.norm(x))
    return x / norm if norm else x.copy()


class MechanismCore(ByteCore):
    def __init__(self, name, seed):
        if name not in CONFIGS or type(seed) is not int or seed < 0:
            raise ValueError('invalid configuration or birth seed')
        super().__init__(CoreConfig(address_seed=seed))
        self.name, self.seed = name, seed
        embed_rng, address_rng, router_rng = [np.random.default_rng(s)
            for s in np.random.SeedSequence(seed).spawn(3)]
        self.embedding = embed_rng.choice((-1., 1.), size=(256, 8)) / np.sqrt(8)
        self.A = address_rng.exponential(1., (96, 192))
        self.A *= address_rng.random(self.A.shape) < .18
        norms = np.linalg.norm(self.A, axis=1, keepdims=True)
        self.A /= np.where(norms > 0, norms, 1.)
        # Distinct random prototypes; FIXED/LOCAL births are IDENTICAL to each
        # other, not four duplicate prototypes (which would route all to bank 0).
        centers = router_rng.normal(size=(4, 96))
        self.centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
        self.V = np.zeros((4, 96, 257), dtype=np.float64)
        full = name.endswith('FULL')
        self.P = (np.repeat((10. * np.eye(96))[None], 4, axis=0)
                  if full else np.full((4, 96), 10.))
        self.counts = np.zeros(4, dtype=np.int64)
        self._question = bytearray()
        self._question_total = 0
        self._reply_offset = 0
        self._previous = None
        self._in_reply = False
        self._bank = 0
        self._route_h = np.zeros(96)

    def begin_sequence(self):
        super().begin_sequence()
        self._question.clear()
        self._question_total = 0
        self._reply_offset = 0
        self._previous = None
        self._in_reply = False

    def _append(self, event):
        super()._append(event)
        if 0 <= event < 256:
            if self._in_reply:
                self._previous = event
                self._reply_offset += 1
            else:
                self._question_total += 1
                self._question.append(event)
                # Shared sensory state follows the visible rolling window.
                excess = len(self._question) - self.config.max_context_events
                if excess > 0:
                    del self._question[:excess]

    def _features(self, routing=False):
        h = np.zeros(96)
        # An early prompt byte must actually still exist in the bounded window.
        after = self._reply_offset + 1 if self._in_reply or routing else 0
        available = max(0, min(len(self._question),
                               self.config.max_context_events - after))
        departed = len(self._question) - available
        origin = self._question_total - len(self._question)
        for local, byte in enumerate(self._question[:max(0, 8-origin)]):
            i = origin + local
            if local < departed:
                continue
            h[8*i:8*i+8] = self.embedding[byte]
            if 48 <= byte <= 57:
                h[64+i] = (byte - 48) / 9.
        if not routing:
            h[72 + min(self._reply_offset, 7)] = 1.
            if self._previous is not None:
                h[80:88] = self.embedding[self._previous]
        tail = bytes(self._question[max(0, 8-origin, departed):])
        if tail:
            raw = hashlib.shake_256(f'TAIL/{self.seed}/'.encode() + tail).digest(8)
            h[88:96] = (np.frombuffer(raw, dtype=np.uint8).astype(float) / 127.5 - 1.)
        return unit(h)

    def _encode(self, h):
        if self.name.startswith('COMP'):
            positive = np.concatenate((np.maximum(h, 0), np.maximum(-h, 0)))
            activation = self.A @ unit(positive)
            indices = np.argsort(activation, kind='stable')[-8:]
            code = np.zeros(96)
            code[indices] = activation[indices]
            return unit(code)
        return unit((self.A[:, :96] - self.A[:, 96:]) @ h)

    def start_reply(self):
        if self._in_reply:
            raise RuntimeError('duplicate reply boundary')
        super().start_reply()
        if self.name == 'HASH':
            self._in_reply = True
            self._reply_offset = 0
            self._previous = None
            return
        self._route_h = self._features(routing=True)
        self._bank = int(np.argmax(self.centers @ self._route_h))
        self._in_reply = True
        self._reply_offset = 0
        self._previous = None

    def _terms(self, address):
        u = self._encode(self._features())
        z = .25 * np.sum(self._weights[address], axis=0, dtype=np.float64)
        z += u @ self.V[self._bank]
        return u, z

    def _prediction_at(self, address):
        if self.name == 'HASH':
            return super()._prediction_at(address)
        _, z = self._terms(address)
        probabilities = np.exp(z - np.max(z))
        probabilities /= probabilities.sum()
        return Prediction(probabilities, int(np.argmax(probabilities)))

    def _observe_event(self, event, learn):
        if self.name == 'HASH':
            return super()._observe_event(event, learn)
        self._require_open()
        address = self._address()
        u, z = self._terms(address)
        probabilities = np.exp(z - np.max(z))
        probabilities /= probabilities.sum()
        prediction = Prediction(probabilities, int(np.argmax(probabilities)))
        if learn:
            if not self._in_reply:
                raise RuntimeError('this transfer runner learns observed replies only')
            q = (self.P[self._bank] @ u if self.name.endswith('FULL')
                 else self.P[self._bank] * u)
            d = 2. + float(u @ q)  # ||x||^2 = 16 * .25^2 = 1
            error = -z.copy()
            error[event] += 1.
            self._weights[address] += (.25 * error / d).astype(np.float32)
            self.V[self._bank] += np.outer(q, error) / d
            if self.name.endswith('FULL'):
                self.P[self._bank] -= np.outer(q, q) / d
                self.P[self._bank] = .5 * (self.P[self._bank] + self.P[self._bank].T)
            else:
                self.P[self._bank] -= q*q / d
            self.learning_events += 1
            if event == EOS:
                self.counts[self._bank] += 1
                if '-LOCAL-' in self.name:
                    eta = min(.05, 1. / int(self.counts[self._bank]))
                    self.centers[self._bank] = unit(
                        (1.-eta)*self.centers[self._bank] + eta*self._route_h)
        self._append(event)
        return prediction

    def persistent_components(self):
        arrays = {'W': self._weights}
        if self.name != 'HASH':
            arrays.update(V=self.V, P=self.P, centers=self.centers, counts=self.counts)
        return {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in arrays.items()}

    def learned_digest(self):
        obj = self.persistent_components()
        obj['learning_events'] = self.learning_events
        return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()

    def birth_digest(self):
        return hashlib.sha256(self.A.tobytes() + self.embedding.tobytes()
                              + self.centers.tobytes()).hexdigest()

    def storage_accounting(self):
        fixed = self.A.nbytes + self.embedding.nbytes
        shared = self.V.nbytes + self.P.nbytes + self.centers.nbytes + self.counts.nbytes
        if self.name == 'HASH':
            # Auxiliary arrays are absent from HASH production and learned state.
            fixed = shared = 0
        return {'fixed_array_bytes': fixed, 'learned_array_bytes': self._weights.nbytes + shared,
                'working_context_bytes': self._context.nbytes + len(self._question),
                'allocated_array_bytes': sum(a.nbytes for a in
                    (self.A, self.embedding, self.V, self.P, self.centers, self.counts,
                     self._weights, self._context)),
                'context_bound': self.config.max_context_events, 'payload_categories': 256,
                'output_categories': 257, 'generation_cap': 16,
                'production_work': 'context hash + 96x192 projection + 96x257 read/write; FULL adds 96x96',
                'audit_digest_work_charged_separately': True}

    def save_checkpoint(self, path):
        extra = {'name': self.name, 'seed': self.seed, 'in_reply': self._in_reply,
                 'reply_offset': self._reply_offset, 'previous': self._previous,
                 'question_total': self._question_total,
                 'bank': self._bank, 'base': self._metadata(), 'schema': 'ARITHMETIC_TRANSFER_V2'}
        with Path(path).open('wb') as out:
            np.savez_compressed(out, metadata=np.frombuffer(json.dumps(extra).encode(), np.uint8),
                weights=self._weights, context=self._context, A=self.A, embedding=self.embedding,
                V=self.V, P=self.P, centers=self.centers, counts=self.counts,
                question=np.frombuffer(bytes(self._question), np.uint8), route_h=self._route_h)

    @classmethod
    def load_checkpoint(cls, path):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(data['metadata'].tobytes())
            if meta['schema'] != 'ARITHMETIC_TRANSFER_V2':
                raise ValueError('checkpoint schema')
            c = cls(meta['name'], meta['seed'])
            for key, attr in [('weights', '_weights'), ('context', '_context'), ('A', 'A'),
                              ('embedding', 'embedding'), ('V', 'V'), ('P', 'P'),
                              ('centers', 'centers'), ('counts', 'counts'), ('route_h', '_route_h')]:
                arr = data[key]
                expected = getattr(c, attr)
                if arr.shape != expected.shape or arr.dtype != expected.dtype or not np.isfinite(arr).all():
                    raise ValueError(f'invalid checkpoint array {key}')
                expected[:] = arr
            base = meta['base']
            for key, attr in [('context_len', '_context_len'), ('next', '_next'), ('open', '_open'),
                              ('generated_end', '_generated_end')]:
                setattr(c, attr, base[key])
            for key in ('learning_events', 'observed_payload_bytes', 'generated_payload_bytes',
                        'context_overflow_events', 'elapsed_seconds'):
                setattr(c, key, base[key])
            c._question = bytearray(data['question'].tobytes())
            c._question_total = meta['question_total']
            c._in_reply, c._reply_offset = meta['in_reply'], meta['reply_offset']
            c._previous, c._bank = meta['previous'], meta['bank']
            c._route_h[:] = data['route_h']
            if not 0 <= c._bank < 4 or not 0 <= c._context_len <= len(c._context):
                raise ValueError('checkpoint bounds')
            return c
