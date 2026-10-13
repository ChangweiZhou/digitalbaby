"""ADAPT-2 engineered retained adapter. No evaluator, arithmetic or task ID.

Only external observed reply events train. All gradient directions precede
all writes. Frozen means frozen INTERNAL representation, not frozen readout.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT / 'vendor'))
from generic_byte_io import ByteCore, CoreConfig, Prediction, EOS
from vendor.legacy_core import MechanismCore as LegacyCore

CONFIGS = ('FROZEN-HASH', 'LEARN-HASH', 'FROZEN-SHARED', 'LEARN-SHARED', 'LEGACY-B')
CANDIDATES = ('LEARN-HASH', 'LEARN-SHARED')
BASELINE = 'LEGACY-B'
EPS = 1e-8
HEAD_RATE, ADAPTER_RATE, STEP_CAP = .6, .025, .1


def build_core(name, seed):
    if name == BASELINE:
        return LegacyCore('LINEAR-FIXED-FULL', seed)
    return AdapterCore(name, seed)


class AdapterCore(ByteCore):
    def __init__(self, name, seed):
        if name not in CONFIGS[:-1] or type(seed) is not int or seed < 0:
            raise ValueError('invalid adapter birth')
        self.name, self.seed = name, seed
        self.hash_enabled = name.endswith('HASH')
        self.internal_learning = name.startswith('LEARN')
        self.config = CoreConfig(address_seed=seed)
        # Do not call ByteCore.__init__: SHARED must never allocate the hash bank.
        if self.hash_enabled:
            self._weights = np.zeros((4096, 257), dtype=np.float32)
        self._context = np.zeros(4096, dtype=np.uint16)
        self._context_len = self._next = 0
        self._open = self._generated_end = False
        self.learning_events = self.observed_payload_bytes = self.generated_payload_bytes = 0
        self.context_overflow_events = 0
        self.elapsed_seconds = 0.
        erng, r1, r2 = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]
        self.embedding = erng.choice((-1., 1.), (256, 8)) / np.sqrt(8)
        bound = np.sqrt(6 / 192)
        self.A1, self.A2 = r1.uniform(-bound, bound, (96, 96)), r2.uniform(-bound, bound, (96, 96))
        self.b1, self.b2 = np.zeros(96), np.zeros(96)
        self.V, self.c = np.zeros((96, 257)), np.zeros(257)
        self._question = bytearray()
        self._question_total = self._reply_offset = 0
        self._previous = None
        self._in_reply = False

    _features = LegacyCore._features

    def begin_sequence(self):
        super().begin_sequence()
        self._question.clear()
        self._question_total = self._reply_offset = 0
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
                excess = len(self._question) - 4096
                if excess > 0:
                    del self._question[:excess]

    def start_reply(self):
        if self._in_reply:
            raise RuntimeError('duplicate reply boundary')
        super().start_reply()
        self._in_reply = True
        self._reply_offset = 0
        self._previous = None

    def _address(self):
        if not self.hash_enabled:
            raise RuntimeError('SHARED attempted hash-address access')
        return super()._address()

    def _forward(self, address=None):
        h = self._features()
        s1 = np.tanh(self.A1 @ h + self.b1)
        s2 = np.tanh(self.A2 @ s1 + self.b2)
        r = np.sqrt(s2 @ s2 + EPS**2)
        u = s2 / r
        z = u @ self.V + self.c
        if self.hash_enabled:
            if address is None:
                address = self._address()
            z += .25 * self._weights[address].sum(axis=0, dtype=np.float64)
        p = np.exp(z - z.max())
        p /= p.sum()
        return (h, s1, s2, r, u), p

    def prediction(self):
        self._require_open()
        _, p = self._forward()
        return Prediction(p, int(p.argmax()))

    def _prediction_at(self, address):
        _, p = self._forward(address)
        return Prediction(p, int(p.argmax()))

    def _directions(self, state, p, target):
        h, s1, s2, r, u = state
        e = -p.copy()
        e[target] += 1.
        v = self.V @ e
        g = (v - u * (u @ v)) / r
        d2 = g * (1 - s2*s2)
        d1 = (self.A2.T @ d2) * (1 - s1*s1)
        return {'A1': np.outer(d1, h), 'b1': d1,
                'A2': np.outer(d2, s1), 'b2': d2,
                'V': np.outer(u, e), 'c': e}

    def _observe_event(self, event, learn):
        self._require_open()
        if learn and not self._in_reply:
            raise RuntimeError('learning requires an actually observed reply event')
        address = self._address() if self.hash_enabled else None
        state, p = self._forward(address)
        predicted = Prediction(p, int(p.argmax()))
        if learn:
            # Internal directions are computed BEFORE head/hash/bias updates.
            if self.internal_learning:
                directions = self._directions(state, p, event)
                for layer in (1, 2):
                    da = ADAPTER_RATE * directions[f'A{layer}']
                    db = ADAPTER_RATE * directions[f'b{layer}']
                    norm = np.sqrt(np.sum(da*da) + db @ db)
                    factor = min(1., STEP_CAP / norm) if norm else 1.
                    getattr(self, f'A{layer}')[:] += factor * da
                    getattr(self, f'b{layer}')[:] += factor * db
            e = -p.copy()
            e[event] += 1.
            self.V += HEAD_RATE * np.outer(state[4], e)
            self.c += HEAD_RATE * e
            if self.hash_enabled:
                self._weights[address] += (HEAD_RATE*.25*e).astype(np.float32)
            self.learning_events += 1
        self._append(event)
        return predicted

    def arrays(self):
        a = {k: getattr(self, k) for k in ('A1', 'A2', 'b1', 'b2', 'V', 'c')}
        if self.hash_enabled:
            a['W'] = self._weights
        return a

    def persistent_components(self):
        return {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in self.arrays().items()}

    def learned_digest(self):
        obj = self.persistent_components()
        obj['learning_events'] = self.learning_events
        return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()

    def birth_digest(self):
        # Called at birth; correspondence is independent of the hash switch.
        return hashlib.sha256(self.A1.tobytes()+self.A2.tobytes()+self.embedding.tobytes()).hexdigest()

    def storage_accounting(self):
        internal = sum(getattr(self, k).nbytes for k in ('A1', 'A2', 'b1', 'b2'))
        mutable = self.V.nbytes+self.c.nbytes + (internal if self.internal_learning else 0)
        mutable += self._weights.nbytes if self.hash_enabled else 0
        return {'fixed_array_bytes': self.embedding.nbytes + (0 if self.internal_learning else internal),
                'learned_array_bytes': mutable,
                'working_context_bytes': self._context.nbytes+len(self._question),
                'allocated_array_bytes': sum(a.nbytes for a in self.arrays().values()) +
                    self.embedding.nbytes+self._context.nbytes,
                'hash_allocated': hasattr(self, '_weights'), 'output_categories': 257,
                'payload_categories': 256, 'context_bound': 4096, 'generation_cap': 16,
                'production_work': 'O(window + 2*96^2 + 96*257 + H*16*257)',
                'audit_digest_work_charged_separately': True}

    def transient_digest(self):
        obj = self._metadata()
        obj.update(question=self._question.hex(), question_total=self._question_total,
                   offset=self._reply_offset, previous=self._previous, in_reply=self._in_reply)
        return hashlib.sha256(self._context.tobytes()+json.dumps(obj, sort_keys=True).encode()).hexdigest()

    def save_checkpoint(self, path):
        metadata = {'schema': 'ADAPT2_CHECKPOINT_V1', 'name': self.name, 'seed': self.seed,
                    'base': self._metadata(), 'question_total': self._question_total,
                    'offset': self._reply_offset, 'previous': self._previous, 'in_reply': self._in_reply}
        with Path(path).open('wb') as f:
            np.savez_compressed(f, metadata=np.frombuffer(json.dumps(metadata).encode(), np.uint8),
                context=self._context, question=np.frombuffer(bytes(self._question), np.uint8),
                embedding=self.embedding, **self.arrays())

    @classmethod
    def load_checkpoint(cls, path):
        with np.load(path, allow_pickle=False) as d:
            m = json.loads(d['metadata'].tobytes())
            if m['schema'] != 'ADAPT2_CHECKPOINT_V1':
                raise ValueError('checkpoint schema')
            c = cls(m['name'], m['seed'])
            expected = set(c.arrays()) | {'metadata', 'context', 'question', 'embedding'}
            if set(d.files) != expected:
                raise ValueError('unexpected checkpoint arrays')
            for k, a in c.arrays().items():
                b = d[k]
                if b.shape != a.shape or b.dtype != a.dtype or not np.isfinite(b).all():
                    raise ValueError(f'invalid array {k}')
                if not c.internal_learning and k in ('A1','A2','b1','b2') and not np.array_equal(a,b):
                    raise ValueError('frozen representation changed in checkpoint')
                a[:] = b
            if not np.array_equal(c.embedding, d['embedding']):
                raise ValueError('fixed embedding changed')
            ctx = d['context']
            if ctx.shape != c._context.shape or ctx.dtype != np.uint16 or np.any(ctx > 258):
                raise ValueError('invalid context')
            if d['question'].dtype != np.uint8 or d['question'].ndim != 1 or len(d['question']) > 4096:
                raise ValueError('invalid question')
            c._context[:] = ctx
            base = m['base']
            if base.get('config') != asdict(c.config):
                raise ValueError('checkpoint configuration mismatch')
            for key in ('context_len','next','learning_events','observed_payload_bytes',
                        'generated_payload_bytes','context_overflow_events'):
                if type(base.get(key)) is not int or base[key] < 0:
                    raise ValueError('invalid integer metadata')
            for key in ('open','generated_end'):
                if type(base.get(key)) is not bool:
                    raise ValueError('invalid boundary metadata')
            if base['generated_end'] and base['open']:
                raise ValueError('contradictory boundary metadata')
            elapsed=base.get('elapsed_seconds')
            if isinstance(elapsed,bool) or not isinstance(elapsed,(int,float)) or not np.isfinite(elapsed) or elapsed<0:
                raise ValueError('invalid elapsed time')
            if type(m.get('question_total')) is not int or type(m.get('offset')) is not int or type(m.get('in_reply')) is not bool:
                raise ValueError('invalid working metadata')
            for k, attr in [('context_len','_context_len'),('next','_next'),('open','_open'),
                            ('generated_end','_generated_end')]:
                setattr(c, attr, base[k])
            if not 0 <= c._context_len <= 4096 or not 0 <= c._next < 4096:
                raise ValueError('context bounds')
            if c._context_len<4096 and c._next!=c._context_len:
                raise ValueError('inconsistent incomplete ring')
            for k in ('learning_events','observed_payload_bytes','generated_payload_bytes',
                      'context_overflow_events','elapsed_seconds'):
                setattr(c, k, base[k])
            c._question = bytearray(d['question'].tobytes())
            c._question_total, c._reply_offset = m['question_total'], m['offset']
            c._previous, c._in_reply = m['previous'], m['in_reply']
            if c._question_total < len(c._question) or c._reply_offset < 0 or (
                    c._previous is not None and (type(c._previous) is not int or not 0 <= c._previous < 256)):
                raise ValueError('working bounds')
            return c
