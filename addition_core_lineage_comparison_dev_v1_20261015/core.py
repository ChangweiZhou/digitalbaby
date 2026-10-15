"""Generic byte/controller adapters. No task generator or numeric feature imports.

Native equations and signed teaching are executed by the retained source objects.
Only the explicitly declared control protocol and common output port are new.
"""
import copy
import hashlib
import json
import math
import time
import bootstrap
import numpy as np
import torch
import stores

OLD = bootstrap.source_module('lineage_recurrent_parent', 'arithmetic_gated_feedback_dev_v2_20261014/core.py')
EOS, BEGIN, REPLY = 256, 257, 258
DELTA = 30 / 14
VERSIONS = ('N151-C16', 'C151-C16', 'RC151-C16', 'KEEP-C16-CE', 'G62-C16-CE')
WORDS = json.loads((bootstrap.HERE / 'OUTPUT_CODEBOOK.json').read_text())['words_by_class']
CODE = np.array([[1. if w & (1 << j) else -1. for j in range(16)] for w in WORDS])
MU = CODE.mean(0)
TCODE = torch.tensor(CODE, dtype=torch.float64)


def logits(values, code=CODE):
    return code @ (np.asarray(values) - np.mean(values)) / math.sqrt(code.shape[1])


def sha_arrays(arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.asarray(a).tobytes())
    return h.hexdigest()


class Compartment(stores._Teach):
    def __init__(self):
        temporary, self.birth = stores.birth('native')
        self.fly = temporary.fly
        self.brain_t = 0.
        self.pending_t = self.pending_x = None
        self.teach_seen = 0

    def clone(self):
        result = copy.copy(self)
        result.fly = self.fly.clone()
        result.pending_x = None if self.pending_x is None else self.pending_x.copy()
        return result


class Native:
    def __init__(self, version):
        if version not in VERSIONS[:3]:
            raise ValueError(version)
        self.version = version
        self.banks = [[Compartment() for _ in range(16)]]
        if version == 'RC151-C16':
            self.banks.append([Compartment() for _ in range(16)])
        self.time = 0.
        self.fronts = []
        self.resource = dict(records=0, answer_bytes=0, EOS_targets=0, native_events=0,
                             production_cpu=0., input_bytes=0, controls=0)
        models = [s.fly.m for bank in self.banks for s in bank]
        for attr in ('fast', 'slow', 'adapt'):
            if len({id(getattr(m, attr)) for m in models}) != len(models):
                raise AssertionError('mutable newborn alias')
            for i,m in enumerate(models):
                if any(np.shares_memory(getattr(m,attr),getattr(other,attr)) for other in models[i+1:]):
                    raise AssertionError('mutable newborn storage alias')

    def clone(self):
        result = copy.copy(self)
        result.banks = [[s.clone() for s in bank] for bank in self.banks]
        result.fronts = [f.clone() for f in self.fronts]
        result.resource = self.resource.copy()
        return result

    def arrays(self):
        return [a for b in self.banks for s in b for a in (s.fly.m.fast, s.fly.m.slow, s.fly.m.adapt)]

    def digest(self):
        return sha_arrays(self.arrays() + [np.array([self.time])])

    def _event(self, token=None, target=None, allowed=False):
        previous_time = self.time
        self.time += DELTA
        dt = self.time - previous_time
        if token in (BEGIN, REPLY):
            if token == BEGIN:
                self.fronts = [stores.bb.bc.FE0()]
                if len(self.banks) == 2:
                    self.fronts.append(stores.content_model.VisibleContextFE())
                for f in self.fronts:
                    f.t = self.time
            for bank in self.banks:
                for s in bank:
                    s.fly.rest(dt)
                    s.brain_t = self.time
            self.resource['controls'] += 1
            self.resource['native_events'] += sum(map(len, self.banks))
            return 0.
        total = 0.
        for k, bank in enumerate(self.banks):
            fe = self.fronts[k]
            fe.advance(self.time)
            x = stores.bb.bc.native().model.encode_sparse(bank[0].fly.m, fe.read())
            for j, s in enumerate(bank):
                if target is None or not allowed:
                    s.fly.event(dt, x, 0., False)
                    s.brain_t = self.time
                else:
                    s.pending_t, s.pending_x = self.time, x
                    if self.version == 'C151-C16' or k == 1:
                        total += s.teach_signed(0., (MU[j] - CODE[target, j]) / 2,
                                                self.time, write=True)
                    else:
                        total += s.teach_logged(int((1 - CODE[target, j]) / 2),
                                                self.time, write=True)
            if token is not None and token != EOS:
                fe.feed(token, self.time)
        self.resource['native_events'] += sum(map(len, self.banks))
        return total

    def _prompt(self, prompt):
        if type(prompt) is not bytes or len(prompt) > 256:
            raise ValueError('bounded raw byte prompt')
        self._event(BEGIN)
        for token in prompt:
            self._event(token)
        self._event(REPLY)

    def train(self, prompt, answer, allowed=True):
        if type(answer) is not bytes or len(answer) > 16 or type(allowed) is not bool:
            raise ValueError('actual bounded observed bytes and explicit permission')
        start = time.process_time()
        pre = [(s.fly.m.fast.copy(), s.fly.m.slow.copy()) for bank in self.banks for s in bank] if not allowed else None
        old_t = self.time
        self._prompt(prompt)
        applied = sum(self._event(b, b, allowed) for b in (*answer, EOS))
        off_error = 0.
        if not allowed:
            elapsed = self.time - old_t
            for (fast, slow), s in zip(pre, [s for b in self.banks for s in b]):
                m = s.fly.m
                factors = np.array([math.exp(-elapsed/m.tg), math.exp(-elapsed/m.kernel['fast_tau'])])
                off_error = max(off_error, float(np.max(np.abs(m.fast - fast * factors))),
                                float(np.max(np.abs(m.slow - slow * math.exp(-elapsed/m.kernel['slow_tau'])))))
            if off_error > 1e-9:
                raise AssertionError(('forbidden native write', off_error))
        self.resource['records'] += 1
        self.resource['answer_bytes'] += len(answer)
        self.resource['EOS_targets'] += 1
        self.resource['input_bytes'] += len(prompt)
        self.resource['production_cpu'] += time.process_time() - start
        return dict(allowed=allowed, raw_alpha_l1=applied, forbidden_delta=off_error,
                    actor_steps=0, loss=None)

    def values(self):
        outputs = []
        for fe, bank in zip(self.fronts, self.banks):
            f = fe.clone()
            f.advance(self.time + DELTA)
            x = stores.bb.bc.native().model.encode_sparse(bank[0].fly.m, f.read())
            values = []
            for s in bank:
                fly = s.fly.clone()
                fly.rest((self.time + DELTA) - self.time)
                dx = stores.bb.bc.native().common.observed_activity(fly.m, x[None, :])
                values.append(float((fly.m.expression(dx) - fly.reader.predict(dx)).mean()))
            outputs.append(np.array(values))
        return outputs[0] if len(outputs) == 1 else outputs[0]/1.4911274663291492 + outputs[1]/1.3452365735750882

    def generate_batch(self, prompts, cap=16):
        result = []
        for prompt in prompts:
            view = self.clone()
            view._prompt(prompt)
            out = bytearray()
            for _ in range(cap + 1):
                token = int(logits(view.values()).argmax())
                if token == EOS:
                    status = 'EOS'
                    break
                if len(out) == cap:
                    status = 'TRUNCATED'
                    break
                out.append(token)
                view._event(token)
            result.append((bytes(out), status))
        return result

    def rest(self, seconds):
        for bank in self.banks:
            for s in bank:
                s.fly.rest(seconds)
                s.brain_t += seconds
        self.time += seconds

    def erased(self):
        result = self.clone()
        for bank in result.banks:
            for s in bank:
                s.fly.m.fast.fill(0)
                s.fly.m.slow.fill(0)
        return result


class Recurrent(OLD.Learner):
    def __init__(self, version, seed):
        super().__init__('KEEP' if version == 'KEEP-C16-CE' else 'G62', seed)
        self.version = version
        self.p['V'] = torch.nn.Parameter(torch.zeros((self.width + 1, 16), dtype=torch.float64))
        del self.p['critic']
        self.mom = {k: torch.zeros_like(v) for k, v in self.p.items()}

    def logits(self, state):
        h = self.hidden(state)
        self.resource['vocabulary_rows'] += len(h)
        u = torch.cat((h, torch.ones(len(h), 1, dtype=torch.float64)), 1)
        values = (u / torch.linalg.vector_norm(u, dim=1, keepdim=True)) @ self.p['V']
        return (values - values.mean(1, keepdim=True)) @ TCODE.T / 4

    def train(self, prompt, answer, allowed=True):
        if type(prompt) is not bytes or type(answer) is not bytes or len(answer) > 16 or type(allowed) is not bool:
            raise ValueError('raw byte feedback and explicit permission')
        start = time.process_time()
        self.resource['records'] += 1
        self.resource['observed_answer_bytes'] += len(answer)
        self.resource['observed_EOS_targets'] += 1
        if not allowed:
            # Same byte stream is consumed, with no optimizer/parameter updates.
            with torch.no_grad():
                loss = float(self.ce_loss(prompt, answer))
            self.resource['production_cpu'] += time.process_time() - start
            return dict(allowed=False, actor_steps=0, loss=loss, forbidden_delta=0., raw_alpha_l1=0.)
        loss = self.ce_loss(prompt, answer)
        gradients = torch.autograd.grad(loss, list(self.p.values()))
        norm = math.sqrt(sum(float((g*g).sum()) for g in gradients))
        if not math.isfinite(norm):
            raise FloatingPointError('nonfinite actor gradient')
        clip = min(1., 1/max(norm, 1e-30))
        self.actor_steps += 1
        with torch.no_grad():
            for (k, p), gradient in zip(self.p.items(), gradients):
                g = gradient * clip
                self.mom[k].mul_(.99).add_(g*g, alpha=.01)
                p.sub_(.001*g/(torch.sqrt(self.mom[k]/(1-.99**self.actor_steps))+1e-8))
        if not all(torch.isfinite(p).all() for p in self.p.values()):
            raise FloatingPointError('nonfinite actor state')
        self.resource['actor_steps'] += 1
        self.resource['production_cpu'] += time.process_time() - start
        return dict(allowed=True, actor_steps=1, loss=float(loss.detach()), gradient_norm=norm,
                    forbidden_delta=0., raw_alpha_l1=0.)

    def rest(self, seconds):
        # Parent recipes have no autonomous parameter decay. Physical wait is a no-op.
        if seconds < 0:
            raise ValueError('negative wait')


def create(version, seed=0):
    if version not in VERSIONS:
        raise ValueError('fixed roster')
    return Native(version) if version in VERSIONS[:3] else Recurrent(version, seed)
