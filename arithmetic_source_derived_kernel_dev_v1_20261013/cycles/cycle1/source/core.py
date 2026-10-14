"""Source-derived generic byte kernels. No task, evaluator or semantic imports."""
import copy
import hashlib
import math
import pickle
import time
from collections import deque
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(1)
EOS, BEGIN, REPLY = 256, 257, 258
ARMS = ('B16', 'B24', 'B69', 'E6', 'E6_FIX')


def child_seed(seed, name):
    return int.from_bytes(hashlib.sha256(f'{seed}|{name}'.encode()).digest()[:8], 'big') % (2**63)


class Kernel(nn.Module):
    def __init__(self, arm, seed):
        super().__init__()
        if arm not in ARMS:
            raise ValueError('unknown generic recipe')
        self.arm = arm
        self.elm = arm.startswith('E')
        self.width = 69 if arm == 'B69' else 64
        self.embedding = 16 if arm == 'B16' else 24
        self.p = nn.ParameterDict()
        def add(name, value, learn=True):
            self.p[name] = nn.Parameter(torch.as_tensor(value, dtype=torch.float64).clone(), requires_grad=learn)
        if not self.elm:
            # Exact original ANSWER-0 birth draw order and equations.
            rng = np.random.default_rng(seed)
            d, e = self.width, self.embedding
            add('E', rng.choice((-1., 1.), (259, e))/np.sqrt(e))
            for gate in ('z', 'r', 'g'):
                add('W'+gate, rng.normal(0, 1/np.sqrt(e), (d, e)))
                q, r = np.linalg.qr(rng.normal(size=(d, d)))
                add('U'+gate, q*np.sign(np.diag(r))[None, :])
                add('b'+gate, np.zeros(d))
        else:
            rng = np.random.default_rng(child_seed(seed, 'embedding'))
            add('E', rng.choice((-1., 1.), (259, 24))/np.sqrt(24))
            graph_rng = np.random.default_rng(child_seed(seed, 'graph'))
            graph = np.empty((64, 8, 4), dtype=np.int64)
            for i in range(64):
                graph[i, :, :3] = graph_rng.permutation(24).reshape(8, 3)
                graph[i, :, 3] = 24+graph_rng.choice([j for j in range(64) if j != i], 8, replace=False)
            self.register_buffer('source_idx', torch.tensor(graph))
            def normal(name, shape, scale):
                g = torch.Generator().manual_seed(child_seed(seed, name))
                return torch.randn(shape, generator=g, dtype=torch.float64)*scale
            syn = torch.empty(64, 8, 4, dtype=torch.float64)
            syn[:, :, :3] = normal('sensory', (64, 8, 3), .24)
            syn[:, :, 3:] = normal('recurrent', (64, 8, 1), .16)
            add('syn_weight', syn)
            add('branch_bias', torch.full((64, 8), -.05))
            add('branch_gain_raw', torch.full((64, 8), math.log(math.expm1(1.))))
            for name, shape in [('internal_w1', (64, 14, 12)), ('internal_w2', (64, 12, 6))]:
                g = torch.Generator().manual_seed(child_seed(seed, name))
                w = torch.empty(shape, dtype=torch.float64)
                for i in range(64):
                    nn.init.xavier_uniform_(w[i], generator=g)
                add(name, w)
            add('internal_b1', torch.zeros(64, 12))
            add('internal_b2', torch.zeros(64, 6))
            tau = torch.logspace(math.log10(1.5), math.log10(32.), 6, dtype=torch.float64)
            k = torch.exp(-1/tau).repeat(64, 1)+normal('kappa', (64, 6), .06)
            k = k.clamp(.0801, .9949)
            u = ((k-.08)/(.995-.08)).clamp(1e-5, 1-1e-5)
            add('kappa_raw', torch.logit(u), arm != 'E6_FIX')
            add('memory_to_activity', normal('memory_to_activity', (64, 6), .30))
            add('branch_to_activity', normal('branch_to_activity', (64, 8), .10))
            add('activity_bias', torch.zeros(64))
            add('threshold_raw', torch.full((64,), math.log(math.expm1(.75))))
        add('V', torch.zeros(self.width+1, 257, dtype=torch.float64))

    def initial(self, batch=1):
        if self.elm:
            return (torch.zeros(batch, 64, 6, dtype=torch.float64), torch.zeros(batch, 64, dtype=torch.float64))
        return torch.zeros(batch, self.width, dtype=torch.float64)

    def step(self, state, tokens):
        e = self.p['E'][tokens]
        p = self.p
        if not self.elm:
            h = state
            z = torch.sigmoid(e@p['Wz'].T+h@p['Uz'].T+p['bz'])
            r = torch.sigmoid(e@p['Wr'].T+h@p['Ur'].T+p['br'])
            g = torch.tanh(e@p['Wg'].T+(r*h)@p['Ug'].T+p['bg'])
            return (1-z)*h+z*g
        m, previous = state
        sources = torch.cat((e, previous), 1)
        gathered = sources[:, self.source_idx.reshape(-1)].reshape(-1, 64, 8, 4)
        branch = torch.tanh(F.softplus(p['branch_gain_raw'])*((gathered*p['syn_weight']).sum(-1)+p['branch_bias']))
        inner = torch.cat((branch, m), -1)
        hidden = torch.tanh(torch.einsum('bnq,nqh->bnh', inner, p['internal_w1'])+p['internal_b1'])
        candidate = torch.tanh(torch.einsum('bnh,nhm->bnm', hidden, p['internal_w2'])+p['internal_b2'])
        k = .08+(.995-.08)*torch.sigmoid(p['kappa_raw'])
        m = k*m+(1-k)*candidate
        v = (m*p['memory_to_activity']).sum(-1)+(branch*p['branch_to_activity']).sum(-1)+p['activity_bias']
        activity = torch.sigmoid(4*(v-F.softplus(p['threshold_raw'])))
        return m, activity

    def logits(self, state):
        activity = state[1] if self.elm else state
        v = torch.cat((activity, torch.ones(activity.shape[0], 1, dtype=torch.float64)), 1)
        return (v/torch.linalg.vector_norm(v, dim=1, keepdim=True))@self.p['V']


def detach(state):
    return tuple(x.detach() for x in state) if isinstance(state, tuple) else state.detach()


def mask_state(old, new, active):
    if isinstance(old, tuple):
        return tuple(mask_state(a, b, active) for a, b in zip(old, new))
    return torch.where(active.reshape((-1,)+(1,)*(old.ndim-1)), new, old)


class Learner:
    def __init__(self, arm, seed):
        self.net = Kernel(arm, seed)
        self.mom = {k: torch.zeros_like(p) for k, p in self.net.p.items() if p.requires_grad}
        self.steps = 0
        self.last_gradient = {}
        self.last_update = {}
        self.resource = dict(external_records=0, external_payload_bytes=0, actual_target_applications=0,
                             optimizer_steps=0, transition_rows=0, vocabulary_rows=0,
                             generated_payload_bytes=0, generation_queries=0,
                             training_cpu=0., generation_cpu=0.)

    def clone(self):
        return copy.deepcopy(self)

    def digest(self):
        h = hashlib.sha256()
        for label, group in [('p', self.net.p), ('buffers', dict(self.net.named_buffers())), ('mom', self.mom)]:
            for k, p in sorted(group.items()):
                h.update((label+'|'+k).encode())
                h.update(p.detach().numpy().tobytes())
                if label == 'p':
                    h.update(str(p.requires_grad).encode())
                    h.update(b'NONE' if p.grad is None else p.grad.detach().numpy().tobytes())
        h.update(pickle.dumps((self.steps, self.last_gradient, self.last_update), protocol=5))
        return h.hexdigest()

    def advance(self, state, token):
        self.resource['transition_rows'] += 1
        return self.net.step(state, torch.tensor([token], dtype=torch.long))

    def decision(self, state):
        self.resource['vocabulary_rows'] += 1
        return self.net.logits(state)

    def record_loss(self, prompt, reply, allowed):
        events = [BEGIN, *prompt, REPLY]
        targets = [*reply, EOS]
        predictions, losses = [], []
        # All native task frames fit32. The fallback preserves forward state and
        # replays only32 ancestors from its actual detached boundary, not zero.
        if len(events)+len(reply) <= 32:
            state = self.net.initial()
            for token in events:
                state = self.advance(state, token)
            for target in targets:
                logits = self.decision(state)
                loss = F.cross_entropy(logits, torch.tensor([target]))
                predictions.append({'target': target, 'choice': int(logits.detach().argmax()), 'loss': float(loss.detach())})
                if allowed:
                    losses.append(loss)
                if target != EOS:
                    state = self.advance(state, target)
            return predictions, torch.stack(losses).mean() if allowed else None
        state = self.net.initial()
        tape = deque(maxlen=32)
        def push(token):
            nonlocal state
            with torch.no_grad():
                tape.append((token, detach(state)))
                state = self.advance(state, token)
        for token in events:
            push(token)
        for target in targets:
            if allowed:
                reconstructed = tape[0][1]
                for token, _ in tape:
                    reconstructed = self.advance(reconstructed, token)
                logits = self.decision(reconstructed)
            else:
                logits = self.decision(state)
            loss = F.cross_entropy(logits, torch.tensor([target]))
            predictions.append({'target': target, 'choice': int(logits.detach().argmax()), 'loss': float(loss.detach())})
            if allowed:
                # Gradients sum across targets; state boundaries intentionally
                # detached according to the source per-target32 credit rule.
                losses.append(loss)
            if target != EOS:
                push(target)
        return predictions, torch.stack(losses).mean() if allowed else None

    def observe(self, prompt, reply, allowed):
        if type(prompt) is not bytes or type(reply) is not bytes or type(allowed) is not bool:
            raise TypeError('raw byte records and explicit boolean learning permission')
        started = time.process_time()
        before_steps = self.steps
        with torch.set_grad_enabled(allowed):
            predictions, loss = self.record_loss(prompt, reply, allowed)
        if allowed:
            loss.backward()
            grads = {k: p.grad for k, p in self.net.p.items() if p.requires_grad}
            if any(g is None or not torch.isfinite(g).all() for g in grads.values()):
                raise FloatingPointError('missing or nonfinite component gradient')
            component = {k: float(torch.linalg.vector_norm(g)) for k, g in grads.items()}
            norm = math.sqrt(sum(float((g*g).sum()) for g in grads.values()))
            clip = min(1., 1/max(norm, 1e-30))
            self.steps += 1
            updates = {}
            with torch.no_grad():
                for k, g in grads.items():
                    g = g*clip
                    self.mom[k].mul_(.99).add_(g*g, alpha=.01)
                    delta = .001*g/(torch.sqrt(self.mom[k]/(1-.99**self.steps))+1e-8)
                    self.net.p[k].sub_(delta)
                    updates[k] = float(torch.linalg.vector_norm(delta))
            self.last_gradient = component
            self.last_update = dict(targets=len(reply)+1, mean_loss=float(loss.detach()),
                                    unclipped_gradient_norm=norm, clip_factor=clip,
                                    component_step_norms=updates)
            for p in self.net.p.values():
                p.grad = None
            self.resource['actual_target_applications'] += len(reply)+1
        self.resource['external_records'] += 1
        self.resource['external_payload_bytes'] += len(prompt)+len(reply)
        self.resource['optimizer_steps'] += self.steps-before_steps
        self.resource['training_cpu'] += time.process_time()-started
        if not all(torch.isfinite(p).all() for p in self.net.p.values()):
            raise FloatingPointError('nonfinite model')
        return predictions

    @torch.no_grad()
    def generate_batch(self, prompts, cap=16):
        if type(cap) is not int or cap < 0 or any(type(p) is not bytes for p in prompts):
            raise TypeError('bounded raw byte generation')
        if not prompts:
            return []
        started = time.process_time()
        batch = len(prompts)
        sequences = [[BEGIN, *p, REPLY] for p in prompts]
        state = self.net.initial(batch)
        # Padding is absence only. No pooling, learned position weight or route.
        for t in range(max(map(len, sequences))):
            active = torch.tensor([t < len(s) for s in sequences])
            tokens = torch.tensor([s[t] if t < len(s) else 0 for s in sequences])
            new = self.net.step(state, tokens)
            self.resource['transition_rows'] += batch
            state = mask_state(state, new, active)
        answers = [bytearray() for _ in prompts]
        statuses = [None]*batch
        active = torch.ones(batch, dtype=torch.bool)
        for _ in range(cap+1):
            logits = self.net.logits(state)
            self.resource['vocabulary_rows'] += batch
            choices = logits.argmax(1)
            for i in range(batch):
                if not active[i]:
                    continue
                token = int(choices[i])
                if token == EOS:
                    statuses[i] = 'EOS'; active[i] = False
                elif len(answers[i]) == cap:
                    statuses[i] = 'TRUNCATED'; active[i] = False
                else:
                    answers[i].append(token)
            if not bool(active.any()):
                break
            new = self.net.step(state, choices)
            self.resource['transition_rows'] += batch
            state = mask_state(state, new, active)
        assert all(s is not None for s in statuses)
        self.resource['generated_payload_bytes'] += sum(map(len, answers))
        self.resource['generation_queries'] += batch
        self.resource['generation_cpu'] += time.process_time()-started
        return [(bytes(a), s) for a, s in zip(answers, statuses)]

    def generate(self, prompt, cap=16):
        return self.generate_batch([prompt], cap)[0]

    def state_bytes(self):
        params = sum(p.numel()*p.element_size() for p in self.net.p.values())
        moments = sum(p.numel()*p.element_size() for p in self.mom.values())
        graph = sum(p.numel()*p.element_size() for p in self.net.buffers())
        return dict(parameter_and_fixed_bytes=params, moment_bytes=moments, fixed_graph_bytes=graph,
                    single_query_transient_scalars=448 if self.net.elm else self.net.width,
                    pending_gradient_bytes=0, raw_history_buffer_bytes=0)
