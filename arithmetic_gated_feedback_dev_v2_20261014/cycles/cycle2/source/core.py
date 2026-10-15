"""Standalone raw-byte GRU/LSTM policies and feedback learning; no task imports."""
import copy
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

torch.set_num_threads(1)
EOS, BEGIN, REPLY = 256, 257, 258
MODELS = ('KEEP', 'G62')
RECIPES = ('CE_ONLY', 'OLD', 'SAFE')
NUMERICAL_TOLERANCE = 1e-12


class Learner:
    def __init__(self, model, seed):
        if model not in MODELS:
            raise ValueError('fixed model roster')
        self.model = model
        self.width = 64 if model == 'KEEP' else 62
        self.p = {}
        rng = np.random.default_rng(seed)
        def add(name, value):
            self.p[name] = torch.nn.Parameter(torch.as_tensor(value, dtype=torch.float64).clone())
        add('E', rng.choice((-1., 1.), (259, 24)) / np.sqrt(24))
        for gate in ('z', 'r', 'g') if model == 'KEEP' else ('i', 'f', 'o', 'g'):
            add('W' + gate, rng.normal(0, 1 / np.sqrt(24), (self.width, 24)))
            q, r = np.linalg.qr(rng.normal(size=(self.width, self.width)))
            add('U' + gate, q * np.sign(np.diag(r))[None, :])
            add('b' + gate, np.ones(self.width) if model == 'G62' and gate == 'f' else np.zeros(self.width))
        add('V', np.zeros((self.width + 1, 257)))
        if model == 'KEEP':
            add('C', np.zeros((64, 64)))
        add('critic', np.zeros(self.width + 1))
        self.mom = {k: torch.zeros_like(v) for k, v in self.p.items()}
        self.actor_steps = self.critic_steps = 0
        self.last = {}
        self.resource = dict(records=0, observed_answer_bytes=0, observed_EOS_targets=0,
            scalar_feedback=0, actor_steps=0, critic_steps=0, transition_rows=0,
            vocabulary_rows=0, generated_payload_bytes=0, queries=0, production_cpu=0.,
            reencoded_prompts=0, safe_proposals=0, candidate_evaluations=0,
            safe_rejections=0, copied_parameter_bytes=0, C_matrix_rows=0)

    def clone(self):
        return copy.deepcopy(self)

    def actor(self):
        return {k: v for k, v in self.p.items() if k != 'critic'}

    def digest(self, actor_only=False):
        h = hashlib.sha256()
        for label, arrays in (('parameter', self.p), ('moment', self.mom)):
            for k, v in sorted(arrays.items()):
                if actor_only and k == 'critic':
                    continue
                h.update((label + k).encode())
                h.update(v.detach().numpy().tobytes())
        values = [self.actor_steps] if actor_only else [self.actor_steps, self.critic_steps, self.last]
        h.update(json.dumps(values, sort_keys=True, allow_nan=False).encode())
        return h.hexdigest()

    def initial(self, n=1):
        z = torch.zeros(n, self.width, dtype=torch.float64)
        return z if self.model == 'KEEP' else (z, z.clone())

    def hidden(self, state):
        return state if self.model == 'KEEP' else state[1]

    def step(self, state, tokens):
        p = self.p
        e = p['E'][tokens]
        h = self.hidden(state)
        self.resource['transition_rows'] += len(tokens)
        if self.model == 'KEEP':
            z = torch.sigmoid(e @ p['Wz'].T + h @ p['Uz'].T + p['bz'])
            r = torch.sigmoid(e @ p['Wr'].T + h @ p['Ur'].T + p['br'])
            g = torch.tanh(e @ p['Wg'].T + (r * h) @ p['Ug'].T + p['bg'])
            return (1 - z) * h + z * g
        m, h = state
        i, f, o = (torch.sigmoid(e @ p['W' + k].T + h @ p['U' + k].T + p['b' + k])
                   for k in ('i', 'f', 'o'))
        g = torch.tanh(e @ p['Wg'].T + h @ p['Ug'].T + p['bg'])
        m = f * m + i * g
        return m, o * torch.tanh(m)

    def decode_step(self, state, tokens, context):
        new = self.step(state, tokens)
        if self.model == 'KEEP':
            self.resource['C_matrix_rows'] += len(tokens)
            return new + context @ self.p['C'].T
        return new

    def logits(self, state):
        h = self.hidden(state)
        self.resource['vocabulary_rows'] += len(h)
        u = torch.cat((h, torch.ones(len(h), 1, dtype=torch.float64)), 1)
        return (u / torch.linalg.vector_norm(u, dim=1, keepdim=True)) @ self.p['V']

    def encode(self, prompt):
        if type(prompt) is not bytes or len(prompt) > 256:
            raise ValueError('bounded raw-byte prompt required')
        self.resource['reencoded_prompts'] += 1
        state = self.initial()
        for b in (BEGIN, *prompt):
            state = self.step(state, torch.tensor([b]))
        context = self.hidden(state)
        return context, self.decode_step(state, torch.tensor([REPLY]), context)

    def critic(self, context):
        u = torch.cat((context.detach(), torch.ones(len(context), 1, dtype=torch.float64)), 1)
        u = u / torch.linalg.vector_norm(u, dim=1, keepdim=True)
        return torch.sigmoid(u @ self.p['critic'])

    def act(self, prompt, seed, credit, cap=16):
        if type(cap) is not int or not 0 <= cap <= 16:
            raise ValueError('bounded payload cap')
        start = time.process_time()
        generator = torch.Generator().manual_seed(int(seed))
        with torch.set_grad_enabled(credit):
            context, state = self.encode(prompt)
            baseline = self.critic(context).detach()[0]
            critic_context = context.detach() if self.model == 'KEEP' else None
            if self.model == 'G62':
                context = None  # no unnecessary prompt snapshot during generation
            selected, full, actions, out = [], [], [], bytearray()
            for _ in range(cap + 1):
                logp = F.log_softmax(self.logits(state)[0], dim=0)
                token = int(torch.multinomial(logp.detach().exp(), 1, generator=generator))
                actions.append(token)
                if credit:
                    selected.append(logp[token])
                    full.append(logp.detach())
                if token == EOS:
                    status = 'EOS'
                    break
                if len(out) == cap:
                    status = 'TRUNCATED'
                    break
                out.append(token)
                state = self.decode_step(state, torch.tensor([token]), context)
        self.resource['production_cpu'] += time.process_time() - start
        self.resource['generated_payload_bytes'] += len(out)
        cache = dict(context=critic_context, baseline=baseline,
                     logp=torch.stack(selected).sum() if selected else None,
                     full_logp=torch.stack(full) if full else None, actions=actions,
                     prompt=prompt, version=(self.actor_steps, self.critic_steps), used=False)
        return bytes(out), status, cache

    def ce_loss(self, prompt, answer, with_context=False):
        if type(answer) is not bytes:
            raise TypeError('only actual observed answer bytes')
        context, state = self.encode(prompt)
        losses = []
        for b in (*answer, EOS):
            losses.append(F.cross_entropy(self.logits(state), torch.tensor([b])))
            if b != EOS:
                state = self.decode_step(state, torch.tensor([b]), context)
        loss = torch.stack(losses).mean()
        return (loss, context.detach()) if with_context else loss

    def trace_logps(self, prompt, actions):
        context, state = self.encode(prompt)
        logs = []
        for index, b in enumerate(actions):
            if not 0 <= b <= EOS or (b == EOS and index != len(actions) - 1):
                raise ValueError('actual bounded action trace')
            logs.append(F.log_softmax(self.logits(state)[0], dim=0))
            if b != EOS and index != len(actions) - 1:
                state = self.decode_step(state, torch.tensor([b]), context)
        return torch.stack(logs)

    @staticmethod
    def _candidate_admissible(kl, loss, old_loss):
        return (math.isfinite(kl) and math.isfinite(loss)
                and -NUMERICAL_TOLERANCE <= kl <= .01 + NUMERICAL_TOLERANCE
                and loss <= old_loss + NUMERICAL_TOLERANCE * (1 + abs(old_loss)))

    def update(self, prompt, observed_answer, reward, cache, phase, recipe='OLD', allowed=True):
        if phase not in ('WARM', 'HALF', 'SPARSE', 'SCALAR') or recipe not in RECIPES:
            raise ValueError('declared feedback mode')
        if type(reward) is not int or reward not in (0, 1) or type(allowed) is not bool:
            raise ValueError('actual bit and explicit permission')
        if phase == 'SCALAR' and recipe != 'CE_ONLY' and observed_answer is not None:
            raise ValueError('hidden scalar answer prohibited')
        if observed_answer is not None and type(observed_answer) is not bytes:
            raise TypeError('observed bytes or None')
        if allowed and (phase == 'WARM' or recipe == 'CE_ONLY') and observed_answer is None:
            raise ValueError('visible answer required for pure answer update')
        if cache['used'] or cache['prompt'] != prompt or cache['version'] != (self.actor_steps, self.critic_steps):
            raise ValueError('used, mismatched or stale actual-action cache')
        cache['used'] = True
        self.resource['records'] += 1
        self.resource['scalar_feedback'] += 1
        if observed_answer is not None:
            self.resource['observed_answer_bytes'] += len(observed_answer)
            self.resource['observed_EOS_targets'] += 1
        if not allowed:
            return dict(actor_steps=0, critic_steps=0, ce=None, pg=None, critic=None,
                        safe=False, candidates=[], accepted=None, alpha=None)
        start = time.process_time()
        critic_context = cache['context']
        ce = None
        if observed_answer is not None:
            ce, supervised_context = self.ce_loss(prompt, observed_answer, with_context=True)
            if critic_context is None:
                critic_context = supervised_context
        if critic_context is None:
            with torch.no_grad():
                critic_context, _ = self.encode(prompt)
        pg = None
        if phase != 'WARM' and recipe != 'CE_ONLY':
            if cache['logp'] is None:
                raise ValueError('actual generated action credit required')
            pg = -(reward - cache['baseline']) * cache['logp']
        loss = ce if pg is None else pg if ce is None else ce + pg
        actor = self.actor()
        grads = torch.autograd.grad(loss, list(actor.values()))
        if not all(torch.isfinite(g).all() for g in grads):
            raise FloatingPointError('nonfinite actor gradient')
        norm = math.sqrt(sum(float((g * g).sum()) for g in grads))
        clip = min(1., 1 / max(norm, 1e-30))
        proposal, deltas = {}, {}
        next_step = self.actor_steps + 1
        with torch.no_grad():
            for (k, p), gradient in zip(actor.items(), grads):
                g = gradient * clip
                proposal[k] = self.mom[k].clone().mul_(.99).add_(g * g, alpha=.01)
                deltas[k] = .001 * g / (torch.sqrt(proposal[k] / (1 - .99 ** next_step)) + 1e-8)
        safe = recipe == 'SAFE' and phase != 'WARM'
        candidates, accepted, alpha = [], True, 1.
        if safe:
            self.resource['safe_proposals'] += 1
            before = {k: p.detach().clone() for k, p in actor.items()}
            parameter_bytes = sum(p.numel() * p.element_size() for p in actor.values())
            self.resource['copied_parameter_bytes'] += parameter_bytes
            accepted, alpha = False, None
            old_loss = float(loss.detach())
            old_logp = cache['full_logp']
            with torch.no_grad():
                for candidate in range(8):
                    scale = 2. ** -candidate
                    for k, p in actor.items():
                        p.copy_(before[k] - scale * deltas[k])
                    self.resource['copied_parameter_bytes'] += parameter_bytes
                    new_logp = self.trace_logps(prompt, cache['actions'])
                    kl = float((old_logp.exp() * (old_logp - new_logp)).sum())
                    chosen = new_logp[torch.arange(len(cache['actions'])), torch.tensor(cache['actions'])].sum()
                    candidate_loss = -(reward - cache['baseline']) * chosen
                    if observed_answer is not None:
                        candidate_loss = candidate_loss + self.ce_loss(prompt, observed_answer)
                    value = float(candidate_loss)
                    accepted = self._candidate_admissible(kl, value, old_loss)
                    candidates.append(dict(alpha=scale, kl=kl if math.isfinite(kl) else None,
                                           loss=value if math.isfinite(value) else None, accepted=accepted))
                    self.resource['candidate_evaluations'] += 1
                    if accepted:
                        alpha = scale
                        break
                if not accepted:
                    for k, p in actor.items():
                        p.copy_(before[k])
                    self.resource['copied_parameter_bytes'] += parameter_bytes
                    self.resource['safe_rejections'] += 1
        else:
            with torch.no_grad():
                for k, p in actor.items():
                    p.sub_(deltas[k])
        if accepted:
            with torch.no_grad():
                for k in actor:
                    self.mom[k].copy_(proposal[k])
            self.actor_steps += 1
            self.resource['actor_steps'] += 1
        critic_loss = (self.critic(critic_context)[0] - reward).square()
        vg = torch.autograd.grad(critic_loss, self.p['critic'])[0]
        critic_norm = float(vg.norm())
        vg = vg * min(1., 1 / max(critic_norm, 1e-30))
        self.critic_steps += 1
        with torch.no_grad():
            self.mom['critic'].mul_(.99).add_(vg * vg, alpha=.01)
            self.p['critic'].sub_(.001 * vg / (torch.sqrt(self.mom['critic'] / (1 - .99 ** self.critic_steps)) + 1e-8))
        self.resource['critic_steps'] += 1
        self.resource['production_cpu'] += time.process_time() - start
        if not all(torch.isfinite(p).all() for p in self.p.values()):
            raise FloatingPointError('nonfinite parameter')
        self.last = dict(actor_steps=int(accepted), critic_steps=1,
                         ce=None if ce is None else float(ce.detach()), pg=None if pg is None else float(pg.detach()),
                         critic=float(critic_loss.detach()), actor_gradient_norm=norm,
                         critic_gradient_norm=critic_norm, clip=clip, safe=safe,
                         candidates=candidates, accepted=accepted, alpha=alpha,
                         old_actor_loss=float(loss.detach()),
                         component_gradient_norms={k: float(g.norm()) for k, g in zip(actor, grads)})
        return copy.deepcopy(self.last)

    def _where(self, active, new, old):
        if self.model == 'KEEP':
            return torch.where(active[:, None], new, old)
        return tuple(torch.where(active[:, None], x, y) for x, y in zip(new, old))

    @torch.no_grad()
    def generate_batch(self, prompts, cap=16):
        if not prompts:
            return []
        if type(cap) is not int or not 0 <= cap <= 16:
            raise ValueError('payload cap')
        if any(type(x) is not bytes or len(x) > 256 for x in prompts):
            raise ValueError('raw prompts')
        start = time.process_time()
        n = len(prompts)
        sequences = [[BEGIN, *p] for p in prompts]
        state = self.initial(n)
        for t in range(max(map(len, sequences))):
            active = torch.tensor([t < len(s) for s in sequences])
            tokens = torch.tensor([s[t] if t < len(s) else 0 for s in sequences])
            state = self._where(active, self.step(state, tokens), state)
        context = self.hidden(state)
        state = self.decode_step(state, torch.full((n,), REPLY), context)
        if self.model == 'G62':
            context = None
        out, status = [bytearray() for _ in prompts], [None] * n
        active = torch.ones(n, dtype=torch.bool)
        for _ in range(cap + 1):
            tokens = self.logits(state).argmax(1)
            for i in range(n):
                if not active[i]:
                    continue
                b = int(tokens[i])
                if b == EOS:
                    status[i], active[i] = 'EOS', False
                elif len(out[i]) == cap:
                    status[i], active[i] = 'TRUNCATED', False
                else:
                    out[i].append(b)
            if not bool(active.any()):
                break
            state = self._where(active, self.decode_step(state, tokens, context), state)
        self.resource['queries'] += n
        self.resource['generated_payload_bytes'] += sum(map(len, out))
        self.resource['production_cpu'] += time.process_time() - start
        return [(bytes(x), s) for x, s in zip(out, status)]

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = dict(model=self.model, actor_steps=self.actor_steps, critic_steps=self.critic_steps,
                        last=self.last, resource=self.resource, digest=self.digest())
        data = {'metadata': np.array(json.dumps(metadata, allow_nan=False))}
        data.update({'p_' + k: v.detach().numpy() for k, v in self.p.items()})
        data.update({'m_' + k: v.numpy() for k, v in self.mom.items()})
        temporary = path.with_name(path.name + '.pending')
        with temporary.open('wb') as f:
            np.savez_compressed(f, **data)
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as arrays:
            metadata = json.loads(str(arrays['metadata']))
            result = cls(metadata['model'], 0)
            with torch.no_grad():
                for k, v in result.p.items():
                    v.copy_(torch.from_numpy(arrays['p_' + k]))
                for k, v in result.mom.items():
                    v.copy_(torch.from_numpy(arrays['m_' + k]))
            for name in ('actor_steps', 'critic_steps', 'last', 'resource'):
                setattr(result, name, metadata[name])
        if result.digest() != metadata['digest']:
            raise ValueError('full state archive mismatch')
        return result
