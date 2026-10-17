"""Native-context graft. No task generator, arithmetic parser or private head."""
import copy
import hashlib
import json
import math
import time
import bootstrap
import numpy as np
import torch
from torch.nn import functional as F
from bind_native import State, tensor, canonical_states

torch.set_num_threads(1)
WORDS = json.loads((bootstrap.HERE / "OUTPUT_CODEBOOK.json").read_text())["words_by_class"]
CODE = tensor([[1. if w & (1 << j) else -1. for j in range(16)] for w in WORDS])
MU = CODE.mean(0)
EOS, BEGIN, REPLY = 256, 257, 258
DELTA = 30 / 14
ARMS = ("NATIVE", "BIND", "BIND-FROZEN", "BIND-1STEP", "BIND-ADAPT", "BIND-FB", "FROZEN-ADAPT", "FROZEN-FB")


def seed(world, purpose, parameter=""):
    value = f"NATIVE-BIND-V1|{world}|{purpose}|{parameter}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little")


def array_hash(arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = a.detach().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


class Learner:
    def __init__(self, arm, world):
        if arm not in ARMS:
            raise ValueError("locked arm roster")
        self.arm, self.world = arm, world
        self.anatomy, self.state, self.births = canonical_states()
        self.p, self.mom = {}, {}
        self.has_context = arm != "NATIVE"
        self.adaptive = arm in ("BIND-ADAPT", "FROZEN-ADAPT")
        self.feedback = arm in ("BIND-FB", "FROZEN-FB")
        self.learns = self.has_context and "FROZEN" not in arm
        self.one_step = arm == "BIND-1STEP"
        if self.has_context:
            for name, shape in (("Wg", (64, 88)), ("Rg", (64, 64)), ("bg", (64,)),
                                ("W", (64, 88)), ("R", (64, 64)), ("b", (64,)),
                                ("V", (16, 64)), ("D0", (88, 16)), ("D1", (88, 16))):
                rng = np.random.default_rng(seed(world, "theta", name))
                if name in ("R", "Rg"):
                    q, r = np.linalg.qr(rng.normal(size=shape))
                    value = q * np.sign(np.diag(r))[None, :]
                elif len(shape) == 1:
                    value = np.zeros(shape)
                elif name.startswith("D"):
                    value = rng.normal(0, .01, shape)
                else:
                    bound = math.sqrt(6 / sum(shape))
                    value = rng.uniform(-bound, bound, shape)
                self.p[name] = torch.nn.Parameter(tensor(value), requires_grad=self.learns)
            if self.adaptive:
                self.p["xi"] = torch.nn.Parameter(torch.full((64,), math.log(.2/.8), dtype=torch.float64), requires_grad=self.learns)
                self.p["zeta"] = torch.nn.Parameter(torch.full((64,), math.log(.1/.9), dtype=torch.float64), requires_grad=self.learns)
            if self.feedback:
                for name, shape in (("H", (4, 128)), ("G", (64, 4))):
                    rng = np.random.default_rng(seed(world, "theta", name))
                    value = rng.normal(0, .01, shape) if name == "G" else rng.uniform(-math.sqrt(6/sum(shape)), math.sqrt(6/sum(shape)), shape)
                    self.p[name] = torch.nn.Parameter(tensor(value), requires_grad=self.learns)
            self.mom = {k: torch.zeros_like(v) for k, v in self.p.items()}
        self.actor_steps, self.time = 0, 0.
        self.h, self.a = torch.zeros(64, dtype=torch.float64), torch.zeros(64, dtype=torch.float64)
        self.fronts = []
        self.resource = dict(records=0, input_bytes=0, answer_bytes=0, EOS_targets=0,
                             native_store_events=0, actor_steps=0, production_cpu=0., readout_cpu=0.,
                             raw_alpha_l1=0., queries=0)
        self.last_trace, self.last_grads = [], {}

    def clone(self):
        result = copy.copy(self)
        result.state = self.state.detached(True)
        result.p = {k: torch.nn.Parameter(v.detach().clone(), requires_grad=self.learns) for k, v in self.p.items()}
        result.mom = {k: v.clone() for k, v in self.mom.items()}
        result.h, result.a = self.h.detach().clone(), self.a.detach().clone()
        result.fronts = [f.clone() for f in self.fronts]
        result.resource = self.resource.copy()
        result.last_trace, result.last_grads = [], {}
        return result

    def digest(self):
        arrays = [self.state.fast, self.state.slow, self.state.adapt, self.h, self.a]
        arrays += [v for _, v in sorted(self.p.items())] + [v for _, v in sorted(self.mom.items())]
        arrays += [f.p for f in self.fronts]
        metadata = [self.time, self.actor_steps, [getattr(f, "visible", b"").hex() for f in self.fronts]]
        return hashlib.sha256((array_hash(arrays) + json.dumps(metadata)).encode()).hexdigest()

    def theta_digest(self):
        return array_hash(list(self.p.values()) + list(self.mom.values()) + [np.array([self.actor_steps])])

    def _context(self, byte, rm):
        if not self.has_context or byte == EOS:
            return
        import stores
        r = tensor(stores.bb.bc.R_TABLE[byte])
        h = torch.zeros_like(self.h) if self.one_step else self.h
        p = self.p
        g = torch.sigmoid(p["Wg"] @ r + p["Rg"] @ h + p["bg"])
        drive = p["W"] @ r + p["R"] @ h + p["b"]
        if self.adaptive:
            tau = 2 + 30 * torch.sigmoid(p["xi"])
            rho = torch.exp(-1/tau)
            self.a = rho * self.a + (1-rho) * h
            drive = drive - torch.sigmoid(p["zeta"]) * self.a
        if self.feedback:
            normalizer = tensor([36.46, 36.46, 19.96, 19.96])
            m = (rm / normalizer).reshape(128)
            addition = .5 * torch.tanh(p["G"] @ torch.tanh(p["H"] @ m))
            self.last_feedback_max = float(addition.detach().abs().max())
            drive = drive + addition
        self.h = (1-g)*h + g*torch.tanh(drive)

    def _encode(self, fronts, at):
        pn = []
        for bank, fe in enumerate(fronts):
            fe.advance(at)
            p = tensor(fe.read())
            if self.has_context:
                raw = self.p[f"D{bank}"] @ (self.p["V"] @ self.h)
                eta = math.log(2) * torch.tanh(raw - raw.mean())
                p = p * torch.exp(eta)
                p = p / p.max().clamp_min(1e-300)
            pn.append(p)
        return self.anatomy.encode(torch.stack(pn), torch.is_grad_enabled() and self.learns)

    def control(self, token):
        import stores
        if token == BEGIN:
            self.h, self.a = torch.zeros_like(self.h), torch.zeros_like(self.a)
            self.fronts = [stores.bb.bc.FE0(), stores.content_model.VisibleContextFE()]
            for f in self.fronts:
                f.t = self.time + DELTA
        elif token != REPLY:
            raise ValueError("typed generic control required")
        self.state = self.anatomy.rest(self.state, DELTA)
        self.time += DELTA
        self.resource["native_store_events"] += 32

    def event(self, byte, target=None, allowed=False):
        if type(byte) is not int or not 0 <= byte <= EOS:
            raise ValueError("payload byte or EOS required")
        x = self._encode(self.fronts, self.time + DELTA)
        if target is None:
            zero = torch.zeros((2,16), dtype=torch.float64)
            self.state, rm, info = self.anatomy.event(self.state, x, DELTA, zero, False)
        else:
            c = {"shared": (1 - CODE[target])/2, "private": (MU - CODE[target])/2}
            self.state, rm, info = self.anatomy.teach(self.state, x, DELTA, c, allowed)
        if byte != EOS:
            for f in self.fronts:
                f.feed(byte, self.time + DELTA)
            self._context(byte, rm)
        self.time += DELTA
        self.resource["native_store_events"] += 32
        self.resource["raw_alpha_l1"] += float(info["alpha"].detach().abs().sum())
        return x, rm

    def prompt(self, prompt):
        if type(prompt) is not bytes or len(prompt)>256:
            raise ValueError("bounded bytes prompt")
        self.control(BEGIN)
        for b in prompt:
            self.event(b)
        self.control(REPLY)

    def values(self, readout="MEM"):
        x = self._encode([f.clone() for f in self.fronts], self.time + DELTA)
        s, dx = self.anatomy.query_activity(self.state, x, DELTA)
        expression = self.anatomy.expression(s, dx)
        if readout == "MEM":
            baseline = self.anatomy.expression(s, dx, True)
        elif readout == "LEGACY":
            if torch.is_grad_enabled() and self.learns:
                raise ValueError("legacy RBF is diagnostic only, not a surrogate training head")
            baseline = tensor(self.anatomy.reader.predict(dx.detach().numpy().reshape(32, -1))).reshape(2,16,2)
        else:
            raise ValueError("locked readout")
        v = (expression-baseline).mean(-1)
        return v[0]/1.4911274663291492 + v[1]/1.3452365735750882

    def forecast(self, readout="MEM"):
        v = self.values(readout)
        return CODE @ (v-v.mean()) / 4

    def train_window(self, records, allowed=True, apply_update=True, detach_between_records=False):
        if len(records)!=4 or type(allowed) is not bool:
            raise ValueError("exactly four records and explicit learning permission")
        start = time.process_time()
        self.state = self.state.detached()
        self.last_trace = []
        losses = []
        with torch.set_grad_enabled(self.learns and allowed):
            for prompt, answer in records:
                if type(answer) is not bytes or len(answer)>16:
                    raise ValueError("bounded actual observed bytes")
                self.prompt(prompt)
                terms = []
                for target in (*answer, EOS):
                    scores = self.forecast()
                    terms.append(F.cross_entropy(scores[None,:], torch.tensor([target])))
                    self.last_trace.append(dict(predicted=int(scores.detach().argmax()), target=target,
                                                prefeedback=True, native_write=allowed, logits=scores.detach().tolist()))
                    self.event(target, target, allowed)
                losses.append(torch.stack(terms).mean())
                self.resource["records"] += 1
                self.resource["input_bytes"] += len(prompt)
                self.resource["answer_bytes"] += len(answer)
                self.resource["EOS_targets"] += 1
                if detach_between_records:
                    self.state = self.state.detached()
            loss = torch.stack(losses).mean()
            norm = 0.
            self.last_grads = {}
            if self.learns and allowed and loss.requires_grad:
                grads = torch.autograd.grad(loss, list(self.p.values()), allow_unused=True)
                grads = [torch.zeros_like(p) if g is None else g for p,g in zip(self.p.values(),grads)]
                norm = math.sqrt(sum(float((g*g).sum()) for g in grads))
                if not math.isfinite(norm):
                    raise FloatingPointError("nonfinite context gradient")
                self.last_grads = {k: g.detach().clone() for k,g in zip(self.p,grads)}
                if apply_update:
                    self.actor_steps += 1
                    clip = min(1., 1/max(norm, 1e-30))
                    with torch.no_grad():
                        for (k,p),g in zip(self.p.items(),grads):
                            g = g*clip
                            self.mom[k].mul_(.99).add_(g*g, alpha=.01)
                            p.sub_(.001*g/(torch.sqrt(self.mom[k]/(1-.99**self.actor_steps))+1e-8))
                    self.resource["actor_steps"] += 1
        self.state = self.state.detached()
        self.h, self.a = self.h.detach(), self.a.detach()
        self.resource["production_cpu"] += time.process_time()-start
        self.assert_finite()
        return dict(loss=float(loss.detach()), gradient_norm=norm, actor_steps=self.actor_steps,
                    records=4, allowed=allowed, applied_update=bool(self.learns and allowed and apply_update))

    def assert_finite(self):
        if not all(torch.isfinite(x).all() for x in (self.state.fast, self.state.slow, self.state.adapt, *self.p.values())):
            raise FloatingPointError("nonfinite native or context state")

    def rest(self, seconds):
        if not math.isfinite(seconds) or seconds<0:
            raise ValueError("nonnegative time")
        self.state = self.anatomy.rest(self.state, seconds)
        self.time += seconds

    def generate(self, prompt, cap=16, readout="MEM"):
        if type(cap) is not int or not 0<=cap<=16:
            raise ValueError("payload cap")
        view = self.clone()
        with torch.no_grad():
            view.prompt(prompt)
            answer, trace = bytearray(), []
            for _ in range(cap+1):
                scores = view.forecast(readout)
                target = int(scores.argmax())
                trace.append(dict(prefix=bytes(answer).hex(), predicted=target))
                if target==EOS:
                    return bytes(answer), "EOS", trace
                if len(answer)==cap:
                    return bytes(answer), "TRUNCATED", trace
                answer.append(target)
                view.event(target)
        raise AssertionError("unreachable")
