"""NATIVE-BIND V2.1 native-derived factorial kernel; no task parser or answer head."""
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
ARMS = ("NATIVE",) + tuple(f"F{e}{a}{c}" for e in (0,1) for a in (0,1) for c in (0,1))


def seed(world, purpose, parameter=""):
    value = f"NATIVE-BIND-V2.1|{world}|{purpose}|{parameter}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little")


def array_hash(arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = a.detach().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def science_identity():
    files=("bootstrap.py","bind_native.py","bind_kernel.py","bind_evaluator.py","protocol.py","SPEC.json","OUTPUT_CODEBOOK.json")
    return {name:hashlib.sha256((bootstrap.HERE/name).read_bytes()).hexdigest() for name in files}


class Learner:
    def __init__(self, arm, world):
        if arm not in ARMS:
            raise ValueError("locked arm roster")
        self.arm, self.world = arm, world
        self.anatomy, self.state, self.births = canonical_states()
        self.p, self.mom = {}, {}
        self.has_context = arm != "NATIVE"
        self.learns = self.has_context
        self.error_rule = self.has_context and arm[1] == "1"
        self.additive = self.has_context and arm[2] == "1"
        self.through = self.has_context and arm[3] == "1"
        if self.has_context:
            for name, shape in (("Wg", (64, 88)), ("Rg", (64, 64)), ("bg", (64,)),
                                ("W", (64, 88)), ("R", (64, 64)), ("b", (64,)),
                                ("V", (16, 64)), ("D", (88, 16))):
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
            self.mom = {k: torch.zeros_like(v) for k, v in self.p.items()}
        self.actor_steps, self.time = 0, 0.
        self.h = torch.zeros(64, dtype=torch.float64)
        self.fronts = []
        self.resource = dict(records=0, input_bytes=0, answer_bytes=0, EOS_targets=0,
                             native_store_events=0, actor_steps=0, production_cpu=0., readout_cpu=0.,
                             raw_alpha_l1=0., queries=0, legacy_readout_cpu=0., mem_readout_cpu=0.,
                             logical_MEM_cpu=0., logical_LEGACY_cpu=0.)
        self.last_trace, self.last_grads = [], {}
        self.frame_version = 0

    def clone(self):
        result = copy.copy(self)
        result.state = self.state.detached(True)
        result.p = {k: torch.nn.Parameter(v.detach().clone(), requires_grad=self.learns) for k, v in self.p.items()}
        result.mom = {k: v.clone() for k, v in self.mom.items()}
        result.h = self.h.detach().clone()
        result.fronts = [f.clone() for f in self.fronts]
        result.resource = self.resource.copy()
        result.last_trace, result.last_grads = [], {}
        return result

    def digest(self):
        arrays = [self.state.fast, self.state.slow, self.state.adapt, self.h]
        arrays += [v for _, v in sorted(self.p.items())] + [v for _, v in sorted(self.mom.items())]
        arrays += [f.p for f in self.fronts]
        metadata = [self.time, self.actor_steps, [(f.t, getattr(f, "visible", b"").hex()) for f in self.fronts]]
        return hashlib.sha256((array_hash(arrays) + json.dumps(metadata)).encode()).hexdigest()

    def theta_digest(self):
        return array_hash(list(self.p.values()) + list(self.mom.values()) + [np.array([self.actor_steps])])

    def _context(self, byte, rm):
        if not self.has_context or byte == EOS:
            return
        import stores
        r = tensor(stores.bb.bc.R_TABLE[byte])
        p = self.p
        g = torch.sigmoid(p["Wg"] @ r + p["Rg"] @ self.h + p["bg"])
        drive = p["W"] @ r + p["R"] @ self.h + p["b"]
        self.h = (1-g)*self.h + g*torch.tanh(drive)

    def _encode(self, fronts, at):
        pn = []
        for bank, fe in enumerate(fronts):
            fe.advance(at)
            p = tensor(fe.read())
            if self.has_context and bank == 0:
                raw = self.p["D"] @ (self.p["V"] @ self.h)
                eta = math.log(2) * torch.tanh(raw - raw.mean())
                p = torch.relu(p + eta) if self.additive else p * torch.exp(eta)
                p = p / p.max().clamp_min(1e-300)
            pn.append(p)
        return self.anatomy.encode(torch.stack(pn), torch.is_grad_enabled() and self.learns)

    def control(self, token):
        import stores
        if token == BEGIN:
            self.h = torch.zeros_like(self.h)
            self.fronts = [stores.bb.bc.FE0(), stores.content_model.VisibleContextFE()]
            for f in self.fronts:
                f.t = self.time + DELTA
        elif token != REPLY:
            raise ValueError("typed generic control required")
        self.state = self.anatomy.rest(self.state, DELTA)
        self.time += DELTA
        self.resource["native_store_events"] += 32
        self.frame_version += 1

    def event(self, byte, target=None, allowed=False, frame=None):
        if type(byte) is not int or not 0 <= byte <= EOS:
            raise ValueError("payload byte or EOS required")
        if type(allowed) is not bool or (target is not None and target != byte):
            raise ValueError("observed target must be arriving byte with explicit permission")
        if target is None and allowed:
            raise ValueError("generated/input byte cannot be labelled as observed feedback")
        if target is not None:
            if frame is None or frame["owner"] != id(self) or frame["version"] != self.frame_version or frame["time"] != self.time or frame["consumed"]:
                raise ValueError("missing,stale,cross-model or consumed prediction frame")
            frame["consumed"] = True
            x = frame["x"]
        else:
            x = self._encode(self.fronts, self.time + DELTA)
        if target is None:
            zero = torch.zeros((2,16), dtype=torch.float64)
            self.state, rm, info = self.anatomy.event(self.state, x, DELTA, zero, False)
            coefficient = None
        elif self.error_rule:
            pi = torch.softmax(frame["logits"].detach(), 0)
            coefficient = (pi @ CODE - CODE[target]) / 2
            self.state, rm, info = self.anatomy.teach_error(self.state, x, DELTA, coefficient, allowed)
        else:
            coefficient = {"shared": (1 - CODE[target])/2, "private": (MU - CODE[target])/2}
            self.state, rm, info = self.anatomy.teach(self.state, x, DELTA, coefficient, allowed)
        if byte != EOS:
            for f in self.fronts:
                f.feed(byte, self.time + DELTA)
            self._context(byte, rm)
        self.time += DELTA
        self.frame_version += 1
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

    def prediction_frame(self, readout="MEM"):
        x = self._encode([f.clone() for f in self.fronts], self.time + DELTA)
        state, dx = self.anatomy.query_activity(self.state, x, DELTA)
        expression = self.anatomy.expression(state, dx)
        if readout == "MEM":
            baseline = self.anatomy.expression(state, dx, True)
        elif readout == "LEGACY":
            if torch.is_grad_enabled() and self.learns:
                raise ValueError("legacy RBF is diagnostic only")
            baseline = tensor(self.anatomy.reader.predict(dx.detach().numpy().reshape(32,-1))).reshape(2,16,2)
        else:
            raise ValueError("locked readout")
        bank_values = (expression-baseline).mean(-1)
        values = bank_values[0]/1.4911274663291492 + bank_values[1]/1.3452365735750882
        logits = CODE @ (values-values.mean())/4
        return dict(owner=id(self),version=self.frame_version,time=self.time,consumed=False,
                    x=x,values=values,logits=logits,bank_values=bank_values,readout=readout)

    def values(self, readout="MEM"):
        return self.prediction_frame(readout)["values"]

    def forecast(self, readout="MEM"):
        return self.prediction_frame(readout)["logits"]

    def train_window(self, records, allowed=True, apply_update=True, detach_between_records=None):
        if len(records)!=4 or type(allowed) is not bool:
            raise ValueError("exactly four records and explicit learning permission")
        if detach_between_records is not None and detach_between_records != (not self.through):
            raise ValueError("credit mode is frozen by arm identity")
        detach_between_records = not self.through
        start = time.process_time()
        mem_read = legacy_read = 0.
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
                    # Separate logical baseline costs: do not charge both readouts
                    # to either deployable single-readout native condition.
                    if self.arm=="NATIVE" and self.resource["records"]%2:
                        t=time.process_time();self.forecast("LEGACY")
                        legacy_read+=time.process_time()-t
                    t=time.process_time();frame = self.prediction_frame();scores = frame["logits"]
                    mem_read+=time.process_time()-t
                    if self.arm=="NATIVE" and not self.resource["records"]%2:
                        t=time.process_time();self.forecast("LEGACY")
                        legacy_read+=time.process_time()-t
                    terms.append(F.cross_entropy(scores[None,:], torch.tensor([target])))
                    self.last_trace.append(dict(predicted=int(scores.detach().argmax()), target=target,
                                                prefeedback=True, native_write=allowed, logits=scores.detach().tolist()))
                    self.last_trace[-1].update(frame_version=frame["version"],pretime=frame["time"],
                        signal=((torch.softmax(scores.detach(),0)@CODE-CODE[target])/2).tolist() if self.error_rule else None,
                        frame_address_reused=True,error_rule=self.error_rule)
                    self.event(target, target, allowed, frame)
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
        self.h = self.h.detach()
        elapsed=time.process_time()-start
        self.resource["production_cpu"] += elapsed
        self.resource["mem_readout_cpu"] += mem_read
        self.resource["legacy_readout_cpu"] += legacy_read
        self.resource["readout_cpu"] += mem_read+legacy_read
        self.resource["logical_MEM_cpu"] += elapsed-legacy_read
        self.resource["logical_LEGACY_cpu"] += elapsed-mem_read if self.arm=="NATIVE" else 0.
        self.assert_finite()
        return dict(loss=float(loss.detach()), gradient_norm=norm, actor_steps=self.actor_steps,
                    records=4, allowed=allowed, applied_update=bool(self.learns and allowed and apply_update))

    def assert_finite(self):
        if not all(torch.isfinite(x).all() for x in (self.state.fast, self.state.slow, self.state.adapt,
                                                     self.h,*self.p.values(),*self.mom.values())):
            raise FloatingPointError("nonfinite native or context state")

    def checkpoint(self, path, cursor):
        """Boundary-only state. No graph, hidden parent, or training examples."""
        if any(v.grad_fn is not None for v in (self.state.fast,self.state.slow,self.state.adapt,self.h)):
            raise ValueError("checkpoint only at a detached complete-window boundary")
        from pathlib import Path
        import os
        value=dict(schema="NATIVE_BIND_CHECKPOINT_V2_1",arm=self.arm,world=self.world,
                   source_identity=science_identity(),
                   cursor=cursor,time=self.time,actor_steps=self.actor_steps,
                   state={k:getattr(self.state,k).clone() for k in ("fast","slow","adapt")},
                   p={k:v.detach().clone() for k,v in self.p.items()},
                   mom={k:v.clone() for k,v in self.mom.items()},h=self.h.clone(),
                   fronts=[dict(p=f.p.copy(),t=f.t,visible=getattr(f,"visible",None)) for f in self.fronts],
                   resource=self.resource.copy(),digest=self.digest())
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix(path.suffix+".tmp")
        torch.save(value,temporary);os.replace(temporary,path)

    @classmethod
    def restore(cls,path):
        import stores
        # Local explicitly produced checkpoint; numpy front arrays require full loading.
        value=torch.load(path,map_location="cpu",weights_only=False)
        if value.get("schema")!="NATIVE_BIND_CHECKPOINT_V2_1":
            raise ValueError("wrong checkpoint")
        if value.get("source_identity")!=science_identity():
            raise ValueError("checkpoint belongs to different scientific source")
        result=cls(value["arm"],value["world"])
        if set(result.p)!=set(value["p"]):
            raise ValueError("parameter set mismatch")
        result.state=State(*(value["state"][k].clone() for k in ("fast","slow","adapt")))
        for k,p in result.p.items():
            with torch.no_grad():p.copy_(value["p"][k])
        result.mom={k:v.clone() for k,v in value["mom"].items()}
        result.h=value["h"].clone()
        result.time,result.actor_steps=value["time"],value["actor_steps"]
        result.fronts=[]
        for i,f in enumerate(value["fronts"]):
            fe=stores.bb.bc.FE0() if i==0 else stores.content_model.VisibleContextFE()
            fe.p=f["p"].copy();fe.t=f["t"]
            if i:fe.visible=f["visible"]
            result.fronts.append(fe)
        result.resource=value["resource"].copy()
        if result.digest()!=value["digest"]:
            raise AssertionError("checkpoint state mismatch")
        return result,value["cursor"]

    def rest(self, seconds):
        if not math.isfinite(seconds) or seconds<0:
            raise ValueError("nonnegative time")
        self.state = self.anatomy.rest(self.state, seconds)
        self.time += seconds
        self.frame_version += 1

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
