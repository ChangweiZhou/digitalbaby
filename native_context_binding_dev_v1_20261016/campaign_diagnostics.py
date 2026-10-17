"""Fixed, read-only snapshot measurements. Never calls a training function."""
import math
import time
import types
import bootstrap
import torch
from bind_kernel import Learner, EOS, DELTA


def generated(view, prompt, mode="NORMAL", donor=None):
    view = view.clone()
    original = Learner._context
    if mode in ("ZERO_CONTEXT", "ZERO_ADAPT"):
        def context(self, byte, rm):
            if mode == "ZERO_ADAPT":
                self.a = torch.zeros_like(self.a)
            original(self, byte, rm)
            if mode == "ZERO_CONTEXT":
                self.h = torch.zeros_like(self.h)
        view._context = types.MethodType(context, view)
    if mode == "ZERO_FEEDBACK" and view.feedback:
        view.p["G"].zero_()
    view.prompt(prompt)
    if mode == "SWAP_CONTEXT" and donor is not None:
        other = view.clone()
        other.prompt(donor)
        view.h,view.a = other.h.clone(),other.a.clone()
    code = view._encode([f.clone() for f in view.fronts], view.time+DELTA)
    summary = dict(h_norm=float(view.h.norm()), a_norm=float(view.a.norm()),
                   h_saturation_fraction=float((view.h.abs()>.99).double().mean()),
                   active_KCs=[int((x>0).sum()) for x in code],
                   feedback_max=getattr(view,"last_feedback_max",0.))
    if view.has_context:
        pns=[]
        for bank,fe in enumerate(view.fronts):
            p=torch.tensor(fe.read().copy(),dtype=torch.float64)
            raw=view.p[f"D{bank}"]@(view.p["V"]@view.h)
            eta=math.log(2)*torch.tanh(raw-raw.mean())
            pns.append(dict(log_modulation_norm=float(eta.norm()),
                            log_modulation_max=float(eta.abs().max())))
        summary["PN_modulation"]=pns
    answer=bytearray()
    for _ in range(17):
        target=int(view.forecast().argmax())
        if target==EOS:
            return bytes(answer).hex(),"EOS",summary,code
        if len(answer)==16:
            return bytes(answer).hex(),"TRUNCATED",summary,code
        answer.append(target);view.event(target)
    raise AssertionError("unreachable")


def measure(model, items, theta_views=None):
    start=time.process_time();before=model.digest();rows=[]
    with torch.no_grad():
        for index,(prompt,answer) in enumerate(items):
            modes=["NORMAL","ZERO_CONTEXT","SWAP_CONTEXT"]
            if model.adaptive:modes.append("ZERO_ADAPT")
            if model.feedback:modes.append("ZERO_FEEDBACK")
            views={}
            basecode=None
            for mode in modes:
                emitted,status,features,code=generated(model,prompt,mode,items[(index+1)%len(items)][0])
                if basecode is None:basecode=code
                views[mode]=dict(emitted=emitted,status=status,correct=status=="EOS" and emitted==answer.hex(),
                                 features=features,hard_KC_difference=int((code!=basecode).sum()))
            for name,theta in (theta_views or {}).items():
                if not model.has_context:continue
                other=model.clone()
                for k,v in other.p.items():v.copy_(theta[k])
                emitted,status,features,code=generated(other,prompt)
                views[name]=dict(emitted=emitted,status=status,correct=status=="EOS" and emitted==answer.hex(),
                                 features=features,hard_KC_difference=int((code!=basecode).sum()))
            zero=model.clone();zero.state.fast.zero_();zero.state.slow.zero_()
            zero.prompt(prompt)
            value=zero.values()
            if not torch.equal(value,torch.zeros_like(value)):
                raise AssertionError("Zero-memory MEM bypass check failed")
            rows.append(dict(prompt=prompt.hex(),views=views,zero_native_MEM_exact=True))
        adaptive=None
        if model.adaptive:
            tau=2+30*torch.sigmoid(model.p["xi"]);beta=torch.sigmoid(model.p["zeta"])
            adaptive=dict(tau=tau.tolist(),beta=beta.tolist(),activity=model.a.tolist())
    # Exact native-memory query derivative, not the top-k surrogate or new training.
    query=model.clone()
    query.prompt(items[0][0])
    query.state.fast=query.state.fast.detach().requires_grad_(True)
    query.state.slow=query.state.slow.detach().requires_grad_(True)
    scores=query.forecast()
    derivative=torch.autograd.grad(scores.max(),(query.state.fast,query.state.slow),allow_unused=True)
    native_query_grad=[0. if g is None else float(g.norm()) for g in derivative]
    if model.digest()!=before:raise AssertionError("Diagnostics changed production state")
    return dict(rows=rows,adaptive=adaptive,native_query_gradient_norms=native_query_grad,
                CPU_seconds=time.process_time()-start,training_records=0,
                interpretation="Descriptive interventions; coordinate mismatch and off-distribution effects are not causal efficacy proof")
