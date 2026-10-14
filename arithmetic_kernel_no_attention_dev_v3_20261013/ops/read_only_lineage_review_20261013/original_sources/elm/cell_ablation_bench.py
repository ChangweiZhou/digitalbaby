#!/usr/bin/env python3
"""P4: isolate which ELM cell properties matter for sample-efficient memory.

All models use the same Fashion-MNIST label pool, presentation schedule, 64-cell
population, and sparse source topology. Parameter counts are reported rather
than silently called 'matched'. The first comparison is therefore a synaptic-
geometry / sample-budget control; parameter-matched follow-ups can be built only
if a cell family shows a meaningful effect.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

import developmental_elm_fashion as base
from research_common import (
    CyclingLoader,
    csv_dump,
    fashion_datasets,
    json_dump,
    make_fashion_model,
    make_loader,
    nested_budget_indices,
    parameter_count,
    stratified_nested_split,
)


class LinearDendriteELM(base.DevelopmentalELM):
    """ELM with dendritic tanh removed; the internal neuron MLP is retained."""
    def forward(self, images, *, record_structure=False, delay_override=None, ablate_neurons=None):
        batch = images.size(0)
        dtype, device = images.dtype, images.device
        memory = torch.zeros(batch, self.n, self.m_dim, dtype=dtype, device=device)
        activity_prev = torch.zeros(batch, self.n, dtype=dtype, device=device)
        source_flat_idx = self.source_idx.reshape(-1)
        kappa = self.kappa().unsqueeze(0)
        branch_gain = F.softplus(self.branch_gain_raw).unsqueeze(0)
        threshold = F.softplus(self.threshold_raw).unsqueeze(0)
        activities, memories = [], []
        for sensory in self.make_episode(images, delay_override=delay_override):
            sources = torch.cat([sensory, activity_prev], dim=1)
            gathered = sources.index_select(1, source_flat_idx).reshape(batch, self.n, self.r, self.s)
            branch_sum = (gathered * self.syn_weight.unsqueeze(0)).sum(-1) + self.branch_bias.unsqueeze(0)
            branch_activity = branch_gain * branch_sum  # only change versus full ELM
            internal_input = torch.cat([branch_activity, memory], dim=-1)
            hidden = torch.tanh(torch.einsum("bnq,nqh->bnh", internal_input, self.internal_w1) + self.internal_b1.unsqueeze(0))
            candidate = torch.tanh(torch.einsum("bnh,nhm->bnm", hidden, self.internal_w2) + self.internal_b2.unsqueeze(0))
            memory = kappa * memory + (1-kappa) * candidate
            voltage = (
                (memory * self.memory_to_activity.unsqueeze(0)).sum(-1)
                + (branch_activity * self.branch_to_activity.unsqueeze(0)).sum(-1)
                + self.activity_bias.unsqueeze(0)
            )
            activity = torch.sigmoid(self.cfg.firing_gain * (voltage - threshold))
            activities.append(activity); memories.append(memory); activity_prev = activity
        ah = torch.stack(activities, 1)
        mh = torch.stack(memories, 1)
        return {"logits": self.readout(activity_prev), "activity": ah, "memory": mh, "final_activity": activity_prev}


class SparseScalarBase(nn.Module):
    input_dim = 784
    def __init__(self, source_idx: torch.Tensor, cfg: base.Config):
        super().__init__()
        self.cfg = cfg
        self.n = cfg.neurons
        self.r = cfg.branches
        self.s = cfg.synapses_per_branch
        self.s_in = cfg.input_synapses_per_branch
        self.register_buffer("source_idx", source_idx.detach().clone().long())
        self.readout = nn.Linear(self.n, 10)

    def make_episode(self, images, delay_override=None):
        flat = images.reshape(images.size(0), -1)
        delay = self.cfg.delay_steps if delay_override is None else int(delay_override)
        episode = [flat] * self.cfg.flash_steps
        zero = torch.zeros_like(flat)
        episode.extend([zero] * delay)
        return episode

    def _sources(self, sensory, prev):
        src = torch.cat([sensory, prev], 1)
        return src.index_select(1, self.source_idx.reshape(-1)).reshape(src.size(0), self.n, self.r, self.s)


class SparseLeaky(SparseScalarBase):
    def __init__(self, source_idx, cfg):
        super().__init__(source_idx, cfg)
        self.w = nn.Parameter(torch.randn(self.n, self.r, self.s) * 0.18)
        self.bias = nn.Parameter(torch.zeros(self.n))
        self.leak_raw = nn.Parameter(torch.zeros(self.n))
        self.threshold = nn.Parameter(torch.full((self.n,), 0.25))

    def forward(self, images, *, delay_override=None, **_):
        h = torch.zeros(images.size(0), self.n, device=images.device, dtype=images.dtype)
        prev = torch.zeros_like(h)
        hist = []
        leak = 0.50 + 0.49 * torch.sigmoid(self.leak_raw)
        for sensory in self.make_episode(images, delay_override):
            g = self._sources(sensory, prev)
            drive = (g * self.w.unsqueeze(0)).sum((2,3)) + self.bias
            cand = torch.tanh(drive)
            h = leak.unsqueeze(0) * h + (1-leak).unsqueeze(0) * cand
            prev = torch.sigmoid(4.0 * (h - self.threshold.unsqueeze(0)))
            hist.append(prev)
        ah = torch.stack(hist, 1)
        return {"logits": self.readout(prev), "activity": ah, "final_activity": prev, "memory": h.unsqueeze(1)}


class SparseALIF(SparseScalarBase):
    def __init__(self, source_idx, cfg):
        super().__init__(source_idx, cfg)
        self.w = nn.Parameter(torch.randn(self.n, self.r, self.s) * 0.18)
        self.bias = nn.Parameter(torch.zeros(self.n))
        self.v_leak_raw = nn.Parameter(torch.zeros(self.n))
        self.a_leak_raw = nn.Parameter(torch.zeros(self.n))
        self.base_threshold = nn.Parameter(torch.full((self.n,), 0.25))
        self.adapt_gain = nn.Parameter(torch.full((self.n,), 0.50))

    def forward(self, images, *, delay_override=None, **_):
        v = torch.zeros(images.size(0), self.n, device=images.device, dtype=images.dtype)
        a = torch.zeros_like(v)
        prev = torch.zeros_like(v)
        hist = []
        vl = 0.50 + 0.49 * torch.sigmoid(self.v_leak_raw)
        al = 0.50 + 0.499 * torch.sigmoid(self.a_leak_raw)
        for sensory in self.make_episode(images, delay_override):
            g = self._sources(sensory, prev)
            drive = (g * self.w.unsqueeze(0)).sum((2,3)) + self.bias
            v = vl.unsqueeze(0) * v + (1-vl).unsqueeze(0) * torch.tanh(drive)
            threshold = self.base_threshold.unsqueeze(0) + F.softplus(self.adapt_gain).unsqueeze(0) * a
            prev = torch.sigmoid(4.0 * (v - threshold))
            a = al.unsqueeze(0) * a + (1-al).unsqueeze(0) * prev
            hist.append(prev)
        ah = torch.stack(hist, 1)
        return {"logits": self.readout(prev), "activity": ah, "final_activity": prev, "memory": torch.stack([v,a],-1).unsqueeze(1)}


class SparseGRU(SparseScalarBase):
    def __init__(self, source_idx, cfg):
        super().__init__(source_idx, cfg)
        self.wz = nn.Parameter(torch.randn(self.n, self.r, self.s) * 0.12)
        self.wr = nn.Parameter(torch.randn(self.n, self.r, self.s) * 0.12)
        self.wh = nn.Parameter(torch.randn(self.n, self.r, self.s) * 0.12)
        self.bz = nn.Parameter(torch.zeros(self.n))
        self.br = nn.Parameter(torch.zeros(self.n))
        self.bh = nn.Parameter(torch.zeros(self.n))
        self.uh = nn.Parameter(torch.ones(self.n) * 0.5)

    def forward(self, images, *, delay_override=None, **_):
        h = torch.zeros(images.size(0), self.n, device=images.device, dtype=images.dtype)
        prev = torch.zeros_like(h)
        hist = []
        for sensory in self.make_episode(images, delay_override):
            g = self._sources(sensory, prev)
            z = torch.sigmoid((g * self.wz.unsqueeze(0)).sum((2,3)) + self.bz)
            r = torch.sigmoid((g * self.wr.unsqueeze(0)).sum((2,3)) + self.br)
            cand = torch.tanh((g * self.wh.unsqueeze(0)).sum((2,3)) + self.bh + self.uh.unsqueeze(0) * r * h)
            h = (1-z) * h + z * cand
            prev = torch.sigmoid(4.0 * h)
            hist.append(prev)
        ah = torch.stack(hist, 1)
        return {"logits": self.readout(prev), "activity": ah, "final_activity": prev, "memory": h.unsqueeze(1)}


def generic_eval(model, loader, device, cfg, delay):
    model.eval(); correct=0; n=0; activity=0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            out = model(images, delay_override=delay)
            correct += int((out["logits"].argmax(1)==labels).sum())
            n += int(labels.numel())
            activity += float(out["activity"].mean().cpu()) * labels.numel()
    return {"accuracy": correct/max(1,n), "mean_activity": activity/max(1,n)}


def build_model(name: str, *, model_seed: int, topology_seed: int, device, corrected_tau: bool):
    overrides = {"rewire_fraction": 0.0, "homeostasis_coef": 4.0, "decorrelation_coef": 0.002}
    # Create a reference ELM first so every control receives exactly its topology.
    ref = make_fashion_model(
        model_seed=model_seed, topology_seed=topology_seed, device=torch.device("cpu"),
        corrected_tau=corrected_tau, cfg_overrides=overrides,
    )
    topology = ref.source_idx.detach().cpu().clone()
    cfg = ref.cfg
    # Reset RNG so ELM variants remain paired in their common parameters.
    base.seed_everything(model_seed)
    if name == "full":
        model = base.DevelopmentalELM(cfg)
        if corrected_tau:
            from research_common import reinitialize_log_tau
            reinitialize_log_tau(model, seed=model_seed+91337)
    elif name == "fixed_tau":
        model = base.DevelopmentalELM(cfg)
        if corrected_tau:
            from research_common import reinitialize_log_tau
            reinitialize_log_tau(model, seed=model_seed+91337)
        model.kappa_raw.requires_grad_(False)
    elif name == "linear_dendrite":
        model = LinearDendriteELM(cfg)
        if corrected_tau:
            from research_common import reinitialize_log_tau
            reinitialize_log_tau(model, seed=model_seed+91337)
    elif name == "leaky":
        model = SparseLeaky(topology, cfg)
    elif name == "alif":
        model = SparseALIF(topology, cfg)
    elif name == "sparse_gru":
        model = SparseGRU(topology, cfg)
    else:
        raise ValueError(name)
    if isinstance(model, base.DevelopmentalELM):
        from research_common import apply_topology
        apply_topology(model, topology)
    return model.to(device), cfg


def main() -> int:
    p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--models", default="full,fixed_tau,linear_dendrite,leaky,alif,sparse_gru")
    p.add_argument("--budget", type=int, default=1000)
    p.add_argument("--model-seeds", default="7,17,29")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--output-root", default="./cell_ablation_runs")
    p.add_argument("--device", default="auto", choices=["auto","mps","cpu","cuda"])
    p.add_argument("--data-seed", type=int, default=101)
    p.add_argument("--loader-seed", type=int, default=202)
    p.add_argument("--topology-seed", type=int, default=303)
    p.add_argument("--corrected-tau", action="store_true", default=True)
    p.add_argument("--legacy-tau", action="store_false", dest="corrected_tau")
    args=p.parse_args()

    device=base.choose_device(args.device)
    train_full,test_full=fashion_datasets(args.data_dir)
    pool,val_idx=stratified_nested_split(train_full,max_budget=max(5000,args.budget),validation_size=5000,data_seed=args.data_seed)
    train_idx=nested_budget_indices(train_full,pool,args.budget)
    val_loader=make_loader(train_full,val_idx,batch_size=512,shuffle=False,seed=0)
    test_loader=make_loader(test_full,list(range(len(test_full))),batch_size=512,shuffle=False,seed=0)
    models=[x.strip() for x in args.models.split(",") if x.strip()]
    seeds=[int(x) for x in args.model_seeds.split(",") if x.strip()]
    root=Path(args.output_root).expanduser().resolve(); root.mkdir(parents=True,exist_ok=True)
    aggregate=[]

    for seed in seeds:
        for name in models:
            model,cfg=build_model(name,model_seed=seed,topology_seed=args.topology_seed,device=device,corrected_tau=args.corrected_tau)
            loader=make_loader(train_full,train_idx,batch_size=args.batch_size,shuffle=True,seed=args.loader_seed)
            stream=CyclingLoader(loader)
            opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.learning_rate,weight_decay=args.weight_decay)
            sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.steps,eta_min=args.learning_rate*0.05)
            hist=[]; preq_ce=0.; preq_correct=0; preq_n=0; start=time.perf_counter()
            for step in range(1,args.steps+1):
                model.train(); images,labels=stream.next(); images,labels=images.to(device),labels.to(device)
                opt.zero_grad(set_to_none=True); out=model(images)
                loss,parts=base.classification_loss(out,labels,cfg)
                preq_ce += float(F.cross_entropy(out["logits"],labels,reduction="sum").detach().cpu())
                preq_correct += int((out["logits"].argmax(1)==labels).sum()); preq_n += labels.numel()
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sched.step()
                if step==1 or step%args.eval_every==0 or step==args.steps:
                    v=generic_eval(model,val_loader,device,cfg,cfg.delay_steps)
                    hist.append({"step":step,"presentations":stream.presentations,"val_accuracy":v["accuracy"],"prequential_accuracy":preq_correct/preq_n,"prequential_ce":preq_ce/preq_n})
            t=generic_eval(model,test_loader,device,cfg,cfg.delay_steps)
            summary={"model":name,"budget":args.budget,"seed":seed,"test_accuracy":t["accuracy"],"test_activity":t["mean_activity"],"presentations":stream.presentations,"prequential_accuracy":preq_correct/preq_n,"prequential_ce":preq_ce/preq_n,"trainable_parameters":parameter_count(model),"elapsed_seconds":time.perf_counter()-start}
            for d in (8,12,16,24):
                m=generic_eval(model,test_loader,device,cfg,d); summary[f"delay_{d}_accuracy"]=m["accuracy"]; summary[f"delay_{d}_activity"]=m["mean_activity"]
            rd=root/f"{name}_n{args.budget}_seed{seed}"; rd.mkdir(exist_ok=True)
            csv_dump(rd/"history.csv",hist); json_dump(rd/"summary.json",summary); aggregate.append(summary)
            print(f"{name:16s} seed={seed}: test={summary['test_accuracy']:.3%} d24={summary['delay_24_accuracy']:.3%} params={summary['trainable_parameters']}")
    csv_dump(root/"cell_ablation_summary.csv",aggregate)
    return 0

if __name__=="__main__": raise SystemExit(main())
