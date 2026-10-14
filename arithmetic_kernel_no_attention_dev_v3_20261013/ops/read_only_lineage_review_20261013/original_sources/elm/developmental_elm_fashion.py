#!/usr/bin/env python3
"""
Developmental ELM network for Fashion-MNIST.

Purpose
-------
This is a small, interpretable experiment in biological machine learning:

* 64 expressive leaky-memory (ELM) neurons arranged on an 8 x 8 sheet.
* Each neuron has nonlinear dendritic branches, multiple learned leak
  timescales, and ONE scalar output communicated to other neurons.
* Images are presented as short episodes (flash, then blank delay), so memory
  is genuinely useful.
* Connectivity changes slowly with a RigL-like prune/regrow rule evaluated on
  a held-out architecture split. The neuron count and synapse budget remain
  fixed in this first version.
* No convolution, self-attention, Q/K/V, Transformer block, persistent global
  hub, teacher model, or reinforcement-learning reward model is used.

The script is designed to run on Apple Silicon through PyTorch MPS, with a CPU
fallback. It automatically produces an HTML report plus CSV/JSON/PNG artifacts
that expose what the network learned: receptive fields, recurrent wiring,
learned timescales, class-selective neurons, activity traces, neuron ablations,
and edge turnover.

Examples
--------
Quick smoke test:
    python developmental_elm_fashion.py --quick

Full default run:
    python developmental_elm_fashion.py

A longer-delay experiment:
    python developmental_elm_fashion.py --delay-steps 12 --epochs 24

Analyze a saved checkpoint without retraining:
    python developmental_elm_fashion.py --analyze-only path/to/checkpoint.pt
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

# Matplotlib is only used after training, so it does not affect the hot path.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


FASHION_CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


# -----------------------------------------------------------------------------
# Configuration and reproducibility
# -----------------------------------------------------------------------------


@dataclass
class Config:
    seed: int = 7
    data_dir: str = "./data"
    output_root: str = "./runs"
    run_name: str = ""
    device: str = "auto"  # auto | mps | cpu | cuda

    epochs: int = 20
    batch_size: int = 256
    arch_batch_size: int = 128
    learning_rate: float = 2.0e-3
    weight_decay: float = 1.0e-4
    grad_clip: float = 1.0
    num_workers: int = 0

    # Split of Fashion-MNIST's official 60,000-example training set.
    weight_train_size: int = 50_000
    architecture_size: int = 5_000
    validation_size: int = 5_000
    train_limit: int = 0
    architecture_limit: int = 0
    validation_limit: int = 0
    test_limit: int = 0

    # Population geometry.
    neurons: int = 64
    grid_height: int = 8
    grid_width: int = 8

    # One ELM neuron.
    memory_dim: int = 6
    branches: int = 8
    synapses_per_branch: int = 4
    input_synapses_per_branch: int = 3
    internal_hidden: int = 12
    kappa_low: float = 0.08
    kappa_high: float = 0.995
    firing_gain: float = 4.0

    # Sensory episode.
    protocol: str = "flash"  # flash | saccade
    flash_steps: int = 4
    delay_steps: int = 8

    # Activity regularization.
    target_activity: float = 0.10
    homeostasis_coef: float = 4.0
    decorrelation_coef: float = 0.002

    # Slow structural plasticity.
    rewire_start_epoch: int = 4
    rewire_end_epoch: int = 14
    rewire_every: int = 2
    rewire_fraction: float = 0.04
    architecture_batches: int = 1
    minimum_edge_age: int = 2
    input_wiring_cost: float = 0.60
    recurrent_wiring_cost: float = 0.30

    # Interpretability pass.
    analysis_samples: int = 5_000
    prototype_neurons: int = 12
    prototypes_per_neuron: int = 8
    explained_samples: int = 6

    quick: bool = False
    analyze_only: str = ""

    def validate(self) -> None:
        if self.neurons != self.grid_height * self.grid_width:
            raise ValueError(
                "neurons must equal grid_height * grid_width "
                f"({self.neurons} != {self.grid_height} * {self.grid_width})"
            )
        if not (0 < self.input_synapses_per_branch < self.synapses_per_branch):
            raise ValueError(
                "input_synapses_per_branch must lie strictly between 0 and "
                "synapses_per_branch"
            )
        if self.protocol not in {"flash", "saccade"}:
            raise ValueError("protocol must be 'flash' or 'saccade'")
        if self.flash_steps < 1 or self.delay_steps < 0:
            raise ValueError("flash_steps must be >= 1 and delay_steps must be >= 0")
        if not (0.0 <= self.rewire_fraction < 1.0):
            raise ValueError("rewire_fraction must be in [0, 1)")
        if self.architecture_batches < 1:
            raise ValueError("architecture_batches must be >= 1")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but torch.backends.mps.is_available() is False")
        return torch.device("mps")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device: {requested}")


def make_run_dir(cfg: Config) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = cfg.run_name.strip() or f"delm_fashion_{stamp}"
    run_dir = Path(cfg.output_root).expanduser().resolve() / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------


def _limit_indices(indices: Sequence[int], limit: int) -> List[int]:
    if limit <= 0 or limit >= len(indices):
        return list(indices)
    return list(indices[:limit])


def build_dataloaders(
    cfg: Config,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, Dataset]:
    transform = transforms.ToTensor()
    root = str(Path(cfg.data_dir).expanduser())
    train_full = datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    test_full = datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)

    total_requested = cfg.weight_train_size + cfg.architecture_size + cfg.validation_size
    if total_requested > len(train_full):
        raise ValueError(
            f"Requested split size {total_requested} exceeds Fashion-MNIST train size {len(train_full)}"
        )

    generator = torch.Generator().manual_seed(cfg.seed)
    permutation = torch.randperm(len(train_full), generator=generator).tolist()
    n_w = cfg.weight_train_size
    n_a = cfg.architecture_size
    n_v = cfg.validation_size

    weight_idx = _limit_indices(permutation[:n_w], cfg.train_limit)
    arch_idx = _limit_indices(permutation[n_w:n_w + n_a], cfg.architecture_limit)
    val_idx = _limit_indices(permutation[n_w + n_a:n_w + n_a + n_v], cfg.validation_limit)
    test_idx = _limit_indices(list(range(len(test_full))), cfg.test_limit)

    weight_ds = Subset(train_full, weight_idx)
    arch_ds = Subset(train_full, arch_idx)
    val_ds = Subset(train_full, val_idx)
    test_ds = Subset(test_full, test_idx)

    loader_kwargs = dict(
        num_workers=cfg.num_workers,
        pin_memory=False,
        persistent_workers=(cfg.num_workers > 0),
    )
    train_loader = DataLoader(
        weight_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        **loader_kwargs,
    )
    arch_loader = DataLoader(
        arch_ds,
        batch_size=cfg.arch_batch_size,
        shuffle=True,
        drop_last=False,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    return train_loader, arch_loader, val_loader, test_loader, test_full


# -----------------------------------------------------------------------------
# Developmental ELM network
# -----------------------------------------------------------------------------


def inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def inverse_bounded_sigmoid(value: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    unit = ((value - lo) / (hi - lo)).clamp(1.0e-5, 1.0 - 1.0e-5)
    return torch.logit(unit)


class DevelopmentalELM(nn.Module):
    """Vectorized population of scalar-output expressive leaky-memory neurons."""

    input_dim: int = 28 * 28

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.n = cfg.neurons
        self.r = cfg.branches
        self.s = cfg.synapses_per_branch
        self.s_in = cfg.input_synapses_per_branch
        self.s_rec = self.s - self.s_in
        self.m_dim = cfg.memory_dim
        self.h_dim = cfg.internal_hidden
        self.total_sources = self.input_dim + self.n

        neuron_positions = self._make_neuron_positions()
        pixel_positions = self._make_pixel_positions()
        self.register_buffer("neuron_positions", neuron_positions)
        self.register_buffer("pixel_positions", pixel_positions)
        self.register_buffer("pixel_distance", self._pixel_distance_matrix(neuron_positions, pixel_positions))
        self.register_buffer("recurrent_distance", self._recurrent_distance_matrix(neuron_positions))

        source_idx = self._initial_topology()
        self.register_buffer("source_idx", source_idx)
        self.register_buffer("initial_source_idx", source_idx.clone())
        self.register_buffer("edge_age", torch.full_like(source_idx, 100, dtype=torch.long))

        # Synaptic weights: [neuron, branch, slot]. Inactive candidate edges are
        # not materialized in the hot path; source_idx identifies the active edge.
        self.syn_weight = nn.Parameter(torch.empty(self.n, self.r, self.s))
        with torch.no_grad():
            self.syn_weight[:, :, : self.s_in].normal_(0.0, 0.24)
            self.syn_weight[:, :, self.s_in :].normal_(0.0, 0.16)

        self.branch_bias = nn.Parameter(torch.full((self.n, self.r), -0.05))
        self.branch_gain_raw = nn.Parameter(
            torch.full((self.n, self.r), inverse_softplus(1.0))
        )

        # Each neuron has its own small nonlinear internal map:
        # [branch activities, old memories] -> hidden -> candidate memories.
        q_dim = self.r + self.m_dim
        self.internal_w1 = nn.Parameter(torch.empty(self.n, q_dim, self.h_dim))
        self.internal_b1 = nn.Parameter(torch.zeros(self.n, self.h_dim))
        self.internal_w2 = nn.Parameter(torch.empty(self.n, self.h_dim, self.m_dim))
        self.internal_b2 = nn.Parameter(torch.zeros(self.n, self.m_dim))
        # Xavier initialization must be applied neuron-by-neuron. Applying it to
        # the entire 3-D tensor would incorrectly treat the neuron axis as a
        # convolutional input-channel axis and make each micro-network too small.
        with torch.no_grad():
            for neuron in range(self.n):
                nn.init.xavier_uniform_(self.internal_w1[neuron])
                nn.init.xavier_uniform_(self.internal_w2[neuron])

        # Initialize heterogeneous memory timescales logarithmically.
        base_tau = torch.logspace(
            math.log10(1.5), math.log10(32.0), steps=self.m_dim
        )
        initial_kappa = torch.exp(-1.0 / base_tau)
        initial_kappa = initial_kappa.unsqueeze(0).repeat(self.n, 1)
        jitter = torch.randn_like(initial_kappa) * 0.06
        initial_kappa = (initial_kappa + jitter).clamp(
            cfg.kappa_low + 1.0e-4, cfg.kappa_high - 1.0e-4
        )
        self.kappa_raw = nn.Parameter(
            inverse_bounded_sigmoid(initial_kappa, cfg.kappa_low, cfg.kappa_high)
        )

        # Scalar motor output per neuron.
        self.memory_to_activity = nn.Parameter(torch.empty(self.n, self.m_dim))
        self.branch_to_activity = nn.Parameter(torch.empty(self.n, self.r))
        nn.init.normal_(self.memory_to_activity, mean=0.0, std=0.30)
        nn.init.normal_(self.branch_to_activity, mean=0.0, std=0.10)
        self.activity_bias = nn.Parameter(torch.zeros(self.n))
        self.threshold_raw = nn.Parameter(
            torch.full((self.n,), inverse_softplus(0.75))
        )

        # The classifier sees only the final scalar population activity.
        self.readout = nn.Linear(self.n, len(FASHION_CLASSES))

        # Four quadrant masks for the optional saccadic protocol.
        self.register_buffer("saccade_masks", self._make_saccade_masks())

    # ---- Geometry and topology ------------------------------------------------

    def _make_neuron_positions(self) -> torch.Tensor:
        ys = torch.linspace(0.0, 1.0, self.cfg.grid_height)
        xs = torch.linspace(0.0, 1.0, self.cfg.grid_width)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

    @staticmethod
    def _make_pixel_positions() -> torch.Tensor:
        ys = torch.linspace(0.0, 1.0, 28)
        xs = torch.linspace(0.0, 1.0, 28)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

    @staticmethod
    def _pixel_distance_matrix(
        neuron_positions: torch.Tensor, pixel_positions: torch.Tensor
    ) -> torch.Tensor:
        return torch.cdist(neuron_positions, pixel_positions) / math.sqrt(2.0)

    @staticmethod
    def _recurrent_distance_matrix(neuron_positions: torch.Tensor) -> torch.Tensor:
        return torch.cdist(neuron_positions, neuron_positions) / math.sqrt(2.0)

    @staticmethod
    def _make_saccade_masks() -> torch.Tensor:
        masks = []
        for y0, y1, x0, x1 in [
            (0, 16, 0, 16),
            (0, 16, 12, 28),
            (12, 28, 0, 16),
            (12, 28, 12, 28),
        ]:
            mask = torch.zeros(28, 28)
            mask[y0:y1, x0:x1] = 1.0
            masks.append(mask.reshape(-1))
        return torch.stack(masks, dim=0)

    def _initial_topology(self) -> torch.Tensor:
        idx = torch.empty(self.n, self.r, self.s, dtype=torch.long)
        rng = np.random.default_rng(self.cfg.seed)
        pixel_dist = self.pixel_distance.cpu().numpy()
        rec_dist = self.recurrent_distance.cpu().numpy()

        for neuron in range(self.n):
            # Most afferents are selected from a local candidate pool; a small
            # fraction are global exploratory contacts.
            nearest_pixels = np.argsort(pixel_dist[neuron])[:128].tolist()
            global_pixels = list(range(self.input_dim))
            chosen_pixels: set[int] = set()

            # One recurrent input slot per branch in the default 3+1 design.
            nearest_neurons = [
                int(j) for j in np.argsort(rec_dist[neuron]).tolist() if int(j) != neuron
            ]
            recurrent_pool = nearest_neurons[:24]
            chosen_recurrent: set[int] = set()

            for branch in range(self.r):
                for slot in range(self.s_in):
                    use_global = rng.random() < 0.15
                    pool = global_pixels if use_global else nearest_pixels
                    available = [p for p in pool if p not in chosen_pixels]
                    if not available:
                        available = [p for p in global_pixels if p not in chosen_pixels]
                    source = int(rng.choice(available))
                    chosen_pixels.add(source)
                    idx[neuron, branch, slot] = source

                for slot in range(self.s_in, self.s):
                    use_global = rng.random() < 0.20
                    pool = nearest_neurons if use_global else recurrent_pool
                    available = [p for p in pool if p not in chosen_recurrent and p != neuron]
                    if not available:
                        available = [
                            p for p in nearest_neurons if p not in chosen_recurrent and p != neuron
                        ]
                    source_neuron = int(rng.choice(available))
                    chosen_recurrent.add(source_neuron)
                    idx[neuron, branch, slot] = self.input_dim + source_neuron
        return idx

    # ---- Dynamics -------------------------------------------------------------

    def kappa(self) -> torch.Tensor:
        cfg = self.cfg
        return cfg.kappa_low + (cfg.kappa_high - cfg.kappa_low) * torch.sigmoid(
            self.kappa_raw
        )

    def timescales(self) -> torch.Tensor:
        k = self.kappa().clamp(1.0e-6, 1.0 - 1.0e-6)
        return -1.0 / torch.log(k)

    def make_episode(
        self, images: torch.Tensor, delay_override: Optional[int] = None
    ) -> List[torch.Tensor]:
        flat = images.reshape(images.size(0), -1)
        delay = self.cfg.delay_steps if delay_override is None else int(delay_override)
        episode: List[torch.Tensor] = []

        if self.cfg.protocol == "flash":
            episode.extend([flat] * self.cfg.flash_steps)
        else:
            for t in range(self.cfg.flash_steps):
                mask = self.saccade_masks[t % self.saccade_masks.size(0)]
                episode.append(flat * mask.unsqueeze(0))

        zero = torch.zeros_like(flat)
        episode.extend([zero] * delay)
        return episode

    def forward(
        self,
        images: torch.Tensor,
        *,
        record_structure: bool = False,
        delay_override: Optional[int] = None,
        ablate_neurons: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        batch = images.size(0)
        dtype = images.dtype
        device = images.device

        memory = torch.zeros(batch, self.n, self.m_dim, dtype=dtype, device=device)
        activity_prev = torch.zeros(batch, self.n, dtype=dtype, device=device)

        source_flat_idx = self.source_idx.reshape(-1)
        kappa = self.kappa().unsqueeze(0)
        branch_gain = F.softplus(self.branch_gain_raw).unsqueeze(0)
        threshold = F.softplus(self.threshold_raw).unsqueeze(0)

        activities: List[torch.Tensor] = []
        memories: List[torch.Tensor] = []
        source_history: List[torch.Tensor] = []
        branch_sum_nodes: List[torch.Tensor] = []

        ablation_idx: Optional[torch.Tensor] = None
        if ablate_neurons:
            clean = sorted({int(i) for i in ablate_neurons if 0 <= int(i) < self.n})
            if clean:
                ablation_idx = torch.tensor(clean, device=device, dtype=torch.long)

        for sensory in self.make_episode(images, delay_override=delay_override):
            sources = torch.cat([sensory, activity_prev], dim=1)
            gathered = sources.index_select(1, source_flat_idx).reshape(
                batch, self.n, self.r, self.s
            )
            branch_sum = (
                gathered * self.syn_weight.unsqueeze(0)
            ).sum(dim=-1) + self.branch_bias.unsqueeze(0)

            if record_structure:
                branch_sum.retain_grad()
                source_history.append(sources)
                branch_sum_nodes.append(branch_sum)

            branch_activity = torch.tanh(branch_gain * branch_sum)
            internal_input = torch.cat([branch_activity, memory], dim=-1)
            hidden = torch.tanh(
                torch.einsum("bnq,nqh->bnh", internal_input, self.internal_w1)
                + self.internal_b1.unsqueeze(0)
            )
            candidate = torch.tanh(
                torch.einsum("bnh,nhm->bnm", hidden, self.internal_w2)
                + self.internal_b2.unsqueeze(0)
            )
            memory = kappa * memory + (1.0 - kappa) * candidate

            voltage = (
                (memory * self.memory_to_activity.unsqueeze(0)).sum(dim=-1)
                + (branch_activity * self.branch_to_activity.unsqueeze(0)).sum(dim=-1)
                + self.activity_bias.unsqueeze(0)
            )
            activity = torch.sigmoid(
                self.cfg.firing_gain * (voltage - threshold)
            )
            if ablation_idx is not None:
                activity = activity.index_fill(1, ablation_idx, 0.0)

            activities.append(activity)
            memories.append(memory)
            activity_prev = activity

        activity_history = torch.stack(activities, dim=1)
        memory_history = torch.stack(memories, dim=1)
        logits = self.readout(activity_prev)

        result: Dict[str, Any] = {
            "logits": logits,
            "activity": activity_history,
            "memory": memory_history,
            "final_activity": activity_prev,
        }
        if record_structure:
            result["source_history"] = source_history
            result["branch_sum_nodes"] = branch_sum_nodes
        return result

    # ---- Structural statistics ------------------------------------------------

    def edge_distance(self, neuron: int, source: int) -> float:
        if source < self.input_dim:
            return float(self.pixel_distance[neuron, source].detach().cpu())
        source_neuron = source - self.input_dim
        return float(self.recurrent_distance[neuron, source_neuron].detach().cpu())

    @torch.no_grad()
    def enforce_no_duplicate_sources(self) -> None:
        # This is a defensive assertion, not a topology mutation.
        idx = self.source_idx.detach().cpu()
        for neuron in range(self.n):
            inputs = idx[neuron, :, : self.s_in].reshape(-1).tolist()
            rec = (idx[neuron, :, self.s_in :] - self.input_dim).reshape(-1).tolist()
            if len(inputs) != len(set(inputs)):
                raise RuntimeError(f"Duplicate input source detected for neuron {neuron}")
            if len(rec) != len(set(rec)):
                raise RuntimeError(f"Duplicate recurrent source detected for neuron {neuron}")
            if neuron in rec:
                raise RuntimeError(f"Self recurrent edge detected for neuron {neuron}")


# -----------------------------------------------------------------------------
# Loss, training, and evaluation
# -----------------------------------------------------------------------------


def classification_loss(
    output: Dict[str, Any], labels: torch.Tensor, cfg: Config
) -> Tuple[torch.Tensor, Dict[str, float]]:
    logits = output["logits"]
    activity = output["activity"]
    final_activity = output["final_activity"]

    ce = F.cross_entropy(logits, labels)
    per_neuron_rate = activity.mean(dim=(0, 1))
    homeostasis = ((per_neuron_rate - cfg.target_activity) ** 2).mean()

    # A small final-state decorrelation term discourages redundant cells.
    if final_activity.size(0) > 1:
        centered = final_activity - final_activity.mean(dim=0, keepdim=True)
        std = centered.pow(2).mean(dim=0, keepdim=True).sqrt().clamp_min(1.0e-4)
        normalized = centered / std
        corr = normalized.transpose(0, 1) @ normalized / float(final_activity.size(0))
        eye = torch.eye(corr.size(0), device=corr.device, dtype=corr.dtype)
        decorrelation = ((corr - eye) ** 2).mean()
    else:
        decorrelation = logits.new_zeros(())

    loss = (
        ce
        + cfg.homeostasis_coef * homeostasis
        + cfg.decorrelation_coef * decorrelation
    )
    metrics = {
        "ce": float(ce.detach().cpu()),
        "homeostasis": float(homeostasis.detach().cpu()),
        "decorrelation": float(decorrelation.detach().cpu()),
        "mean_activity": float(activity.mean().detach().cpu()),
    }
    return loss, metrics


@torch.no_grad()
def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[int, int]:
    pred = logits.argmax(dim=1)
    correct = int((pred == labels).sum().item())
    return correct, int(labels.numel())


def train_one_epoch(
    model: DevelopmentalELM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: Config,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_activity = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        output = model(images)
        loss, parts = classification_loss(output, labels, cfg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        batch = labels.size(0)
        correct, count = accuracy_from_logits(output["logits"], labels)
        total_loss += float(loss.detach().cpu()) * batch
        total_ce += parts["ce"] * batch
        total_activity += parts["mean_activity"] * batch
        total_correct += correct
        total_examples += count

    denom = max(1, total_examples)
    return {
        "loss": total_loss / denom,
        "ce": total_ce / denom,
        "accuracy": total_correct / denom,
        "mean_activity": total_activity / denom,
    }


@torch.no_grad()
def evaluate(
    model: DevelopmentalELM,
    loader: DataLoader,
    device: torch.device,
    cfg: Config,
    *,
    delay_override: Optional[int] = None,
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    total_activity = 0.0
    confusion = torch.zeros(10, 10, dtype=torch.long)

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        output = model(images, delay_override=delay_override)
        loss, parts = classification_loss(output, labels, cfg)
        pred = output["logits"].argmax(dim=1)

        batch = labels.size(0)
        total_loss += float(loss.detach().cpu()) * batch
        total_correct += int((pred == labels).sum().item())
        total_examples += batch
        total_activity += parts["mean_activity"] * batch

        for true_value, pred_value in zip(labels.detach().cpu(), pred.detach().cpu()):
            confusion[int(true_value), int(pred_value)] += 1

    per_class = []
    for cls in range(10):
        row = confusion[cls]
        per_class.append(float(row[cls].item() / max(1, row.sum().item())))

    denom = max(1, total_examples)
    return {
        "loss": total_loss / denom,
        "accuracy": total_correct / denom,
        "mean_activity": total_activity / denom,
        "confusion": confusion,
        "per_class_accuracy": per_class,
    }


# -----------------------------------------------------------------------------
# RigL-like structural plasticity
# -----------------------------------------------------------------------------


def _next_batches(loader: DataLoader, count: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
    iterator = iter(loader)
    for _ in range(count):
        try:
            batches.append(next(iterator))
        except StopIteration:
            iterator = iter(loader)
            batches.append(next(iterator))
    return batches


def collect_structure_scores(
    model: DevelopmentalELM,
    arch_loader: DataLoader,
    device: torch.device,
    cfg: Config,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Return active-edge utility and zero-edge growth scores.

    For a hypothetical zero-weight edge from source s into branch r of neuron n,
    the local first-order score is

        | sum_{batch,time} dL/d(branch_sum[n,r]) * source[s] |.

    This is exact for the immediate branch contribution and is a practical local
    approximation for recurrent edges, whose addition would also change future
    source activities.
    """
    model.eval()
    active_utility = torch.zeros_like(model.syn_weight, device=device)
    growth_score = torch.zeros(
        model.n, model.r, model.total_sources, device=device
    )
    total_ce = 0.0

    batches = _next_batches(arch_loader, cfg.architecture_batches)
    for images, labels in batches:
        images = images.to(device)
        labels = labels.to(device)
        model.zero_grad(set_to_none=True)

        output = model(images, record_structure=True)
        ce = F.cross_entropy(output["logits"], labels)
        ce.backward()
        total_ce += float(ce.detach().cpu())

        if model.syn_weight.grad is None:
            raise RuntimeError("syn_weight gradient was not produced during architecture scoring")
        active_utility += (model.syn_weight.detach() * model.syn_weight.grad.detach()).abs()

        branch_grad = torch.stack(
            [node.grad.detach() for node in output["branch_sum_nodes"]], dim=1
        )
        source_history = torch.stack(
            [source.detach() for source in output["source_history"]], dim=1
        )
        local_gradient = torch.einsum(
            "btnr,bts->nrs", branch_grad, source_history
        ) / float(images.size(0) * source_history.size(1))
        growth_score += local_gradient.abs()

    scale = float(len(batches))
    active_utility /= scale
    growth_score /= scale
    model.zero_grad(set_to_none=True)
    return active_utility.detach().cpu(), growth_score.detach().cpu(), total_ce / scale


def _reset_adam_slots(
    optimizer: torch.optim.Optimizer,
    parameter: nn.Parameter,
    changed: torch.Tensor,
) -> None:
    state = optimizer.state.get(parameter)
    if not state:
        return
    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        value = state.get(key)
        if isinstance(value, torch.Tensor) and value.shape == parameter.shape:
            value[changed.to(value.device)] = 0.0


def _adjust_growth_for_wiring(
    model: DevelopmentalELM,
    neuron: int,
    raw_score: torch.Tensor,
    source_kind: str,
    cfg: Config,
) -> torch.Tensor:
    score = raw_score.clone()
    if source_kind == "input":
        distance = model.pixel_distance[neuron].detach().cpu()
        return score * torch.exp(-cfg.input_wiring_cost * distance)
    distance = model.recurrent_distance[neuron].detach().cpu()
    return score * torch.exp(-cfg.recurrent_wiring_cost * distance)


def rewire_model(
    model: DevelopmentalELM,
    optimizer: torch.optim.Optimizer,
    active_utility: torch.Tensor,
    growth_score: torch.Tensor,
    cfg: Config,
    epoch: int,
) -> Dict[str, Any]:
    """Prune low-utility active edges and regrow high-gradient dormant edges."""
    if cfg.rewire_fraction <= 0.0:
        return {
            "epoch": epoch,
            "input_rewired": 0,
            "recurrent_rewired": 0,
            "architecture_ce": None,
        }

    source_idx = model.source_idx.detach().cpu().clone()
    edge_age = model.edge_age.detach().cpu().clone()
    changed = torch.zeros_like(source_idx, dtype=torch.bool)

    records: List[Dict[str, Any]] = []

    def process_kind(kind: str) -> int:
        if kind == "input":
            slots = range(0, model.s_in)
            source_start, source_end = 0, model.input_dim
        else:
            slots = range(model.s_in, model.s)
            source_start, source_end = model.input_dim, model.total_sources

        candidates: List[Tuple[float, int, int, int]] = []
        for neuron in range(model.n):
            for branch in range(model.r):
                for slot in slots:
                    if int(edge_age[neuron, branch, slot]) >= cfg.minimum_edge_age:
                        candidates.append(
                            (
                                float(active_utility[neuron, branch, slot]),
                                neuron,
                                branch,
                                slot,
                            )
                        )
        candidates.sort(key=lambda item: item[0])
        total_edges = model.n * model.r * len(list(slots))
        n_replace = min(len(candidates), max(1, int(round(cfg.rewire_fraction * total_edges))))
        selected = candidates[:n_replace]

        for old_utility, neuron, branch, slot in selected:
            old_source = int(source_idx[neuron, branch, slot])

            if kind == "input":
                currently_used = set(
                    source_idx[neuron, :, : model.s_in].reshape(-1).tolist()
                )
                raw = growth_score[neuron, branch, source_start:source_end]
                score = _adjust_growth_for_wiring(model, neuron, raw, kind, cfg)
                for used in currently_used:
                    if 0 <= used < model.input_dim:
                        score[used] = -float("inf")
                new_local = int(torch.argmax(score).item())
                new_source = new_local
                new_growth = float(raw[new_local])
            else:
                currently_used_rec = set(
                    (
                        source_idx[neuron, :, model.s_in :].reshape(-1)
                        - model.input_dim
                    ).tolist()
                )
                raw = growth_score[neuron, branch, source_start:source_end]
                score = _adjust_growth_for_wiring(model, neuron, raw, kind, cfg)
                score[neuron] = -float("inf")
                for used in currently_used_rec:
                    if 0 <= used < model.n:
                        score[used] = -float("inf")
                new_local = int(torch.argmax(score).item())
                new_source = model.input_dim + new_local
                new_growth = float(raw[new_local])

            if not math.isfinite(float(score[new_local])):
                # Extremely unlikely fallback when all candidates are excluded.
                available = [
                    s
                    for s in range(source_start, source_end)
                    if s not in set(source_idx[neuron].reshape(-1).tolist())
                    and not (kind == "recurrent" and s == model.input_dim + neuron)
                ]
                if not available:
                    continue
                new_source = int(random.choice(available))
                new_growth = 0.0

            source_idx[neuron, branch, slot] = new_source
            edge_age[neuron, branch, slot] = 0
            changed[neuron, branch, slot] = True
            records.append(
                {
                    "epoch": epoch,
                    "kind": kind,
                    "target_neuron": neuron,
                    "branch": branch,
                    "slot": slot,
                    "old_source": old_source,
                    "new_source": new_source,
                    "old_utility": old_utility,
                    "new_growth_score": new_growth,
                    "old_distance": model.edge_distance(neuron, old_source),
                    "new_distance": model.edge_distance(neuron, new_source),
                }
            )
        return len(selected)

    input_rewired = process_kind("input")
    recurrent_rewired = process_kind("recurrent")

    with torch.no_grad():
        model.source_idx.copy_(source_idx.to(model.source_idx.device))
        model.edge_age.copy_(edge_age.to(model.edge_age.device))
        model.syn_weight[changed.to(model.syn_weight.device)] = 0.0
    _reset_adam_slots(optimizer, model.syn_weight, changed)
    model.enforce_no_duplicate_sources()

    def mean_record(field: str, kind: Optional[str] = None) -> float:
        values = [
            float(record[field])
            for record in records
            if kind is None or record["kind"] == kind
        ]
        return float(np.mean(values)) if values else 0.0

    return {
        "epoch": epoch,
        "input_rewired": input_rewired,
        "recurrent_rewired": recurrent_rewired,
        "old_input_distance": mean_record("old_distance", "input"),
        "new_input_distance": mean_record("new_distance", "input"),
        "old_recurrent_distance": mean_record("old_distance", "recurrent"),
        "new_recurrent_distance": mean_record("new_distance", "recurrent"),
        "records": records,
    }


def should_rewire(cfg: Config, epoch: int) -> bool:
    return (
        cfg.rewire_fraction > 0.0
        and cfg.rewire_start_epoch <= epoch <= cfg.rewire_end_epoch
        and (epoch - cfg.rewire_start_epoch) % cfg.rewire_every == 0
    )


# -----------------------------------------------------------------------------
# Checkpointing
# -----------------------------------------------------------------------------


def save_checkpoint(
    path: Path,
    model: DevelopmentalELM,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    cfg: Config,
    history: List[Dict[str, Any]],
    rewire_history: List[Dict[str, Any]],
    epoch: int,
) -> None:
    payload = {
        "config": asdict(cfg),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "history": history,
        "rewire_history": rewire_history,
        "epoch": epoch,
    }
    torch.save(payload, path)


def load_model_from_checkpoint(
    checkpoint_path: str, device: torch.device
) -> Tuple[DevelopmentalELM, Config, Dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = Config(**payload["config"])
    cfg.validate()
    model = DevelopmentalELM(cfg)
    model.load_state_dict(payload["model"])
    model.to(device)
    return model, cfg, payload


# -----------------------------------------------------------------------------
# Interpretability and reporting
# -----------------------------------------------------------------------------


def save_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def save_training_plots(
    run_dir: Path,
    history: List[Dict[str, Any]],
    rewire_history: List[Dict[str, Any]],
) -> List[str]:
    produced: List[str] = []
    if not history:
        return produced

    epochs = [row["epoch"] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["train_loss"] for row in history], label="train")
    plt.plot(epochs, [row["val_loss"] for row in history], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.tight_layout()
    filename = "training_loss.png"
    plt.savefig(run_dir / filename, dpi=150)
    plt.close()
    produced.append(filename)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["train_accuracy"] for row in history], label="train")
    plt.plot(epochs, [row["val_accuracy"] for row in history], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Classification accuracy")
    plt.legend()
    plt.tight_layout()
    filename = "accuracy.png"
    plt.savefig(run_dir / filename, dpi=150)
    plt.close()
    produced.append(filename)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["train_activity"] for row in history], label="train")
    plt.plot(epochs, [row["val_activity"] for row in history], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Mean scalar activity")
    plt.title("Population activity")
    plt.legend()
    plt.tight_layout()
    filename = "population_activity.png"
    plt.savefig(run_dir / filename, dpi=150)
    plt.close()
    produced.append(filename)

    if rewire_history:
        plt.figure(figsize=(8, 5))
        x = [row["epoch"] for row in rewire_history]
        plt.plot(x, [row["input_rewired"] for row in rewire_history], marker="o", label="input")
        plt.plot(
            x,
            [row["recurrent_rewired"] for row in rewire_history],
            marker="o",
            label="recurrent",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Edges replaced")
        plt.title("Slow structural plasticity events")
        plt.legend()
        plt.tight_layout()
        filename = "rewiring_events.png"
        plt.savefig(run_dir / filename, dpi=150)
        plt.close()
        produced.append(filename)

        if all("input_retained_fraction" in row for row in rewire_history):
            plt.figure(figsize=(8, 5))
            plt.plot(
                x,
                [row["input_retained_fraction"] for row in rewire_history],
                marker="o",
                label="input",
            )
            plt.plot(
                x,
                [row["recurrent_retained_fraction"] for row in rewire_history],
                marker="o",
                label="recurrent",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Fraction of initial edges still present")
            plt.title("Cumulative architectural change")
            plt.legend()
            plt.tight_layout()
            filename = "edge_retention.png"
            plt.savefig(run_dir / filename, dpi=150)
            plt.close()
            produced.append(filename)

        if all("mean_input_wiring_distance" in row for row in rewire_history):
            plt.figure(figsize=(8, 5))
            plt.plot(
                x,
                [row["mean_input_wiring_distance"] for row in rewire_history],
                marker="o",
                label="input",
            )
            plt.plot(
                x,
                [row["mean_recurrent_wiring_distance"] for row in rewire_history],
                marker="o",
                label="recurrent",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Mean normalized wiring distance")
            plt.title("How wiring geometry changes")
            plt.legend()
            plt.tight_layout()
            filename = "wiring_distance.png"
            plt.savefig(run_dir / filename, dpi=150)
            plt.close()
            produced.append(filename)

    return produced


@torch.no_grad()
def collect_analysis_data(
    model: DevelopmentalELM,
    loader: DataLoader,
    device: torch.device,
    sample_limit: int,
) -> Dict[str, Any]:
    model.eval()
    all_images: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    all_preds: List[torch.Tensor] = []
    all_probs: List[torch.Tensor] = []
    all_final_activity: List[torch.Tensor] = []
    all_activity_history: List[torch.Tensor] = []
    total = 0

    for images, labels in loader:
        if total >= sample_limit:
            break
        take = min(images.size(0), sample_limit - total)
        images = images[:take]
        labels = labels[:take]
        output = model(images.to(device))
        probs = F.softmax(output["logits"], dim=1)
        pred = probs.argmax(dim=1)

        all_images.append(images.cpu())
        all_labels.append(labels.cpu())
        all_preds.append(pred.cpu())
        all_probs.append(probs.cpu())
        all_final_activity.append(output["final_activity"].cpu())
        all_activity_history.append(output["activity"].cpu())
        total += take

    return {
        "images": torch.cat(all_images, dim=0),
        "labels": torch.cat(all_labels, dim=0),
        "preds": torch.cat(all_preds, dim=0),
        "probs": torch.cat(all_probs, dim=0),
        "final_activity": torch.cat(all_final_activity, dim=0),
        "activity_history": torch.cat(all_activity_history, dim=0),
    }


def compute_receptive_fields(model: DevelopmentalELM) -> Tuple[np.ndarray, np.ndarray]:
    idx = model.source_idx.detach().cpu().numpy()
    weight = model.syn_weight.detach().cpu().numpy()
    signed = np.zeros((model.n, model.input_dim), dtype=np.float32)
    absolute = np.zeros((model.n, model.input_dim), dtype=np.float32)
    for neuron in range(model.n):
        for branch in range(model.r):
            for slot in range(model.s_in):
                pixel = int(idx[neuron, branch, slot])
                value = float(weight[neuron, branch, slot])
                signed[neuron, pixel] += value
                absolute[neuron, pixel] += abs(value)
    return signed, absolute


def topology_statistics(model: DevelopmentalELM) -> Dict[str, Any]:
    current = model.source_idx.detach().cpu()
    initial = model.initial_source_idx.detach().cpu()
    weights = model.syn_weight.detach().cpu()

    input_now = current[:, :, : model.s_in]
    input_initial = initial[:, :, : model.s_in]
    rec_now = current[:, :, model.s_in :]
    rec_initial = initial[:, :, model.s_in :]

    input_retained = float((input_now == input_initial).float().mean())
    rec_retained = float((rec_now == rec_initial).float().mean())

    input_distances = []
    recurrent_distances = []
    recurrent_pairs = set()
    for neuron in range(model.n):
        for branch in range(model.r):
            for slot in range(model.s):
                source = int(current[neuron, branch, slot])
                if slot < model.s_in:
                    input_distances.append(model.edge_distance(neuron, source))
                else:
                    source_neuron = source - model.input_dim
                    recurrent_distances.append(model.edge_distance(neuron, source))
                    recurrent_pairs.add((source_neuron, neuron))
    reciprocal = sum(1 for edge in recurrent_pairs if (edge[1], edge[0]) in recurrent_pairs) // 2

    out_degree = np.zeros(model.n, dtype=np.int64)
    for source_neuron, _target in recurrent_pairs:
        out_degree[source_neuron] += 1

    return {
        "input_retained_fraction": input_retained,
        "recurrent_retained_fraction": rec_retained,
        "mean_input_wiring_distance": float(np.mean(input_distances)),
        "mean_recurrent_wiring_distance": float(np.mean(recurrent_distances)),
        "reciprocal_recurrent_pairs": int(reciprocal),
        "highest_recurrent_out_degree": int(out_degree.max()),
        "highest_out_degree_neuron": int(out_degree.argmax()),
        "mean_absolute_synaptic_weight": float(weights.abs().mean()),
    }


def class_neuron_statistics(
    model: DevelopmentalELM, analysis: Dict[str, Any]
) -> Dict[str, np.ndarray]:
    activity = analysis["final_activity"].numpy()
    labels = analysis["labels"].numpy()
    class_mean = np.zeros((10, model.n), dtype=np.float32)
    for cls in range(10):
        mask = labels == cls
        if mask.any():
            class_mean[cls] = activity[mask].mean(axis=0)

    preferred = class_mean.argmax(axis=0)
    sorted_means = np.sort(class_mean, axis=0)
    selectivity = sorted_means[-1] - sorted_means[-2]
    mean_activity = activity.mean(axis=0)
    readout = model.readout.weight.detach().cpu().numpy()
    readout_importance = np.max(np.abs(readout), axis=0)
    return {
        "class_mean": class_mean,
        "preferred_class": preferred,
        "selectivity": selectivity,
        "mean_activity": mean_activity,
        "readout_importance": readout_importance,
    }


def save_confusion_matrix(run_dir: Path, confusion: torch.Tensor) -> str:
    matrix = confusion.detach().cpu().numpy()
    plt.figure(figsize=(8, 7))
    plt.imshow(matrix, aspect="auto")
    plt.colorbar(label="Count")
    plt.xticks(range(10), FASHION_CLASSES, rotation=45, ha="right")
    plt.yticks(range(10), FASHION_CLASSES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Test confusion matrix")
    plt.tight_layout()
    filename = "confusion_matrix.png"
    plt.savefig(run_dir / filename, dpi=150)
    plt.close()
    return filename


def save_timescale_plot(run_dir: Path, model: DevelopmentalELM) -> str:
    tau = model.timescales().detach().cpu().numpy()
    order = np.argsort(tau.mean(axis=1))
    plt.figure(figsize=(8, 9))
    plt.imshow(tau[order], aspect="auto")
    plt.colorbar(label="Effective time constant (steps)")
    plt.xlabel("Memory channel")
    plt.ylabel("Neuron (sorted fast to slow)")
    plt.title("Learned ELM memory timescales")
    plt.tight_layout()
    filename = "learned_timescales.png"
    plt.savefig(run_dir / filename, dpi=150)
    plt.close()
    return filename


def save_class_neuron_map(
    run_dir: Path, stats: Dict[str, np.ndarray]
) -> str:
    preferred = stats["preferred_class"]
    selectivity = stats["selectivity"]
    order = np.lexsort((-selectivity, preferred))
    plt.figure(figsize=(12, 5))
    plt.imshow(stats["class_mean"][:, order], aspect="auto")
    plt.colorbar(label="Mean final activity")
    plt.yticks(range(10), FASHION_CLASSES)
    plt.xlabel("Neuron (grouped by preferred class)")
    plt.ylabel("True class")
    plt.title("Which classes activate which neurons?")
    plt.tight_layout()
    filename = "class_neuron_activity.png"
    plt.savefig(run_dir / filename, dpi=150)
    plt.close()
    return filename


def save_connectivity_plot(
    run_dir: Path,
    model: DevelopmentalELM,
    stats: Dict[str, np.ndarray],
) -> str:
    pos = model.neuron_positions.detach().cpu().numpy()
    idx = model.source_idx.detach().cpu().numpy()
    weight = model.syn_weight.detach().cpu().numpy()
    edges: List[Tuple[float, int, int]] = []
    for target in range(model.n):
        for branch in range(model.r):
            for slot in range(model.s_in, model.s):
                source = int(idx[target, branch, slot] - model.input_dim)
                edges.append((abs(float(weight[target, branch, slot])), source, target))
    edges.sort(reverse=True)
    edges = edges[: min(160, len(edges))]

    plt.figure(figsize=(8, 8))
    max_weight = max((edge[0] for edge in edges), default=1.0)
    for magnitude, source, target in edges:
        x = [pos[source, 0], pos[target, 0]]
        y = [1.0 - pos[source, 1], 1.0 - pos[target, 1]]
        plt.plot(x, y, linewidth=0.3 + 1.8 * magnitude / max_weight, alpha=0.30)
    sizes = 45.0 + 260.0 * (
        stats["readout_importance"]
        / max(1.0e-8, float(stats["readout_importance"].max()))
    )
    scatter = plt.scatter(
        pos[:, 0],
        1.0 - pos[:, 1],
        c=stats["preferred_class"],
        s=sizes,
    )
    plt.colorbar(scatter, ticks=range(10), label="Preferred Fashion-MNIST class")
    for neuron in range(model.n):
        plt.text(pos[neuron, 0], 1.0 - pos[neuron, 1], str(neuron), fontsize=6, ha="center", va="center")
    plt.axis("off")
    plt.title("Strongest learned recurrent connections\n(node size = readout importance)")
    plt.tight_layout()
    filename = "learned_recurrent_graph.png"
    plt.savefig(run_dir / filename, dpi=170)
    plt.close()
    return filename


def save_afferent_coverage(run_dir: Path, absolute_rf: np.ndarray) -> str:
    coverage = absolute_rf.sum(axis=0).reshape(28, 28)
    plt.figure(figsize=(6, 6))
    plt.imshow(coverage)
    plt.colorbar(label="Total absolute afferent weight")
    plt.axis("off")
    plt.title("Where the population samples the image")
    plt.tight_layout()
    filename = "population_afferent_coverage.png"
    plt.savefig(run_dir / filename, dpi=150)
    plt.close()
    return filename


def save_receptive_fields(
    run_dir: Path,
    absolute_rf: np.ndarray,
    model: DevelopmentalELM,
    stats: Dict[str, np.ndarray],
    selected_neurons: Sequence[int],
) -> List[str]:
    directory = run_dir / "receptive_fields"
    directory.mkdir(exist_ok=True)
    tau = model.timescales().detach().cpu().numpy()
    files: List[str] = []
    for neuron in selected_neurons:
        plt.figure(figsize=(5, 5))
        plt.imshow(absolute_rf[neuron].reshape(28, 28))
        plt.colorbar(label="Absolute afferent weight")
        plt.axis("off")
        title = (
            f"Neuron {neuron}: prefers {FASHION_CLASSES[int(stats['preferred_class'][neuron])]}\n"
            f"mean time constant {tau[neuron].mean():.1f} steps"
        )
        plt.title(title)
        plt.tight_layout()
        filename = f"receptive_fields/neuron_{neuron:02d}.png"
        plt.savefig(run_dir / filename, dpi=150)
        plt.close()
        files.append(filename)
    return files


def _tensor_image_to_pil(image: torch.Tensor, scale: int = 5) -> Image.Image:
    array = (image.squeeze(0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    pil = Image.fromarray(array, mode="L")
    return pil.resize((28 * scale, 28 * scale), resample=Image.Resampling.NEAREST)


def save_neuron_prototypes(
    run_dir: Path,
    analysis: Dict[str, Any],
    selected_neurons: Sequence[int],
    per_neuron: int,
) -> List[str]:
    directory = run_dir / "neuron_prototypes"
    directory.mkdir(exist_ok=True)
    images = analysis["images"]
    labels = analysis["labels"]
    activity = analysis["final_activity"]
    files: List[str] = []

    tile_w, tile_h = 140, 166
    for neuron in selected_neurons:
        top_idx = torch.topk(activity[:, neuron], k=min(per_neuron, activity.size(0))).indices
        canvas = Image.new("L", (tile_w * len(top_idx), tile_h), 255)
        draw = ImageDraw.Draw(canvas)
        for j, sample_idx in enumerate(top_idx.tolist()):
            tile = _tensor_image_to_pil(images[sample_idx], scale=5)
            canvas.paste(tile, (j * tile_w, 0))
            label = FASHION_CLASSES[int(labels[sample_idx])]
            value = float(activity[sample_idx, neuron])
            draw.text((j * tile_w + 2, 142), f"{label}\na={value:.3f}", fill=0)
        filename = f"neuron_prototypes/neuron_{neuron:02d}.png"
        canvas.save(run_dir / filename)
        files.append(filename)
    return files


def save_activity_raster(
    run_dir: Path,
    activity_history: torch.Tensor,
    sample_index: int,
    true_label: int,
    pred_label: int,
) -> str:
    values = activity_history[sample_index].numpy().T
    plt.figure(figsize=(9, 7))
    plt.imshow(values, aspect="auto")
    plt.colorbar(label="Scalar neuron activity")
    plt.xlabel("Episode step")
    plt.ylabel("Neuron")
    plt.title(
        f"Activity raster: true {FASHION_CLASSES[true_label]}, "
        f"predicted {FASHION_CLASSES[pred_label]}"
    )
    plt.tight_layout()
    filename = f"sample_{sample_index:04d}_activity_raster.png"
    plt.savefig(run_dir / filename, dpi=150)
    plt.close()
    return filename


def save_sample_image(
    run_dir: Path, image: torch.Tensor, sample_index: int, true_label: int
) -> str:
    pil = _tensor_image_to_pil(image, scale=8)
    filename = f"sample_{sample_index:04d}_{true_label}.png"
    pil.save(run_dir / filename)
    return filename


def recurrent_degrees(model: DevelopmentalELM) -> Tuple[np.ndarray, np.ndarray]:
    in_degree = np.zeros(model.n, dtype=np.int64)
    out_degree = np.zeros(model.n, dtype=np.int64)
    idx = model.source_idx.detach().cpu().numpy()
    for target in range(model.n):
        for branch in range(model.r):
            for slot in range(model.s_in, model.s):
                source = int(idx[target, branch, slot] - model.input_dim)
                in_degree[target] += 1
                out_degree[source] += 1
    return in_degree, out_degree


def neuron_rows(
    model: DevelopmentalELM,
    stats: Dict[str, np.ndarray],
    absolute_rf: np.ndarray,
) -> List[Dict[str, Any]]:
    tau = model.timescales().detach().cpu().numpy()
    pos = model.neuron_positions.detach().cpu().numpy()
    pixels = model.pixel_positions.detach().cpu().numpy()
    in_degree, out_degree = recurrent_degrees(model)
    rows: List[Dict[str, Any]] = []
    for neuron in range(model.n):
        weights = absolute_rf[neuron]
        denom = max(1.0e-8, float(weights.sum()))
        centroid = (weights[:, None] * pixels).sum(axis=0) / denom
        rows.append(
            {
                "neuron": neuron,
                "sheet_x": float(pos[neuron, 0]),
                "sheet_y": float(pos[neuron, 1]),
                "preferred_class": FASHION_CLASSES[int(stats["preferred_class"][neuron])],
                "class_selectivity": float(stats["selectivity"][neuron]),
                "mean_activity": float(stats["mean_activity"][neuron]),
                "readout_importance": float(stats["readout_importance"][neuron]),
                "mean_timescale": float(tau[neuron].mean()),
                "minimum_timescale": float(tau[neuron].min()),
                "maximum_timescale": float(tau[neuron].max()),
                "receptive_field_centroid_x": float(centroid[0]),
                "receptive_field_centroid_y": float(centroid[1]),
                "recurrent_in_degree": int(in_degree[neuron]),
                "recurrent_out_degree": int(out_degree[neuron]),
            }
        )
    return rows


def edge_rows(model: DevelopmentalELM) -> List[Dict[str, Any]]:
    current = model.source_idx.detach().cpu()
    initial = model.initial_source_idx.detach().cpu()
    age = model.edge_age.detach().cpu()
    weight = model.syn_weight.detach().cpu()
    rows: List[Dict[str, Any]] = []
    for target in range(model.n):
        for branch in range(model.r):
            for slot in range(model.s):
                source = int(current[target, branch, slot])
                kind = "input" if source < model.input_dim else "recurrent"
                source_id = source if kind == "input" else source - model.input_dim
                row = {
                    "kind": kind,
                    "source": source_id,
                    "target_neuron": target,
                    "branch": branch,
                    "slot": slot,
                    "weight": float(weight[target, branch, slot]),
                    "absolute_weight": float(abs(weight[target, branch, slot])),
                    "distance": model.edge_distance(target, source),
                    "age": int(age[target, branch, slot]),
                    "changed_from_initial": int(source != int(initial[target, branch, slot])),
                }
                if kind == "input":
                    row["pixel_y"] = source // 28
                    row["pixel_x"] = source % 28
                else:
                    row["pixel_y"] = ""
                    row["pixel_x"] = ""
                rows.append(row)
    return rows


def top_pixel_inputs(model: DevelopmentalELM, neuron: int, k: int = 5) -> List[Dict[str, Any]]:
    idx = model.source_idx.detach().cpu()
    weight = model.syn_weight.detach().cpu()
    values: List[Tuple[float, int, float, int]] = []
    for branch in range(model.r):
        for slot in range(model.s_in):
            pixel = int(idx[neuron, branch, slot])
            w = float(weight[neuron, branch, slot])
            values.append((abs(w), pixel, w, branch))
    values.sort(reverse=True)
    return [
        {
            "pixel_x": pixel % 28,
            "pixel_y": pixel // 28,
            "weight": w,
            "branch": branch,
        }
        for _magnitude, pixel, w, branch in values[:k]
    ]


def top_recurrent_inputs(model: DevelopmentalELM, neuron: int, k: int = 5) -> List[Dict[str, Any]]:
    idx = model.source_idx.detach().cpu()
    weight = model.syn_weight.detach().cpu()
    values: List[Tuple[float, int, float, int]] = []
    for branch in range(model.r):
        for slot in range(model.s_in, model.s):
            source = int(idx[neuron, branch, slot] - model.input_dim)
            w = float(weight[neuron, branch, slot])
            values.append((abs(w), source, w, branch))
    values.sort(reverse=True)
    return [
        {"source_neuron": source, "weight": w, "branch": branch}
        for _magnitude, source, w, branch in values[:k]
    ]


@torch.no_grad()
def explain_samples(
    model: DevelopmentalELM,
    analysis: Dict[str, Any],
    device: torch.device,
    run_dir: Path,
    cfg: Config,
) -> List[Dict[str, Any]]:
    images = analysis["images"]
    labels = analysis["labels"]
    preds = analysis["preds"]
    probs = analysis["probs"]
    activity = analysis["final_activity"]
    history = analysis["activity_history"]
    readout = model.readout.weight.detach().cpu()

    # Choose confident correct examples from distinct classes, then uncertain or
    # incorrect examples. This produces a more informative report than random picks.
    selected: List[int] = []
    for cls in range(10):
        candidates = torch.where((labels == cls) & (preds == labels))[0]
        if candidates.numel() > 0:
            confidence = probs[candidates, cls]
            selected.append(int(candidates[confidence.argmax()]))
        if len(selected) >= max(3, cfg.explained_samples - 2):
            break

    wrong = torch.where(preds != labels)[0]
    if wrong.numel() > 0:
        wrong_confidence = probs[wrong].max(dim=1).values
        selected.append(int(wrong[wrong_confidence.argmax()]))
    uncertainty = probs.max(dim=1).values
    for idx_value in torch.argsort(uncertainty).tolist():
        if idx_value not in selected:
            selected.append(int(idx_value))
        if len(selected) >= cfg.explained_samples:
            break
    selected = selected[: cfg.explained_samples]

    tau = model.timescales().detach().cpu()
    explanations: List[Dict[str, Any]] = []
    for sample_idx in selected:
        true_label = int(labels[sample_idx])
        pred_label = int(preds[sample_idx])
        confidence = float(probs[sample_idx, pred_label])
        contribution = activity[sample_idx] * readout[pred_label]
        top_neurons = torch.topk(contribution, k=min(5, model.n)).indices.tolist()

        base_image = images[sample_idx : sample_idx + 1].to(device)
        baseline_prob = confidence
        neuron_details = []
        for neuron in top_neurons:
            ablated = model(base_image, ablate_neurons=[neuron])
            ablated_prob = float(
                F.softmax(ablated["logits"], dim=1)[0, pred_label].cpu()
            )
            neuron_details.append(
                {
                    "neuron": int(neuron),
                    "final_activity": float(activity[sample_idx, neuron]),
                    "readout_contribution": float(contribution[neuron]),
                    "mean_timescale": float(tau[neuron].mean()),
                    "probability_drop_when_silenced": baseline_prob - ablated_prob,
                    "top_pixel_inputs": top_pixel_inputs(model, neuron),
                    "top_recurrent_inputs": top_recurrent_inputs(model, neuron),
                }
            )

        image_file = save_sample_image(run_dir, images[sample_idx], sample_idx, true_label)
        raster_file = save_activity_raster(
            run_dir, history, sample_idx, true_label, pred_label
        )
        explanations.append(
            {
                "sample_index": sample_idx,
                "true_label": FASHION_CLASSES[true_label],
                "predicted_label": FASHION_CLASSES[pred_label],
                "confidence": confidence,
                "correct": bool(true_label == pred_label),
                "image_file": image_file,
                "raster_file": raster_file,
                "top_neurons": neuron_details,
            }
        )
    return explanations


def write_html_report(
    run_dir: Path,
    cfg: Config,
    test_metrics: Dict[str, Any],
    delay_metrics: List[Dict[str, Any]],
    topology: Dict[str, Any],
    neuron_table: List[Dict[str, Any]],
    explanations: List[Dict[str, Any]],
    figures: Dict[str, Any],
    history: List[Dict[str, Any]],
    rewire_history: List[Dict[str, Any]],
) -> Path:
    sorted_neurons = sorted(
        neuron_table,
        key=lambda row: row["class_selectivity"] * row["readout_importance"],
        reverse=True,
    )
    top_neurons = sorted_neurons[:12]

    fastest = min(neuron_table, key=lambda row: row["mean_timescale"])
    slowest = max(neuron_table, key=lambda row: row["mean_timescale"])
    architecture_changes = sum(
        int(row.get("input_rewired", 0)) + int(row.get("recurrent_rewired", 0))
        for row in rewire_history
    )

    css = """
    body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif;
           max-width: 1120px; margin: 30px auto; padding: 0 18px; line-height: 1.45; }
    h1, h2, h3 { line-height: 1.15; }
    .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; }
    .card { border: 1px solid #bbb; border-radius: 10px; padding: 14px; }
    img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 8px; }
    table { border-collapse: collapse; width: 100%; font-size: 0.92rem; }
    th, td { border: 1px solid #ccc; padding: 6px; text-align: left; }
    th { position: sticky; top: 0; background: Canvas; }
    code { background: #eee; padding: 1px 4px; border-radius: 4px; }
    .sample { margin: 20px 0 34px; padding-top: 12px; border-top: 2px solid #aaa; }
    .two { display: grid; grid-template-columns: minmax(180px, 0.6fr) minmax(320px, 1.4fr); gap: 16px; }
    @media (max-width: 750px) { .two { grid-template-columns: 1fr; } }
    """

    def img_tag(path: str, alt: str) -> str:
        return f'<img src="{html.escape(path)}" alt="{html.escape(alt)}">'

    lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<style>{css}</style></head><body>",
        "<h1>Developmental ELM on Fashion-MNIST</h1>",
        "<p>This report explains the learned circuit, not only its classification score. "
        "The model uses scalar-output ELM neurons, a flash-then-delay sensory episode, "
        "and slow held-out-data prune–regrow structural plasticity.</p>",
        "<div class='cards'>",
        f"<div class='card'><strong>Test accuracy</strong><br>{test_metrics['accuracy']:.3%}</div>",
        f"<div class='card'><strong>Mean activity</strong><br>{test_metrics['mean_activity']:.4f}</div>",
        f"<div class='card'><strong>Input edges retained</strong><br>{topology['input_retained_fraction']:.1%}</div>",
        f"<div class='card'><strong>Recurrent edges retained</strong><br>{topology['recurrent_retained_fraction']:.1%}</div>",
        f"<div class='card'><strong>Structural replacements</strong><br>{architecture_changes}</div>",
        f"<div class='card'><strong>Reciprocal recurrent pairs</strong><br>{topology['reciprocal_recurrent_pairs']}</div>",
        "</div>",
        "<h2>Automatically generated interpretation</h2>",
        "<ul>",
        f"<li>The fastest neuron is <strong>{fastest['neuron']}</strong> with mean time constant "
        f"{fastest['mean_timescale']:.2f} steps; the slowest is <strong>{slowest['neuron']}</strong> "
        f"at {slowest['mean_timescale']:.2f} steps.</li>",
        f"<li>The learned recurrent graph's largest out-degree is "
        f"{topology['highest_recurrent_out_degree']} at neuron "
        f"{topology['highest_out_degree_neuron']}. A hard-coded hub was not supplied.</li>",
        f"<li>Mean normalized afferent wiring distance is "
        f"{topology['mean_input_wiring_distance']:.3f}; mean recurrent wiring distance is "
        f"{topology['mean_recurrent_wiring_distance']:.3f}.</li>",
        "<li>Use the receptive-field and prototype panels together: the former shows "
        "which pixels are physically connected; the latter shows which full images actually "
        "drive the neuron after recurrent processing.</li>",
        "</ul>",
    ]

    if history:
        lines += [
            "<h2>Training dynamics</h2>",
            img_tag(figures.get("training_loss", ""), "training loss"),
            img_tag(figures.get("population_activity", ""), "population activity"),
        ]
        if figures.get("rewiring_events"):
            lines.append(img_tag(figures["rewiring_events"], "rewiring events"))
        if figures.get("edge_retention"):
            lines.append(img_tag(figures["edge_retention"], "edge retention"))
        if figures.get("wiring_distance"):
            lines.append(img_tag(figures["wiring_distance"], "wiring distance"))

    lines += [
        "<h2>Performance structure</h2>",
        img_tag(figures["confusion"], "test confusion matrix"),
        "<h2>Learned circuit</h2>",
        img_tag(figures["connectivity"], "learned recurrent graph"),
        img_tag(figures["afferent_coverage"], "population afferent coverage"),
        img_tag(figures["timescales"], "learned timescales"),
        img_tag(figures["class_neuron"], "class-neuron activity map"),
        "<h2>Memory generalization</h2>",
        "<table><tr><th>Blank delay</th><th>Accuracy</th><th>Mean activity</th></tr>",
    ]
    for row in delay_metrics:
        lines.append(
            f"<tr><td>{row['delay_steps']}</td><td>{row['accuracy']:.3%}</td>"
            f"<td>{row['mean_activity']:.4f}</td></tr>"
        )
    lines.append("</table>")

    lines += [
        "<h2>Most interpretable neurons</h2>",
        "<table><tr><th>Neuron</th><th>Preferred class</th><th>Selectivity</th>"
        "<th>Mean activity</th><th>Mean timescale</th><th>Recurrent out-degree</th>"
        "<th>Receptive field</th><th>Top-activating images</th></tr>",
    ]
    receptive_map = {
        int(Path(path).stem.split("_")[-1]): path
        for path in figures["receptive_fields"]
    }
    prototype_map = {
        int(Path(path).stem.split("_")[-1]): path
        for path in figures["prototypes"]
    }
    for row in top_neurons:
        neuron = int(row["neuron"])
        lines.append(
            "<tr>"
            f"<td>{neuron}</td>"
            f"<td>{html.escape(str(row['preferred_class']))}</td>"
            f"<td>{row['class_selectivity']:.4f}</td>"
            f"<td>{row['mean_activity']:.4f}</td>"
            f"<td>{row['mean_timescale']:.2f}</td>"
            f"<td>{row['recurrent_out_degree']}</td>"
            f"<td>{img_tag(receptive_map.get(neuron, ''), 'receptive field')}</td>"
            f"<td>{img_tag(prototype_map.get(neuron, ''), 'top activating images')}</td>"
            "</tr>"
        )
    lines.append("</table>")

    lines += [
        "<h2>Example decisions and causal neuron silencing</h2>",
        "<p>For each example, the report identifies neurons that support the predicted "
        "class and reruns the recurrent episode with each neuron silenced at every step. "
        "A positive probability drop is evidence that the neuron was causally useful for "
        "that particular decision.</p>",
    ]
    for example in explanations:
        lines += [
            "<div class='sample'>",
            f"<h3>Sample {example['sample_index']}: true {html.escape(example['true_label'])}; "
            f"predicted {html.escape(example['predicted_label'])} "
            f"({example['confidence']:.2%})</h3>",
            "<div class='two'>",
            img_tag(example["image_file"], "Fashion-MNIST sample"),
            img_tag(example["raster_file"], "neuron activity raster"),
            "</div>",
            "<table><tr><th>Neuron</th><th>Final activity</th><th>Readout contribution</th>"
            "<th>Mean timescale</th><th>Predicted-probability drop when silenced</th>"
            "<th>Strong pixel contacts</th><th>Strong recurrent parents</th></tr>",
        ]
        for detail in example["top_neurons"]:
            pixel_text = "; ".join(
                f"({item['pixel_x']},{item['pixel_y']}) w={item['weight']:+.2f}"
                for item in detail["top_pixel_inputs"][:3]
            )
            rec_text = "; ".join(
                f"n{item['source_neuron']} w={item['weight']:+.2f}"
                for item in detail["top_recurrent_inputs"][:3]
            )
            lines.append(
                "<tr>"
                f"<td>{detail['neuron']}</td>"
                f"<td>{detail['final_activity']:.4f}</td>"
                f"<td>{detail['readout_contribution']:+.4f}</td>"
                f"<td>{detail['mean_timescale']:.2f}</td>"
                f"<td>{detail['probability_drop_when_silenced']:+.4f}</td>"
                f"<td>{html.escape(pixel_text)}</td>"
                f"<td>{html.escape(rec_text)}</td>"
                "</tr>"
            )
        lines += ["</table>", "</div>"]

    lines += [
        "<h2>Machine-readable artifacts</h2>",
        "<ul>",
        "<li><code>neurons.csv</code>: selectivity, activity, time constants, receptive-field centroid, degrees.</li>",
        "<li><code>edges.csv</code>: every active input and recurrent edge, its weight, distance, age, and whether it changed.</li>",
        "<li><code>rewire_events.csv</code>: every prune/regrow replacement.</li>",
        "<li><code>sample_explanations.json</code>: sample-level support and causal ablation results.</li>",
        "<li><code>summary.json</code>: configuration and aggregate findings.</li>",
        "</ul>",
        "</body></html>",
    ]

    report_path = run_dir / "report.html"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_interpretability(
    model: DevelopmentalELM,
    cfg: Config,
    test_loader: DataLoader,
    device: torch.device,
    run_dir: Path,
    history: List[Dict[str, Any]],
    rewire_history: List[Dict[str, Any]],
) -> Path:
    print("\nRunning interpretability pass...")
    analysis = collect_analysis_data(
        model, test_loader, device, sample_limit=cfg.analysis_samples
    )
    test_metrics = evaluate(model, test_loader, device, cfg)

    delay_values = sorted(
        {
            cfg.delay_steps,
            max(cfg.delay_steps + 4, 12),
            max(cfg.delay_steps + 8, 16),
            max(cfg.delay_steps + 16, 24),
        }
    )
    delay_metrics = []
    for delay in delay_values:
        metrics = evaluate(model, test_loader, device, cfg, delay_override=delay)
        delay_metrics.append(
            {
                "delay_steps": delay,
                "accuracy": metrics["accuracy"],
                "mean_activity": metrics["mean_activity"],
            }
        )

    signed_rf, absolute_rf = compute_receptive_fields(model)
    stats = class_neuron_statistics(model, analysis)
    topology = topology_statistics(model)
    neuron_table = neuron_rows(model, stats, absolute_rf)
    edges = edge_rows(model)

    ranking = np.argsort(
        -(stats["selectivity"] * stats["readout_importance"])
    )
    selected_neurons = ranking[: min(cfg.prototype_neurons, model.n)].tolist()

    plot_files = save_training_plots(run_dir, history, rewire_history)
    plot_lookup = {Path(name).stem: name for name in plot_files}
    figures: Dict[str, Any] = {
        "training_loss": plot_lookup.get("training_loss", ""),
        "accuracy": plot_lookup.get("accuracy", ""),
        "population_activity": plot_lookup.get("population_activity", ""),
        "rewiring_events": plot_lookup.get("rewiring_events", ""),
        "edge_retention": plot_lookup.get("edge_retention", ""),
        "wiring_distance": plot_lookup.get("wiring_distance", ""),
        "confusion": save_confusion_matrix(run_dir, test_metrics["confusion"]),
        "timescales": save_timescale_plot(run_dir, model),
        "class_neuron": save_class_neuron_map(run_dir, stats),
        "connectivity": save_connectivity_plot(run_dir, model, stats),
        "afferent_coverage": save_afferent_coverage(run_dir, absolute_rf),
    }
    figures["receptive_fields"] = save_receptive_fields(
        run_dir, absolute_rf, model, stats, selected_neurons
    )
    figures["prototypes"] = save_neuron_prototypes(
        run_dir,
        analysis,
        selected_neurons,
        per_neuron=cfg.prototypes_per_neuron,
    )

    explanations = explain_samples(model, analysis, device, run_dir, cfg)

    neuron_fields = list(neuron_table[0].keys()) if neuron_table else []
    edge_fields = list(edges[0].keys()) if edges else []
    save_csv(run_dir / "neurons.csv", neuron_table, neuron_fields)
    save_csv(run_dir / "edges.csv", edges, edge_fields)

    flat_rewire_records: List[Dict[str, Any]] = []
    for event in rewire_history:
        flat_rewire_records.extend(event.get("records", []))
    rewire_fields = [
        "epoch",
        "kind",
        "target_neuron",
        "branch",
        "slot",
        "old_source",
        "new_source",
        "old_utility",
        "new_growth_score",
        "old_distance",
        "new_distance",
    ]
    save_csv(run_dir / "rewire_events.csv", flat_rewire_records, rewire_fields)
    save_csv(
        run_dir / "training_history.csv",
        history,
        list(history[0].keys()) if history else ["epoch"],
    )

    np.save(run_dir / "signed_receptive_fields.npy", signed_rf)
    np.save(run_dir / "absolute_receptive_fields.npy", absolute_rf)
    (run_dir / "sample_explanations.json").write_text(
        json.dumps(explanations, indent=2), encoding="utf-8"
    )

    summary = {
        "config": asdict(cfg),
        "test": {
            "loss": test_metrics["loss"],
            "accuracy": test_metrics["accuracy"],
            "mean_activity": test_metrics["mean_activity"],
            "per_class_accuracy": {
                FASHION_CLASSES[i]: test_metrics["per_class_accuracy"][i]
                for i in range(10)
            },
        },
        "delay_generalization": delay_metrics,
        "topology": topology,
        "selected_neurons": selected_neurons,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    report = write_html_report(
        run_dir,
        cfg,
        test_metrics,
        delay_metrics,
        topology,
        neuron_table,
        explanations,
        figures,
        history,
        rewire_history,
    )
    print(f"Interpretability report: {report}")
    return report


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------


def apply_quick_mode(cfg: Config) -> None:
    if not cfg.quick:
        return
    cfg.epochs = min(cfg.epochs, 6)
    cfg.batch_size = min(cfg.batch_size, 256)
    cfg.arch_batch_size = min(cfg.arch_batch_size, 256)
    cfg.train_limit = cfg.train_limit or 12_000
    cfg.architecture_limit = cfg.architecture_limit or 2_000
    cfg.validation_limit = cfg.validation_limit or 2_000
    cfg.test_limit = cfg.test_limit or 2_000
    cfg.rewire_start_epoch = 2
    cfg.rewire_end_epoch = 4
    cfg.rewire_every = 1
    cfg.architecture_batches = 1
    cfg.analysis_samples = min(cfg.analysis_samples, 2_000)
    cfg.prototype_neurons = min(cfg.prototype_neurons, 8)
    cfg.explained_samples = min(cfg.explained_samples, 4)


def train_experiment(cfg: Config) -> Tuple[Path, Path]:
    cfg.validate()
    seed_everything(cfg.seed)
    device = choose_device(cfg.device)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    run_dir = make_run_dir(cfg)
    print(f"Device: {device}")
    print(f"Run directory: {run_dir}")
    print(json.dumps(asdict(cfg), indent=2))

    train_loader, arch_loader, val_loader, test_loader, _test_dataset = build_dataloaders(cfg)
    model = DevelopmentalELM(cfg).to(device)
    model.enforce_no_duplicate_sources()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, cfg.epochs), eta_min=cfg.learning_rate * 0.10
    )

    history: List[Dict[str, Any]] = []
    rewire_history: List[Dict[str, Any]] = []
    best_val = -float("inf")
    best_path = run_dir / "best_checkpoint.pt"
    start = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, cfg)

        # Structural updates happen before validation so the reported score and
        # any saved checkpoint describe the same architecture. The following
        # epoch provides the intended post-rewiring self-repair period.
        rewire_event: Optional[Dict[str, Any]] = None
        if should_rewire(cfg, epoch):
            active_utility, growth_score, architecture_ce = collect_structure_scores(
                model, arch_loader, device, cfg
            )
            rewire_event = rewire_model(
                model,
                optimizer,
                active_utility,
                growth_score,
                cfg,
                epoch,
            )
            rewire_event["architecture_ce"] = architecture_ce
            current_topology = topology_statistics(model)
            rewire_event["input_retained_fraction"] = current_topology[
                "input_retained_fraction"
            ]
            rewire_event["recurrent_retained_fraction"] = current_topology[
                "recurrent_retained_fraction"
            ]
            rewire_event["mean_input_wiring_distance"] = current_topology[
                "mean_input_wiring_distance"
            ]
            rewire_event["mean_recurrent_wiring_distance"] = current_topology[
                "mean_recurrent_wiring_distance"
            ]
            rewire_history.append(rewire_event)

        val_metrics = evaluate(model, val_loader, device, cfg)

        with torch.no_grad():
            model.edge_age.add_(1)

        scheduler.step()
        elapsed = time.perf_counter() - epoch_start
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_activity": train_metrics["mean_activity"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_activity": val_metrics["mean_activity"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": elapsed,
            "input_rewired": int(rewire_event["input_rewired"]) if rewire_event else 0,
            "recurrent_rewired": int(rewire_event["recurrent_rewired"]) if rewire_event else 0,
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | "
            f"train loss {row['train_loss']:.4f} acc {row['train_accuracy']:.3%} | "
            f"val loss {row['val_loss']:.4f} acc {row['val_accuracy']:.3%} | "
            f"activity {row['val_activity']:.4f} | "
            f"rewired {row['input_rewired']}+{row['recurrent_rewired']} | "
            f"{elapsed:.1f}s"
        )

        if val_metrics["accuracy"] > best_val:
            best_val = float(val_metrics["accuracy"])
            save_checkpoint(
                best_path,
                model,
                optimizer,
                scheduler,
                cfg,
                history,
                rewire_history,
                epoch,
            )

        save_checkpoint(
            run_dir / "latest_checkpoint.pt",
            model,
            optimizer,
            scheduler,
            cfg,
            history,
            rewire_history,
            epoch,
        )

    total_seconds = time.perf_counter() - start
    print(f"Training complete in {total_seconds / 60.0:.1f} minutes")

    # Analyze the best validation checkpoint rather than merely the last epoch.
    model, loaded_cfg, payload = load_model_from_checkpoint(str(best_path), device)
    report = run_interpretability(
        model,
        loaded_cfg,
        test_loader,
        device,
        run_dir,
        payload.get("history", history),
        payload.get("rewire_history", rewire_history),
    )
    return run_dir, report


def analyze_checkpoint(checkpoint_path: str, cfg_overrides: Config) -> Tuple[Path, Path]:
    device = choose_device(cfg_overrides.device)
    model, cfg, payload = load_model_from_checkpoint(checkpoint_path, device)

    # Output and analysis limits may be overridden from the CLI.
    cfg.output_root = cfg_overrides.output_root
    cfg.run_name = cfg_overrides.run_name or f"analysis_{Path(checkpoint_path).stem}"
    cfg.data_dir = cfg_overrides.data_dir
    cfg.analysis_samples = cfg_overrides.analysis_samples
    cfg.test_limit = cfg_overrides.test_limit
    cfg.num_workers = cfg_overrides.num_workers
    cfg.device = cfg_overrides.device
    run_dir = make_run_dir(cfg)

    _train_loader, _arch_loader, _val_loader, test_loader, _test_dataset = build_dataloaders(cfg)
    report = run_interpretability(
        model,
        cfg,
        test_loader,
        device,
        run_dir,
        payload.get("history", []),
        payload.get("rewire_history", []),
    )
    return run_dir, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Developmental scalar-output ELM network for Fashion-MNIST",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--quick", action="store_true", help="Short smoke test with dataset limits")
    parser.add_argument("--analyze-only", default="", help="Checkpoint to analyze without training")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"])
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-root", default="./runs")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--arch-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--architecture-limit", type=int, default=0)
    parser.add_argument("--validation-limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)

    parser.add_argument("--neurons", type=int, default=64)
    parser.add_argument("--grid-height", type=int, default=8)
    parser.add_argument("--grid-width", type=int, default=8)
    parser.add_argument("--memory-dim", type=int, default=6)
    parser.add_argument("--branches", type=int, default=8)
    parser.add_argument("--synapses-per-branch", type=int, default=4)
    parser.add_argument("--input-synapses-per-branch", type=int, default=3)
    parser.add_argument("--internal-hidden", type=int, default=12)

    parser.add_argument("--protocol", choices=["flash", "saccade"], default="flash")
    parser.add_argument("--flash-steps", type=int, default=4)
    parser.add_argument("--delay-steps", type=int, default=8)
    parser.add_argument("--target-activity", type=float, default=0.10)
    parser.add_argument("--homeostasis-coef", type=float, default=4.0)
    parser.add_argument("--decorrelation-coef", type=float, default=0.002)

    parser.add_argument("--rewire-start-epoch", type=int, default=4)
    parser.add_argument("--rewire-end-epoch", type=int, default=14)
    parser.add_argument("--rewire-every", type=int, default=2)
    parser.add_argument("--rewire-fraction", type=float, default=0.04)
    parser.add_argument("--architecture-batches", type=int, default=1)
    parser.add_argument("--minimum-edge-age", type=int, default=2)
    parser.add_argument("--input-wiring-cost", type=float, default=0.60)
    parser.add_argument("--recurrent-wiring-cost", type=float, default=0.30)

    parser.add_argument("--analysis-samples", type=int, default=5_000)
    parser.add_argument("--prototype-neurons", type=int, default=12)
    parser.add_argument("--prototypes-per-neuron", type=int, default=8)
    parser.add_argument("--explained-samples", type=int, default=6)
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    cfg = Config(
        seed=args.seed,
        data_dir=args.data_dir,
        output_root=args.output_root,
        run_name=args.run_name,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        arch_batch_size=args.arch_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        num_workers=args.num_workers,
        train_limit=args.train_limit,
        architecture_limit=args.architecture_limit,
        validation_limit=args.validation_limit,
        test_limit=args.test_limit,
        neurons=args.neurons,
        grid_height=args.grid_height,
        grid_width=args.grid_width,
        memory_dim=args.memory_dim,
        branches=args.branches,
        synapses_per_branch=args.synapses_per_branch,
        input_synapses_per_branch=args.input_synapses_per_branch,
        internal_hidden=args.internal_hidden,
        protocol=args.protocol,
        flash_steps=args.flash_steps,
        delay_steps=args.delay_steps,
        target_activity=args.target_activity,
        homeostasis_coef=args.homeostasis_coef,
        decorrelation_coef=args.decorrelation_coef,
        rewire_start_epoch=args.rewire_start_epoch,
        rewire_end_epoch=args.rewire_end_epoch,
        rewire_every=args.rewire_every,
        rewire_fraction=args.rewire_fraction,
        architecture_batches=args.architecture_batches,
        minimum_edge_age=args.minimum_edge_age,
        input_wiring_cost=args.input_wiring_cost,
        recurrent_wiring_cost=args.recurrent_wiring_cost,
        analysis_samples=args.analysis_samples,
        prototype_neurons=args.prototype_neurons,
        prototypes_per_neuron=args.prototypes_per_neuron,
        explained_samples=args.explained_samples,
        quick=args.quick,
        analyze_only=args.analyze_only,
    )
    apply_quick_mode(cfg)
    cfg.validate()
    return cfg


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    cfg = config_from_args(args)

    try:
        if cfg.analyze_only:
            run_dir, report = analyze_checkpoint(cfg.analyze_only, cfg)
        else:
            run_dir, report = train_experiment(cfg)
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise

    print("\nDone.")
    print(f"Artifacts: {run_dir}")
    print(f"Open this report in a browser: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
