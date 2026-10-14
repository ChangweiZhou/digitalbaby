"""Qualification-only old-source oracles; never imported by operative core."""
import ast
import dataclasses
import importlib.util
import math
import sys
import typing
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ARCHIVE = Path(__file__).resolve().parents[1]/'arithmetic_kernel_no_attention_dev_v3_20261013/ops/read_only_lineage_review_20261013/original_sources'


def old_gru(seed, width, embedding):
    name = 'source_cell_original_answer0'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ARCHIVE/'answer0/core.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    module = sys.modules[name]
    return module.Core(seed, module.Config(width=width, embedding=embedding, gradient_mode='per_target'))


def original_elm():
    # Execute the original class and its original helpers only. Exclude dataset,
    # training and plotting imports/functions; original arithmetic-free dynamics
    # and geometry remain in the oracle, not the operative byte model.
    source = ARCHIVE/'elm/developmental_elm_fashion.py'
    tree = ast.parse(source.read_text())
    names = {'Config', 'DevelopmentalELM', 'inverse_softplus', 'inverse_bounded_sigmoid'}
    chosen = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    module = type(sys)('source_cell_original_elm')
    sys.modules[module.__name__] = module
    scope = module.__dict__
    scope.update(math=math, np=np, torch=torch, nn=nn, F=F,
                 dataclass=dataclasses.dataclass, asdict=dataclasses.asdict,
                 FASHION_CLASSES=tuple(range(10)))
    scope.update({name: getattr(typing, name) for name in ('Any', 'Dict', 'List', 'Optional', 'Sequence', 'Tuple')})
    exec(compile(ast.Module(body=chosen, type_ignores=[]), str(source), 'exec'), scope)
    return scope['DevelopmentalELM'](scope['Config']()).double()


def install_elm(reference, kernel):
    with torch.no_grad():
        for name, p in kernel.p.items():
            if name not in ('E', 'V'):
                getattr(reference, name).copy_(p)
        graph = kernel.source_idx.clone()
        graph[:, :, 3] = graph[:, :, 3]-24+784
        reference.source_idx.copy_(graph)
