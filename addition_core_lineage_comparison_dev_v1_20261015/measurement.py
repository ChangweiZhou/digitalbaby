"""Read-only resource accounting; no budget or evaluator state enters learning."""
import resource
import sys
import numpy as np
import torch
import core


def state_bytes(model):
    if isinstance(model,core.Recurrent):
        return dict(native_stores=0,parameter_bytes=sum(p.numel()*p.element_size() for p in model.p.values()),
                    optimizer_bytes=sum(p.numel()*p.element_size() for p in model.mom.values()),
                    fixed_output_code_bytes=core.CODE.nbytes+core.TCODE.numel()*core.TCODE.element_size())
    seen=set();arrays={}
    def visit(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value,np.ndarray):
            while isinstance(value.base,np.ndarray):
                value=value.base
            arrays[id(value)]=value.nbytes
        elif isinstance(value,dict):
            for x in value.values():visit(x)
        elif isinstance(value,(tuple,list)):
            for x in value:visit(x)
        elif not callable(value) and hasattr(value,'__dict__') and not isinstance(value,type(sys)):
            visit(vars(value))
    visit(model.banks)
    plastic=sum(s.fly.m.fast.nbytes+s.fly.m.slow.nbytes for b in model.banks for s in b)
    adaptation=sum(s.fly.m.adapt.nbytes for b in model.banks for s in b)
    total=sum(arrays.values())
    return dict(native_stores=sum(map(len,model.banks)),native_plastic_state_bytes=plastic,
                native_adaptation_bytes=adaptation,unique_resident_numpy_bytes=total,
                fixed_and_other_numpy_bytes=total-plastic-adaptation,
                fixed_output_code_bytes=core.CODE.nbytes+core.TCODE.numel()*core.TCODE.element_size(),
                note='Includes independently born resident arrays; excludes Python object overhead and transient allocations. Peak RSS is separately reported.')


def peak_rss_bytes():
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform=='darwin' else value*1024)
