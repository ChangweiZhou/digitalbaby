"""Portable learning-state archive, not a new forward or update policy."""
import json
import os
from pathlib import Path
import numpy as np
import torch
from core import Learner


def save(core, path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    metadata=dict(arm=core.net.arm,steps=core.steps,last_gradient=core.last_gradient,
                  last_update=core.last_update,resource=core.resource,digest=core.digest())
    arrays={'metadata':np.array(json.dumps(metadata,allow_nan=False))}
    arrays.update({'parameter_'+k:p.detach().numpy() for k,p in core.net.p.items()})
    arrays.update({'moment_'+k:p.numpy() for k,p in core.mom.items()})
    arrays.update({'buffer_'+k:p.numpy() for k,p in core.net.named_buffers()})
    temp=path.with_name(path.name+'.pending-'+str(os.getpid()))
    with temp.open('wb') as f:np.savez_compressed(f,**arrays)
    os.replace(temp,path)


def load(path):
    with np.load(path,allow_pickle=False) as data:
        metadata=json.loads(str(data['metadata']))
        core=Learner(metadata['arm'],0)
        with torch.no_grad():
            for k,p in core.net.p.items():p.copy_(torch.from_numpy(data['parameter_'+k]))
            for k,p in core.mom.items():p.copy_(torch.from_numpy(data['moment_'+k]))
            for k,p in core.net.named_buffers():p.copy_(torch.from_numpy(data['buffer_'+k]))
        for name in ('steps','last_gradient','last_update','resource'):setattr(core,name,metadata[name])
        if core.digest()!=metadata['digest']:raise ValueError('archived learning state mismatch')
        return core
