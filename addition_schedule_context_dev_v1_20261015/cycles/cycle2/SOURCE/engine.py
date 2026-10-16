"""Record application and actual write evidence. No policy receives task metadata."""
import copy
import time
import models


def versions(model):
    return dict(parameters={k:p._version for k,p in model.p.items()},moments={k:m._version for k,m in model.mom.items()},actor_steps=model.actor_steps)


def apply_addition(model,row,*,capture=False):
    before=versions(model);record=model.train(bytes.fromhex(row['prompt']),bytes.fromhex(row['answer']),row['allowed'],capture=capture)
    after=versions(model)
    if not row['allowed'] and before!=after:raise AssertionError('forbidden parameter or RMS write')
    return dict(id=row['id'],kind=row['kind'],source_index=row['source_index'],prompt=row['prompt'],answer=row['answer'],
                versions_before=before,versions_after=after,**record)


def apply_native(model,item,allowed):
    start=model.time;record=model.train(item['cue'],item['answer'],allowed)
    elapsed=model.time-start
    if elapsed>165:raise AssertionError('native frame exceeds fixed cadence')
    model.rest(165-elapsed)
    return dict(id=item['id'],cue=item['cue'].hex(),answer=item['answer'].hex(),**record)
