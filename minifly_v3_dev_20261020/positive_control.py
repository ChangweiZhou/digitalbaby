"""Pinned v0 FULL recipe, adaptor off, using the parity-qualified native batch."""
import bootstrap
import json
import random
import time
from core import Learner
from data import split
from evaluate import score
from artifacts import HERE,atomic_json


def run():
    output=[]
    for world in (202610209901,202610209902):
        start=time.process_time();sets,_=split(world);recs=sets['OLD']+sets['NEW']
        m=Learner(202610208920,plastic=False,source_parent=True)
        rng=random.Random(world)
        for _ in range(24):
            order=list(range(80));rng.shuffle(order)
            for i in order:m.train(*recs[i])
        before=m.digest();taught=score(m,recs);assert m.digest()==before
        m.rest(86400.);after=score(m,recs)
        row=dict(world=world,record_applications=1920,adaptor_disabled=True,
                 taught=taught,after_day=after,cpu_s=time.process_time()-start,diagnostics=m.diagnostics())
        output.append(row);atomic_json(HERE/'cycles/cycle3/PARENT_POSITIVE_CONTROL.json',dict(
            scientific_records=0,record_applications=sum(x['record_applications'] for x in output),worlds=output))
        print('PARENT_CONTROL',world,taught['accuracy'],after['accuracy'],flush=True)
    assert all(r['taught']['accuracy']>.5 for r in output),'parent memory acquisition not reproduced'
    return output


if __name__=='__main__':run()
