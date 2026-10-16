"""Read-only scoring, separate from operative learning and its timing."""
import copy
import math
import time
import models
import data
import torch
import numpy as np
from collections import Counter,defaultdict


def own(model,examples):
    before=model.digest();resource=copy.deepcopy(model.resource);start=time.process_time()
    outputs=model.generate_batch([p for p,y in examples])
    if model.digest()!=before:raise AssertionError('query altered learned state')
    model.resource=resource
    return dict(rows=[dict(prompt=p.hex(),expected=y.hex(),emitted=z.hex(),status=s,correct=z==y and s=='EOS') for (p,y),(z,s) in zip(examples,outputs)],
                query_cpu=time.process_time()-start)


def addition(model,groups,seen):
    result={};cpu=0.
    for name,pairs in groups.items():
        r=own(model,[data.example(p) for p in pairs]);cpu+=r['query_cpu']
        for row in r['rows']:row['actually_exposed']=row['prompt'] in seen
        result[name]=r['rows']
    return dict(scores=result,query_cpu=cpu)


def modes(training,examples):
    frequencies=Counter();raw={};positions=[defaultdict(Counter) for _ in range(4)]
    for row in training:
        p,y=bytes.fromhex(row['prompt']),bytes.fromhex(row['answer']);frequencies[y]+=1;raw[p]=y
        for i,b in enumerate(p):positions[i][b][y]+=1
    def mode(c):return min(c,key=lambda x:(-c[x],x)) if c else b''
    default=mode(frequencies);scores={k:[] for k in ('FREQUENCY','RAW_KEY','CANONICAL_KEY','BYTE0','BYTE1','BYTE2','BYTE3')}
    for p,y in examples:
        outputs=dict(FREQUENCY=default,RAW_KEY=raw.get(p,default),CANONICAL_KEY=raw.get(p,default))
        outputs.update({f'BYTE{i}':mode(positions[i][p[i]]) if positions[i][p[i]] else default for i in range(4)})
        for key,z in outputs.items():scores[key].append(dict(emitted=z.hex(),correct=z==y))
    return dict(scores=scores,envelope=max(sum(r['correct'] for r in rows)/len(rows) for rows in scores.values()))


def conditional(model,examples):
    before=model.digest();resource=copy.deepcopy(model.resource);start=time.process_time();result=[]
    with torch.no_grad():
        for prompt,answer in examples:
            context,state=model.encode(prompt);trace=[]
            for target in (*answer,models.EOS):
                logits=model.logits(state)[0];values=logits.numpy().copy()
                competing=np.delete(values,target)
                trace.append(dict(target=target,predicted=int(values.argmax()),logits=values.tolist(),
                    target_margin=float(values[target]-competing.max()),NLL=float(torch.logsumexp(logits,0)-logits[target])))
                if target!=models.EOS:state=model.decode_step(state,torch.tensor([target]),context)
            result.append(dict(prompt=prompt.hex(),expected=answer.hex(),trace=trace,
                 minimum_target_margin=min(t['target_margin'] for t in trace),mean_NLL=sum(t['NLL'] for t in trace)/len(trace),
                 autonomous_score=False))
    model.resource=resource
    if before!=model.digest():raise AssertionError('conditional diagnostic mutated learning')
    return dict(rows=result,query_cpu=time.process_time()-start)


def native(model,fixture,cap=16):
    before=model.digest();cache=[(f.cache_hits,f.cache_misses,len(f.cache)) for f in model.fronts if hasattr(f,'cache')]
    start=time.process_time();rows=[]
    for item in fixture:
        emitted,status,trace=model.generate_trace(item['cue'],cap)
        teacher=model.teacher_conditioned_trace(item['cue'],item['answer'])
        rows.append(dict(id=item['id'],stratum=item['stratum'],expected=item['answer'].hex(),emitted=emitted.hex(),status=status,
            correct=emitted==item['answer'] and status=='EOS',first_byte_correct=bool(emitted) and emitted[:1]==item['answer'][:1],
            premature_EOS=status=='EOS' and len(emitted)<len(item['answer']),trace=trace,
            conditional=teacher,conditional_is_autonomous_score=False))
    if model.digest()!=before or cache!=[(f.cache_hits,f.cache_misses,len(f.cache)) for f in model.fronts if hasattr(f,'cache')]:
        raise AssertionError('native query changed continuing state or production cache')
    return dict(rows=rows,query_cpu=time.process_time()-start)


def native_geometry(model,fixture):
    start=time.process_time();before=model.digest();rows=[]
    for item in fixture:
        view=model.clone();view._prompt(item['cue'])
        for i,target in enumerate((*item['answer'],models.EOS)):
            components=[];codes=[]
            for fe,bank in zip(view.fronts,view.banks):
                f=fe.clone();f.advance(view.time+models.PARENT.DELTA);pn=f.read()
                kc=models.stores.bb.bc.native().model.encode_sparse(bank[0].fly.m,pn)
                values=[]
                for s in bank:
                    fly=s.fly.clone();fly.rest((view.time+models.PARENT.DELTA)-view.time)
                    dx=models.stores.bb.bc.native().common.observed_activity(fly.m,kc[None,:])
                    values.append(float((fly.m.expression(dx)-fly.reader.predict(dx)).mean()))
                components.append(values)
                codes.append(dict(PN=pn.tolist(),KC=kc.tolist(),active=int((kc!=0).sum())))
            combined=np.array(components[0])/models.PREVIOUS.SCALES[0]+np.array(components[1])/models.PREVIOUS.SCALES[1]
            np.testing.assert_allclose(models.PARENT.logits(combined),view.forecast()[1],rtol=0,atol=1e-12)
            rows.append(dict(id=item['id'],stratum=item['stratum'],prefix=item['answer'][:i].hex(),target=target,
                address= codes,shared_values=components[0],private_values=components[1],conditional_only=True))
            view._event(target)
    if before!=model.digest():raise AssertionError('geometry diagnostic mutated continuing native state')
    return dict(rows=rows,query_cpu=time.process_time()-start)
