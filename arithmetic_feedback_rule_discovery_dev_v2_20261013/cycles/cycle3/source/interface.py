"""Generic byte adapter; no evaluator imports or arithmetic-specific learner inputs."""
import copy,sys
from pathlib import Path
import numpy as np
from core import Core,Config,EOS
from vendor.raw_reference_core import Core as RawCore,Config as RawConfig
sys.path.insert(0,str(Path(__file__).parent/'vendor'))
from vendor.legacy_core import MechanismCore

CONFIGS=('RAW-RLS','G-SQ','G-CE','L-SQ','L-CE','G-FROZEN-CE','L-FROZEN-CE','LEGACY-B')
def birth(name,seed):
    if name=='RAW-RLS':return RawCore(seed,RawConfig(depth=1,full=True,learned=True))
    if name=='LEGACY-B':return MechanismCore('LINEAR-FIXED-FULL',seed)
    if name not in CONFIGS:raise ValueError(name)
    return Core(seed,Config(reader='GRU' if name.startswith('G') else 'LAG32',
                objective='SQ' if name.endswith('SQ') else 'CE',learned='FROZEN' not in name))

def arrays(c):
    if isinstance(c,Core):return list(c.parameters().values())+list(c.mom.values())
    if isinstance(c,RawCore):return list(c.theta.values())+list(c.mom.values())+[c.V,c.P]
    return [c._weights,c.V,c.P,c.centers,c.counts]
def digest(c):return c.learning_digest() if isinstance(c,(Core,RawCore)) else c.learned_digest()
def counters(c):
    if isinstance(c,Core):return (c.learning_events,c.encoder_events,c.steps)
    if isinstance(c,RawCore):return (c.learning_events,c.steps,c.steps)
    return (c.learning_events,0,0)
def begin(c):c.begin() if isinstance(c,(Core,RawCore)) else c.begin_sequence()
def observe(c,b,allowed):
    if isinstance(c,(Core,RawCore)):return c.observe(b,allowed)
    before=c.learning_events;p=c.observe_end(allowed) if b==EOS else c.observe(b,allowed)
    return {'target':b,'permitted':allowed,'choice':int(p.event),'readout_updates':c.learning_events-before,'encoder_steps':0}

class ReadOnly:
    def __init__(self,c):self.c=c
    def __enter__(self):
        self.items=arrays(self.c);self.flags=[a.flags.writeable for a in self.items]
        for a in self.items:a.flags.writeable=False
    def __exit__(self,*args):
        for a,f in zip(self.items,self.flags):a.flags.writeable=f

def teach(c,prompt,target,allowed,all_bytes=False):
    begin(c);events=[]
    # Per-record array flags enforce the gate independently from reported counters.
    with ReadOnly(c) if not allowed else NullContext():
        for b in prompt:events.append(observe(c,b,allowed and all_bytes))
        c.start_reply()
        for b in target:events.append(observe(c,b,allowed))
        events.append(observe(c,EOS,allowed))
    return events
class NullContext:
    def __enter__(self):return self
    def __exit__(self,*args):pass

def working(c):
    if isinstance(c,Core):keys=('h','ids','tape','open','permitted','grad','loss_sum','streaming','evictions','aborted','last_update')
    elif isinstance(c,RawCore):keys=('h','tape','permitted','trace','open','aborted','evictions','dropped_targets','last_update')
    else:keys=('_context','_context_len','_next','_open','_generated_end','observed_payload_bytes','generated_payload_bytes',
        'context_overflow_events','elapsed_seconds','_question','_question_total','_reply_offset','_previous','_in_reply','_bank','_route_h')
    return {k:copy.deepcopy(getattr(c,k)) for k in keys}

def probe(c,queries):
    state=working(c);before=digest(c);rows=[]
    try:
        with ReadOnly(c):
            for q in queries:
                begin(c)
                for b in bytes.fromhex(q['prompt_hex']):observe(c,b,False)
                c.start_reply()
                if isinstance(c,(Core,RawCore)):choice,_,_=c.predict();payload,status=c.generate(16)
                else:
                    choice=c.prediction().event;g=c.generate(16);payload=g.payload;status='EOS' if g.status=='ENDED' else 'TRUNCATED'
                rows.append(dict(q,payload_hex=payload.hex(),status=status,first_choice=int(choice)))
    finally:
        for k,v in state.items():setattr(c,k,v)
    if before!=digest(c):raise AssertionError('probe learned state changed')
    return rows

def learning_record(c,prompt,target,allowed,all_bytes=False):
    before=counters(c);events=teach(c,prompt,target,allowed,all_bytes);after=counters(c)
    delta=tuple(a-b for a,b in zip(after,before))
    if delta[:2]!=(sum(e['readout_updates'] for e in events),sum(e['encoder_steps'] for e in events)):
        raise AssertionError('actual component counter mismatch')
    return {'prompt_hex':prompt.hex(),'answer_hex':target.hex(),'allowed':allowed,'all_bytes':all_bytes,
            'targets':[e['target'] for e in events],'permissions':[e['permitted'] for e in events],
            'choices':[e['choice'] for e in events],'counters_before':before,'counters_after':after,
            'readout_updates':delta[0],'encoder_steps':delta[1],'optimizer_steps':delta[2],
            'update':copy.deepcopy(c.last_update) if isinstance(c,Core) and allowed else None}
