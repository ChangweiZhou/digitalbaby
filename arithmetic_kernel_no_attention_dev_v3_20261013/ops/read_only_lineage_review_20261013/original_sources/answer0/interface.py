"""Generic joint-core byte transport; imports no task generator or legacy core."""
import copy
from core import Core,Config,EOS
CONFIGS=('ALL-0','ALL-D','ANSWER-0','ANSWER-D')
def birth(name,seed):
    if name not in CONFIGS:raise ValueError(name)
    return Core(seed,Config(gradient_mode='per_target',decay=.01 if name.endswith('-D') else 0.))
def digest(c):return c.learning_digest()
def counters(c):return (c.learning_events,c.encoder_events,c.steps)
class ReadOnly:
    def __init__(self,c):self.c=c
    def __enter__(self):
        self.items=list(self.c.parameters().values())+list(self.c.mom.values());self.flags=[a.flags.writeable for a in self.items]
        for a in self.items:a.flags.writeable=False
    def __exit__(self,*args):
        for a,f in zip(self.items,self.flags):a.flags.writeable=f
class NullContext:
    def __enter__(self):return self
    def __exit__(self,*args):pass
def teach(c,prompt,target,allowed,all_bytes):
    c.begin();events=[]
    with ReadOnly(c) if not allowed else NullContext():
        for byte in prompt:events.append(c.observe(byte,allowed and all_bytes))
        c.start_reply()
        for byte in target:events.append(c.observe(byte,allowed))
        events.append(c.observe(EOS,allowed))
    return events
def learning_record(c,prompt,target,allowed,all_bytes):
    before=counters(c);events=teach(c,prompt,target,allowed,all_bytes);after=counters(c)
    delta=tuple(a-b for a,b in zip(after,before))
    if delta!=(sum(e['readout_updates'] for e in events),sum(e['encoder_steps'] for e in events),sum(e['optimizer_steps'] for e in events)):
        raise AssertionError('actual component counters disagree')
    return {'prompt_hex':prompt.hex(),'answer_hex':target.hex(),'allowed':allowed,'all_bytes':all_bytes,
        'targets':[e['target'] for e in events],'permissions':[e['permitted'] for e in events],
        'choices':[e['choice'] for e in events],'counters_before':before,'counters_after':after,
        'readout_updates':delta[0],'encoder_steps':delta[1],'optimizer_steps':delta[2],
        'update':dict(c.last_update) if allowed else None}
def probe(c,queries):
    keys=('h','ids','tape','open','permitted','grad','loss_sum','streaming','evictions','aborted','last_update')
    saved={k:copy.deepcopy(getattr(c,k)) for k in keys};initial=digest(c);rows=[]
    try:
        with ReadOnly(c):
            for q in queries:
                c.begin()
                for b in bytes.fromhex(q['prompt_hex']):c.observe(b,False)
                c.start_reply();choice,_,_=c.predict();payload,status=c.generate(16)
                rows.append(dict(q,payload_hex=payload.hex(),status=status,first_choice=choice))
    finally:
        for k,v in saved.items():setattr(c,k,v)
    if digest(c)!=initial:raise AssertionError('probe persistent-state mutation')
    return rows
