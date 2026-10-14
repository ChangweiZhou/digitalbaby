"""Generic learner adapters. Evaluator data do not enter these constructors."""
import copy
import hashlib
import sys
from pathlib import Path
import numpy as np
from core import Core,Config,EOS
sys.path.insert(0,str(Path(__file__).parent/'vendor'))
from vendor.legacy_core import MechanismCore
CONFIGS=('R1-DIAG','R1-FULL','R4-DIAG','R4-FULL','F1-FULL','F4-FULL','LEGACY-B')

def birth(name,seed):
 if name=='LEGACY-B': return MechanismCore('LINEAR-FIXED-FULL',seed)
 if name not in CONFIGS: raise ValueError(name)
 return Core(seed,Config(depth=int(name[1]),full=name.endswith('FULL'),learned=name.startswith('R')))

def begin(c):
 c.begin() if isinstance(c,Core) else c.begin_sequence()

def observe(c,byte,learn):
 if isinstance(c,Core): return c.observe(byte,learn)
 before=c.learning_events
 p=c.observe_end(learn) if byte==EOS else c.observe(byte,learn)
 return {'target':byte,'permitted':learn,'choice':int(p.event),'readout_updates':c.learning_events-before,'encoder_steps':0}

def digest(c):
 return c.learning_digest() if isinstance(c,Core) else c.learned_digest()

def arrays(c):
 if isinstance(c,Core):return list(c.theta.values())+list(c.mom.values())+[c.V,c.P]
 return [c._weights,c.V,c.P,c.centers,c.counts]

class ReadOnly:
 def __init__(self,c):self.c=c
 def __enter__(self):
  self.items=arrays(self.c); self.flags=[a.flags.writeable for a in self.items]
  for a in self.items:a.flags.writeable=False
 def __exit__(self,*args):
  for a,f in zip(self.items,self.flags):a.flags.writeable=f

def teach(c,payload,target,learn):
 begin(c)
 for byte in payload: observe(c,int(byte),False)
 c.start_reply()
 events=[]
 if learn:
  for b in target:events.append(observe(c,int(b),True))
  events.append(observe(c,EOS,True))
 else:
  with ReadOnly(c):
   for b in target:events.append(observe(c,int(b),False))
   events.append(observe(c,EOS,False))
 return events

def working(c):
 if isinstance(c,Core):
  fields=('h','tape','permitted','trace','open','aborted','evictions','dropped_targets','last_update')
 else:
  fields=('_context','_context_len','_next','_open','_generated_end','observed_payload_bytes',
  'generated_payload_bytes','context_overflow_events','elapsed_seconds','_question','_question_total',
  '_reply_offset','_previous','_in_reply','_bank','_route_h')
 return {k:copy.deepcopy(getattr(c,k)) for k in fields}

def restore(c,state):
 for k,v in state.items():setattr(c,k,v)

def probe(c,queries):
 state=working(c); before=digest(c); predictions=[]
 try:
  with ReadOnly(c):
   for item in queries:
    begin(c)
    for byte in bytes.fromhex(item['prompt_hex']): observe(c,byte,False)
    c.start_reply()
    if isinstance(c,Core):
     choice,p,z=c.predict(); feature,_=c.features(); payload,status=c.generate(16)
    else:
     pr=c.prediction(); choice,p=pr.event,pr.probabilities
     feature,z=c._terms(c._address()); g=c.generate(16); payload=g.payload;status='EOS' if g.status=='ENDED' else 'TRUNCATED'
    predictions.append(dict(item,payload_hex=payload.hex(),status=status,choice=int(choice),
       first_probabilities=p.tolist(),feature=feature.tolist(),empty=len(payload)==0))
 finally:restore(c,state)
 if digest(c)!=before:raise AssertionError('probe mutated learning state')
 return predictions
