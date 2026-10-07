"""Read-only cost qualification on existing final checkpoints; no teaching."""
import sys,time,dataclasses,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'code'))
import runtime
from bridge_checkpoint import load
from identity import identity
from core import Core as ParentCore
from bridge_core import RetrievalCore,ChoiceOrgan
from assays import make_world,DT
from io_utils import atomic_json,digest

def baseline_clone(c):
 d=ParentCore.clone(c);d.__class__=ParentCore;return d

def score_queries(c,w,mode):
 start=time.process_time();control=0.;emissions=[]
 rs=w['sets'];at=w['clocks']['final']
 if w['assay']=='lifetime':
  for group in ('old','new','revision'):
   for r in rs[group]:
    d=baseline_clone(c) if mode=='ERROR' else c.clone()
    for i,b in enumerate(bytes.fromhex(r['cue_hex'])):d.feed(b,at+i*DT)
    p=d.predict(at+12*DT)
    if mode=='ERROR':emissions.append(p.emitted)
    else:
     emissions.append(p.policies[mode]['emitted'])
     control+=p.costs['control_retrieval_cpu_s' if mode=='LINK' else 'retrieval_cpu_s']
 else:
  for group in ('old','heldout','new'):
   pairs=rs[group]
   for i in range(len(pairs)//2):
    a,b=pairs[2*i:2*i+2]
    for order in (0,1):
     first,second=(a,b) if order==0 else (b,a)
     d=baseline_clone(c) if mode=='ERROR' else c.clone();vals=[]
     for j,r in enumerate((first,second)):
      t=at+13*j*DT
      for k,byte in enumerate(bytes.fromhex(r['cue_hex'])):d.feed(byte,t+k*DT)
      p=d.predict(t+12*DT)
      if mode=='ERROR':u=p.combined
      else:
       u=p.policies[mode]['combined'];control+=p.costs['control_retrieval_cpu_s' if mode=='LINK' else 'retrieval_cpu_s']
      vals.append(u[1]-u[0]);d.clear_unreinforced_prediction()
      d.feed(ord('|') if j==0 else ord('?'),t+12*DT);d.cue_count=0
     emissions.append(ord('L') if vals[0]>=vals[1] else ord('R'))
 return {'operational_cpu_s':time.process_time()-start-control,'countermodel_CPU_excluded_s':control,
         'queries':len(emissions),'emissions_digest':digest(emissions)}

results={}
for assay in ('lifetime','reuse'):
 old_identity=json.loads((root/'cycles/cycle3_before_ops/SOURCE.json').read_text())['identity']
 c,cur=load(root/f'technical/cycle3/71008003_{assay}_W.npz',old_identity)
 before=c.state_digest();w=make_world(71008003,assay);rows=[]
 for repeat in range(3):
  modes=('ERROR','LINK','PERM') if repeat%2==0 else ('PERM','LINK','ERROR')
  rows.append({m:score_queries(c,w,m) for m in modes})
 assert c.state_digest()==before
 results[assay]={'runs':rows,'continuing_state_unchanged':True,
                 'mean_final_query_cpu_s':{m:sum(r[m]['operational_cpu_s'] for r in rows)/len(rows) for m in ('ERROR','LINK','PERM')}}
out={'input_build_identity':old_identity,'evidence':'E0 read-only timing on existing checkpoints','new_native_lives':0,'native_teaching_calls':0,
     'identity':identity(),'results':results,'limits':'Three timing repeats on one state; no tail guarantee; excludes cold import/birth, adds final-query API to measured teaching API.'}
atomic_json(root/'evidence/FINAL_QUERY_COST.json',out);print(json.dumps(out),flush=True)
