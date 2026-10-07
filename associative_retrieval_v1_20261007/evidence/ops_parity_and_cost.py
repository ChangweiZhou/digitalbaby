"""Final cycle-3 operations revision; old native lives are not rerun."""
import sys,json,time,copy,hashlib,difflib
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'code'))
import runtime
from bridge_checkpoint import load
from bridge_core import RetrievalCore
from core import Core as ParentCore
from dev_worker import record
from exercise import parent_record
from assays import make_world,DT
from identity import identity,source_map
from io_utils import atomic_json,read_receipt
from association import Associations

old=json.loads((root/'cycles/cycle3_before_ops/SOURCE.json').read_text())
new=source_map();changes={p for p,h in new.items() if old['files'].get(p)!=h}
assert changes=={str(root/'code/bridge_core.py')},changes
old_text=(root/'cycles/cycle3_before_ops/bridge_core.py').read_text();new_text=(root/'code/bridge_core.py').read_text()
removed="        before = self.bank_digests()\n"
removed2="        if before != self.bank_digests():\n            raise AssertionError('association retrieval mutated native memory')\n"
assert old_text.replace(removed,'').replace(removed2,'')==new_text
checks=[];costs={}
for assay in ('lifetime','reuse'):
 w=make_world(71008003,assay)
 # Every saved full branch retains its original data and original source identity.
 for branch in w['branches']:
  c,cur=load(root/f'technical/cycle3/71008003_{assay}_{branch}.npz',old['identity'])
  before=c.state_digest(); bd=read_receipt(root/f'technical/cycle3/71008003_{assay}_{branch}.json.gz')['branches'][branch]
  final=next(p for p in bd['probes'] if p['name']=='final');rows=0
  from dev_worker import probe
  now=probe(c,w,w['clocks']['final'],'final',len(w['events']))
  for a,b in zip(final['rows'],now['rows']):
   assert a['policies']==b['policies']
   for x,y in zip(a['predictions'],b['predictions']):
    for key in ('emitted','shared','private','combined','retrieval','permutation_retrieval','shared_ids','private_ids','query','policies'):
     assert x[key]==y[key]
   rows+=1
  assert c.state_digest()==before
  checks.append({'assay':assay,'branch':branch,'final_query_rows':rows,'exact_numeric_parity':True,'continuing_state_unchanged':True})
 # Sample new technical event primitives after a saved final state, at full occupancy.
 # These disposable copies are not formal events or complete trajectories.
 c,_=load(root/f'technical/cycle3/71008003_{assay}_W.npz',old['identity']);before=c.state_digest()
 examples=w['sets']['old'][:2]+w['sets']['new'][:2];samples=[]
 for i,e in enumerate(examples*3):
  e=dict(e,at=w['clocks']['final'],index=c.records,stage='old',item=i)
  a=c.clone();b=ParentCore.clone(c);b.__class__=ParentCore
  if i%2:
   st=time.process_time();p=parent_record(b,e);base=time.process_time()-st
   r=record(a,e,'W')
  else:
   r=record(a,e,'W');st=time.process_time();p=parent_record(b,e);base=time.process_time()-st
  assert a.bank_digests()==b.bank_digests() and r['prediction']['policies']['ERROR']['combined']==list(p.combined)
  samples.append({'baseline_cpu_s':base,'LINK_cpu_s':r['LINK_online_cpu_s'],'ratio':r['LINK_online_cpu_s']/base})
 assert c.state_digest()==before
 costs[assay]={'samples':samples,'mean_ERROR_cpu_s':sum(x['baseline_cpu_s'] for x in samples)/len(samples),
               'mean_LINK_cpu_s':sum(x['LINK_cpu_s'] for x in samples)/len(samples),'occupied_slots':c.associations.count}
 costs[assay]['ratio']=costs[assay]['mean_LINK_cpu_s']/costs[assay]['mean_ERROR_cpu_s']
out={'verdict':'PASS','old_identity':old['identity'],'new_identity':identity(),'only_change':'remove two redundant read-only full-bank digest scans and their assertion; no numerical/learning path change',
     'complete_original_native_lives_preserved':6,'new_full_native_lives':0,'final_query_parity':checks,'future_disposable_primitive_tests':24,
     'new_API_cost_samples':costs,'scope':'Costs are local samples at final-table occupancy, not a complete new-source lifetime or runtime-tail guarantee.'}
atomic_json(root/'evidence/CYCLE3_OPS_REVISION.json',out);print(json.dumps(out),flush=True)
