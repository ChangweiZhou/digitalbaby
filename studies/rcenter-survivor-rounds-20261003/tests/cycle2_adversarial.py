# SPDX-License-Identifier: GPL-3.0-or-later
import ast,copy,hashlib,json,os,resource,sys,time,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'source/runtime'))
from survivor_fixture import make_world,DT,RECORD_SECONDS,digest,ALPHABET
from engine import Learner,SCALES
from assay import cue,scores,probe,clock_vector,state_identity,suppress_value_writes,operative_state_identity,fixed_identity
from integrity import require_technical,hashes
import numpy as np
start=time.monotonic();checks=[];w=make_world(310101)
# Poison dormant inherited fixture interfaces before any newborn/learning.
import common_platform,fixture as inherited_fixture
poison=lambda *a,**k: (_ for _ in ()).throw(AssertionError('evaluator generator leaked into learner'))
common_platform.make_world=poison;inherited_fixture.make_world=poison
if hasattr(common_platform,'fact_variants'):common_platform.fact_variants=poison
if hasattr(inherited_fixture,'fact_variants'):inherited_fixture.fact_variants=poison
frozen=ROOT/'source/baseline/src/learner.py'
tree=ast.parse(frozen.read_text())
assert not any(isinstance(n,ast.ImportFrom) and n.module in ('survivor_fixture','environment','assay') for n in ast.walk(tree))
base=Learner('R_center',SCALES)
assert not set(base.__dict__) & {'world','stage','target','mapping','relation','item','cue_id'}
checks.append('learner_boundary_and_poisoned_inherited_generators')

full_state=operative_state_identity
def nonplastic(m):
    out=[]
    for x in m.shared+m.private:
        out.append({'adapt':hashlib.sha256(x.fly.m.adapt.tobytes()).hexdigest(),'fe':hashlib.sha256(x.fe.p.tobytes()).hexdigest(),
                    'visible':getattr(x.fe,'visible',b'').hex(),'w':hashlib.sha256(x.w.tobytes()).hexdigest(),
                    'bias':hashlib.sha256(x.bias.tobytes()).hexdigest(),'prev1':x.prev1,'prev2':x.prev2,
                    'pending_x':None if x.pending_x is None else hashlib.sha256(x.pending_x.tobytes()).hexdigest(),
                    'event_count':int(x.fly.m.event_count),'presentation_count':int(x.fly.m.presentation_count)})
    return digest({'stores':out,'clocks':clock_vector(m),'cue_count':m.cue_count,'prediction_time':m.prediction_time})
def check_clock(m,t):
    for x in m.shared+m.private:
        assert abs(x.brain_t-t)<1e-6
        assert abs(float(x.fly.m.elapsed)-x.elapsed_base-t)<1e-6
        assert abs(float(x.fe.t)-t)<1e-6
        assert x.pending_t is None and x.pending_x is None
        assert x.last_byte_t is None or x.last_byte_t<=t+1e-6
        assert (x.bytes_seen==0)==(x.last_byte_t is None)
    return clock_vector(m)
def reject(fn):
    try:fn()
    except (ValueError,RuntimeError,AssertionError):return
    raise AssertionError('invalid call accepted')
# Invalid calls use disposable fresh clones, no extra valid teaches.
reject(lambda:base.clone().observe_outcome(48,0.))
e=base.clone()
for i,b in enumerate(bytes.fromhex(w['events'][0]['cue_hex'])[:11]):e.feed(b,i*DT)
reject(lambda:e.predict(12*DT))
e.feed(bytes.fromhex(w['events'][0]['cue_hex'])[11],11*DT);e.predict(12*DT)
reject(lambda:e.predict(12*DT));reject(lambda:e.observe_outcome(88,12*DT));reject(lambda:e.observe_outcome(48,12*DT+1));reject(lambda:e.flush(165.))
reject(lambda:base.clone().feed(48,-1.));reject(lambda:base.clone().feed(256,0.))
reject(lambda:base.clone().flush(-1.));reject(lambda:e.shared[0].association_value(-1.))
back=base.clone()
for i,b in enumerate(bytes.fromhex(w['events'][0]['cue_hex'])):back.feed(b,i*DT)
reject(lambda:back.predict(9*DT))
checks.append('invalid_outcome_sequence_and_time_rejected')
# Operative mutable clone isolation only: frozen parameters intentionally share.
for store_index in range(8):
 for field in ('fast','slow','adapt','fe_p','w','bias','clock'):
  a=base.clone();before=full_state(base);st=(a.shared+a.private)[store_index]
  if field in ('fast','slow','adapt'):getattr(st.fly.m,field).flat[0]+=1
  elif field=='fe_p':st.fe.p.flat[0]+=1
  elif field in ('w','bias'):getattr(st,field).flat[0]+=1
  else:st.brain_t+=1;st.fe.t+=1
  assert full_state(base)==before
 a=base.clone();a.private[0].fe.visible=b'1234';assert full_state(base)==before
pending=base.clone();cue(pending,w['events'][0]['cue_hex'],0.)
for field in ('pending_x','cached'):
 a=pending.clone();before=full_state(pending)
 if field=='cached':a.cached[0]+=1
 else:a.private[0].pending_x.flat[0]+=1
 assert full_state(pending)==before
for field in ('brain','elapsed','fe'):
 bad=base.clone()
 if field=='brain':bad.shared[0].brain_t+=1
 elif field=='elapsed':bad.shared[0].fly.m.elapsed+=1
 else:bad.shared[0].fe.t+=1
 reject(lambda:check_clock(bad,0.))
checks+=['operative_mutable_clone_isolation','independent_brain_elapsed_fe_corruption_detected']
fixed_anchor=fixed_identity(base)
for field in ('fe_tau','arm','scales','reader','connectome','predictor'):
 a=base.clone();before=full_state(base)
 if field=='fe_tau':a.shared[0].fe.tau+=1
 elif field=='arm':a.arm='test-only-altered-arm'
 elif field=='scales':a.scales=(a.scales[0]+1,a.scales[1])
 elif field=='reader':
  rr=copy.copy(a.shared[0].fly.reader);rr.head={k:v.copy() for k,v in rr.head.items()};rr.head['intercept'].flat[0]+=1;a.shared[0].fly.reader=rr
 elif field=='connectome':
  a.shared[0].fly.m.B=a.shared[0].fly.m.B.copy();a.shared[0].fly.m.B.data.flat[0]+=1
 else:a.shared[0].w.flat[0]+=1
 assert full_state(a)!=before and fixed_identity(a)!=fixed_anchor and full_state(base)==before
checks.append('full_and_fixed_fingerprint_sensitivity')
branches={name:base.clone() for name in ('W','N_old_relation','N_new','half','shared','forced')}
for name,alpha in [('half',.5),('shared',0.),('forced',None)]:
 m=branches[name];original=m.predict
 def policy(self,t,_original=original,_alpha=alpha):
  out=_original(t);s=np.array(out['shared'])/SCALES[0];p=np.array(out['private'])/SCALES[1]
  if _alpha is None:out['emitted']=int(ALPHABET[(ALPHABET.index(out['emitted'])+1)%4])
  else:
   out['combined']=(s+_alpha*p).tolist();out['emitted']=int(ALPHABET[int(np.argmax(out['combined']))])
  return out
 m.predict=types.MethodType(policy,m)
policy_names=('W','half','shared','forced')
steps=[];clock_rows=[];phase_rows=[]
def verify_phase(record,phase):
 assert len({full_state(branches[n]) for n in policy_names})==1
 assert len({nonplastic(m) for m in branches.values()})==1
 assert all(fixed_identity(m)==fixed_anchor for m in branches.values())
 for name,m in branches.items():
  for x in m.shared+m.private:
   assert abs(float(x.fly.m.elapsed)-x.elapsed_base-x.brain_t)<1e-6
   if x.pending_t is None:assert x.pending_x is None and abs(x.fe.t-x.brain_t)<1e-6
   else:assert x.pending_x is not None and x.pending_t>=x.brain_t-1e-6 and abs(x.fe.t-x.pending_t)<1e-6 and x.last_byte_t==x.pending_t
  phase_rows.append({'record':record,'phase':phase,'branch':name,'operative':full_state(m),'nonplastic':nonplastic(m),'fixed':fixed_identity(m)})
for j,event in enumerate(w['events'][:4]+w['events'][192:196]):
 if j==4:
  for m in branches.values():
   assert m.flush(w['old_end'])<1e-6;clock_rows.append({'branch':next(n for n,x in branches.items() if x is m),'at':w['old_end'],'clocks':check_clock(m,w['old_end'])})
   assert m.flush(w['new_start'])<1e-6;clock_rows.append({'branch':next(n for n,x in branches.items() if x is m),'at':w['new_start'],'clocks':check_clock(m,w['new_start'])})
 predictions={};blocked_map={}
 for name,m in branches.items():
  predictions[name]=cue(m,event['cue_hex'],event['at'])
  if name in ('W','N_old_relation','N_new'):scores(predictions[name],SCALES)
 verify_phase(j,'prediction')
 for name,m in branches.items():
  blocked=(name=='N_old_relation' and event['stage']=='old') or (name=='N_new' and event['stage']=='new');blocked_map[name]=blocked
  with suppress_value_writes(m,blocked):m.observe_outcome(event['outcome'],event['at']+12*DT)
  if blocked:assert m.audit[-1]['shared_l1']==[0.]*4 and m.audit[-1]['private_l1']==[0.]*4
 verify_phase(j,'outcome')
 for m in branches.values():m.feed(10,event['at']+13*DT)
 verify_phase(j,'newline')
 for name,m in branches.items():
  err=m.flush(event['at']+RECORD_SECONDS);assert err<1e-6;check_clock(m,event['at']+RECORD_SECONDS)
  steps.append({'record':j,'branch':name,'stage':event['stage'],'predicted':predictions[name]['emitted'],'state':full_state(m),'nonplastic':nonplastic(m),'blocked':blocked_map[name],'clock_error':err})
 verify_phase(j,'flush')
 assert len({nonplastic(m) for m in branches.values()})==1
for at in (w['new_end'],w['final']):
 for name,m in branches.items():
  assert m.flush(at)<1e-6;clock_rows.append({'branch':name,'at':at,'clocks':check_clock(m,at)})
assert len({full_state(branches[n]) for n in policy_names})==1
assert all(fixed_identity(m)==fixed_anchor for m in branches.values())
checks+=['48_locked_teaching_records','all_policy_operative_states_identical','causal_nonplastic_states_identical','all_branch_clocks_and_delays','write_clamps_correct']
probes={n:probe(branches[n],w,w['final'],('old','heldout','new')) for n in ('W','N_old_relation','N_new')}
assert all(len(v['rows'])==32 and v['state_before']==v['state_after'] for v in probes.values())
# Actual target-bearing evaluator path on two disposable clones, no teaching.
ch=w['sets']['heldout']['cues'][0];actual=w['sets']['heldout']['outcomes'][0];other=ALPHABET[(ALPHABET.index(actual)+1)%4]
wa={'sets':{'label_adversary':{'cues':[ch],'outcomes':[actual]}}};wb=copy.deepcopy(wa);wb['sets']['label_adversary']['outcomes'][0]=int(other)
before=full_state(branches['W']);pa=probe(branches['W'],wa,w['final'],('label_adversary',));pb=probe(branches['W'],wb,w['final'],('label_adversary',))
a=pa['rows'][0];b=pb['rows'][0]
for field in ('emitted','shared','private','combined'):assert a[field]==b[field]
for name in a['policies']:
 for field in ('scores','emitted','ties'):assert a['policies'][name][field]==b['policies'][name][field]
 for row in (a,b):assert row['policies'][name]['correct']==int(row['policies'][name]['emitted']==row['target'])
assert a['target']!=b['target'] and full_state(branches['W'])==before
assert all(fixed_identity(m)==fixed_anchor for m in branches.values())
label_check={'first_probe':pa,'second_probe':pb,'prediction_unchanged':True,'continuing_state':before}
checks+=['96_disposable_first_pre_feedback_probes','scoring_label_independence']
# Source/report tampering exercises gates without learner execution; restore in finally.
for path in [ROOT/'source/runtime/survivor_fixture.py',ROOT/'source/baseline/src/learner.py',ROOT/'audits/CYCLE2_SOURCE_REVIEW.md']:
 original=path.read_bytes()
 try:
  path.write_bytes(original+b'\n# TEMPORARY TAMPER PROBE\n');reject(require_technical)
 finally:path.write_bytes(original)
require_technical();checks.append('source_and_report_tamper_rejected_and_restored')
d={'schema':'RC-SURVIVOR-CYCLE2-TEST-v1','passed':True,'fixture':w['sha256'],'checks':checks,'steps':steps,'phase_rows':phase_rows,'clocks':clock_rows,
   'probes':probes,'label_check':label_check,'source':hashes(),'state_equivalence_scope':'fixed open-loop observed-outcome stream; returned policy output only',
   'resources':{'wall_s':time.monotonic()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}}
p=ROOT/'receipts/cycle2/smoke.json';p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(d,sort_keys=True,indent=2)+'\n')
with q.open('rb') as f:os.fsync(f.fileno())
try:os.link(q,p)
finally:q.unlink()
print(json.dumps({'passed':True,'resources':d['resources'],'checks':checks}))
