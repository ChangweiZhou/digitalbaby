"""No training: outcome-gate fixtures, streamed metrics, causal IDs and snapshot plumbing."""
from common import *
import copy,tempfile,io
import learner as L
import evaluator as E
from sensor import Sensor
import metrics as M
from reduce import decide,components

def fake():
 metrics={}
 for w in ('M1','M2'):
  metrics[w]={}
  for arm in ('NORMAL','FIXED-LOW','RMS-SHAM','CGLP'):
   val=1. if arm=='CGLP' else 2.;a={}
   for stage in ('birth','A1','A2','B1','B2','B3','EDIT1','EDIT2','NOEDIT1','NOEDIT2'):
    a[stage]={'A_held':{'bits':1.},'B_held':{'bits':val},'EDIT_held':{'partitions':{'SAME_KC':{'bits':val},'CHANGED':{'bits':1.3 if stage=='NOEDIT2' else 1.}}}}
   metrics[w][arm]=a
 return metrics
counts={'worlds':{w:{'A_unigram':{'bits':2.}} for w in ('M1','M2')}};m=fake();assert decide(m,counts)['verdict']=='SELECTIVE_PLASTICITY_MINI_DEV'
assert decide(m,{'worlds':{w:{'A_unigram':{'bits':.5}} for w in ('M1','M2')}})['verdict']=='MINI_CAPACITY_OR_LEARNING_GAP'
# A candidate cannot pass from NORMAL comparison or one world alone.
one=copy.deepcopy(m);one['M2']['CGLP']['EDIT2']['EDIT_held']['partitions']['CHANGED']['bits']=4.;assert decide(one,counts)['verdict']!='SELECTIVE_PLASTICITY_MINI_DEV'
feedback=copy.deepcopy(m)
for w in feedback:feedback[w]['CGLP']['NOEDIT2']['EDIT_held']['partitions']['CHANGED']['bits']=1.
assert decide(feedback,counts)['verdict']!='SELECTIVE_PLASTICITY_MINI_DEV'
# Read-only metrics include real EOS and uneven minibatch tail, no optimizer call.
s=Sensor();net=L.Learner();records=[b'ab',b'x'*128];packed=E.pack(s,records);out,sites=M.score(net,records,packed,records);assert out['targets']==132 and out['record_target_counts']==[3,129]
ref=E.score(net,records,packed);assert abs(ref['bits']-out['bits'])<1e-10
parts=M.corrected_partitions(records,records,packed,packed,sites);assert parts['SAME_KC']['n']==132 and parts['CHANGED']['n']==0
st=copy.deepcopy(net.state());buf=io.BytesIO();L.T.save(st,buf);restored=L.T.load(io.BytesIO(buf.getvalue()),weights_only=True);q=L.Learner();q.restore(restored);assert E.state_hash(q)==E.state_hash(net)
# Expected fixed dose, target-batch conservation across complete record boundaries.
lengths=[3,129,65,129,1];batch=[];left=0
for n in lengths:
 for i in range(n):
  batch.append(i==n-1)
  if len(batch)==256:left+=sum(batch);batch=[]
left+=sum(batch);assert left==len(lengths)
print(json.dumps({'status':'PASS','new_training_records':0,'new_optimizer_updates':0,'synthetic_outcome_gate_tests':4,'read_only_metric_targets':132,'normalization':'full257','serialization_and_batch_tail':True}),flush=True)
