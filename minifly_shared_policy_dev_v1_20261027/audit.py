"""Independent saved-artifact reduction. Does not import or run native models."""
import os,time,json,pickle,hashlib,resource,sys
from pathlib import Path
from collections import Counter
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
T0=time.process_time()
import numpy as np
H=Path(__file__).resolve().parent;ROOT=H.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads((H/p).read_text())
def save(p,x):
 f=H/p;f.parent.mkdir(parents=True,exist_ok=True);f.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
class Ghost:pass
class Reader(pickle.Unpickler):
 def find_class(self,module,name):
  if module=='sharedlib' and name=='Shared':return Ghost
  return super().find_class(module,name)
def main():
 result=load('RUN_RESULTS.json');spec=load('EXECUTION_SPEC.json');source=load('MAIN_SOURCE_LOCK.json');assert result['main_arms_all_completed']
 assert all(sha(ROOT/p)==v for p,v in source.items());assert all(sha(ROOT/p)==v for p,v in spec['inputs_sha256'].items())
 words=json.loads((ROOT/'minifly_cls_dev_20261017/OUTPUT_CODEBOOK.json').read_text())['words_by_class'];C=np.array([[1 if w&(1<<j) else -1 for j in range(16)] for w in words]);assert np.all(C.sum(1)==0)
 data=json.loads((ROOT/'minifly_v171_dev_20261026/local/TRAIN_DATA.json').read_text());arms=spec['fitted_arms'];panels=0;sites=0;arrays=0;records=0;tables=[];checks=Counter();momentcheckpoints=0;history=[]
 for world in spec['worlds']:
  w=data[str(world)];wr=result['worlds'][str(world)];rows={r['id']:r for group in ('OLD','NEW') for r in w[group]};old={r['id'] for r in w['OLD']};new={r['id'] for r in w['NEW']};allids=old|new
  cachepath=H/wr['cache_receipt']['path'];assert sha(cachepath)==wr['cache_receipt']['sha256']
  with cachepath.open('rb') as f:cache=pickle.load(f)
  assert set(cache)==allids;prompt_norms=[]
  for identifier,row in rows.items():
   r=cache[identifier];assert r['prompt']==row['prompt'] and r['answer']==row['answer'];assert len(r['prompt_v'])==len(bytes.fromhex(row['prompt']));assert len(r['target_v'])==len(bytes.fromhex(row['answer']))+1;assert np.array_equal(r['boundary_v'],r['target_v'][0]);prompt_norms.extend(np.linalg.norm(r['prompt_v'],axis=1).tolist())
  assert abs(wr['constant_scale']-np.median(prompt_norms)/4)<1e-14
  log=[json.loads(x) for x in (H/'local'/f'{world}_RECORDS.jsonl').read_text().splitlines()];assert len(log)==1280
  expected_order=[]
  for phase in ('OLD','NEW'):
   for p in range(1,17):
    seq=[w['OLD'][int(j)]['id'] for j in w['orders'][(p-1)%8]] if phase=='OLD' else [r['id'] for r in w['NEW']]
    expected_order.extend((phase,p,j) for j in seq)
  assert [(r['phase'],r['pass_no'],r['id']) for r in log]==expected_order
  clocks={a:0 for a in arms};ts={a:0 for a in arms}
  for r in log:
   count=len(bytes.fromhex(rows[r['id']]['answer']))+1
   for a in arms:
    u=r['updates'][a];assert u['n_start']==clocks[a] and u['n_end']==clocks[a]+count;assert u['targets']==count;clocks[a]+=count;ts[a]+=1;records+=1
    ws=[1/16+15/16*min(n/2048,1) for n in range(u['n_start'],u['n_end'])];assert abs(u['mean_w']-np.mean(ws))<1e-14 and abs(u['min_w']-min(ws))<1e-14
    assert u['opt_t']==(0 if a=='SGD3' else ts[a]);assert u['gradient_norm']['R']==0 and u['applied_step_norm']['R']==0 if a=='ADAM3-R0' else True
    checks['optimizer_record_clocks']+=1
  assert clocks==wr['controller_target_clocks'];assert all(t==1280 for a,t in wr['Adam_record_clocks'].items() if a!='SGD3')
  expected_n={a:0 for a in arms}
  for receipt in wr['training_array_receipts']:
   path=H/receipt['path'];assert sha(path)==receipt['sha256']
   with np.load(path,allow_pickle=False) as z:
    md=json.loads(z['metadata'].tobytes());P=z['p'];V=z['v'];G=z['g'];A=z['a'];assert P.shape==(len(md),257)
    logits=(G*V)@C.T/4;logits-=logits.max(1,keepdims=True);lp=logits-np.log(np.exp(logits).sum(1,keepdims=True));assert np.allclose(P,np.exp(lp),rtol=1e-11,atol=1e-14)
    assert np.allclose(G,1+.5*np.tanh(A),rtol=0,atol=1e-14)
    for i,m in enumerate(md):
     arm=m['arm'];y=m['y'];identifier=m['id'];seq=list(bytes.fromhex(rows[identifier]['answer']))+[256];assert seq[m['site']]==y and m['n']==expected_n[arm];expected_n[arm]+=1
     warm=1/16+15/16*min(m['n']/2048,1);assert abs(m['actual_w']-warm)<1e-14;assert abs(m['warmup']-(warm if arm=='SGD3' else 1))<1e-14
     assert m['argmax']==int(P[i].argmax()) and abs(m['lp']-lp[i,y])<1e-10
     if arm=='ADAM1':assert len(m['ages'])<=1
     else:assert len(m['ages'])<=3
     expected_v=np.full(16,wr['constant_scale']) if arm=='ADAM3-CONST' else cache[identifier]['target_v'][m['site']];assert np.array_equal(V[i],expected_v)
     err=P[i].copy();err[y]-=1;dv=err@C/4;dv-=dv.mean();direct=.5*(1-np.tanh(A[i])**2)*V[i]*dv;assert np.allclose(direct,z['direct_error'][i],rtol=1e-10,atol=1e-12)
    sites+=len(md);arrays+=1
  assert expected_n==clocks
  parameter_panels={}
  for receipt in wr['checkpoint_receipts']:
   p=H/receipt['path'];assert sha(p)==receipt['sha256']
   with p.open('rb') as f:actors=Reader(f).load()
   parameter_panels[Path(receipt['path']).stem.split('_')[-1]]={a:{k:getattr(c,k).copy() for k in spec['parameter_caps']} for a,c in actors.items()}
   for a,c in actors.items():
    assert not c.h.any() and not c.r.any() and not c.frames
    for k,limit in spec['parameter_caps'].items():assert np.max(abs(getattr(c,k)))<=limit+1e-14
    if a=='ADAM3-R0':assert not c.R.any() and not c.mom1['R'].any() and not c.mom2['R'].any()
    momentcheckpoints+=1
  for rec in (wr['native'],wr['native_delayed']):
   path=H/rec['path'];assert sha(path)==rec['sha256'];rr=json.loads(path.read_text())['rows'];assert len(rr)==80
   good={x['id'] for x in rr if x['EOS'] and x['output_hex']==rows[x['id']]['answer']};assert good==set(rec['correct_ids'])
   for x in rr:assert x['correct']==(x['EOS'] and x['output_hex']==rows[x['id']]['answer']) and x['correct']==all(t['argmax']==t['y'] for t in x['teacher_forced'])
   checks['native_baseline_pairs']+=len(rr)
  panelsets={};panel_extra={}
  for tag,mapping in wr['panels'].items():
   panelsets[tag]={};panel_extra[tag]={}
   for arm,rec in mapping.items():
    path=H/rec['path'];assert sha(path)==rec['sha256'];rr=json.loads(path.read_text())['rows'];assert len(rr)==80
    good=set();nll=0;targets=0;EOS_correct=0;payload_correct=0;payload_sites=0;first_correct=0
    for r in rr:
     chosen=r['choices'];assert r['EOS']==(bool(chosen) and chosen[-1]==256);assert r['CAP']==(not r['EOS']);assert len(chosen)<=4
     assert r['output_hex']==bytes(x for x in chosen if x!=256).hex();correct=r['EOS'] and r['output_hex']==rows[r['id']]['answer'];assert r['correct']==correct;assert correct==all(t['argmax']==t['y'] for t in r['teacher_forced'])
     assert [t['y'] for t in r['teacher_forced']]==list(bytes.fromhex(rows[r['id']]['answer']))+[256]
     if correct:good.add(r['id'])
     EOS_correct+=int(r['teacher_forced'][-1]['argmax']==256);payload_correct+=sum(t['argmax']==t['y'] for t in r['teacher_forced'][:-1]);payload_sites+=len(r['teacher_forced'])-1;first_correct+=int(r['teacher_forced'][0]['argmax']==r['teacher_forced'][0]['y']);nll+=sum(-x['lp'] for x in r['teacher_forced']);targets+=len(r['teacher_forced']);panels+=1
    assert good==set(rec['correct_ids']) and len(good)==rec['correct'];assert abs(nll-rec['TF_nll'])<1e-9 and targets==rec['TF_targets'];panelsets[tag][arm]=good;panel_extra[tag][arm]=dict(TF_EOS_correct=EOS_correct,TF_payload_correct=payload_correct,TF_payload_sites=payload_sites,TF_first_correct=first_correct,empty_EOS=rec['empty_EOS'],CAP=rec['CAP'])
  native=set(wr['native']['correct_ids']);native_delay=set(wr['native_delayed']['correct_ids'])
  for a in arms:
   initial=panelsets['OLD16'][a]&old;final=panelsets['NEW16'][a];delayed=panelsets['POST_IDLE128'][a];sgd=panelsets['NEW16']['SGD3'];rec=wr['panels']['NEW16'][a]
   movement={k:dict(change_L2=float(np.linalg.norm(parameter_panels['NEW16'][a][k]-parameter_panels['BIRTH'][a][k])),at_cap=int(np.count_nonzero(abs(parameter_panels['NEW16'][a][k])>=limit-1e-14)),coordinates=int(parameter_panels['NEW16'][a][k].size)) for k,limit in spec['parameter_caps'].items()}
   tables.append(dict(world=world,arm=a,final_OLD=len(final&old),final_NEW=len(final&new),final_all80=len(final),delayed_OLD=len(delayed&old),delayed_NEW=len(delayed&new),delayed_all80=len(delayed),initially_correct_OLD=len(initial),initial_OLD_retained=len(initial&final),initial_OLD_lost=len(initial-final),TF_bits_per_target=rec['TF_nll']/rec['TF_targets']/np.log(2),vs_native_gained=len(final-native),vs_native_lost=len(native-final),vs_SGD3_gained=len(final-sgd),vs_SGD3_lost=len(sgd-final),parameter_movement=movement,**panel_extra['NEW16'][a]))
  tables.append(dict(world=world,arm='NATIVE',final_OLD=len(native&old),final_NEW=len(native&new),final_all80=len(native),delayed_all80=len(native_delay)))
  for tag in wr['panels']:history.append(dict(world=world,panel=tag,correct={a:len(v) for a,v in panelsets[tag].items()}))
  # Equal births imply exactly equal forward output for these unlesioned roles.
  assert panelsets['BIRTH']['SGD3']==panelsets['BIRTH']['ADAM3']==panelsets['BIRTH']['ADAM1']
 assert records==12800 and sites==36800 and arrays==320 and panels==4800
 assert result['counts']['main_fit_records']==12800 and result['counts']['main_fit_targets']==36800
 assert result['counts']['paired_queries']==5122 and result['main_native_teaching_records']==0 and result['held_assessment_decisions']==0
 save('RESULTS.json',dict(identity='SP-COND-V1',status='AUDITED_FIXED_DOSE_COMPLETE',table=tables,history=history,independent_worlds=2,all_fitted_arms_reference_instruments=True,E3_assessment_decisions=0,new_native_training_records=0,controller_main_record_applications=records,controller_main_target_exposures=sites,new_core_adopted=False))
 save('FINAL_AUDIT.json',dict(status='ACCEPTED',source_files_verified=len(source),source_and_inputs_unchanged=True,training_array_files=arrays,training_target_distributions_verified=sites,optimizer_records_verified=records,panel_own_TF_pairs_verified=panels,native_baseline_pairs_verified=checks['native_baseline_pairs'],controller_checkpoint_instances_checked=momentcheckpoints,technical_records=5,technical_targets=10,technical_own_TF_pairs=2,technical_synthetic_loss_evaluations=24,counts=result['counts'],worlds=2,core_trajectories_replayed=False,held_assessment_opened=False,new_native_teaching=False,all_arms_same_fixed_dose=True,exact_gradients_reference_only=True,all_payload_and_EOS_classes_unmasked=True,CONST_also_removes_centered_feedback=True,no_attention_or_arithmetic_structure_added=True,model_queries_for_this_reducer=0,CPU_seconds=time.process_time()-T0,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,new_core_adopted=False))
 print(json.dumps(tables))
if __name__=='__main__':main()
