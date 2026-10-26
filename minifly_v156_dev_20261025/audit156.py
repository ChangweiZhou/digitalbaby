"""Independent saved-artifact reducer. Never loads a model or repeats a history."""
import json
import numpy as np
from scipy import sparse
import support156 as s

def response(choices,answer):
 seq=[];ended=False
 for v in choices:
  if v<0:break
  if v==256:ended=True;break
  seq.append(int(v))
 return ended and bytes(seq)==answer,ended,bytes(seq).hex()
def check_panel(path,rows):
 with np.load(path,allow_pickle=False) as a:z=a['z'];lp=a['lp'];off=a['offsets'];y=a['y'];margin=a['margin'];rival=a['rival'];correct=a['correct'];choices=a['choices'];eos=a['EOS'];common=a['common_logits'];clip=a['clipping']
 assert len(rows)==len(correct) and len(off)==len(rows)+1 and z.shape==lp.shape==(len(y),257)
 assert np.isfinite(z).all() and np.isfinite(lp).all() and (clip>=0).all()
 zz=z-z.max(1,keepdims=True);expected_lp=zz-np.log(np.exp(zz).sum(1,keepdims=True));assert np.max(abs(lp-expected_lp))<1e-11
 for i,r in enumerate(rows):
  a=bytes.fromhex(r['answer']);assert list(y[off[i]:off[i+1]])==list(a)+[256]
  co,en,_=response(choices[i],a);assert co==bool(correct[i]) and en==bool(eos[i])
 masked=z.copy();masked[np.arange(len(y)),y]=-np.inf;rv=masked.argmax(1);ma=z[np.arange(len(y)),y]-z[np.arange(len(y)),rv]
 assert np.array_equal(rv,rival) and np.array_equal(ma,margin)
 assert np.max(abs(common-z[off[:-1]].mean(0)))<1e-12
 return dict(margin=margin,correct=correct,z=z,y=y,offsets=off,clipping=clip)

def independent_witness(X,y,w,b,family,C):
 # Different reduction path from the rowwise long-double fit-side certificate.
 v=X@w.T;A=(C-C.mean(1,keepdims=True))/4
 if family!='FQ':z=v@A.T+b;mag=(abs(X)@abs(w.T))@abs(A.T)+abs(b)
 else:z=v+b;mag=abs(X)@abs(w.T)+abs(b)
 # Inflated gamma bound covers both products, subtractions and norm computation.
 eps=np.finfo(float).eps;err=64*eps*(X.shape[1]+64)*(mag+1)
 lower=z-err;upper=z+err;upper[np.arange(len(y)),y]=-np.inf
 minimum=float(np.min(lower[np.arange(len(y)),y]-upper.max(1)))
 s.COUNTS['audit_witness_row_rival_evaluations']+=len(y)*256
 return dict(valid=minimum>=1,independent_lower_margin=minimum)

def audit_case(front,world,C):
 key=f'{front}_{world}';d=s.inputs()[str(world)];rows=d['panel'];base=check_panel(s.H/'local'/f'{key}_initial.npz',rows);prior=base;result=s.load(s.H/f'cases/{key}/MARGIN_RESULT.json');initial=np.flatnonzero(base['correct'][:64]).tolist();assert result['initial_old_correct']==initial
 n0=result['initial_n'];raw_events=[];max_res=0.;max_saved=0.
 for e,row in zip(result['events'],d['selected_NEW']):
  i=e['index'];assert e['id']==row['id'];nn=len(bytes.fromhex(row['answer']))+1
  assert len(e['sites'])==len(e['matched_sites'])==nn
  for j,t in enumerate(e['sites']):
   expected=1/16+15/16*min((n0+j)/1024,1);assert abs(t['warmup']-expected)<1e-14
  n0+=nn;assert e['n']==n0
  idle=check_panel(s.H/'local'/f'{key}_e{i:02d}_idle.npz',rows);obs=check_panel(s.H/'local'/f'{key}_e{i:02d}_observed.npz',rows);actual=check_panel(s.H/'local'/f'{key}_e{i:02d}_actual.npz',rows)
  I=idle['margin']-prior['margin'];O=obs['margin']-idle['margin'];W=actual['margin']-obs['margin'];res=I+O+W-(actual['margin']-prior['margin']);max_res=max(max_res,float(abs(res).max()))
  assert e['correct']==int(actual['correct'].sum())
  for name,value in [('mean_I',I.mean()),('mean_O',O.mean()),('mean_W',W.mean())]:max_saved=max(max_saved,abs(e[name]-float(value)))
  raw_events.append(dict(i=i,I=I,O=O,W=W,prior_correct=prior['correct'],actual_correct=actual['correct']));prior=actual
 assert max_res<1e-10 and max_saved<1e-10
 delay=check_panel(s.H/'local'/f'{key}_final_idle.npz',rows);archived=check_panel(s.H/'local'/f'{key}_archived_final.npz',rows)
 assert result['final_complete']==int(prior['correct'].sum()) and result['final_idle_complete']==int(delay['correct'].sum()) and result['archived_final_complete']==int(archived['correct'].sum())
 f=s.load(s.H/f'cases/{key}/FEASIBILITY_RESULT.json');stem=s.ROOT/f'minifly_v154_dev_20261025/local/FEATURE_{front}_{world}'
 allowed={r['id'] for r in d['OLD']+d['NEW']}
 with np.load(str(stem)+'.npz',allow_pickle=False) as a:ids=a['record_ids'];ix=np.flatnonzero([v in allowed for v in ids]);y=a['y'][ix]
 X=sparse.load_npz(str(stem)+'_OBS.npz')[ix];witnesses=[]
 for cell in f['checks']:
  if cell['status']=='FEASIBLE':
   p=s.H/cell['witness']['path'];assert s.sha(p)==cell['witness']['sha256']
   with np.load(p,allow_pickle=False) as a:w=a['weights'];b=a['bias']
   cert=independent_witness(X,y,w,b,cell['family'],C);assert cert['valid'];rr=cell['rollout'];assert sum(r['correct'] for r in rr['rows'])==rr['correct']
   for r in rr['rows']:
    expected=next(a for a in d['OLD']+d['NEW'] if a['id']==r['id']);co,en,out=response(r['choices'],bytes.fromhex(expected['answer']));assert co==r['correct'] and en==r['EOS'] and out==r['output_hex']
   witnesses.append(dict(family=cell['family'],**cert))
  elif cell['status']=='CERTIFIED_INFEASIBLE':
   c=cell['certificate'];i,j=c['rows'];assert y[i]!=y[j] and (X.getrow(i)!=X.getrow(j)).nnz==0 and c['lambda_values']==[1,1]
  else:assert cell['status']=='UNRESOLVED'
 out=dict(status='ACCEPTED',case=key,panel_files_checked=51,telescoping_residual_max=max_res,saved_attribution_discrepancy_max=max_saved,independent_witnesses=witnesses,model_queries=0,training_replayed=False)
 s.save(f'cases/{key}/AUDIT.json',out);return out

def negative_fixtures():
 # Reducer rejects mutated answers/EOS; no observations are supplied to a learner.
 correct=response([65,256,-1,-1],b'A')[0];bad_answer=response([66,256,-1,-1],b'A')[0];bad_eos=response([65,66,67,68],b'A')[0]
 assert correct and not bad_answer and not bad_eos
 return dict(correct_fixture='ACCEPTED',mutated_answer='REJECTED',mutated_EOS='REJECTED')
