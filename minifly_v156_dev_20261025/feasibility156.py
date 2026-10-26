"""Finite evaluator witnesses. Nothing in this file teaches or answers for MiniFly."""
import time,warnings
import numpy as np
from scipy import sparse,linalg,optimize
import support156 as s
import native156 as n

def features(front,world):
 p=s.ROOT/f'minifly_v154_dev_20261025/local/FEATURE_{front}_{world}'
 rows=s.inputs()[str(world)]['OLD']+s.inputs()[str(world)]['NEW'];allowed={r['id'] for r in rows}
 with np.load(str(p)+'.npz',allow_pickle=False) as z:
  ids=z['record_ids'];ix=np.flatnonzero(np.asarray([r in allowed for r in ids]));ids=ids[ix];y=z['y'][ix];positions=z['positions'][ix]
 X=sparse.load_npz(str(p)+'_OBS.npz')[ix].tocsr();assert X.shape[1]==5177 and len(ix)<=384
 assert set(ids)==allowed and len(rows)==128
 for row in rows:
  ii=np.flatnonzero(ids==row['id']);assert list(y[ii])==list(bytes.fromhex(row['answer']))+[256] and list(positions[ii])==list(range(len(ii)))
 return X,y,ids,positions,rows

def verify(X,y,weights,b,family):
 """Independent rowwise forward-error bounds; stored floats denote exact finite numbers.

 IEEE round-to-nearest dot products are enclosed with gamma_(n+32),
 an 8-fold safety inflation (including bound computation), and underflow terms.
 A is dyadic and exactly representable. Never accepts an ambiguous sign.
 """
 if not np.isfinite(weights).all() or not np.isfinite(b).all():return dict(valid=False,reason='NONFINITE_PARAMETERS')
 ld=np.longdouble;u=np.finfo(ld).eps/2;tiny=np.finfo(ld).tiny;maxerr=ld(0);minimum=ld(np.inf)
 for i,target in enumerate(y):
  row=X.getrow(i);vv=weights[:,row.indices].astype(ld)*row.data.astype(ld)[None,:];v=vv.sum(1,dtype=ld)
  g=(len(row.data)+32)*u/(1-(len(row.data)+32)*u)
  ve=np.nextafter(8*g*(abs(vv).sum(1,dtype=ld)+tiny)+tiny*(len(row.data)+32),ld(np.inf))
  if family!='FQ':
   prod=n.A.astype(ld)*v[None,:];z=prod.sum(1,dtype=ld);ge=48*u/(1-48*u)
   err=np.nextafter(abs(n.A.astype(ld))@ve+8*ge*(abs(prod).sum(1,dtype=ld)+tiny)+tiny*48,ld(np.inf))
  else:z=v;err=ve
  zz=z+b.astype(ld);err=np.nextafter(err+8*u*(abs(z)+abs(b.astype(ld))+tiny)+tiny,ld(np.inf))
  lo=zz[target]-err[target];upper=zz+err;upper[target]=-np.inf;minimum=min(minimum,lo-upper.max());maxerr=max(maxerr,err.max())
  s.COUNTS['certificate_row_rival_evaluations']+=256
 return dict(valid=bool(np.isfinite(minimum) and minimum>=1),strictly_positive=bool(np.isfinite(minimum) and minimum>0),lower_margin=float(minimum) if np.isfinite(minimum) else None,max_logit_error_bound=float(maxerr),norm=float(np.linalg.norm(weights)),parameter_bytes=weights.nbytes+b.nbytes,verification='independent rowwise gamma bound, IEEE stored floats',longdouble_mantissa=np.finfo(ld).nmant)

def certify_candidate(X,y,w,b,family):
 cert=verify(X,y,w,b,family)
 if cert.get('strictly_positive') and not cert['valid']:
  scale=2/cert['lower_margin'];w=w*scale;b=b*scale;cert=verify(X,y,w,b,family)
 return w,b,cert

def duplicate_certificate(X,y):
 seen={}
 for i in range(X.shape[0]):
  row=X.getrow(i);key=(row.indices.tobytes(),row.data.tobytes())
  if key in seen and int(y[seen[key]])!=int(y[i]):
   return dict(type='IDENTICAL_ROWS_CONFLICT',rows=[seen[key],i],targets=[int(y[seen[key]]),int(y[i])],lambda_values=[1,1],proof='Opposite required class differences at identical exact feature rows sum to 0 >= 2')
  seen[key]=i
 return None
def check_duplicate(X,y,cert):
 i,j=cert['rows'];return int(y[i])!=int(y[j]) and np.array_equal(X.getrow(i).indices,X.getrow(j).indices) and np.array_equal(X.getrow(i).data,X.getrow(j).data) and cert['lambda_values']==[1,1]

def lp_witness(X,y,family):
 """Bounded sparse all-rival separation; numerical infeasibility alone is unresolved."""
 start=time.process_time();cols=np.unique(X.indices);xx=X[:,cols].tocsr();d=len(cols);k=257 if family=='FQ' else 16;nb=0 if family=='C0' else 257;nv=k*d+nb
 constraints=[];seen=set();params=np.zeros(nv);iterations=0;scans=0;status='UNRESOLVED';reason='SCAN_LIMIT'
 for scan in range(64):
  remaining=60-(time.process_time()-start);s.budget()
  if remaining<=0 or iterations>=5000:reason='CELL_RESOURCE_LIMIT';break
  w=params[:k*d].reshape(k,d);b=params[k*d:] if nb else np.zeros(257)
  z=(xx@w.T)@(n.A.T if k==16 else np.eye(257))+b
  gaps=z[np.arange(len(y)),y,None]-z;gaps[np.arange(len(y)),y]=np.inf;scans+=1;s.COUNTS['separation_row_rival_evaluations']+=len(y)*256
  if gaps.min()>0:
   full=np.zeros((k,X.shape[1]));full[:,cols]=w;full,b,cert=certify_candidate(X,y,full,b,family)
   if cert['valid']:return full,b,cert,dict(scans=scans,iterations=iterations,cpu=time.process_time()-start)
  candidates=np.flatnonzero(gaps.ravel()<1);order=sorted(candidates,key=lambda q:(float(gaps.ravel()[q]),int(q)//257,int(q)%257))
  added=0
  for flat in order:
   pair=(int(flat)//257,int(flat)%257)
   if pair not in seen:seen.add(pair);constraints.append(pair);added+=1
   if added==256:break
  if not added:reason='NO_NEW_CONSTRAINT_NUMERICAL';break
  rid=[];cid=[];val=[]
  for ri,(i,rival) in enumerate(constraints):
   row=xx.getrow(i);target=int(y[i])
   if family=='FQ':
    for c,sign in [(target,1),(rival,-1)]:rid.extend([ri]*row.nnz);cid.extend((c*d+row.indices).tolist());val.extend((sign*row.data).tolist())
   else:
    a=n.A[target]-n.A[rival]
    for j in np.flatnonzero(a):rid.extend([ri]*row.nnz);cid.extend((int(j)*d+row.indices).tolist());val.extend((a[j]*row.data).tolist())
   if nb:rid.extend([ri,ri]);cid.extend([k*d+target,k*d+rival]);val.extend([1.,-1.])
  DD=sparse.coo_matrix((val,(rid,cid)),shape=(len(constraints),nv)).tocsr();del rid,cid,val
  s.COUNTS['LP_constraints_materialized']+=len(constraints);s.COUNTS['LP_nonzeros_materialized']+=DD.nnz
  # Keep the actual temporary sparse matrix within the common RSS cap.
  s.budget();remaining=max(.001,60-(time.process_time()-start))
  with warnings.catch_warnings():
   warnings.simplefilter('ignore',optimize.OptimizeWarning)
   fit=optimize.linprog(np.zeros(nv),A_ub=-DD,b_ub=np.full(len(constraints),-2.),bounds=(None,None),method='highs',options={'time_limit':remaining,'maxiter':5000-iterations,'threads':1,'parallel':False})
  iterations+=int(fit.nit or 0);s.COUNTS['LP_iterations']+=int(fit.nit or 0)
  if fit.x is None:reason='NUMERICAL_INFEASIBLE_WITHOUT_EXACT_CERTIFICATE' if fit.status==2 else 'SOLVER_LIMIT_OR_FAILURE';break
  params=fit.x
 return None,None,dict(valid=False,status=status,reason=reason),dict(scans=scans,iterations=iterations,cpu=time.process_time()-start)

def policy_query(base,prompt,start_time,w,b,family):
 """No expected output, target, record ID or cache access is accepted by this policy."""
 m=base.clone();m.time=start_time
 for bank in m.bank:bank.brain_t=start_time
 m._prompt(prompt);features=[];choices=[];ended=False;payload=[]
 for _ in range(4):
  raw=n.n.taps(m)['OBS'];phi=raw/max(1.,np.linalg.norm(raw));v=w@phi;z=n.A@v+b if family!='FQ' else v+b;y=int(z.argmax());features.append(phi);choices.append(y)
  s.COUNTS['evaluator_distributions']+=1;s.COUNTS['witness_distributions']+=1
  if y==256:ended=True;break
  payload.append(y);m._event(y,None)
 return bytes(payload),ended,choices,features

def rollout(X,y,ids,positions,rows,w,b,family,front,world):
 base=n.n.Native('N_2B',front);base.bank=base.bank[:1];clock=0.;correct=0;parity=0;bound_preserved=0;unknown=0;query_rows=[]
 for row in rows:
  p=bytes.fromhex(row['prompt']);a=bytes.fromhex(row['answer']);s.COUNTS['witness_queries']+=1;s.budget()
  got,ended,choices,actual=policy_query(base,p,clock,w,b,family)
  # Audit only after the autonomous rollout; expected prefixes never choose policy inputs.
  ix=np.flatnonzero(ids==row['id']);flags=[]
  for j,phi in enumerate(actual):
   if j>len(a) or choices[:j]!=list(a[:j]):break
   cached=X.getrow(int(ix[j])).toarray()[0]
   if np.array_equal(phi,cached):parity+=1;flags.append('EXACT')
   else:
    one=sparse.csr_matrix(phi[None,:]);cert=verify(one,np.asarray([int(y[ix[j]])]),w,b,family)
    if cert.get('strictly_positive'):bound_preserved+=1;flags.append('RECOMPUTED_FEATURE_MARGIN_CERTIFIED')
    else:unknown+=1;flags.append('FEATURE_DISCREPANCY_UNRESOLVED')
  ok=ended and got==a;correct+=int(ok);query_rows.append(dict(id=row['id'],correct=ok,output_hex=got.hex(),EOS=ended,choices=choices,feature_checks=flags))
  # The cached collector advanced every prompt byte, outcome byte and EOS, plus two controls.
  # This reconstructs evaluator event timing from already observed record lengths only.
  for _ in range(len(p)+len(a)+3):clock+=n.n.DELTA
 result=dict(records=len(rows),correct=correct,exact_feature_rows=parity,recomputed_feature_decisions_certified=bound_preserved,unresolved_feature_rows=unknown,interface_qualified=correct==len(rows) and unknown==0,rows=query_rows)
 del base
 return result

def case(front,world):
 key=f'{front}_{world}';X,y,ids,positions,rows=features(front,world);start=time.process_time();zero=duplicate_certificate(X,y);results=[];inherited=None
 if zero:
  assert check_duplicate(X,y,zero)
  for family in ['C0','CQ','FQ']:results.append(dict(family=family,status='CERTIFIED_INFEASIBLE',certificate=zero))
 else:
  cols=np.unique(X.indices);dense=X[:,cols].toarray();V=n.C[y];s.COUNTS['constructive_target_rows']+=len(y);s.COUNTS['constructive_input_nonzeros']+=X.nnz
  solution,res,rank,sv=linalg.lstsq(dense,V,lapack_driver='gelsd');w=np.zeros((16,5177));w[:,cols]=solution.T
  w,b,cert=certify_candidate(X,y,w,np.zeros(257),'C0');condition=float(sv[0]/sv[-1]) if len(sv) and sv[-1]>0 else None
  construction=dict(numerical_rank=int(rank),rows=X.shape[0],nonzero_columns=len(cols),condition_estimate=condition,cpu=time.process_time()-start,rank_is_not_a_negative_certificate=True)
  if cert['valid']:inherited=(w,b)
  del dense,solution
  for family in ['C0','CQ','FQ']:
   s.budget();method='CONSTRUCTIVE_OR_NESTED_WITNESS';ops={}
   if inherited is not None:
    w,b=inherited;w=n.A@w if family=='FQ' else w
    w,b,cert=certify_candidate(X,y,w,b,family)
   else:
    method='SPARSE_CONSTRAINT_GENERATION';w,b,cert,ops=lp_witness(X,y,family)
    if cert.get('valid') and family in ['C0','CQ']:inherited=(w,b)
   if cert.get('valid'):
    path=s.H/'local'/f'WITNESS_{key}_{family}.npz';np.savez_compressed(path,weights=w,bias=b)
    r=rollout(X,y,ids,positions,rows,w,b,family,front,world)
    results.append(dict(family=family,status='FEASIBLE',certificate=cert,method=method,solver=ops,witness=dict(path=str(path.relative_to(s.H)),sha256=s.sha(path)),rollout=r))
   else:results.append(dict(family=family,status='UNRESOLVED',method=method,certificate=cert,solver=ops))
 result=dict(case=key,front=front,world=world,view='normalized OBS only',records=128,target_rows=len(y),feature_columns=5177,construction=construction if not zero else None,checks=results,labels='OLD/NEW only',cpu=time.process_time()-start)
 s.save(f'cases/{key}/FEASIBILITY_RESULT.json',result);s.log('FEASIBILITY_CASE_COMMIT',case=key,statuses=[r['status'] for r in results]);return result
