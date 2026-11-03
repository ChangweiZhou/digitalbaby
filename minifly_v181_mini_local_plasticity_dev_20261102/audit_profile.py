"""No new fitted trajectories: independent arithmetic tests and full-cost preflight."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,time,json,ast,math,resource,hashlib
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parent
sys.path.insert(0,str(ROOT/'minifly_v178_recurrent_failure_localization_dev_20261102/local/compat'))
import learner as L
from sensor import Sensor,sha
from data import prepare,corrected
from evaluator import pack,score,generate
import numpy as np
from collections import defaultdict
T=L.T
start=time.process_time();s=Sensor();net=L.Learner();data,audit=prepare(s)
def save(n,v):(H/n).write_text(json.dumps(v,indent=2)+'\n')
# Independent float64 finite differences of output loss and two expressly local
# surrogate objectives. Synthetic arrays only: no new learner updates or records.
rng=np.random.default_rng(181);n=7;xx=rng.random((n,64,256));wh=net.p['H'].numpy().astype(float);bh=np.ones(64)*.7;wt=net.p['T'].numpy().astype(float);bt=np.ones(64)*1.;wo=net.p['O'].numpy().astype(float);bo=np.zeros(257)
ah=np.einsum('njr,jr->nj',xx,wh)+bh;h=np.maximum(0,ah);at=h@wt.T+bt;drop=(rng.random(at.shape)>=.1)/.9;z=np.maximum(0,at)*drop;log=z@wo.T+bo;lp=log-log.max(1,keepdims=True);p=np.exp(lp);p/=p.sum(1,keepdims=True);y=rng.integers(257,size=n);e=p.copy();e[np.arange(n),y]-=1;e/=n;dt=(e@net.ft.numpy())*(at>0)*drop;dh=(e@net.fh.numpy())*(ah>0)
expected={'H':np.einsum('nj,njr->jr',dh,xx),'T':dt.T@h,'O':e.T@z}
def hobj(w):return np.sum(np.maximum(0,np.einsum('njr,jr->nj',xx,w)+bh)*(e@net.fh.numpy()))
def tobj(w):return np.sum(np.maximum(0,h@w.T+bt)*drop*(e@net.ft.numpy()))
def oobj(w):
 a=z@w.T+bo;a-=a.max(1,keepdims=True);return np.mean(np.log(np.exp(a).sum(1))-a[np.arange(n),y])
errors={}
for name,w,fun in [('H',wh,hobj),('T',wt,tobj),('O',wo,oobj)]:
 err=[]
 for _ in range(16):
  ix=tuple(rng.integers(q) for q in w.shape);old=w[ix];eps=1e-5;w[ix]=old+eps;f=fun(w);w[ix]=old-eps;g=fun(w);w[ix]=old;err.append(abs((f-g)/(2*eps)-expected[name][ix]))
 errors[name]=max(err);assert max(err)<1e-6
# Same-state RMS energy control and explicit decay-included scalar step.
delta=rng.normal(size=(64,257));g=rng.uniform(.25,1,64);en=(delta**2).sum(1);r=math.sqrt(np.sum(en*g*g)/en.sum());assert np.isclose(np.sum(en*r*r),np.sum(en*g*g),rtol=1e-12)
mask=(rng.random(n)>.3);scaled=np.array([e[mask].sum(0)*n/max(1,mask.sum()),e[~mask].sum(0)*n/max(1,(~mask).sum())]);assert np.allclose((scaled[0]*mask.sum()+scaled[1]*(~mask).sum())/n,e.sum(0))
save('INDEPENDENT_NUMERIC_AUDIT.json',{'status':'PASS','synthetic_not_science':True,'new_training_records':0,'finite_difference_max_abs_error':errors,'same_state_RMS_candidate_energy_relative_tolerance':1e-12,'unequal_group_weighted_recombination':True,'local_surrogates_not_global_gradient_claim':True})
# All evaluation panels at fixed birth; only costs are retained, no neural
# efficacy metric is used for selection. Compressed active indices, never an
# all-corpus dense N x5177 cache.
encoding=evalcpu=0.;cachebytes=0;evaltargets=0
for w,sp in data.items():
 sp=dict(sp);sp['EDIT_held']=[corrected(r,(bytes.fromhex(audit[w]['correction']['context_hex']),audit[w]['correction']['old'],audit[w]['correction']['new'])) for r in sp['A_held']]
 for name,rs in sp.items():
  t=time.process_time();packed=pack(s,rs);encoding+=time.process_time()-t;cachebytes+=sum(row.nbytes for code in packed for row in code)
  t=time.process_time();v=score(net,rs,packed);evalcpu+=time.process_time()-t;evaltargets+=v['targets']
t=time.process_time();gg=generate(net,s,data['M1']['A_held'][:16]+data['M1']['B_held'][:16]);gen_cpu=time.process_time()-t;steps=sum(g['outputs'] for g in gg)
# Actual source-locked countermodel implementations; calibration only; include
# all k=1..5 and D=.5/.75/.9. Profile, do not change a neural policy with them.
def definitions(path,names):
 tree=ast.parse(path.read_text());return compile(ast.Module(body=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names],type_ignores=[]),str(path),'exec')
env={'EOS':256,'V':257,'np':np,'math':math,'defaultdict':defaultdict};source=ROOT/'minifly_v179_continuous_local_life_dev_20261102/local/source'
exec(definitions(source/'minifly_v148_pretrain_probe_dev_20261025/ladder.py',['stream','Unigram']),env);exec(definitions(source/'minifly_v169_native_dev_v1_20261027/v169_count.py',['IKN','line_bits']),env)
t=time.process_time();count_fit_targets=count_fit_records=0;models=0
for w,sp in data.items():
 for role in ('A','B'):
  tr,ca,held=[sp[role+'_'+k] for k in ('train','calib','held')]
  uni=env['Unigram']().fit(tr);env['line_bits'](uni,held);count_fit_targets+=sum(len(r)+1 for r in tr);count_fit_records+=len(tr);models+=1;best=None
  for k in range(1,6):
   for d in (.5,.75,.9):
    model=env['IKN'](k,d).fit(tr);cb=env['line_bits'](model,ca).sum()/sum(len(r)+1 for r in ca);count_fit_targets+=sum(len(r)+1 for r in tr);count_fit_records+=len(tr);models+=1
    if best is None or cb<best[0]:best=(cb,model)
  env['line_bits'](best[1],held)
countcpu=time.process_time()-t
old=json.loads((H/'RESOURCE_ADMISSION.json').read_text());train=old['training_only_CPU_projection_with_1_5_reserve']/1.5-json.loads((H/'CYCLE3.json').read_text())['CPU_seconds'];generation_bound=gen_cpu/max(1,steps)*(2*4*3*32*64)
# All10 checkpoints x4 arms for both worlds already represented in per-world sum.
projected=1.5*(train+encoding+evalcpu*40+generation_bound+countcpu+30)
peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
output=old['projected_checkpoint_bytes_with_reserve']+cachebytes*1.5+10*1024**2
admitted=projected<=900 and projected<=1800 and output<=268435456 and peak<=1073741824
old.update(status='ADMITTED' if admitted else 'MINI_RESOURCE_NOT_ADMITTED',complete_plan_admitted=admitted,evaluation_generation_count_baseline_profile_still_required=False,full_CPU_projection_with_1_5_reserve=projected,full_wall_projection_assumes_single_process_no_external_contention=True,observed_costs={'encoding_seconds':encoding,'one_full_panel_roster_seconds':evalcpu,'countermodel_seconds':countcpu,'generation_seconds':gen_cpu,'generation_outputs':steps},own_generation_worstcase_CPU=generation_bound,compressed_index_cache_bytes=cachebytes,output_projection_bytes=output,profile_peak_RSS_bytes=peak,countermodels={'fit_models':models,'fit_record_presentations':count_fit_records,'fit_target_presentations':count_fit_targets,'not_neural_training':True},profile_CPU_seconds=time.process_time()-start)
save('RESOURCE_ADMISSION.json',old);save('EVALUATOR_AUDIT.json',{'status':'PASS','new_training_records':0,'profile_target_sites_per_roster':evaltargets,'all_panels_read_only':True,'generation_max_outputs':64,'EOS_selected_by_model':True,'no_forced_stop':True,'no_future_targets_fed':True,'full_model_state_hash_unchanged':True,'efficacy_values_not_used_or_exported':True})
print(json.dumps({'admission':old['status'],'full_CPU_projection':projected,'peak_RSS':peak,'output_projection':output,'finite_diff':errors,'profile_CPU':time.process_time()-start}),flush=True)
