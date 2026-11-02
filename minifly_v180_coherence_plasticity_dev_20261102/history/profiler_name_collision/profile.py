"""Read-only execution profile and resource revision; no optimizer updates."""
import util as U
import learner as M
import gate as G
import pickle,time,json,sys,copy
T=M.torch;np=M.np
sys.path.append('/Users/pencilbard/Downloads/second_try_copy/V88/.venv-rce-v87-followup/lib/python3.11/site-packages');sys.path.insert(0,str(U.ROOT/'minifly_v176_feature_deployment_dev_20261101'));import native_stage as N

def generate(net,recs,batch=True):
 m=N.Model();net.eval();fes=[];times=[];out=[[] for r in recs];active=list(range(len(recs)));q=0
 for r in recs:
  f=m._new_front();t=m.time+2*N.F.DELTA;f.advance(t)
  for b in r[:3]:t+=N.F.DELTA;f.feed(b,t)
  fes.append(f);times.append(t)
 with T.no_grad():
  for step in range(128):
   if not active:break
   rows=[]
   for i in active:
    f=fes[i].clone();f.advance(times[i]+N.F.DELTA);x=np.flatnonzero(N.F.FastMemory.encode(m,f.read())[0]);row=np.full(185,5178,np.int64);row[:len(x) if len(x) else 1]=x if len(x) else 5177;rows.append(row)
   idx=T.from_numpy(np.array(rows));ys=net(idx).argmax(1).tolist() if batch else [int(net(idx[i:i+1]).argmax(1)) for i in range(len(idx))];q+=len(ys);nxt=[]
   for i,y in zip(active,ys):
    out[i].append(y)
    if y!=256:times[i]+=N.F.DELTA;fes[i].feed(y,times[i]);nxt.append(i)
   active=nxt
 return out,q

def main():
 net,opt=M.restore(T.load(U.H/'local/CGLP_EDIT.pt',weights_only=False));fx=pickle.loads((U.H/'local/fixtures.pkl').read_bytes())['A'];idx=T.from_numpy(fx['idx'].astype(np.int64));y=T.from_numpy(fx['y']);before=M.state_hash(net,opt,T.get_rng_state());recs=[bytes(range(31)),bytes(range(31,78))]
 literal,n=generate(net,recs,False);batched,n2=generate(net,recs,True);assert literal==batched and n==n2
 t=time.process_time();zz,qq=generate(net,recs*64,True);gen_cpu=time.process_time()-t;assert M.state_hash(net,opt,T.get_rng_state())==before
 net.eval();t=time.process_time()
 with T.no_grad():
  logits=net(idx.repeat((2,1)));lp=T.log_softmax(logits.double(),1);p=lp.exp();loss=-lp[T.arange(512),y.repeat(2)];entropy=-(p*lp).sum(1);brier=(p[:,256]-(y.repeat(2)==256)).square()
 eval_cpu=time.process_time()-t
 # Gradient/optimizer work profiled without applying an update: full source and two microgroups.
 # No trajectories are repeated, and these extra label/gradient queries are explicitly counted.
 prof={}
 for arm in ('NORMAL','FIXED-LOW','RMS-SHAM','CGLP'):
  net.train();opt.zero_grad();t=time.process_time();g,info,_,_=G.gradients(net,idx,y,fx['ids']);gradient_cpu=time.process_time()-t
  # Moment-and-parameter arithmetic on disposable arrays; does not mutate net/optimizer.
  t=time.process_time()
  with T.no_grad():
   for par in net.parameters():
    st=opt.state[par];mm=.9*st['exp_avg']+.1*par.grad;vv=.999*st['exp_avg_sq']+.001*par.grad.square();tt=float(st['step'])+1;direction=(mm/(1-.9**tt))/(vv.sqrt()/np.sqrt(1-.999**tt)+1e-8)+.0001*par;trial=par-.001*.35*direction;assert T.isfinite(trial).all()
  arithmetic_cpu=time.process_time()-t;prof[arm]=dict(gradient_CPU=gradient_cpu,optimizer_arithmetic_CPU=arithmetic_cpu,targets=256,actual_optimizer_updates=0)
 assert M.state_hash(net,opt,T.get_rng_state())!=before # training dropout RNG advanced; no learned/optimizer values changed
 # RNG work is disposable. Restore original RNG to verify no learning mutation.
 # Recorded before RNG is recovered from saved state; only state tensors are compared separately here.
 snapshot=T.load(U.H/'local/CGLP_EDIT.pt',weights_only=False);T.set_rng_state(snapshot['rng']);assert M.state_hash(net,opt,T.get_rng_state())==M.state_hash(*M.restore(snapshot),T.get_rng_state())
 data=pickle.loads((U.H/'local/records.pkl').read_bytes());train_targets=0;eval_targets=0;encoded=0;main_records=34816
 for w,sp in data.items():
  train_targets+=4*(3*sum(len(r)+1 for r in sp['A_train'])+3*sum(len(r)+1 for r in sp['B_train'])+2*sum(len(r)+1 for r in sp['EDIT_train'])+2*sum(len(r)+1 for r in sp['NOEDIT_train']))
  panel=sum(sum(len(r)+1 for r in rr) for rr in sp.values());encoded+=panel;eval_targets+=4*11*panel
 output_queries=4*4*11*2*128*128
 source_rate=U.load('SENSORY_AUDIT.json')['per_target_export_CPU'];per_target=max((r['gradient_CPU']+r['optimizer_arithmetic_CPU'])/256 for r in prof.values())
 source_cpu=encoded*source_rate;training_cpu=train_targets*per_target;scoring_cpu=eval_targets*eval_cpu/512;generation_cpu=output_queries*gen_cpu/qq;other=600;forecast=1.5*(source_cpu+training_cpu+scoring_cpu+generation_cpu+other+sum(x['CPU'] for x in U.load('PROCESS_LEDGER.json')['completed'])+time.process_time())
 admission=U.load('RESOURCE_ADMISSION.json');admission.update(CPU_full_plan_forecast_with_reserve=forecast,CPU_cap=7200,CPU_forecast_pass=forecast<=7200,CPU_components=dict(input_encoding=source_cpu,training=training_cpu,read_only_panels=scoring_cpu,own128_generation=generation_cpu,other_reserve=other),actual_main_training_targets=train_targets,estimated_all_scoring_targets=eval_targets,generation_query_upper=output_queries,profile_scope='four arm source/microgradient + disposable moment arrays, no optimizer updates',full_fit_bench_not_run=True)
 # Include uncompressed per-site metrics/export/trace allowance in a concrete prospective output amendment.
 admission['proposed_owned_output_cap_bytes']=8*1024**3;admission['revision_reason']='At least64 unique full phase states alone need4.87GB with1.5reserve before panels/exports. No trained checkpoint is omitted or quantized.';admission['status']='RESOURCE_NOT_ADMITTED';U.save('RESOURCE_ADMISSION.json',admission)
 U.save('EXECUTION_PROFILE.json',dict(status='READ_ONLY_PROFILE_COMPLETE',four_arm_profiling=prof,evaluator_CPU_for512=eval_cpu,batched_generation_records=128,batched_generation_outputs=qq,batched_generation_CPU=gen_cpu,literal_batched_output_exact_on_technical_fixture=True,extra_gradient_targets=1024,extra_readonly_prediction_targets=512+n+n2+qq,optimizer_updates=0,model_state_unchanged=True,CPU=time.process_time()))
 U.finish_process('read_only_whole_plan_resource_profile');print(json.dumps(admission,indent=2))
if __name__=='__main__':main()
