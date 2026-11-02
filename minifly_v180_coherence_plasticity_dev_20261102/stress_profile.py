"""Synthetic fixed-length execution workload; NOT model-generated trajectories or science."""
import util as U
import learner as M
import execution_profile as P
import time,json
T=M.torch;np=M.np;N=P.N

def main():
 net,opt=M.restore(T.load(U.H/'local/CGLP_EDIT.pt',weights_only=False));net.eval();before=M.state_hash(net,opt,T.get_rng_state());t=time.process_time();m=N.Model();fe=[];ts=[]
 for j in range(128):
  f=m._new_front();tt=m.time+2*N.F.DELTA;f.advance(tt)
  for b in bytes((j%256,(j+1)%256,(j+2)%256)):tt+=N.F.DELTA;f.feed(b,tt)
  fe.append(f);ts.append(tt)
 setup=time.process_time()-t;t=time.process_time()
 with T.no_grad():
  for step in range(128):
   rows=[]
   for j in range(128):
    f=fe[j].clone();f.advance(ts[j]+N.F.DELTA);x=np.flatnonzero(N.F.FastMemory.encode(m,f.read())[0]);row=np.full(185,5178,np.int64);row[:len(x) if len(x) else 1]=x if len(x) else 5177;rows.append(row)
   logits=net(T.from_numpy(np.array(rows)));assert logits.shape==(128,257)
   # Externally supplied synthetic stream for hardware stress ONLY. Predictions are not used as observations or scores.
   for j in range(128):ts[j]+=N.F.DELTA;fe[j].feed((step*37+j)%256,ts[j])
 loop=time.process_time()-t;assert M.state_hash(net,opt,T.get_rng_state())==before
 r=U.load('RESOURCE_ADMISSION.json');U.save('history/early_EOS_amortization/RESOURCE_ADMISSION.json',r)
 panels=r['generation_query_upper']//(128*128);r['CPU_components']['own128_generation']=panels*(setup+loop);r['CPU_full_plan_forecast_with_reserve']=1.5*(sum(r['CPU_components'].values())+sum(v['CPU'] for v in U.load('PROCESS_LEDGER.json')['completed'])+time.process_time());r['CPU_forecast_pass']=r['CPU_full_plan_forecast_with_reserve']<=7200;r['generation_forecast_revision']='Per-panel fixed initialization plus full128-step synthetic execution workload, not first-step EOS total CPU multiplied by128.';r['CPU_full_plan_not_yet_admitted']=not r['CPU_forecast_pass'];r['status']='RESOURCE_NOT_ADMITTED';U.save('RESOURCE_ADMISSION.json',r)
 U.save('STRESS_PROFILE.json',dict(status='PASS_READ_ONLY',records=128,synthetic_observation_steps=128,read_only_classification_sites=16384,model_predictions_not_fed_as_observations=True,not_free_generation_or_science=True,setup_CPU=setup,loop_CPU=loop,source_network_unchanged=True,source_generation_still_stops_on_own_EOS=True,optimizer_updates=0,CPU=time.process_time()));U.finish_process('synthetic_execution_ceiling_profile');print(json.dumps(r,indent=2))
if __name__=='__main__':main()
