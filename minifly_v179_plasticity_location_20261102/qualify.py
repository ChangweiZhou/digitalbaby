"""No efficacy selection: source parity, exact freeze and restored plasticity."""
import util as U
import learner as M
import plasticity as P
import instrument as I
import evaluate as E
import json,time
T=M.torch;np=M.np

def main():
 parent=U.ROOT/'minifly_v179_lifetime_diagnostic_20261102';src=U.load('PARENT_RESULTS.json');assert all(U.sha(U.ROOT/p)==h for p,h in json.loads((parent/'SOURCE_LOCK.json').read_text())['files'].items())
 assert json.loads((parent/'FINAL_AUDIT.json').read_text())['status']=='ACCEPTED'
 arow=next(r for r in src['checkpoints'] if r['world']=='W1' and r['checkpoint']=='AFTER_A');assert U.sha(U.H/'local/W1_A_START.pt')==arow['checkpoint_sha256'];initial=T.load(U.H/'local/W1_A_START.pt',weights_only=False)
 data=M.cached('W1_B_train');n=int(data['n'][:2].sum());idx=data['idx'][:n];y=T.from_numpy(data['y'][:n]);a,oa=M.restore(initial)
 for j in range(0,n,256):M.step(a,oa,idx[j:j+256],y[j:j+256])
 expected=M.state_hash(a,oa,T.get_rng_state());b,ob=M.restore(initial);t=time.process_time()
 for j in range(0,n,256):I.measured_step(b,ob,idx[j:j+256],y[j:j+256],dict(technical=True,arm='NORMAL'),'TECHNICAL_TRACE.jsonl',False)
 normal_cpu=time.process_time()-t;checks=dict(normal_equals_literal_source=expected==M.state_hash(b,ob,T.get_rng_state()))
 c,oc=M.restore(initial);hidden_before=P.hidden_digest(c,oc);out_before=c.top.out.weight.detach().clone();t=time.process_time()
 for j in range(0,n,256):I.measured_step(c,oc,idx[j:j+256],y[j:j+256],dict(technical=True,arm='CLAMP'),'TECHNICAL_TRACE.jsonl',True)
 clamp_cpu=time.process_time()-t;checks['hidden_parameters_decay_moments_steps_unchanged']=hidden_before==P.hidden_digest(c,oc);checks['own_output_learns']=not T.equal(out_before,c.top.out.weight)
 checks['first_batch_output_delta_matches_normal']=T.equal(b.top.out.weight,c.top.out.weight) if n<=256 else True
 assert all(p.grad is None for _,p in P.hidden_named(c));saved=M.snap(c,oc);T.save(saved,U.H/'local/technical_clamp.pt');c,oc=M.restore(T.load(U.H/'local/technical_clamp.pt',weights_only=False));checks['checkpoint_roundtrip']=P.hidden_digest(c,oc)==hidden_before
 cd=M.cached('W1_EDITED_train');cn=int(cd['n'][:2].sum());cx=cd['idx'][:cn];cy=T.from_numpy(cd['y'][:cn]);step_before={name:float(oc.state[p]['step']) for name,p in P.hidden_named(c)}
 for j in range(0,cn,256):I.measured_step(c,oc,cx[j:j+256],cy[j:j+256],dict(technical=True,arm='RESUMED_C'),'TECHNICAL_TRACE.jsonl',False)
 checks['hidden_plasticity_resumes']=P.hidden_digest(c,oc)!=hidden_before and all(float(oc.state[p]['step'])==step_before[name]+int(np.ceil(cn/256)) for name,p in P.hidden_named(c))
 before=M.state_hash(c,oc,T.get_rng_state());seen=np.isin(np.arange(257),np.unique(data['y']));t=time.process_time();s=E.panel(c,dict(idx=idx,y=y.numpy(),n=data['n'][:2]),'technical',seen,seen);panel_cpu=time.process_time()-t;checks['read_only_panel_state_RNG_unchanged']=M.state_hash(c,oc,T.get_rng_state())==before;checks['full257_normalized']=s['max_probability_normalization_error']<1e-9
 checks={k:bool(v) for k,v in checks.items()};assert all(checks.values()),checks
 counts=dict(records=8,targets=3*n+cn,source_records=2,normal_records=2,clamp_records=2,reactivated_records=2,read_only_panel_targets=n);U.save('TECHNICAL_COUNTS.json',counts)
 expected_records=expected_targets=0
 for w in ('W1','W2'):
  z=M.cached(w+'_B_train');expected_records+=len(z['n'])*3*2;expected_targets+=len(z['y'])*3*2
  for name in ('EDITED_train','NOEDIT_train'):
   z=M.cached(w+'_'+name);expected_records+=len(z['n'])*2*2;expected_targets+=len(z['y'])*2*2
 assert expected_records==2980 and expected_targets==187606 and counts['targets']+expected_targets<200000
 size=sum(p.numel()*p.element_size() for p in c.parameters());storage=1.5*(14*3*size+4000000*7*4+30000000);forecast=1.5*(max(normal_cpu,clamp_cpu)/n*expected_targets+panel_cpu/n*4000000+600+time.process_time());assert storage<2147483648 and forecast<3600
 U.save('RESOURCE_ADMISSION.json',dict(status='PASS',reserve=1.5,forecast_CPU=forecast,forecast_owned_bytes=storage,expected_main_records=expected_records,expected_main_targets=expected_targets,limits=U.load('AUTHORIZATION.json')));U.save('QUALIFICATION.json',dict(status='PASS',checks=checks,technical_counts=counts,parent_source_lock_unchanged=True,CPU=time.process_time()));U.finish_process('technical_parity_freeze_resumption')
 print('QUALIFICATION_PASS',forecast,storage,flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:U.save('QUALIFICATION_FAILURE.json',dict(error=repr(e),CPU=time.process_time()));raise
