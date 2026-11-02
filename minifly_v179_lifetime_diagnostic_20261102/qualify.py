import util as U
import learner as M
import instrument as I
import evaluate as E
import time,json
T=M.torch;np=M.np

def main():
 parent=U.ROOT/'minifly_v179_continuous_local_life_dev_20261102'
 assert all(U.sha(U.ROOT/p)==h for p,h in json.loads((parent/'SOURCE_LOCK.json').read_text())['files'].items())
 assert json.loads((parent/'FINAL_AUDIT.json').read_text())['status']=='ACCEPTED'
 data=M.cached('W1_A_train');n=int(data['n'][:4].sum());idx=data['idx'][:n];y=T.from_numpy(data['y'][:n]);net=M.new();opt=M.optimizer(net);st=M.snap(net,opt)
 a,oa=M.restore(st);t=time.process_time()
 for j in range(0,n,256):M.step(a,oa,idx[j:j+256],y[j:j+256])
 plain=time.process_time()-t;final=M.snap(a,oa);reference=M.state_hash(a,oa,final['rng'])
 b,ob=M.restore(st);t=time.process_time()
 for j in range(0,n,256):I.measured_step(b,ob,idx[j:j+256],y[j:j+256],dict(technical=True),'TECHNICAL_TRACE.jsonl')
 measured=time.process_time()-t;assert reference==M.state_hash(b,ob,T.get_rng_state())
 before=M.state_hash(b,ob,T.get_rng_state());z=dict(idx=idx,y=y.numpy(),n=data['n'][:4]);seen=np.isin(np.arange(257),np.unique(data['y']));t=time.process_time();s=E.panel(b,z,'technical',seen,seen);panel=time.process_time()-t;assert before==M.state_hash(b,ob,T.get_rng_state())
 assert s['max_probability_normalization_error']<1e-9 and s['targets']==n
 counts=dict(records=8,targets=2*n,plain_reference_records=4,instrumented_records=4,read_only_panel_targets=n)
 U.save('TECHNICAL_COUNTS.json',counts)
 # Count only newly owned output; 20 extra full states, parent-identical endpoints referenced.
 param_bytes=sum(p.numel()*p.element_size() for p in b.parameters());new_checkpoint_bytes=20*3*param_bytes;metric_upper=4000000*7*4;storage=1.5*(new_checkpoint_bytes+metric_upper+30000000)
 forecast=1.5*(measured/n*463277+panel/n*4000000+600+time.process_time());assert storage<2147483648 and forecast<7200
 U.save('RESOURCE_ADMISSION.json',dict(status='PASS',training_CPU_per_target=measured/n,evaluation_CPU_per_target=panel/n,forecast_CPU_seconds=forecast,forecast_new_output_bytes_with_reserve=storage,reserve=1.5,existing_assets_not_new_output=True,checkpoint_dedup_requires_exact_state_hash=True,plain_technical_CPU=plain,instrumented_technical_CPU=measured))
 U.save('QUALIFICATION.json',dict(status='PASS',exact_parameters_moments_step_RNG_parity=True,read_only_panel_state_RNG_unchanged=True,full257_normalization=True,technical_counts=counts,parent_source_lock_unchanged=True,instrumentation_uses_no_rng=True,CPU=time.process_time()))
 U.finish_process('instrumentation_qualification')
 files=list(U.H.glob('*.py'))+[U.H/n for n in ['DESIGN.md','AUTHORIZATION.json','PARENT_RESULTS.json','DATA_AUDIT.json','EXPORT.json','QUALIFICATION.json','RESOURCE_ADMISSION.json']]
 hashes={str(p.relative_to(U.ROOT)):U.sha(p) for p in files};hashes.update(json.loads((parent/'SOURCE_LOCK.json').read_text())['files'])
 U.save('SOURCE_LOCK.json',dict(epoch=time.time(),before_any_scored_replay_training=True,files=hashes))
 U.save('STATUS.json',dict(status='QUALIFIED_FOR_INSTRUMENTED_REPLAY',main_training_started=False));print(json.dumps(U.load('RESOURCE_ADMISSION.json')),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:U.save('QUALIFICATION_FAILURE.json',dict(error=repr(e),CPU=time.process_time()));raise
