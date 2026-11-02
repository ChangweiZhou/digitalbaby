"""Three bounded cycles: source math, matched controls, continuing technical life."""
import util as U
import learner as M
import gate as G
import evaluator as E
import pickle,time,copy,gc,json,gzip,io
T=M.torch;np=M.np
COUNTS=dict(records=0,targets=0,optimizer_steps=0,gradient_only_targets=0);LOG=[]

def fixture(name):
 x=pickle.loads((U.H/'local/fixtures.pkl').read_bytes())[name];return dict(x,idx=T.from_numpy(x['idx'].astype(np.int64)))
def charge(f):
 COUNTS['records']+=len(f['record_n']);COUNTS['targets']+=len(f['y']);COUNTS['optimizer_steps']+=1;U.save('COUNTS.json',COUNTS)
 if COUNTS['records']>64 or COUNTS['targets']>8256:raise RuntimeError('TECHNICAL_DOSE_CAP')
 U.budget()
def step(net,opt,f,arm):
 info=G.step(net,opt,f['idx'],T.from_numpy(f['y']),f['ids'],arm);charge(f);LOG.append(G.public(info));U.save('UPDATE_LOG.json',LOG);return info

def main():
 global COUNTS,LOG
 assert U.load('DATA_AUDIT.json')['status']=='PASS_PENDING_SENSORY_MASKS';assert U.load('SENSORY_AUDIT.json')['status']=='PASS';A=fixture('A');B=fixture('B');edit=fixture('EDIT');no=fixture('NOEDIT')
 initial=U.load('CYCLE1_INITIAL_CHECKS.json');COUNTS=dict(records=8,targets=512,optimizer_steps=2,gradient_only_targets=initial['gradient_only_targets']);LOG=U.load('UPDATE_LOG.json')
 # Planned technical revision: a partial real-batch prefix creates unequal group target counts.
 # It is a gradient-only numerical fixture, not a repeated optimizer trajectory.
 net=M.new();net.train();net.zero_grad();inf=G.gradients(net,A['idx'][:-1],T.from_numpy(A['y'][:-1]),A['ids'][:-1])[1];COUNTS['gradient_only_targets']+=255;checks=initial['checks'];checks['reliable_uneven_disjoint_groups']=inf['reliable'] and inf['group_targets'][0]!=inf['group_targets'][1] and sum(inf['group_targets'])==255;assert all(checks.values()),checks
 U.save('CYCLE1.json',dict(status='PASS_AFTER_FIXTURE_REVISION',checks=checks,initial_balanced_fixture_history='history/cycle1_balanced_fixture',uneven_gradient_fixture=inf,no_optimizer_trajectory_repeated=True,counts=COUNTS.copy()));U.save('COUNTS.json',COUNTS);del net;gc.collect()
 # Four independently born paired technical arms; all do the same microgradient work.
 U.save('STATUS.json',dict(status='CYCLE2_MATCHED_CONTROL_QUALIFICATION',science_training_started=False));arms=('NORMAL','FIXED-LOW','RMS-SHAM','CGLP');results=[];first={};first_opt={}
 for arm in arms:
  net=M.new();opt=M.optimizer(net);T.manual_seed(180);inf=step(net,opt,A,arm);first[arm]=inf;first_opt[arm]=copy.deepcopy(opt.state_dict());T.save(M.snap(net,opt),U.H/f'local/{arm}_A.pt');results.append(dict(arm=arm,phase='A',state_hash=M.state_hash(net,opt,T.get_rng_state()),score=E.score(net,A),step=G.public(inf)))
 template=first_opt['NORMAL'];moment_equal=True
 for arm in arms:
  moment_equal &= all(T.equal(v,first_opt[arm]['state'][k][n]) for k,d in template['state'].items() for n,v in d.items())
 energy_checks={k:dict(selective=first['CGLP']['applied_squared_norm'][k],sham=first['RMS-SHAM']['applied_squared_norm'][k]) for k in ('H','T')};assert moment_equal;assert all(abs(d['selective']-d['sham'])<=1e-6*max(1.,d['selective']) for d in energy_checks.values())
 U.save('CYCLE2.json',dict(status='PASS',identical_full_moments_all_arms=moment_equal,same_state_sham_energy=energy_checks,first_steps={k:G.public(v) for k,v in first.items()},counts=COUNTS.copy()));del first_opt,template;gc.collect()
 # Continue B then same-state EDIT/NOEDIT; no phase flag enters G.step.
 U.save('STATUS.json',dict(status='CYCLE3_TECHNICAL_CONTINUOUS_LIFE',science_training_started=False));forks=[];sample_checkpoint=None
 for arm in arms:
  st=T.load(U.H/f'local/{arm}_A.pt',weights_only=False);net,opt=M.restore(st);a_hash=M.state_hash(net,opt,T.get_rng_state());assert a_hash==next(r['state_hash'] for r in results if r['arm']==arm and r['phase']=='A');inf=step(net,opt,B,arm);bst=M.snap(net,opt);T.save(bst,U.H/f'local/{arm}_B.pt');sample_checkpoint=U.H/f'local/{arm}_B.pt';results.append(dict(arm=arm,phase='B',state_hash=M.state_hash(net,opt,T.get_rng_state()),score=E.score(net,B),step=G.public(inf)))
  hashes=[]
  for phase,fx in [('EDIT',edit),('NOEDIT',no)]:
   net,opt=M.restore(bst);hashes.append(M.state_hash(net,opt,T.get_rng_state()));inf=step(net,opt,fx,arm);same=np.all(edit['idx'].numpy()==no['idx'].numpy(),axis=1);score=E.score(net,edit,old_y=no['y'],same_code=same);T.save(M.snap(net,opt),U.H/f'local/{arm}_{phase}.pt');results.append(dict(arm=arm,phase=phase,state_hash=M.state_hash(net,opt,T.get_rng_state()),score=score,step=G.public(inf)))
  assert hashes[0]==hashes[1];forks.append(dict(arm=arm,EDIT_origin=hashes[0],NOEDIT_origin=hashes[1],identical=True));del net,opt,st,bst;gc.collect()
 assert COUNTS['records']==64 and COUNTS['targets']==3840
 # Autonomous generation is a read-only interface check, never arbitrary-prose efficacy.
 gen_net,gen_opt=M.restore(T.load(U.H/'local/CGLP_EDIT.pt',weights_only=False));original_rng=T.get_rng_state().clone();original_state=M.state_hash(gen_net,gen_opt,original_rng);generation=E.free_run(gen_net,[bytes(range(31)),bytes(range(31,78))]);assert original_state==M.state_hash(gen_net,gen_opt,T.get_rng_state())
 U.save('CYCLE3.json',dict(status='PASS',forks=forks,all_arms_fixed_full_moment_age=True,source_gate_no_phase_input=True,unreliable_C_batches_use_registered_floor=True,own_generation=generation,evaluator_same_KC_altered_KC_changed_separate=True,counts=COUNTS.copy(),no_main_or_guard_model_scores=True));U.save('TECHNICAL_RESULTS.json',dict(status='COMPLETE',records=results,not_science=True));U.save('COUNTS.json',COUNTS)
 # Measure a real full mutable checkpoint; forecast mandatory phases with honest storage reserve.
 raw=sample_checkpoint.read_bytes();gz=gzip.compress(raw,compresslevel=1);params=sum(p.numel()*p.element_size() for p in gen_net.parameters());minimum_phase_checkpoints=4*4*4;uncompressed_forecast=minimum_phase_checkpoints*(3*params+4096)*1.5
 full_bytes=sum(v['bytes'] for v in U.load('DATA_AUDIT.json')['assets'].values());U.save('RESOURCE_ADMISSION.json',dict(status='RESOURCE_NOT_ADMITTED' if uncompressed_forecast>2147483648 else 'PASS_PENDING_FULL_CPU',phase_checkpoints_minimum=minimum_phase_checkpoints,phase_checkpoint_scope='A3,B3,C_EDIT2,C_NOEDIT2 for all16 arm-world lives; after-pass state copies would add more',parameter_bytes=params,one_full_checkpoint_bytes=len(raw),one_technical_checkpoint_gzip1_bytes=len(gz),conservative_checkpoint_output_with_reserve=uncompressed_forecast,new_output_cap=2147483648,storage_reserve=1.5,corpus_bytes=full_bytes,checkpoint_compression_not_yet_proven_safe_at_main_scale=True,CPU_full_plan_not_yet_admitted=True,main_training_started=False))
 U.finish_process('three_cycle_technical_exercise');print('TECHNICAL_THREE_CYCLES_PASS',COUNTS,uncompressed_forecast,flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:U.save('TECHNICAL_FAILURE.json',dict(error=repr(e),counts=COUNTS,CPU=time.process_time()));raise
