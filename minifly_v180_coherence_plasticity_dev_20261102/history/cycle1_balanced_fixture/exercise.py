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
 assert U.load('DATA_AUDIT.json')['status']=='PASS_PENDING_SENSORY_MASKS';assert U.load('SENSORY_AUDIT.json')['status']=='PASS';A=fixture('A');B=fixture('B');edit=fixture('EDIT');no=fixture('NOEDIT');checks={};U.save('STATUS.json',dict(status='CYCLE1_NUMERIC_SOURCE_PARITY',science_training_started=False))
 # Literal gamma=1 parity with nonzero historical Adam state, on new technical bytes only.
 old=U.ROOT/'minifly_v179_continuous_local_life_dev_20261102/local/W1_AFTER_A.pt';st=T.load(old,weights_only=False);a,oa=M.restore(st);M.step(a,oa,A['idx'],T.from_numpy(A['y']));charge(A);expected=M.state_hash(a,oa,T.get_rng_state());refgr={n:p.grad.detach().clone() for n,p in a.named_parameters()};b,ob=M.restore(st);before={n:p.detach().clone() for n,p in b.named_parameters()};mom={n:{k:v.clone() for k,v in ob.state[p].items()} for n,p in b.named_parameters()};info=step(b,ob,A,'NORMAL');checks['gamma1_literal_all_state_RNG_parity']=expected==M.state_hash(b,ob,T.get_rng_state());checks['full_source_gradients_exact']=all(T.equal(refgr[n],p.grad) for n,p in b.named_parameters());checks['reliable_uneven_disjoint_groups']=info['reliable'] and info['group_targets'][0]!=info['group_targets'][1] and sum(info['group_targets'])==256
 # Independent NumPy Adam coordinate reference, including historical moments and decoupled decay.
 analytic_errors={}
 for name,p in b.named_parameters():
  delta=p.detach()-before[name];flat=int(delta.abs().argmax());ix=np.unravel_index(flat,tuple(p.shape));g=float(p.grad[ix]);theta=float(before[name][ix]);m0=float(mom[name]['exp_avg'][ix]);v0=float(mom[name]['exp_avg_sq'][ix]);t=float(mom[name]['step'])+1;mm=.9*m0+.1*g;vv=.999*v0+.001*g*g;u=(mm/(1-.9**t))/(np.sqrt(vv/(1-.999**t))+1e-8)+.0001*theta;gate=.25 if name in ('W','b') or name.startswith('top.first') else 1.;manual=theta-.001*gate*u;measured=theta+gate*float(delta[ix]);analytic_errors[name]=abs(manual-measured)
 checks['independent_Adam_effective_step']=max(analytic_errors.values())<3e-7
 # Verify actual quarter movement and H-column/T-row axes without another optimizer step.
 delta={n:p.detach()-before[n] for n,p in b.named_parameters()};gains={'H':T.full((512,),.25),'T':T.full((2048,),.25)};O=b.top.out.weight.detach().clone();G.apply_gains(b,before,delta,gains);ratios={n:float((p.detach()-before[n]).norm()/delta[n].norm()) for n,p in b.named_parameters() if n in ('W','b') or n.startswith('top.first')};checks['quarter_real_hidden_movement']=all(abs(v-.25)<1e-5 for v in ratios.values());checks['output_never_scaled']=T.equal(O,b.top.out.weight)
 # Independent analytic aligned/opposed/orthogonal/zero neuron-vector fixtures.
 aa={'w':T.tensor([[1.,1.,1.,0.],[0.,0.,0.,0.]],dtype=T.float64),'b':T.zeros(4,dtype=T.float64)};cc={'w':T.tensor([[1.,-1.,0.,0.],[0.,0.,1.,0.]],dtype=T.float64),'b':T.zeros(4,dtype=T.float64)};co=G.coherence(aa,cc,'w','b',0,True);checks['analytic_coherence_cases']=T.allclose(co,T.tensor([1.,0.,0.,0.],dtype=T.float64),atol=2e-12,rtol=0);checks['unreliable_floor']=T.equal(G.coherence(aa,cc,'w','b',0,False),T.zeros(4,dtype=T.float64));checks['reversible_coherence_gain']=float((.25+.75*co)[0])>.99 and float((.25+.75*co)[1])==.25
 # Finite differences of local surrogate, not a false claim of global CE hidden gradients.
 f=M.new().double();f.eval();xx=A['idx'][:8];yy=T.from_numpy(A['y'][:8]);f.zero_grad();e,mask,dt,dh=M.L.local_step(f,M.FH.double(),M.FT.double(),xx,yy);COUNTS['gradient_only_targets']+=8
 h=(M.F.embedding_bag(xx,f.W,mode='sum',padding_idx=M.V.PAD)+f.b).relu().detach();fd={}
 for name in ('W','top.first.weight','top.out.weight'):
  p=dict(f.named_parameters())[name];ix=np.unravel_index(int(p.grad.abs().argmax()),tuple(p.shape));expected_gradient=float(p.grad[ix]);eps=1e-5
  def objective():
   if name=='W':return float(((M.F.embedding_bag(xx,f.W,mode='sum',padding_idx=M.V.PAD)+f.b)*dh.detach()).sum())
   if name=='top.first.weight':return float((f.top.first(h)*dt.detach()).sum())
   return float(M.F.cross_entropy(f.top.out(f.top.first(h).relu()),yy))
  with T.no_grad():
   value=float(p[ix]);p[ix]=value+eps;up=objective();p[ix]=value-eps;down=objective();p[ix]=value
  fd[name]=dict(analytic=expected_gradient,finite_difference=(up-down)/(2*eps));assert abs(expected_gradient-fd[name]['finite_difference'])<=1e-8+1e-6*abs(expected_gradient)
 checks['finite_difference_local_and_output']=True;checks={k:bool(v) for k,v in checks.items()};assert all(checks.values()),checks
 U.save('CYCLE1.json',dict(status='PASS',checks=checks,Adam_coordinate_errors=analytic_errors,quarter_ratios=ratios,finite_differences=fd,split=G.public(info),counts=COUNTS.copy()));del a,oa,b,ob,f,st,refgr,before,mom,delta;gc.collect()
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
