"""Frozen full-vocabulary scoring and own-generation; no learner feedback."""
import util as U
import learner as M
import pickle,sys,time
T=M.torch;np=M.np
NATIVE=None

def native():
 global NATIVE
 if NATIVE is None:
  sys.path.append('/Users/pencilbard/Downloads/second_try_copy/V88/.venv-rce-v87-followup/lib/python3.11/site-packages')
  sys.path.insert(0,str(U.ROOT/'minifly_v176_feature_deployment_dev_20261101'));import native_stage as N
  NATIVE=(N,N.Model())
 return NATIVE

def panel(net,z,name):
 net.eval();out=[]
 with T.no_grad():
  for i in range(0,len(z['y']),512):out.append(T.log_softmax(net(z['idx'][i:i+512]).double(),1).numpy()/np.log(2))
 lp=np.concatenate(out);y=z['y'];bits=-lp[np.arange(len(y)),y];pred=lp.argmax(1);ends=np.cumsum(np.r_[0,z['n']]);lines=np.array([bits[ends[i]:ends[i+1]].sum() for i in range(len(z['n']))]);eos=y==256
 path=U.H/'local/panels'/f'{name}.npz';path.parent.mkdir(exist_ok=True);np.savez_compressed(path,log2p=lp.astype(np.float32),y=y,n=z['n'])
 return dict(bits=float(bits.mean()),accuracy=float(np.mean(pred==y)),targets=len(y),records=len(z['n']),record_bit_sums=lines.tolist(),record_targets=z['n'].tolist(),eos_bits=float(bits[eos].mean()),payload_bits=float(bits[~eos].mean()),eos_top1=float(np.mean(pred[eos]==256)),panel=str(path.relative_to(U.H)),sha256=U.sha(path))

def generation(net,recs,name):
 N,m=native();fe=[];times=[];expected=[];generated=[[] for _ in recs];active=list(range(len(recs)))
 for r in recs:
  f=m._new_front();t=m.time+2*N.F.DELTA;f.advance(t)
  for b in r[:3]:t+=N.F.DELTA;f.feed(b,t)
  fe.append(f);times.append(t);expected.append([*r[3:],256])
 before=M.state_hash(net);rng=T.get_rng_state().clone();net.eval()
 with T.no_grad():
  for step in range(32):
   if not active:break
   rows=[]
   for j in active:
    f=fe[j].clone();f.advance(times[j]+N.F.DELTA);x=np.flatnonzero(N.F.FastMemory.encode(m,f.read())[0]);row=np.full(185,5178,np.int64);row[:len(x) if len(x) else 1]=x if len(x) else 5177;rows.append(row)
   ys=net(T.from_numpy(np.array(rows))).argmax(1).tolist();next_active=[]
   for j,y in zip(active,ys):
    generated[j].append(y)
    if y!=256:times[j]+=N.F.DELTA;fe[j].feed(y,times[j]);next_active.append(j)
   active=next_active
 assert M.state_hash(net)==before and T.equal(T.get_rng_state(),rng)
 exact=sum(g==e for g,e in zip(generated,expected));stopped=sum(bool(g) and g[-1]==256 for g in generated)
 path=U.H/'local/panels'/f'{name}_generation.json';path.write_text(__import__('json').dumps(dict(generated=generated,expected=expected)))
 return dict(records=len(recs),complete_exact=exact,complete_exact_rate=exact/len(recs),own_EOS=stopped,truncated_without_EOS=len(active),max_generated=32,prompt_observed_bytes=3,output_targets=sum(map(len,generated)),raw=str(path.relative_to(U.H)),sha256=U.sha(path),no_learning_state_change=True)

def checkpoint(net,opt,world,label,data,records,results,save_state=True):
 t=time.process_time();snap=M.snap(net,opt);state=M.state_hash(net,opt,snap['rng']);sub=np.load(U.H/f'local/{world}_subsets.npz');scores={}
 for split in ('A_held','B_held','EDITED_held'):scores[split]=panel(net,data[split],world+'_'+label+'_'+split)
 orig=np.load(U.H/scores['A_held']['panel']);edit=np.load(U.H/scores['EDITED_held']['panel']);ch=sub['changed'];stable=sub['stable'];old=sub['old'];y=edit['y'];lp=edit['log2p'];ob=-orig['log2p'][np.arange(len(old)),old]
 newloss=-lp[np.arange(len(y)),y];marg=lp[np.arange(len(y)),y]-lp[np.arange(len(y)),old]
 subsets=dict(A_stable_bits=float(ob[stable].mean()),A_stable_positions=int(stable.sum()),changed_new_bits=float(newloss[ch].mean()),changed_margin=float(marg[ch].mean()),changed_accuracy=float(np.mean(lp.argmax(1)[ch]==y[ch])),changed_positions=int(ch.sum()))
 # Per-record signed contrasts are reduced from saved full distributions later.
 gen={k:generation(net,records[k],world+'_'+label+'_'+k) for k in ('A_held','B_held')}
 assert M.state_hash(net,opt,T.get_rng_state())==state
 # Restore training/eval mode is irrelevant to persistent state; RNG remains bit-identical.
 path=U.H/f'local/{world}_{label}.pt'
 if save_state:T.save(snap,path)
 entry=dict(world=world,checkpoint=label,state_hash=state,steps=max((float(v['step']) for v in opt.state.values()),default=0),scores=scores,subsets=subsets,generation=gen,checkpoint_file=str(path.relative_to(U.H)) if save_state else None,checkpoint_sha256=U.sha(path) if save_state else None,CPU=time.process_time()-t)
 results.append(entry);U.save('RESULTS.json',dict(status='RUNNING',checkpoints=results));U.budget();return snap
