"""Candidate speed tests only. Does not modify an active E14 process or locked source."""
import os
os.environ['EXP45_HIDDEN']='512';os.environ['EXP45_WINDOW']='18'
import time,json,cProfile,pstats,io,copy
from pathlib import Path
from random import Random
import exp8_life as E
import e14_fast_inference as F
X,T=E.X,E.T
R=Path(__file__).resolve().parent;O=R/'exp45';old=X.respond

def equal(a,b):
 if T.is_tensor(a):return T.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b

def main():
 m=X.new_net(73129);m.load_state_dict(T.load(O/'e8_dev_73129.pt',weights_only=True));store=X.Store(73129)
 pools=[[E.render(task,a,b) for a,b in E.split(task,.25,73129)[0]] for task in E.LIFE]
 for pool in pools:
  x,y,mask=X.batch_tensor(pool[:256]);store.write(x,y,mask)
 prompts=[p for pool in pools for p,_ in pool[:128]]
 # Include different lengths, BOS padding and partially generated prompts.
 prompts += [p[:n] for pool in pools[:2] for p,_ in pool[:16] for n in (1,3,5)]
 result={};maxdiff=0.
 for count in (1,32,64,256):
  p=T.tensor([[X.BOS,*q] for q in prompts[:count]])
  a=m(p)[:,-1];b,_=F.last_logits(m,p);maxdiff=max(maxdiff,float((a-b).abs().max()));assert T.equal(a.argmax(1),b.argmax(1))
 for label,st in [('none',None),('episodic',store)]:
  start=time.perf_counter();a=old(m,prompts,st);told=time.perf_counter()-start
  start=time.perf_counter();b=F.respond(m,prompts,st);tnew=time.perf_counter()-start
  assert a==b,label;result[label]=dict(prompts=len(prompts),old_seconds=told,new_seconds=tnew,speedup=told/tnew,generated_bytes_exact=True)
 # A matched replay/training history: optimizer, weights, RNG and store must match exactly.
 batches=Random(17);current=[batches.sample(pools[1],32) for _ in range(128)];replay=[batches.sample([p for p,_ in pools[0][:256]],32) for _ in range(128)]
 histories=[]
 for fn in (old,F.respond):
  net=copy.deepcopy(m);mem=copy.deepcopy(store);opt=T.optim.AdamW(net.parameters(),lr=.001,weight_decay=1.,betas=(.9,.98));losses=[];replies=[];X.respond=fn
  prof=cProfile.Profile();start=time.perf_counter();prof.enable()
  for items,rp in zip(current,replay):
   outs=fn(net,rp,mem);ans=[o if len(o)==3 and o[-1]==X.EOS else [48,48,X.EOS] for o in outs];replies.append(outs);losses.append(X.update(net,opt,items+list(zip(rp,ans)),True,mem))
  prof.disable();elapsed=time.perf_counter()-start
  stream=io.StringIO();pstats.Stats(prof,stream=stream).sort_stats('cumtime').print_stats(12)
  histories.append(dict(weights=copy.deepcopy(net.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),store=copy.deepcopy(mem.mem),order=mem.order,losses=losses,replies=replies,rng=T.get_rng_state().clone(),seconds=elapsed,profile=stream.getvalue()))
 X.respond=old
 a,b=histories;assert all(equal(a[k],b[k]) for k in ('weights','optimizer','store','order','losses','replies','rng'))
 result['matched_128_update_history']=dict(exact_weights_optimizer_rng_store_losses_replies=True,old_seconds=a['seconds'],new_seconds=b['seconds'],speedup=a['seconds']/b['seconds'],old_profile=a['profile'],new_profile=b['profile'])
 result['maximum_logit_difference']=maxdiff;result['status']='MICROBENCHMARK_AND_128_UPDATE_PARITY_PASS_NOT_YET_FULL_LIFE_QUALIFIED';result['active_campaign_changed']=False
 (R/'E14_SPEED_BENCHMARK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='matched_128_update_history'},indent=2));print('Replay/training speedup',result['matched_128_update_history']['speedup'])
if __name__=='__main__':main()
