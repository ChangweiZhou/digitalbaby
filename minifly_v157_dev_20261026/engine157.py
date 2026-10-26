"""Fixed scientific roster; actual teaching stays in the unchanged native parent."""
from collections import Counter
import time,json
import numpy as np
import support157 as s
import native157 as n

def train(m,row,kind,write):
 s.record(row,kind,write);t=time.process_time();sites=m.train(bytes.fromhex(row['prompt']),bytes.fromhex(row['answer']),write);s.TIMES[kind+'_CPU']+=time.process_time()-t;s.COUNTS['native_training_distributions']+=len(sites);return sites
def write_trace(path,record):
 with (s.H/path).open('a') as f:f.write(json.dumps(s.clean(record),allow_nan=False)+'\n')
def old(d,prior):
 key=f"{d['world']}_{prior}";m=n.birth();s.ACTIVE=m;s.ACTIVE_INFO=dict(stage='OLD',case=key);path=f'local/{key}_OLD_TRAIN.jsonl';count=0
 for pass_no,order in enumerate(d['OLD_orders'],1):
  for ix in order:
   s.budget();row=d['OLD'][ix];sites=train(m,row,'OLD_updating' if prior=='W' else 'OLD_inhibited',prior=='W');count+=1
   write_trace(path,dict(id=row['id'],pass_no=pass_no,n=m.n,time=m.time,sites=sites,write=prior=='W'))
  s.log('OLD_PASS_COMMIT',case=key,pass_no=pass_no,records=count,n=m.n);s.save('ACTIVE.json',dict(case=key,phase='OLD',next_record=count,**s.state()))
 assert count==512 and m.n>=1024 and not m.q.any()
 panel=n.evaluate(m,d['panel'],key+'_initial');cp=s.checkpoint(m,key+'_OLD.pkl')
 result=dict(world=d['world'],prior=prior,records=512,targets=m.n,n=m.n,time=m.time,checkpoint=cp,panel=panel['receipt'],trace=path,initial_correct_OLD=np.flatnonzero(panel['correct'][:64]).tolist())
 s.save(f'roots/{key}.json',result);s.ACTIVE=None;s.ACTIVE_INFO=None;return result

def branch(d,root,schedule):
 key=f"{d['world']}_{root['prior']}_{schedule}";m=s.restore(root['checkpoint']);s.ACTIVE=m;s.ACTIVE_INFO=dict(stage='NEW',case=key);by={r['id']:r for r in d['NEW']};exposure=Counter();immediate=[];boundaries=[];probes=[];trace=f'local/{key}_NEW_TRAIN.jsonl';n0=m.n;t0=m.time
 for index,rid in enumerate(d['schedules'][schedule],1):
  s.budget();row=by[rid];probe=index in [16,32,64,128]
  if probe:
   idle=m.clone();passive=m.clone();parent=n.pair(m,row);parent_margin=parent['margin'].copy()
  sites=train(m,row,'NEW_updating',True);exposure[rid]+=1
  write_trace(trace,dict(id=rid,index=index,n=m.n,time=m.time,sites=sites,write=True))
  if probe:
   zero_sites=train(passive,row,'NEW_inhibited_probes',False);elapsed=m.time-idle.time;n.n.fastmem.FastMemory.rest(idle,elapsed)
   assert abs(m.time-passive.time)<1e-9 and abs(m.time-idle.time)<1e-9 and m.n==passive.n
   assert idle.n==m.n-len(sites)
   actual=n.evaluate(m,d['panel'],key+f'_at{index}');zero=n.evaluate(passive,d['panel'],key+f'_at{index}_zero');rest=n.evaluate(idle,d['panel'],key+f'_at{index}_idle')
   own_index=next(i for i,r in enumerate(d['panel']) if r['id']==rid);lo,hi=actual['offsets'][own_index:own_index+2]
   r=dict(id=rid,output_hex=bytes([v for v in actual['choices'][own_index] if v>=0 and v!=256]).hex(),EOS=bool(actual['EOS'][own_index]),choices=[int(v) for v in actual['choices'][own_index] if v>=0],correct=bool(actual['correct'][own_index]))
   boundaries.append(dict(index=index,panel=actual['receipt'],exposure=dict(exposure),unique_taught=len(exposure),NEW_targets=m.n-n0,NEW_seconds=m.time-t0,OLD_retained=sum(bool(actual['correct'][i]) for i in root['initial_correct_OLD'])))
   probes.append(dict(index=index,id=rid,elapsed_seconds=elapsed,actual=actual['receipt'],inhibited=zero['receipt'],idle=rest['receipt'],parent_current=n.summary_pair(parent),parent_current_margins=parent_margin,zero_sites=zero_sites,write_mean_margin=float((actual['margin']-zero['margin']).mean()),observed_mean_margin=float((zero['margin']-rest['margin']).mean()),write_complete_losses=int(np.sum(zero['correct']&~actual['correct'])),write_complete_gains=int(np.sum(~zero['correct']&actual['correct'])),own_written_margins=actual['margin'][lo:hi],own_inhibited_margins=zero['margin'][lo:hi],own_idle_margins=rest['margin'][lo:hi]))
   s.COUNTS['immediate_boundary_pair_reuses']+=1
   del passive,idle,zero,rest
  else:r=n.summary_pair(n.pair(m,row))
  immediate.append(dict(index=index,**r));s.COUNTS['immediate_current_scores']+=1
  if index%16==0:s.log('NEW_WINDOW_COMMIT',case=key,records=index,n=m.n,current_correct=sum(r['correct'] for r in immediate[-16:]));s.save('ACTIVE.json',dict(case=key,phase='NEW',next_record=index,**s.state()))
 assert len(immediate)==128 and set(exposure.values())=={8} and len(exposure)==16
 final=s.checkpoint(m,key+'_FINAL.pkl');delay=m.clone();delay.rest(128);delayed=n.evaluate(delay,d['panel'],key+'_delayed');post=s.checkpoint(delay,key+'_DELAYED.pkl')
 result=dict(case=key,world=d['world'],prior=root['prior'],schedule=schedule,parent=root['checkpoint'],initial_panel=root['panel'],initial_correct_OLD=root['initial_correct_OLD'],records=128,NEW_targets=m.n-n0,NEW_seconds=m.time-t0,final_n=m.n,final_time=m.time,final=final,delayed=post,trace=trace,immediate=immediate,boundaries=boundaries,probes=probes,delayed_panel=delayed['receipt'])
 s.save(f'local/{key}_RESULT.json',result);s.log('BRANCH_COMMIT',case=key,final_correct=boundaries[-1]['panel']['correct'],delayed_correct=delayed['receipt']['correct']);s.ACTIVE=None;s.ACTIVE_INFO=None;return result

def run_world(d):
 roots={p:old(d,p) for p in ['W','SH_OLD']};assert roots['W']['n']==roots['SH_OLD']['n'] and roots['W']['time']==roots['SH_OLD']['time'];results=[]
 for p in ['W','SH_OLD']:
  for schedule in ['INTERLEAVED','CLUSTERED']:results.append(branch(d,roots[p],schedule))
 for p in ['W','SH_OLD']:
  aa=[r for r in results if r['prior']==p];assert aa[0]['final_n']==aa[1]['final_n'] and aa[0]['final_time']==aa[1]['final_time'] and aa[0]['NEW_targets']==aa[1]['NEW_targets']
 s.save(f'worlds/{d["world"]}/EXECUTION.json',dict(world=d['world'],roots=roots,branches=[dict(case=r['case'],result=f'local/{r["case"]}_RESULT.json') for r in results]));return results
