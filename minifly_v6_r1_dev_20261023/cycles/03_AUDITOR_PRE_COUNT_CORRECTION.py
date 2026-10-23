"""Artifact-only independent reducer: no learner imports or training calls."""
import json,hashlib,collections,math,time
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent

def read(p):return json.loads(p.read_text())
def save(name,x):(P/name).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def orbit(p):
 a,b=bytes(p)[:-1].split(b'+');return tuple(sorted((int(a),int(b))))
def cm(exposure,probes):
 raw={};can={};freq=collections.Counter();length=collections.defaultdict(collections.Counter);suf=[collections.defaultdict(collections.Counter) for _ in range(5)]
 for p,a in exposure:
  p,a=bytes(p),bytes(a);raw[p]=a;can[orbit(p)]=a;freq[a]+=1;length[len(p)][a]+=1
  for k in range(5):suf[k][p[-k:] if k else b''][a]+=1
 def top(x):return min(x,key=lambda a:(-x[a],a)) if x else None
 out={k:0 for k in ['raw_lookup','canonical_lookup','frequency','prompt_length_frequency',*[f'suffix{k}' for k in range(5)]]}
 for p,a in probes:
  p,a=bytes(p),bytes(a);out['raw_lookup']+=raw.get(p)==a;out['canonical_lookup']+=can.get(orbit(p))==a;out['frequency']+=top(freq)==a;out['prompt_length_frequency']+=top(length[len(p)])==a
  for k in range(5):out[f'suffix{k}']+=top(suf[k][p[-k:] if k else b''])==a
 return {k:v/len(probes) for k,v in out.items()}
def stream(d,phase,arm):
 out=[]
 for q in d[phase+'_order']:
  r=d['shuffled' if arm=='SH_OLD' and phase=='old' else phase][q['index']];p=r['prompt']
  if q['reverse']:
   a,b=bytes(p)[:-1].split(b'+');p=list(b+b'+'+a+b'=')
  out.append((p,r['answer']))
 return out

def run():
 session=read(P/'results/SESSION_RESULT.json');locks=read(P/'SOURCE_LOCK.json');bad=[k for k,v in locks.items() if hashlib.sha256((R/k).read_bytes()).hexdigest()!=v];assert not bad,bad
 worlds={int(p.stem):read(p) for p in (P/'manifests').glob('*.json')};expectedevents=0;checked=0;rowschecked=0;metrics=[];stats=[]
 for d in worlds.values():
  sets=[{orbit(r['prompt']) for r in d[k]} for k in ('held','old','new')];assert [len(s) for s in sets]==[64,128,32] and not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
  for pool in ('old','new','held'):
   for r in d[pool]:assert bytes(r['answer'])==str(sum(orbit(r['prompt']))).encode()
  assert collections.Counter(tuple(r['answer']) for r in d['old'])==collections.Counter(tuple(r['answer']) for r in d['shuffled'])
  expectedevents+=4*sum(len(p)+len(a)+1 for p,a in stream(d,'old','LEARN')+stream(d,'new','LEARN'))
 for file in sorted((P/'results/probes').glob('*.json')):
  v=read(file);d=worlds[v['seed']];rs=v['rows'];seen=set()
  for r in rs:
   ident=(r['pool'],r['index'],r['orientation'],r['tape']);assert ident not in seen;seen.add(ident)
   assert 0<=r['tape']<3 and 0<=r['orientation']<2
   ref=d[r['pool']][r['index']];assert orbit(r['prompt'])==orbit(ref['prompt']) and r['target']==ref['answer']
   margins=r['margins'];assert all(len(m)==9 and all(math.isfinite(x) for x in m) for m in margins)
   emitted=[];eos=False;trunc=False
   for j,m in enumerate(margins):
    if m[8]>0:eos=True;assert j==len(margins)-1;break
    if j==16:trunc=True;assert j==len(margins)-1;break
    emitted.append(sum(1<<i for i,x in enumerate(m[:8]) if x>0))
   assert emitted==r['output'] and eos==r['EOS'] and trunc==r['truncated'];assert r['exact']==(eos and emitted==r['target']);assert r['payload_exact']==(emitted==r['target'])
   assert not(eos and trunc);assert len(emitted)<=16
   rowschecked+=1
  # Refit declared shortcuts independently from only the prefix actually exposed.
  old=stream(d,'old',v['arm']);new=stream(d,'new',v['arm']);n=v['records'];exposure=old[:min(n,4096)]+new[:max(0,n-4096)]
  assert len(exposure)==n
  for pool in sorted({r['pool'] for r in rs}):
   rr=[r for r in rs if r['pool']==pool];assert len(rr)%6==0
   for i in {r['index'] for r in rr}:assert sum(r['index']==i for r in rr)==6
   correct=sum(r['exact'] for r in rr);charhit=sum(sum(a==b for a,b in zip(r['output'],r['target'])) for r in rr);charn=sum(max(len(r['output']),len(r['target'])) for r in rr)
   models=cm(exposure,[(r['prompt'],r['target']) for r in rr]);assert pool!='held' or models['raw_lookup']==models['canonical_lookup']==0
   metrics.append(dict(seed=v['seed'],arm=v['arm'],stage=v['stage'],records=n,pool=pool,queries=len(rr),correct=correct,exact=correct/len(rr),payload_exact=sum(r['payload_exact'] for r in rr)/len(rr),EOS=sum(r['EOS'] for r in rr)/len(rr),truncated=sum(r['truncated'] for r in rr)/len(rr),empty=sum(not r['output'] for r in rr)/len(rr),own_prefix_byte_accuracy=charhit/max(1,charn),CPU_cumulative_seconds=v['CPU_seconds'],countermodels=models))
   checked+=1
 for file in sorted((P/'results/histories').glob('*.json')):
  v=read(file);d=worlds[v['seed']];assert v['records']==4352;assert v['observed_events']==sum(len(p)+len(a)+1 for p,a in stream(d,'old',v['arm'])+stream(d,'new',v['arm']));assert v['summary'][-1]['native']['native_events']==v['observed_events']
  if v['arm']=='FROZEN':assert all(s['controller_weight_delta']==0 for s in v['summary'])
  if v['arm']=='NO_OLD':
   s=[s for s in v['summary'] if s['stage']=='REST1'][0];assert s['controller_weight_delta']==s['native']['write_l1']==s['baseline']==s['updates']==0
  stats.append(v)
 assert len({(h['seed'],h['arm']) for h in stats})==len(stats)
 assert all((h['seed'],h['arm']) in {(w,a) for w in worlds for a in ('LEARN','FROZEN','NO_OLD','SH_OLD')} for h in stats)
 for h in stats:
  pr=[r for r in metrics if r['seed']==h['seed'] and r['arm']==h['arm']]
  assert {r['stage'] for r in pr}=={'BASELINE','OLD_128','OLD_512','OLD_1024','OLD_2048','OLD_END','REST1','NEW_64','NEW_END','FINAL'}
  assert sum(r['queries'] for r in pr)==6000
 complete=len(stats)==8 and session['main_full_records']==34816 and session['main_observed_native_events']==expectedevents
 if session['status']=='MAIN_COMPLETE_AWAITING_AUDIT':assert complete
 gate=[]
 def cell(seed,arm,stage,pool):return next(x for x in metrics if (x['seed'],x['arm'],x['stage'],x['pool'])==(seed,arm,stage,pool))
 if complete:
  for seed in worlds:
   L=cell(seed,'LEARN','FINAL','held');N=cell(seed,'NO_OLD','FINAL','held');a=cell(seed,'LEARN','OLD_END','held');b=cell(seed,'NO_OLD','OLD_END','held')
   g=dict(seed=seed,held_beats_all_arms=all(L['exact']>cell(seed,arm,'FINAL','held')['exact'] for arm in ('FROZEN','NO_OLD','SH_OLD')),held_beats_all_countermodels=L['exact']>max(L['countermodels'].values()),OLD_end_causal=a['exact']>b['exact'],final_causal=L['exact']>N['exact'],OLD_retention_guard=cell(seed,'LEARN','FINAL','old')['exact']>cell(seed,'NO_OLD','FINAL','old')['exact'],NEW_acquisition_guard=cell(seed,'LEARN','FINAL','new')['exact']>cell(seed,'LEARN','REST1','new')['exact'])
   g['pass']=all(v for k,v in g.items() if k!='seed');gate.append(g)
 verdict='FAVORABLE_EXPLORATORY_DEVELOPMENT' if complete and all(g['pass'] for g in gate) else 'NO_DEMONSTRATED_UNSEEN_ADDITION_GAIN' if complete else 'INCOMPLETE_BOUNDED_RUN'
 audit=dict(status='PASS',source_files_checked=len(locks),probe_rows_checked=rowschecked,metric_cells_recomputed=checked,completed_histories=len(stats),complete=complete,main_records=session['main_full_records'],main_observed_events=session['main_observed_native_events'],expected_main_events=expectedevents,technical_full_records=4,technical_full_record_events=20,technical_partial_native_events=4,source_hash_mismatches=bad,exposure_overlap=0,all_rows_decoded_from_native_margins=True,no_training_or_revalidation_performed=True)
 summary=dict(verdict=verdict,formal_confirmation=False,new_core_adopted=False,independent_worlds=len(worlds),histories=len(stats),main_records=session['main_full_records'],main_observed_events=session['main_observed_native_events'],gates=gate,metrics=metrics,history_statistics=stats,session=session)
 save('results/METRICS.json',metrics);save('results/SUMMARY.json',summary);save('results/FINAL_AUDIT.json',audit)
if __name__=='__main__':run()
