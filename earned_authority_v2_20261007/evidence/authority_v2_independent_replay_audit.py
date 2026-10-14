import gzip,hashlib,json,sys
from pathlib import Path
import numpy as np
R=Path('/Users/pencilbard/Downloads/second_try_copy/DIGITALBABY_CLAUDE_B_D4E4578_20260929')
D=R/'earned_authority_v2_20261007';P=R/'next_core_mechanism_v1_20261006'
sys.dont_write_bytecode=True;sys.path.insert(0,str(P))
from assays import make_world
S=(1.4911274663291492,1.3452365735750882)
RULE=json.loads((D/'RULE_LOCK.json').read_text())['rule']
H=json.loads((D/'replay/HISTORIES.json').read_text());H={(h['world'],h['assay'],h['branch'],h['checkpoint']):h for h in H}
ROWS=json.load(gzip.open(D/'replay/ROWS.json.gz','rt'));ROWIDX={}
for row in ROWS:
 k=(row['world'],row['assay'],row['branch'],row['checkpoint'],row['group'],row['item'],row.get('order'));assert k not in ROWIDX;ROWIDX[k]=row

def key(c):
 assert 10 not in c
 k=bytes(x for x in c if x!=32)[-4:];assert len(k)==4;return k

def emit(preds,a):
 vs=[np.asarray(p['shared'])/S[0]+a*np.asarray(p['private'])/S[1] for p in preds]
 if len(vs)==2:return b'LR'[int(np.argmax(np.array([v[1]-v[0] for v in vs])))]
 assert len(vs)==1;return b'0123'[int(np.argmax(vs[0]))]

def allowed(branch,stage):
 assert branch in ('W','N_old','N_new','N_revision')
 return (branch,stage) not in (('N_old','old'),('N_new','new'),('N_revision','revision'))

def cues(row,w):
 if 'cue_hex' in row:return [bytes.fromhex(c) for c in row['cue_hex']]
 if w['assay']=='reuse':
  a,b=w['sets'][row['stage']][row['pair']*2:row['pair']*2+2]
  return [bytes.fromhex(x['cue_hex']) for x in ((a,b) if row['order']==0 else (b,a))]
 return [bytes.fromhex(w['sets'][row['stage']][row['item']]['cue_hex'])]

paths=[R/'next_core_budget_v2_20261006/science/confirm/receipts/61008101_ERROR_reuse.json.gz',R/'next_core_budget_v2_20261006/science/confirm/receipts/61008101_ERROR_lifetime.json.gz',R/'earned_authority_v1_20261007/cycles/cycle3/full/latin/receipts/62010003_ERROR_latin.json.gz']
result={'status':'PASS','independent_implementation':True,'imported_live_ExperienceAuthority':False,'new_native_learning':False,'scope':'three selected archived jobs, not full science validation','jobs':[]};nrec=nprobe=nhist=0;totallow=totdisabled=0
for path in paths:
 d=json.load(gzip.open(path,'rt'));w=make_world(d['world'],d['assay']);assert w['sha256']==d['fixture_sha256'];jr={'world':d['world'],'assay':d['assay'],'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'branches':[]}
 for branch,bd in d['branches'].items():
  cache=[];st=[[0,0,0],[0,0,0]];probes={p['name']:p for p in bd['probes']};visited=[];diag={'branch':branch,'records_allowed':0,'records_disabled':0,'histories_checked':0,'probe_rows_checked':0,'final_alpha_half':0,'final_changed':0,'final_help':0,'final_hurt':0,'evictions':0}
  assert len(bd['records'])==len(w['events'])
  for idx,(rec,event) in enumerate(zip(bd['records'],w['events'])):
   expected=allowed(branch,event['stage']);assert type(rec['learn']) is bool and rec['learn']==expected
   assert rec['index']==event['index']==idx and rec['stage']==event['stage']
   assert rec['prediction_precedes_outcome'] is True and rec['predicted_at']==rec['observed_at']==rec['write']['time']
   assert rec['write']['outcome']==event['outcome'] and rec['write']['learn']==expected
   calls=rec['actual_calls'];assert len(calls)==8 and [c['store'] for c in calls]==list(range(8));assert all(type(c['write']) is bool and c['write']==expected for c in calls)
   if not expected:assert all(c['applied_l1']==0 and c['no_write_reference_equal'] is True for c in calls)
   k=key(bytes.fromhex(event['cue_hex']));stratum=int(k in cache);e=emit([rec['prediction']],1);h=emit([rec['prediction']],.5)
   assert e==rec['prediction']['emitted'];halfold=rec.get('policies',{}).get('Q_HALF',rec.get('readouts',{}).get('Q_HALF'));assert h==halfold['emitted']
   before=(list(cache),[list(a) for a in st])
   if expected:
    diag['records_allowed']+=1;totallow+=1;t=rec['write']['outcome'];st[stratum][0]+=1;st[stratum][1]+=int(h==t and e!=t);st[stratum][2]+=int(e==t and h!=t)
    if k not in cache:
     if len(cache)==64:cache.pop(0);diag['evictions']+=1
     cache.append(k)
   else:
    diag['records_disabled']+=1;totdisabled+=1;assert (cache,st)==before
   nrec+=1
   for pname in w['boundaries'].get(str(idx+1),[]):
    pr=probes[pname];visited.append(pname);assert pr['at']==w['clocks'][pname] and pr['state_before']==pr['state_after']
    snap={'schema':RULE,'cache':[k.hex() for k in cache],'stats':[list(a) for a in st],'pending':None}
    saved=H[(d['world'],d['assay'],branch,pname)];assert saved['records_seen']==idx+1 and saved['state']==snap
    digest=hashlib.sha256(json.dumps(snap,sort_keys=True,separators=(',',':')).encode()).hexdigest();assert saved['state_sha256']==digest
    nhist+=1;diag['histories_checked']+=1
    for row in pr['rows']:
     assert row['first_pre_feedback'] is True
     if w['assay']=='lifetime' and row['stage']=='old' and row['item'] not in w['intact_old_items']:continue
     ps=row.get('predictions') or [row['prediction']];ks=[key(c) for c in cues(row,w)];stratum=int(all(k in cache for k in ks));a=.5 if st[stratum][1]>st[stratum][2] else 1
     emissions={'ERROR':emit(ps,1),'Q_HALF':emit(ps,.5),'EXPERIENCE':emit(ps,a),'GLOBAL_CALIBRATION':emit(ps,.5 if sum(s[1] for s in st)>sum(s[2] for s in st) else 1),'SUPPORT_ONLY':emit(ps,1 if stratum else .5)}
     rk=(d['world'],d['assay'],branch,pname,row['stage'],row.get('item',row.get('pair')),row.get('order'));savedrow=ROWIDX[rk];assert savedrow['supported']==bool(stratum) and savedrow['alpha']==a
     assert all(savedrow['correct'][p]==int(v==row['target']) for p,v in emissions.items())
     assert savedrow['different_from_ERROR']==(emissions['EXPERIENCE']!=emissions['ERROR']);nprobe+=1;diag['probe_rows_checked']+=1
     if pname=='final':
      diag['final_alpha_half']+=int(a==.5);diag['final_changed']+=int(emissions['EXPERIENCE']!=emissions['ERROR']);diag['final_help']+=int(emissions['EXPERIENCE']==row['target'] and emissions['ERROR']!=row['target']);diag['final_hurt']+=int(emissions['ERROR']==row['target'] and emissions['EXPERIENCE']!=row['target'])
    assert snap=={'schema':RULE,'cache':[k.hex() for k in cache],'stats':[list(a) for a in st],'pending':None}
  assert sorted(visited)==sorted(probes);diag['final_stats']=st;diag['final_cache_length']=len(cache);jr['branches'].append(diag)
 result['jobs'].append(jr)
result['checked']={'teacher_records':nrec,'probe_rows':nprobe,'checkpoint_histories':nhist,'observer_allowed':totallow,'observer_disabled':totdisabled}
summary=json.loads((D/'replay/SUMMARY.json').read_text());result['full_replay_descriptive']={k:summary[k] for k in ('native_jobs_replayed','baseline_parity','final_half_weight_queries','final_changed_choices')}
Path('/private/tmp/authority_v2_independent_replay_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
