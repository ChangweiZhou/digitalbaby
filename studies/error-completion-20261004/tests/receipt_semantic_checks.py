# SPDX-License-Identifier: GPL-3.0-or-later
"""Synthetic structural receipts only; never a native learning run."""
import copy,json,hashlib,shutil
from pathlib import Path
from survivor_fixture import make_world,DT,ALPHABET,digest
from durable import create,sha,canonical
from validate_receipt import load,ROOT,BRANCHES,SCALES

def run_checks(temp_root,source):
 root=Path(temp_root);root.mkdir(parents=True,exist_ok=True);create(root/'SYNTHETIC_ONLY.json',{'native_events':0,'purpose':'semantic tamper validation'})
 w=make_world(320200);contract=json.loads((ROOT/'source/runtime/BIRTH_CONTRACT.json').read_text());h='f'*64;fixed='e'*64
 def clocks(at,count):
  return [dict(brain_t=at,elapsed=at,elapsed_base=0.,fe_t=at,pending_t=None,last_byte_t=w['events'][count-1]['at']+13*DT,bytes_seen=14*count,teach_seen=count) for _ in range(8)]
 def prediction():return {'shared':[0.]*4,'private':[0.]*4,'combined':[0.]*4,'emitted':48}
 parts={};parts['header.json']={'schema':'ERROR-COMPLETION-WORLD-v1','world':320200,'kind':'technical','arms':['R_center','ERROR'],'fixture_sha256':w['sha256'],'source':source,'source_digest':digest(source),'acceptance_sha256':source['protocol/COMPARISON_DESIGN_CYCLE3.md'],'runtime':contract['runtime'],'scales':list(SCALES),'birth_contract_sha256':sha(ROOT/'source/runtime/BIRTH_CONTRACT.json'),'birth_digests':contract['birth_digests'],'births':contract['births'],'state_budget':[dict(contract['budget_common'],content_window_bound=0 if i<4 else 4) for i in range(8)],'fixed_identity':{b:fixed for b in BRANCHES},'tie_policy':'lowest ASCII via np.argmax','endpoint':'taught-dependent held-out table completion'}
 parts['fixture.json']=w
 parts['boundaries.json']={b:{when:clocks(w[when],192 if when in ('old_end','new_start') else 384) for when in ('old_end','new_start','new_end','final')} for b in BRANCHES}
 for start in range(0,384,32):
  preds=[];cs=[];inter=[];aud={b:[] for b in BRANCHES}
  for i in range(start,start+32):
   e=w['events'][i]
   for b in BRANCHES:
    blocked=(b.endswith('__N_old_relation') and e['stage']=='old') or (b.endswith('__N_new') and e['stage']=='new')
    preds.append(dict(index=i,branch=b,**prediction()));cs.append(dict(index=i,branch=b,clocks=clocks(e['at']+165.,i+1)));inter.append(dict(index=i,branch=b,write=not blocked))
    aud[b].append({'outcome':e['outcome'],'prediction_time':e['at']+12*DT,'outcome_time':e['at']+12*DT,'c':0.,'s':[.25-float(x==e['outcome']) for x in ALPHABET],'cached_shared':[0.]*4,'p':[.25]*4,'private_pre_outcome_probabilities':[.25]*4,'shared_state':[h]*4,'shared_l1':[0. if blocked else 1.]*4,'private_l1':[0. if blocked else 1.]*4})
  parts[f'predictions-{start:03d}.json']=preds;parts[f'clocks-{start:03d}.json']=cs;parts[f'interventions-{start:03d}.json']=inter
  for b in BRANCHES:parts[f'audit-{b}-{start:03d}.json']=aud[b]
 i=0
 for b in BRANCHES:
  for when in ('old_end','new_end','final'):
   rows=[]
   for name in (('old','heldout') if when=='old_end' else ('old','heldout','new')):
    for j,ch in enumerate(w['sets'][name]['cues']):
     target=w['sets'][name]['outcomes'][j];pol={k:{'scores':[0.]*4,'emitted':48,'ties':[0,1,2,3],'correct':int(target==48)} for k in ('combined','shared','private')}
     rows.append(dict(set=name,item=j,cue_hex=ch,target=target,at=w[when],first_pre_feedback=True,continuing_state_before=h,continuing_state_after=h,fixed_identity=fixed,policies=pol,**prediction()))
   parts[f'probe-{i:02d}.json']={'branch':b,'when':when,'state_before':h,'state_after':h,'fixed_identity':fixed,'store_digests':contract['birth_digests'],'clocks':parts['boundaries.json'][b][when],'rows':rows};i+=1
 resources={'wall_s':1.,'peak_rss_bytes':1024,'process_cpu_s':.9}
 tags=['birth']+[f'event-{i:03d}' for i in range(192)]+['new_start']+[f'event-{i:03d}' for i in range(192,384)]+['final']
 parts['shared-invariants.json']=[{'at_boundary':tag,'control':c,'R_center':h,'ERROR':h,'identical':True} for tag in tags for c in ('W','N_old_relation','N_new')]
 parts['end.json']={'end_state':{b:contract['birth_digests'] for b in BRANCHES},'end_wrapper':{b:h for b in BRANCHES},'end_clocks':{b:parts['boundaries.json'][b]['final'] for b in BRANCHES},'clock_error':0.,'resources':resources}
 def write(path,values):
  meta={}
  for name,d in values.items():meta[name]={'sha256':create(path/name,d),'bytes':len(canonical(d))}
  create(path/'manifest.json',{'schema':'ERROR-COMPLETION-RECEIPT-v1','complete':True,'world':320200,'kind':'technical','parts':meta,'source_digest':digest(source),'fixture_sha256':w['sha256'],'resources':resources})
 base=root/'valid';write(base,parts);load(base,320200,source,expected_kind='technical')
 cases=[]
 changes=[('missing_probe_clocks',lambda d:d['probe-00.json'].pop('clocks')),
 ('missing_final_clocks',lambda d:d['end.json'].pop('end_clocks')),
 ('missing_final_wrapper',lambda d:d['end.json'].pop('end_wrapper')),
 ('missing_runtime',lambda d:d['header.json'].pop('runtime')),
 ('wrong_runtime',lambda d:d['header.json']['runtime'].update(python='0')),
 ('missing_budget',lambda d:d['header.json'].pop('state_budget')),
 ('wrong_birth',lambda d:d['header.json']['birth_digests'].__setitem__(0,'a'*64)),
 ('wrong_endpoint',lambda d:d['header.json'].update(endpoint='unregistered')),
 ('wrong_acceptance',lambda d:d['header.json'].update(acceptance_sha256='0'*64)),
 ('negative_flush_error',lambda d:d['end.json'].update(clock_error=-1.)),
 ('lost_old_delay_clock',lambda d:d['boundaries.json']['R_center__W'].pop('new_start')),
 ('desynchronized_probe_fe',lambda d:d['probe-00.json']['clocks'][0].update(fe_t=-1.)),
 ('bad_final_counter',lambda d:d['end.json']['end_clocks']['R_center__W'][0].update(teach_seen=383)),
 ('wrong_final_link',lambda d:d['end.json']['end_wrapper'].update({'R_center__W':'a'*64})),
 ('wrong_tie_policy',lambda d:d['header.json'].update(tie_policy='random')),
 ('tiny_paired_event_clock_divergence',lambda d:d['clocks-000.json'][0]['clocks'][0].update(fe_t=d['clocks-000.json'][0]['clocks'][0]['fe_t']+5e-7)),
 ('tiny_paired_delay_clock_divergence',lambda d:d['boundaries.json']['R_center__W']['new_start'][0].update(fe_t=d['boundaries.json']['R_center__W']['new_start'][0]['fe_t']+5e-7)),
 ('wrong_error_pi',lambda d:d['audit-ERROR__W-000.json'][0]['private_pre_outcome_probabilities'].__setitem__(0,.3)),
 ('wrong_error_coefficient',lambda d:d['audit-ERROR__W-000.json'][0]['s'].__setitem__(0,42.)),
 ('wrong_rule_count',lambda d:d['header.json']['arms'].append('REPLACE')),
 ('shared_trajectory_diverged',lambda d:d['shared-invariants.json'][0].update(ERROR='a'*64)),
 ('extra_probe_outcome',lambda d:d['probe-00.json']['rows'][0].update(first_pre_feedback=False))]
 for name,change in changes:
  d=copy.deepcopy(parts);change(d);path=root/name;write(path,d)
  try:load(path,320200,source,expected_kind='technical')
  except (AssertionError,KeyError,ValueError,TypeError):cases.append(name)
  else:raise AssertionError('semantic tamper accepted: '+name)
 return {'synthetic_base_accepted':True,'semantic_tampers_rejected':cases,'manifests_recomputed':True,'native_events':0,'parts':len(parts)}
