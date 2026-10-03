# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent receipt-structure/formula/clock validator; no learner imports."""
import hashlib,json,math
from pathlib import Path
import numpy as np
from survivor_fixture import make_world,DT,ALPHABET,digest
from durable import sha
ROOT=Path(__file__).resolve().parents[2]
BRANCHES=('W','N_old_relation','N_new')
SCALES=(1.4911274663291492,1.3452365735750882)

def valid_hash(s):return isinstance(s,str) and len(s)==64 and all(c in '0123456789abcdef' for c in s)
def load(directory,expected_world=None,expected_source=None,expected_kind=None,expected_acceptance=None):
 p=Path(directory);manifest=json.loads((p/'manifest.json').read_text());assert manifest['complete'] is True
 assert manifest['schema']=='RC-SURVIVOR-RECEIPT-v1'
 if expected_world is not None:assert manifest['world']==expected_world
 expected={'header.json','fixture.json','end.json','boundaries.json'}|{f'probe-{i:02d}.json' for i in range(9)}
 for start in range(0,384,64):
  expected|={f'{name}-{start:03d}.json' for name in ('predictions','clocks','interventions')}
  expected|={f'audit-{b}-{start:03d}.json' for b in BRANCHES}
 assert set(manifest['parts'])==expected
 assert {x.name for x in p.iterdir() if x.is_file()}==expected|{'manifest.json'},'unexpected or partial receipt file'
 parts={}
 for name,r in manifest['parts'].items():
  assert name==Path(name).name and name.endswith('.json')
  b=(p/name).read_bytes();assert len(b)==r['bytes'] and len(b)<=512*1024
  assert hashlib.sha256(b).hexdigest()==r['sha256'];parts[name]=json.loads(b)
 header=parts['header.json'];world=make_world(manifest['world'])
 required_header={'schema','world','kind','arm','fixture_sha256','source','source_digest','acceptance_sha256','runtime','scales','birth_contract_sha256','birth_digests','births','state_budget','fixed_identity','tie_policy','endpoint'}
 assert required_header<=set(header),'mandatory header fields missing'
 assert header['schema']=='RC-SURVIVOR-WORLD-v1' and header['kind'] in ('technical','science')
 if expected_kind is not None:assert header['kind']==expected_kind
 if header['kind']=='science':
  from survivor_fixture import OFFICIAL_WORLDS
  assert manifest['world'] in OFFICIAL_WORLDS and p.name==str(manifest['world']) and p.parent.name=='science','noncanonical official directory/kind'
 else:assert manifest['world']==310200
 contract=json.loads((ROOT/'source/runtime/BIRTH_CONTRACT.json').read_text())
 assert header['runtime']==contract['runtime'] and header['births']==contract['births'] and header['birth_digests']==contract['birth_digests']
 assert header['birth_contract_sha256']==sha(ROOT/'source/runtime/BIRTH_CONTRACT.json')
 expected_budget=[dict(contract['budget_common'],content_window_bound=0 if i<4 else 4) for i in range(8)]
 assert header['state_budget']==expected_budget
 assert header['tie_policy']=='lowest ASCII via np.argmax' and header['endpoint']=='taught-dependent held-out table completion'
 assert valid_hash(header['acceptance_sha256'])
 if expected_acceptance is not None:assert header['acceptance_sha256']==expected_acceptance
 elif header['kind']=='technical':assert header['acceptance_sha256']==header['source']['protocol/ROUND1_DESIGN_CYCLE3.md']
 elif (ROOT/'protocol/OFFICIAL_LOCK.json').exists():assert header['acceptance_sha256']==sha(ROOT/'protocol/OFFICIAL_LOCK.json')
 else:raise AssertionError('science acceptance identity required')

 assert parts['fixture.json']==world and manifest['fixture_sha256']==header['fixture_sha256']==world['sha256']
 assert header['world']==manifest['world'] and header['kind']==manifest['kind'] and header['arm']=='R_center'
 assert valid_hash(header['fixed_identity'])
 assert header['scales']==list(SCALES) and header['source_digest']==manifest['source_digest']==digest(header['source'])
 if expected_source is not None:assert header['source']==expected_source
 first=[];clocks=[];interventions=[];audits={b:[] for b in BRANCHES}
 for start in range(0,384,64):
  first+=parts[f'predictions-{start:03d}.json'];clocks+=parts[f'clocks-{start:03d}.json'];interventions+=parts[f'interventions-{start:03d}.json']
  for b in BRANCHES:audits[b]+=parts[f'audit-{b}-{start:03d}.json']
 sequence=[(i,b) for i in range(384) for b in BRANCHES]
 for rows in (first,clocks,interventions):assert [(r['index'],r['branch']) for r in rows]==sequence
 fidx={(r['index'],r['branch']):r for r in first}
 def check_values(pred):
  ss=np.asarray(pred['shared'],float);pp=np.asarray(pred['private'],float);cc=np.asarray(pred['combined'],float)
  assert ss.shape==pp.shape==cc.shape==(4,) and np.isfinite(np.r_[ss,pp,cc]).all()
  expected=ss/SCALES[0]+pp/SCALES[1];assert np.array_equal(expected,cc)
  assert pred['emitted']==ALPHABET[int(np.argmax(cc))]
 for r in first:check_values(r)
 for row in interventions:
  e=world['events'][row['index']];blocked=(row['branch']=='N_old_relation' and e['stage']=='old') or (row['branch']=='N_new' and e['stage']=='new')
  assert row['write']==(not blocked)
 for b,rows in audits.items():
  assert len(rows)==384
  for i,r in enumerate(rows):
   e=world['events'][i];assert r['outcome']==e['outcome'] and r['prediction_time']==r['outcome_time']==e['at']+12*DT
   assert r['c']==0. and r['s']==[.25-float(x==e['outcome']) for x in ALPHABET]
   assert r['cached_shared']==fidx[i,b]['shared'] and len(r['shared_state'])==4 and all(valid_hash(h) for h in r['shared_state'])
   for key in ('shared_l1','private_l1'):assert len(r[key])==4 and all(math.isfinite(x) and x>=0 for x in r[key])
   if (b=='N_old_relation' and e['stage']=='old') or (b=='N_new' and e['stage']=='new'):assert r['shared_l1']==[0.]*4 and r['private_l1']==[0.]*4
 def clockset(rows,at,count,last):
  assert isinstance(rows,list) and len(rows)==8
  needed={'brain_t','elapsed','elapsed_base','fe_t','pending_t','last_byte_t','bytes_seen','teach_seen'}
  for c in rows:
   assert needed<=set(c),'mandatory clock fields missing'
   assert all(isinstance(c[k],(int,float)) and math.isfinite(c[k]) for k in ('brain_t','elapsed','elapsed_base','fe_t','last_byte_t'))
   assert c['elapsed_base']==0. and abs(c['brain_t']-at)<1e-6 and abs(c['elapsed']-at)<1e-6 and abs(c['fe_t']-at)<1e-6
   assert c['pending_t'] is None and c['last_byte_t']==last and c['bytes_seen']==14*count and c['teach_seen']==count
 for row in clocks:
  e=world['events'][row['index']];clockset(row['clocks'],e['at']+165.,row['index']+1,e['at']+13*DT)
 boundary=parts['boundaries.json'];assert set(boundary)==set(BRANCHES)
 for branch,checkpoints in boundary.items():
  assert set(checkpoints)=={'old_end','new_start','new_end','final'}
  for when,rows in checkpoints.items():
   count=192 if when in ('old_end','new_start') else 384;last=world['events'][count-1]['at']+13*DT
   clockset(rows,world[when],count,last)
 probes=[parts[f'probe-{i:02d}.json'] for i in range(9)]
 assert {(r['branch'],r['when']) for r in probes}=={(b,w) for b in BRANCHES for w in ('old_end','new_end','final')}
 for batch in probes:
  assert {'branch','when','state_before','state_after','fixed_identity','store_digests','clocks','rows'}<=set(batch),'mandatory probe fields missing'
  count=192 if batch['when']=='old_end' else 384;clockset(batch['clocks'],world[batch['when']],count,world['events'][count-1]['at']+13*DT)
  assert batch['clocks']==boundary[batch['branch']][batch['when']] and len(batch['store_digests'])==8 and all(valid_hash(h) for h in batch['store_digests'])
  names=('old','heldout') if batch['when']=='old_end' else ('old','heldout','new')
  expectedq=[(name,i) for name in names for i in range(len(world['sets'][name]['cues']))]
  assert [(r['set'],r['item']) for r in batch['rows']]==expectedq
  assert batch['state_before']==batch['state_after'] and valid_hash(batch['state_before']) and valid_hash(batch['fixed_identity']) and batch['fixed_identity']==header['fixed_identity']
  for r in batch['rows']:
   check_values(r);assert r['at']==world[batch['when']] and r['first_pre_feedback'] is True
   assert r['cue_hex']==world['sets'][r['set']]['cues'][r['item']] and r['target']==world['sets'][r['set']]['outcomes'][r['item']]
   assert r['continuing_state_before']==r['continuing_state_after']==batch['state_before'] and r['fixed_identity']==batch['fixed_identity']
   shared=np.array(r['shared'])/SCALES[0];private=np.array(r['private'])/SCALES[1]
   expectedpol={'alpha1':shared+private,'alpha_half':shared+.5*private,'alpha0':shared,'private_only':private}
   assert set(r['policies'])==set(expectedpol)
   for name,score in expectedpol.items():
    pol=r['policies'][name];assert np.array_equal(pol['scores'],score) and pol['emitted']==ALPHABET[int(np.argmax(score))]
    assert pol['ties']==np.flatnonzero(score==score.max()).tolist() and pol['correct']==int(pol['emitted']==r['target'])
 end=parts['end.json'];assert {'end_state','end_wrapper','end_clocks','clock_error','resources'}<=set(end),'mandatory final fields missing'
 assert math.isfinite(end['clock_error']) and 0<=end['clock_error']<1e-6
 assert set(end['end_wrapper'])==set(end['end_clocks'])==set(BRANCHES)
 for branch in BRANCHES:
  final=next(r for r in probes if r['branch']==branch and r['when']=='final')
  assert end['end_wrapper'][branch]==final['state_before']==final['state_after'] and end['end_state'][branch]==final['store_digests']
  assert end['end_clocks'][branch]==final['clocks']==boundary[branch]['final']
  clockset(end['end_clocks'][branch],world['final'],384,world['events'][383]['at']+13*DT)
 assert set(end['end_state'])==set(BRANCHES) and all(len(x)==8 and all(valid_hash(h) for h in x) for x in end['end_state'].values())
 assert manifest['resources']==end['resources']
 assert set(manifest['resources'])=={'wall_s','peak_rss_bytes'}
 assert math.isfinite(manifest['resources']['wall_s']) and 0<manifest['resources']['wall_s']<=900
 assert isinstance(manifest['resources']['peak_rss_bytes'],int) and 0<manifest['resources']['peak_rss_bytes']<=768*1024**2
 # Resource values are intentionally excluded from deterministic replay identity.
 scientific={k:v for k,v in parts.items()};scientific['end.json']={k:v for k,v in end.items() if k!='resources'}
 return {'manifest':manifest,'manifest_sha256':sha(p/'manifest.json'),'header':header,'probes':probes,'scientific_digest':digest(scientific),'parts_count':len(parts)}
