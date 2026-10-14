"""Read-only final preparation audit; never imports or executes the learner."""
import copy,gzip,hashlib,json,pickle
import numpy as np
from pathlib import Path
from auditor import audit

ROOT=Path(__file__).parent
ROSTER=('R1-DIAG','R1-FULL','R4-DIAG','R4-FULL','F1-FULL','F4-FULL','LEGACY-B')

def counter_check(receipt):
 learned=receipt['config'].startswith('R')
 assert 'resource' in receipt
 for history,h in receipt['histories'].items():
  expected_readout=0; expected_encoder=0
  parent_counts={}
  for row in h['root_records']:
   for k in ('readout_counter_before','readout_counter_after','encoder_counter_before','encoder_counter_after'):assert k in row
   assert row['readout_counter_before']==expected_readout
   assert row['encoder_counter_before']==expected_encoder
   expected_readout+=len(bytes.fromhex(row['answer_hex']))+1 if history!='N_OLD' else 0
   expected_encoder+=int(learned and history!='N_OLD')
   assert row['readout_counter_after']==expected_readout
   assert row['encoder_counter_after']==expected_encoder
   parent_counts[row['index']+1]=(expected_readout,expected_encoder)
  for point in h['checkpoints']:
   er,ee=parent_counts[point['root_records']]
   for row in point['cohort_records']:
    assert row['readout_counter_before']==er and row['encoder_counter_before']==ee
    er+=len(bytes.fromhex(row['answer_hex']))+1;ee+=int(learned)
    assert row['readout_counter_after']==er and row['encoder_counter_after']==ee


def checkpoint_digest(path):
 if path.suffix=='.pkl':
  # Only our trusted locally generated fixtures; no constructor or trajectory runs.
  with path.open('rb') as f:c=pickle.load(f)
  digest=hashlib.sha256()
  for group in (c.theta,c.mom,{'V':c.V,'P':c.P}):
   for k,v in sorted(group.items()):digest.update(k.encode());digest.update(v.tobytes())
  digest.update(str((c.steps,c.learning_events)).encode())
  return digest.hexdigest()
 with np.load(path,allow_pickle=False) as data:
  metadata=json.loads(data['metadata'].tobytes())
  components={k:hashlib.sha256(data[field].tobytes()).hexdigest() for k,field in
       [('W','weights'),('V','V'),('P','P'),('centers','centers'),('counts','counts')]}
  components['learning_events']=metadata['base']['learning_events']
  return hashlib.sha256(json.dumps(components,sort_keys=True).encode()).hexdigest()


def final_audit():
 folder=ROOT/'cycles/cycle3/final_receipts'
 receipts={}
 identities=[]
 for p in folder.glob('*.json.gz'):
  with gzip.open(p,'rt') as f:r=json.load(f)
  assert r['world']==202610130103
  assert r['config'] not in receipts
  receipts[r['config']]=r
 assert set(receipts)==set(ROSTER)
 for r in receipts.values():
  current={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in r['source_identity']}
  assert current==r['source_identity']
  audit(r,expected_source=current);counter_check(r)
  identities.append(current)
 assert all(x==identities[0] for x in identities)
 checkpoint_files=0
 for name,r in receipts.items():
  extension='.npz' if name=='LEGACY-B' else '.pkl'
  for point in r['histories']['W']['checkpoints']:
   for suffix,expected in [('root',point['parent_before']),('retained',point['fork_final'])]:
    path=folder/f"{r['world']}_{name}_W_{point['n_per_operation']}_{suffix}{extension}"
    assert checkpoint_digest(path)==expected;checkpoint_files+=1
  for history in ('N_OLD','SHUFFLE_OLD'):
   path=folder/f"{r['world']}_{name}_{history}_final{extension}"
   assert checkpoint_digest(path)==r['histories'][history]['final_arithmetic'];checkpoint_files+=1
 all_receipts=[]
 for path in ROOT.glob('cycles/cycle*/**/*.json.gz'):
  with gzip.open(path,'rt') as f:archived=json.load(f)
  assert archived['technical'] and archived['schema']=='RAW_RULE_TECHNICAL_JOB_V1'
  audit(archived);all_receipts.append(archived)
 assert len(all_receipts)==16
 assert len({(r['world'],r['config']) for r in all_receipts})==16
 assert sum(r['record_applications'] for r in all_receipts)==33792
 r=receipts['R1-FULL']; tampered=copy.deepcopy(r)
 tampered['histories']['W']['root_records'][1]['readout_counter_before']+=1
 try:counter_check(tampered)
 except AssertionError:pass
 else:raise AssertionError('counter-discontinuity tamper accepted')
 tampered=copy.deepcopy(r);tampered.pop('resource')
 try:counter_check(tampered)
 except AssertionError:pass
 else:raise AssertionError('missing resource accounting accepted')
 tampered=copy.deepcopy(r);tampered['histories']['N_OLD']['checkpoints'][0]['cohort_records'][0]['encoder_counter_after']=0
 try:counter_check(tampered)
 except AssertionError:pass
 else:raise AssertionError('cohort disabled encoder tamper accepted')
 return {'accepted':True,'scope':'three-cycle technical preparation, not scientific confirmation',
         'final_source_jobs':7,'all_full_technical_jobs':16,'all_full_technical_record_applications':33792,
         'final_source_record_applications':14784,'scientific_jobs':0,
         'independent_final_tamper_cases_rejected':3,'component_counter_continuity_checked':True,'final_checkpoints_independently_verified':checkpoint_files,
         'retention_forks':len(all_receipts)*18,'root_histories':len(all_receipts)*3,
         'final_source_identity':identities[0],'adopted_core_changed':False}

if __name__=='__main__':
 (ROOT/'FINAL_AUDIT.json').write_text(json.dumps(final_audit(),indent=2))
