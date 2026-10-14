"""Independent receipt reducer. Does not import runner, interface or learning core."""
import json,hashlib
from worlds import make_world,OPS,CHECKPOINTS
from vendor.task_source import prompt,answer,recover_family

def exact(r):return r['status']=='EOS' and r['payload_hex']==r['answer_hex']

def rates(rows):
 return {op:sum(exact(r) for r in rows if r['operation']==op)/sum(r['operation']==op for r in rows) for op in OPS}

def audit(receipt,plan=None):
 if plan is None:plan=make_world(receipt['world'],True)
 assert receipt['technical'] is True and receipt['schema']=='RAW_RULE_TECHNICAL_JOB_V1'
 assert receipt['stream_digest']==plan['stream_digest']
 assert set(receipt['histories'])=={'W','N_OLD','SHUFFLE_OLD'}
 assert receipt['correlated_trajectories']==21
 train={recover_family(r) for r in plan['training_pool']}
 held={recover_family(r) for r in plan['held_pool']}
 assert train.isdisjoint(held)
 expected_held={(prompt(r).encode().hex(),answer(r).encode().hex(),r[0]) for r in plan['held_pool']}
 total=0; summaries={}
 learned=receipt['config'].startswith('R')
 for history,h in receipt['histories'].items():
  assert len(h['root_records'])==512 and len(h['checkpoints'])==6
  writing=history!='N_OLD'
  if not writing:assert h['initial']==h['final_arithmetic']
  for i,r in enumerate(h['root_records']):
   expected=plan['records'][i]
   target=expected['answer'] if history!='SHUFFLE_OLD' else plan['shuffled_answers'][str(i)]
   assert r['index']==i and r['prompt_hex']==expected['prompt'].encode().hex() and r['answer_hex']==target.encode().hex()
   check_record(r,writing,learned)
  summaries[history]=[]
  for stop,c in zip(CHECKPOINTS,h['checkpoints']):
   assert c['n_per_operation']==stop and c['root_records']==4*stop
   assert c['parent_before']==c['parent_after']==c['fork_initial']
   if not writing:assert c['parent_before']==h['initial']
   assert len(c['cohort_records'])==32
   for row,expected in zip(c['cohort_records'],plan['cohort_records']):
    assert row['prompt_hex']==expected['prompt_hex'] and row['answer_hex']==expected['answer_hex']
    check_record(row,True,learned)
   assert abs(c['scenario_cpu']-(c['root_cpu']+c['fork_cpu']))<1e-8
   assert c['root_cpu']>0 and c['fork_cpu']>0
   for phase in ('held_pre','held_post','held_retained'):
    rows=c[phase]
    assert len(rows)==len(expected_held)
    assert {(r['prompt_hex'],r['answer_hex'],r['operation']) for r in rows}==expected_held
    check_predictions(rows)
   assert [(r['payload_hex'],r['status']) for r in c['held_post']]==[(r['payload_hex'],r['status']) for r in c['held_retained']]
   expected_taught={r['prompt'].encode().hex():r['answer'].encode().hex() for r in plan['records'] if r['checkpoint']<=stop}
   for phase in ('taught_pre','taught_retained'):
    assert {r['prompt_hex']:r['answer_hex'] for r in c[phase]}==expected_taught
    check_predictions(c[phase])
   for phase in ('cohort_pre','cohort_post'):
    assert [(r['prompt_hex'],r['answer_hex']) for r in c[phase]]==[(r['prompt_hex'],r['answer_hex']) for r in plan['cohort_pairs']]
    check_predictions(c[phase])
   summaries[history].append({'n':stop,'scenario_records':4*stop+32,'cpu':c['scenario_cpu'],
     'taught':sum(exact(r) for r in c['taught_pre'])/len(c['taught_pre']),
     'taught_retained':sum(exact(r) for r in c['taught_retained'])/len(c['taught_retained']),
     'held_pre':rates(c['held_pre']),'held_retained':rates(c['held_retained']),
     'empty_held':sum(r['empty'] for r in c['held_retained'])/len(c['held_retained']),
     'cohort_pre':sum(exact(r) for r in c['cohort_pre'])/8,
     'cohort_post':sum(exact(r) for r in c['cohort_post'])/8})
  total+=512+6*32
 assert total==2112==receipt['record_applications']
 return {'accepted':True,'technical':True,'world':receipt['world'],'config':receipt['config'],
         'record_applications':total,'correlated_trajectories':21,'summary':summaries}

def check_record(r,allowed,learned):
 assert r['allowed'] is allowed
 targets=list(bytes.fromhex(r['answer_hex']))+[256]
 assert [e['target'] for e in r['events']]==targets
 assert r['readout_updates']==len(targets)*int(allowed)
 assert r['encoder_steps']==int(allowed and learned)
 for event,target in zip(r['events'],targets):
  assert event['permitted'] is allowed and event['readout_updates']==int(allowed)
  assert event['encoder_steps']==int(allowed and learned and target==256)


def check_predictions(rows):
 for r in rows:
  output=bytes.fromhex(r['payload_hex'])
  assert r['status'] in ('EOS','TRUNCATED') and len(output)<=16
  assert r['empty']==(len(output)==0)
  p=r['first_probabilities']
  assert len(p)==257 and all(0<=x<=1 for x in p) and abs(sum(p)-1)<1e-10
  assert r['choice']==max(range(257),key=lambda i:p[i])
