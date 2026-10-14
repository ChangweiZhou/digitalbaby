"""Reserved technical execution only. No DEV/confirmation admission or launch path."""
import argparse,copy,gzip,hashlib,json,time
from pathlib import Path
import numpy as np
from core import Core
from interface import birth,teach,probe,digest,CONFIGS
from worlds import make_world,OPS,CHECKPOINTS
from threadpoolctl import threadpool_limits

def source_identity():
 root=Path(__file__).parent
 files=['core.py','interface.py','worlds.py','runner.py','auditor.py','metrics.py',
        'vendor/task_source.py','vendor/legacy_core.py','vendor/generic_byte_io/core.py',
        'vendor/__init__.py','vendor/generic_byte_io/__init__.py']
 return {f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files if (root/f).exists()}

def queries(rows):
 return [{'prompt_hex':r['prompt'].encode().hex(),'answer_hex':r['answer'].encode().hex(),'operation':r['operation']} for r in rows]

def held_queries(plan):
 from vendor.task_source import prompt,answer
 return [{'prompt_hex':prompt(r).encode().hex(),'answer_hex':answer(r).encode().hex(),'operation':r[0]} for r in plan['held_pool']]

def learning_record(c,prompt,target,allowed):
 before=c.learning_events; step=c.steps if isinstance(c,Core) else 0
 events=teach(c,prompt,target,allowed)
 updates=c.learning_events-before
 after_step=c.steps if isinstance(c,Core) else 0
 if updates!=sum(e['readout_updates'] for e in events) or after_step-step!=sum(e['encoder_steps'] for e in events):
  raise AssertionError('component ledger disagrees with actual counters')
 return {'prompt_hex':prompt.hex(),'answer_hex':target.hex(),'allowed':allowed,'events':events,
         'readout_updates':updates,'encoder_steps':after_step-step,
         'encoder_update':copy.deepcopy(c.last_update) if isinstance(c,Core) and allowed else None}

def checkpoint(c,path):
 if isinstance(c,Core):c.save(path.with_suffix('.pkl'))
 else:c.save_checkpoint(path.with_suffix('.npz'))

def run(world,name,out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 result_path=out/f'{world}_{name}.json.gz'
 if result_path.exists():raise FileExistsError('technical receipts immutable; choose fresh reserved fixture')
 plan=make_world(world,True)
 held=held_queries(plan)
 result={'schema':'RAW_RULE_TECHNICAL_JOB_V1','world':world,'config':name,'technical':True,
         'source_identity':source_identity(),'stream_digest':plan['stream_digest'],
         'record_applications':0,'correlated_trajectories':21,'histories':{}}
 start=time.perf_counter()
 for history in ('W','N_OLD','SHUFFLE_OLD'):
  c=birth(name,world);initial=digest(c);root_cpu=0.;records=[];checkpoints=[]
  for stop in CHECKPOINTS:
   needed=[r for r in plan['records'] if r['checkpoint']==stop]
   for row in needed:
    target=row['answer'] if history!='SHUFFLE_OLD' else plan['shuffled_answers'][str(row['index'])]
    before=time.process_time(); rec=learning_record(c,row['prompt'].encode(),target.encode(),history!='N_OLD');elapsed=time.process_time()-before
    root_cpu+=elapsed
    records.append(dict(rec,index=row['index'],operation=row['operation'],cpu_seconds=elapsed))
   parent=digest(c)
   if history=='N_OLD' and parent!=initial:raise AssertionError('old no-write state changed')
   taught={r['prompt']:r for r in plan['records'] if r['checkpoint']<=stop}
   taught_query=queries(list(taught.values()))
   pre=probe(c,held);taught_pre=probe(c,taught_query)
   fork=c.clone() if isinstance(c,Core) else copy.deepcopy(c)
   fork_initial=digest(fork)
   cohort_pre=probe(fork,plan['cohort_pairs'])
   fork_records=[];fork_cpu=0.
   for row in plan['cohort_records']:
    before=time.process_time();rec=learning_record(fork,bytes.fromhex(row['prompt_hex']),bytes.fromhex(row['answer_hex']),True);elapsed=time.process_time()-before
    fork_cpu+=elapsed;fork_records.append(dict(rec,index=row['index'],cpu_seconds=elapsed))
   cohort_post=probe(fork,plan['cohort_pairs']);post=probe(fork,held)
   fork.rest(86400);retained=probe(fork,held);taught_retained=probe(fork,taught_query)
   if digest(c)!=parent:raise AssertionError('fork altered parent')
   if history=='W':
    checkpoint(c,out/f'{world}_{name}_W_{stop}_root')
    checkpoint(fork,out/f'{world}_{name}_W_{stop}_retained')
   checkpoints.append({'n_per_operation':stop,'root_records':len(records),'root_cpu':root_cpu,
                       'fork_cpu':fork_cpu,'scenario_cpu':root_cpu+fork_cpu,
                       'parent_before':parent,'parent_after':digest(c),'fork_initial':fork_initial,
                       'fork_final':digest(fork),'cohort_records':fork_records,
                       'held_pre':pre,'held_post':post,'held_retained':retained,
                       'taught_pre':taught_pre,'taught_retained':taught_retained,
                       'cohort_pre':cohort_pre,'cohort_post':cohort_post})
   print(json.dumps({'technical':True,'world':world,'config':name,'history':history,'n':stop,
     'taught_correct':sum(x['payload_hex']==x['answer_hex'] and x['status']=='EOS' for x in taught_pre),
     'taught_total':len(taught_pre),'held_retained_correct':sum(x['payload_hex']==x['answer_hex'] and x['status']=='EOS' for x in retained),
     'held_total':len(retained),'root_cpu':root_cpu}),flush=True)
  if history!='W': checkpoint(c,out/f'{world}_{name}_{history}_final')
  result['histories'][history]={'initial':initial,'final_arithmetic':digest(c),'root_records':records,
                               'checkpoints':checkpoints,'root_cpu':root_cpu,
                               'storage':c.bytes_state() if isinstance(c,Core) else c.storage_accounting()}
  result['record_applications']+=len(records)+sum(len(x['cohort_records']) for x in checkpoints)
 result['wall_seconds']=time.perf_counter()-start
 from auditor import audit
 cert=audit(result,plan,source_identity())
 if not cert['accepted']:raise AssertionError(cert)
 with gzip.open(result_path,'xt') as f:json.dump(result,f)
 result_path.with_suffix('.audit.json').write_text(json.dumps(cert,indent=2))
 return result_path

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--world',type=int,required=True);p.add_argument('--config',choices=CONFIGS,required=True);p.add_argument('--out',required=True);args=p.parse_args()
 with threadpool_limits(limits=1):run(args.world,args.config,args.out)
