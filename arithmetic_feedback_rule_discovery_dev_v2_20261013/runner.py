"""Reserved complete technical jobs only. DEV/confirmation are not admitted."""
import argparse,copy,gzip,hashlib,json,time
from pathlib import Path
from threadpoolctl import threadpool_limits
from interface import birth,learning_record,probe,digest,CONFIGS
from worlds import make_world,CHECKPOINTS
from vendor.task_source import prompt,answer

def source_identity():
    root=Path(__file__).parent
    files=list(root.glob('*.py'))+list((root/'vendor').rglob('*.py'))
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
def run(world,name,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);dest=out/f'{world}_{name}.json.gz'
    if dest.exists():raise FileExistsError('reserved receipt already exists')
    source=source_identity();plan=make_world(world,True)
    held=[{'prompt_hex':prompt(r).encode().hex(),'answer_hex':answer(r).encode().hex(),'operation':r[0]} for r in plan['held_pool']]
    receipt={'schema':'RAW_RULE_V2_TECHNICAL','technical':True,'world':world,'config':name,
             'source_identity':source,'stream_digest':plan['stream_digest'],'histories':{},'record_applications':0}
    start=time.perf_counter();job_cpu=time.process_time();cost={'probe_cpu':0.,'snapshot_cpu':0.}
    def probes(c,queries):
        t=time.process_time();rows=probe(c,queries);cost['probe_cpu']+=time.process_time()-t;return rows
    for history in ('W','N_OLD','SHUFFLE_OLD'):
        t=time.process_time();c=birth(name,world);root_cpu=time.process_time()-t
        initial=digest(c);initial_held=probes(c,held);records=[];checkpoints=[]
        for stop in CHECKPOINTS:
            for row in [r for r in plan['records'] if r['checkpoint']==stop]:
                target=row['answer'] if history!='SHUFFLE_OLD' else plan['shuffled_answers'][str(row['index'])]
                t=time.process_time();rec=learning_record(c,row['prompt'].encode(),target.encode(),history!='N_OLD')
                elapsed=time.process_time()-t;root_cpu+=elapsed;rec['cpu_seconds']=elapsed;records.append(rec)
            parent=digest(c)
            if history=='N_OLD' and parent!=initial:raise AssertionError('N_OLD learned state changed')
            taught={r['prompt']:r for r in plan['records'] if r['checkpoint']<=stop}
            tq=[{'prompt_hex':r['prompt'].encode().hex(),'answer_hex':r['answer'].encode().hex(),'operation':r['operation']} for r in taught.values()]
            hp=probes(c,held);tp=probes(c,tq)
            t=time.process_time();fork=c.clone() if hasattr(c,'clone') else copy.deepcopy(c);cost['snapshot_cpu']+=time.process_time()-t
            fi=digest(fork);cohort_pre=probes(fork,plan['cohort_pairs']);fr=[];fork_cpu=0.
            for q in plan['cohort_records']:
                t=time.process_time();rec=learning_record(fork,bytes.fromhex(q['prompt_hex']),bytes.fromhex(q['answer_hex']),True)
                elapsed=time.process_time()-t;rec['cpu_seconds']=elapsed;fork_cpu+=elapsed;fr.append(rec)
            cohort_post=probes(fork,plan['cohort_pairs']);post=probes(fork,held);fork.rest(86400);retained=probes(fork,held)
            tr=probes(fork,tq)
            checkpoints.append({'budget':stop,'root_records':len(records),'root_cpu':root_cpu,'fork_cpu':fork_cpu,
              'scenario_cpu':root_cpu+fork_cpu,'parent_before':parent,'parent_after':digest(c),'fork_initial':fi,
              'cohort_records':fr,'held_pre':hp,'held_post':post,'held_retained':retained,
              'taught_pre':tp,'taught_retained':tr,'cohort_pre':cohort_pre,'cohort_post':cohort_post})
            print(json.dumps({'technical':True,'world':world,'config':name,'history':history,'n':stop,
                'taught_correct':sum(r['status']=='EOS' and r['payload_hex']==r['answer_hex'] for r in tp),'taught_total':len(tp),
                'held_correct':sum(r['status']=='EOS' and r['payload_hex']==r['answer_hex'] for r in retained),'held_total':len(held)}),flush=True)
            if history=='W' and stop==CHECKPOINTS[-1]:
                t=time.process_time()
                if hasattr(c,'save'):c.save(out/f'{world}_{name}_W_final.pkl')
                else:c.save_checkpoint(out/f'{world}_{name}_W_final.npz')
                cost['snapshot_cpu']+=time.process_time()-t
        receipt['histories'][history]={'initial':initial,'final':digest(c),'initial_held':initial_held,'records':records,'checkpoints':checkpoints}
        receipt['record_applications']+=len(records)+sum(len(x['cohort_records']) for x in checkpoints)
    from auditor import audit
    t=time.process_time();certificate=audit(receipt,source);cost['audit_cpu']=time.process_time()-t
    if source_identity()!=source:raise AssertionError('source edited during job')
    receipt['resource']={'wall_seconds':time.perf_counter()-start,'total_job_cpu':time.process_time()-job_cpu,**cost}
    receipt['resource']['production_cpu']=sum(h['checkpoints'][-1]['root_cpu']+sum(c['fork_cpu'] for c in h['checkpoints']) for h in receipt['histories'].values())
    receipt['resource']['other_cpu']=receipt['resource']['total_job_cpu']-receipt['resource']['production_cpu']-sum(cost.values())
    commit=time.process_time()
    with gzip.open(dest,'xt') as f:json.dump(receipt,f)
    certificate['resource']=dict(receipt['resource'],commit_cpu=time.process_time()-commit)
    dest.with_suffix('.audit.json').write_text(json.dumps(certificate,indent=2)+'\n')
    return dest
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--world',type=int,required=True);parser.add_argument('--config',choices=CONFIGS,required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=1):run(args.world,args.config,args.out)
