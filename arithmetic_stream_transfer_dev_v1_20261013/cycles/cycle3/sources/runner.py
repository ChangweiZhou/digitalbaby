"""B complete technical jobs; no DEV/confirmation or background science admission."""
import argparse,gzip,hashlib,json,time
from pathlib import Path
from threadpoolctl import threadpool_limits
from interface import birth,learning_record,probe,digest,CONFIGS,counters
from worlds import make_world,PASSES,seed

def source_identity():
    root=Path(__file__).parent
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob('*.py'))}
def run(label,name,out):
    plan=make_world(label);source=source_identity();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    dest=out/(label.split('/')[-1]+'_'+name+'.json.gz')
    if dest.exists():raise FileExistsError('technical receipt exists; never overwrite or repeat it')
    all_bytes=name.startswith('ALL');start=time.perf_counter();job_start=time.process_time()
    cost={'probe_cpu':0.,'snapshot_cpu':0.}
    def probes(c,q):
        t=time.process_time();v=probe(c,q);cost['probe_cpu']+=time.process_time()-t;return v
    def clone(c):
        t=time.process_time();v=c.clone();cost['snapshot_cpu']+=time.process_time()-t;return v
    def record(c,q,allowed):
        t=time.process_time();r=learning_record(c,bytes.fromhex(q['prompt_hex']),bytes.fromhex(q['answer_hex']),allowed,all_bytes)
        elapsed=time.process_time()-t;r['cpu_seconds']=elapsed;return r,elapsed
    t=time.process_time();base=birth(name,seed(label));birth_cpu=time.process_time()-t
    initial={op:probes(base,plan['tasks'][op]['held']) for op in ('+','*')}
    counting=[];count_cpu=birth_cpu
    for q in plan['counting']:
        r,elapsed=record(base,q,True);count_cpu+=elapsed;counting.append(r)
    count_digest=digest(base);count_mul=probes(base,plan['tasks']['*']['held'])
    result={'schema':'STREAM_TRANSFER_TECHNICAL_V1','technical':True,'label':label,'config':name,
        'source_identity':source,'stream_digest':plan['stream_digest'],'initial_held':initial,
        'counting':{'records':counting,'cpu':count_cpu,'final':count_digest,'multiplication_zero_shot':count_mul},
        'addition':{},'multiplication':{},'record_applications':128}
    addition_states={}
    def execute(c,history,phase,prefix_cpu):
        op='+' if phase=='addition' else '*';task=plan['tasks'][op]
        allowed=history!=('N_ADD' if phase=='addition' else 'N_MUL')
        shuffle=history==('SH_ADD' if phase=='addition' else 'SH_MUL')
        initial_digest=digest(c);records=[];checkpoints=[];root_cpu=0.;position=0
        for pass_n in PASSES:
            stop=pass_n*70
            while position<stop:
                q=dict(task['records'][position])
                if shuffle:q['answer_hex']=task['shuffled_answers'][position]
                r,elapsed=record(c,q,allowed);root_cpu+=elapsed;records.append(r);position+=1
                if position%700==0:
                    print(json.dumps({'technical':True,'label':label,'config':name,'phase':phase,'history':history,
                        'completed_records':position,'required_records':17920}),flush=True)
            parent=digest(c)
            if not allowed and parent!=initial_digest:raise AssertionError('disabled phase altered persistent state')
            hp=probes(c,task['held']);tp=probes(c,task['taught']);fork=clone(c);fi=digest(fork)
            cohort_pre=probes(fork,plan['cohort_pairs']);fr=[];fork_cpu=0.
            for q in plan['cohort_records']:
                r,elapsed=record(fork,q,True);fork_cpu+=elapsed;fr.append(r)
            cohort_post=probes(fork,plan['cohort_pairs']);post=probes(fork,task['held'])
            fork.rest(86400);retained=probes(fork,task['held'])
            cp={'pass':pass_n,'root_records':len(records),'root_cpu':root_cpu,'prefix_cpu':prefix_cpu,
                'fork_cpu':fork_cpu,'scenario_cpu':prefix_cpu+root_cpu+fork_cpu,
                'parent_before':parent,'parent_after':digest(c),'fork_initial':fi,
                'cohort_records':fr,'held_pre':hp,'held_post':post,'held_retained':retained,'taught_pre':tp,
                'cohort_pre':cohort_pre,'cohort_post':cohort_post}
            if phase=='multiplication':cp['addition_retained']=probes(fork,plan['tasks']['+']['held'])
            else:cp['multiplication_zero_shot']=probes(c,plan['tasks']['*']['held'])
            checkpoints.append(cp);result['record_applications']+=len(fr)
            from analysis import rate
            print(json.dumps({'technical':True,'label':label,'config':name,'phase':phase,'history':history,
                'pass':pass_n,'taught':rate(tp),'held_retained':rate(retained),'root_cpu':root_cpu}),flush=True)
        result['record_applications']+=len(records)
        return {'initial':initial_digest,'final':digest(c),'final_counters':counters(c),
                'records':records,'checkpoints':checkpoints,'cpu':root_cpu}
    for history in ('W','N_ADD','SH_ADD'):
        c=clone(base);result['addition'][history]=execute(c,history,'addition',count_cpu);addition_states[history]=c
    if digest(addition_states['N_ADD'])!=digest(base):raise AssertionError('scratch state parity failed')
    result['scratch_state_parity']={'accepted':True,'counting':digest(base),'after_disabled_addition':digest(addition_states['N_ADD'])}
    for history in ('W','N_ADD','SH_ADD','N_MUL','SH_MUL'):
        parent=history if history in ('N_ADD','SH_ADD') else 'W';c=clone(addition_states[parent])
        prefix=count_cpu+result['addition'][parent]['cpu']
        result['multiplication'][history]=execute(c,history,'multiplication',prefix)
        if history=='W':
            t=time.process_time();c.save(out/(label.split('/')[-1]+'_'+name+'_W_final.pkl'));cost['snapshot_cpu']+=time.process_time()-t
    from auditor import audit
    t=time.process_time();certificate=audit(result,source);cost['audit_cpu']=time.process_time()-t
    if source_identity()!=source:raise AssertionError('source changed during technical job')
    production=count_cpu+sum(h['cpu']+sum(cp['fork_cpu'] for cp in h['checkpoints']) for phase in ('addition','multiplication') for h in result[phase].values())
    result['resource']={'wall_seconds':time.perf_counter()-start,'total_job_cpu':time.process_time()-job_start,
        'production_cpu':production,**cost}
    result['resource']['other_cpu']=result['resource']['total_job_cpu']-production-sum(cost.values())
    commit=time.process_time()
    with gzip.open(dest,'xt') as f:json.dump(result,f)
    certificate['resource']=dict(result['resource'],commit_cpu=time.process_time()-commit)
    dest.with_suffix('.audit.json').write_text(json.dumps(certificate,indent=2)+'\n')
    return dest
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--config',choices=CONFIGS,required=True);p.add_argument('--out',required=True)
    a=p.parse_args()
    with threadpool_limits(limits=1):run(a.label,a.config,a.out)
