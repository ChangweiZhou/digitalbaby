"""Shared schedules and paired measurements. Metadata never reaches the model."""
from common import *
from smallfly import SmallFly,FrozenReader
import argparse,time,resource,csv

def measure_pair(m,A,B,role,reader,serial=False):
    scores=[];memory=[];mse=[];clips=[];active=[]
    for p in (A,B):
        n=m if serial else m.clone();r=n.step(5.,p)
        dx=r['sensory_activity'];observed=r['rates'][6:8]
        predicted=reader.predict(dx);zero=analytic_baseline(dx@n.Q,n.k0,n.MM)
        scores.append(float((observed-predicted).mean()));memory.append(float((observed-zero).mean()))
        mse.append(float(np.mean((predicted-zero)**2)));active.append(r['active_KCs'])
        clips.append(float(np.mean((observed<=-11.2+1e-10)|(observed>=19.96-1e-10))))
    delta=scores[0]-scores[1];actual=memory[0]-memory[1];direction=2*role-1
    return dict(score_Hz=delta,memory_Hz=actual,margin_Hz=direction*delta,oracle_margin_Hz=direction*actual,
        correct=direction*delta>=reader.pair_threshold,oracle_correct=direction*actual>=1.,
        abstain=abs(delta)<reader.pair_threshold,single_decisions=sum(abs(s)>=reader.cue_threshold for s in scores),
        baseline_mse=float(np.mean(mse)),clipped_fraction=float(np.mean(clips)),mean_active_KCs=float(np.mean(active)))

def run_model(path,stage):
    start=time.perf_counter();template=SmallFly(path);reader=FrozenReader(template.Q);bankfile=np.load(DATA/'input_banks.npz')
    rows=[];traces=[];maxbytes=0
    def record(m,b,pair,role,seed,task,phase,load=0,delay=0,schedule='snapshot',dose=-1,**extra):
        values=measure_pair(m,b[2*pair],b[2*pair+1],int(role),reader,serial=schedule=='serial')
        rows.append(dict(model=template.model_name,stage=stage,stream=seed,task=task,phase=phase,load=load,delay=delay,
            schedule=schedule,dose=dose,pair=pair,role=int(role),old_correct=False,old_oracle_correct=False,**values))
        rows[-1].update(extra)
        return values
    seeds=CONFIG['screen_stream_seeds'] if stage=='screen' else CONFIG['confirmation_stream_seeds']
    for seed in seeds:
        b=bankfile[f'{stage}_{seed}'];roles=bankfile[f'roles_{seed}'];m=template.clone();null=template.clone();null.mode='no_learning'
        for k in range(96):
            for dt,p,pun in events(b[2*k],b[2*k+1],int(roles[k])):
                m.step(dt,p,pun);null.step(dt,p,pun)
            load=k+1
            if load not in CONFIG['loads']:continue
            assert m.elapsed==load*1980 and null.elapsed==m.elapsed
            for delay in CONFIG['delays_seconds']:
                n=m.clone();n.step(delay);z=null.clone();z.step(delay)
                for pair in range(load):record(n,b,pair,roles[pair],seed,'load','loaded',load,delay)
                online=n.clone()
                for pair in range(load):record(online,b,pair,roles[pair],seed,'load','loaded',load,delay,'serial')
                assert online.event_count-n.event_count==2*load
                for pair in range(96,104):record(n,b,pair,0,seed,'novel','untrained',load,delay)
                for pair in list(range(8))+list(range(96,104)):record(z,b,pair,0,seed,'null','no_learning',load,delay)
        for pair in range(CONFIG['isolated_pairs']):
            isolated=template.clone();train(isolated,b[2*pair],b[2*pair+1],int(roles[pair]))
            for delay in CONFIG['delays_seconds']:
                n=isolated.clone();n.step(delay);record(n,b,pair,roles[pair],seed,'isolated','isolated',1,delay)
            for load in CONFIG['age_control_loads']:
                n=isolated.clone();n.step(86400+1980*(load-1-pair));record(n,b,pair,roles[pair],seed,'age_control','isolated_same_age',load,86400)
        # A separate serial prelude reproduces the deployed reversal challenge.
        m=template.clone()
        for pair in range(4):train(m,b[2*pair],b[2*pair+1],int(roles[pair]))
        for pair in range(4):measure_pair(m,b[2*pair],b[2*pair+1],int(roles[pair]),reader,True)
        for pair in range(4,12):train(m,b[2*pair],b[2*pair+1],int(roles[pair]))
        for pair in range(4):measure_pair(m,b[2*pair],b[2*pair+1],int(roles[pair]),reader,True)
        before=measure_pair(m,b[0],b[1],int(roles[0]),reader)
        states={0:m.clone()};newrole=1-int(roles[0])
        for bout in range(13):
            if bout:
                train(m,b[0],b[1],newrole,1)
                if bout in CONFIG['revision_doses']:states[bout]=m.clone()
            target=measure_pair(m,b[0],b[1],newrole,reader)
            traces.append(dict(model=template.model_name,stage=stage,stream=seed,bout=bout,
                margin_Hz=target['margin_Hz'],oracle_margin_Hz=target['oracle_margin_Hz'],
                slow_norm=float(np.linalg.norm(m.slow)),fast_alpha_norm=float(np.linalg.norm(m.fast[:,1])),
                old_correct=before['correct'],old_oracle_correct=before['oracle_correct']))
        for dose,state in states.items():
            for phase,rest in [('immediate',0),('day',330*(12-dose)+86400)]:
                n=state.clone()
                if rest:n.step(rest)
                for pair in range(12):
                    role=newrole if pair==0 else int(roles[pair])
                    record(n,b,pair,role,seed,'revision' if pair==0 else 'collateral',phase,12,rest,'serial',dose,
                        old_correct=before['correct'],old_oracle_correct=before['oracle_correct'])
        maxbytes=max(maxbytes,m.mutable_bytes());assert maxbytes==32*len(template.ids)+24
    d=pd.DataFrame(rows);dest=DATA/stage;dest.mkdir(exist_ok=True)
    d.to_csv(dest/(template.model_name+'.csv'),index=False)
    pd.DataFrame(traces).to_csv(dest/(template.model_name+'_trajectory.csv'),index=False)
    receipt=dict(model=template.model_name,stage=stage,rows=len(d),seconds=time.perf_counter()-start,
        max_RSS_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),mutable_bytes=maxbytes,model_sha=template.model_sha)
    (dest/(template.model_name+'_run.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['screen','confirmation'],default='screen')
    ap.add_argument('--model');ap.add_argument('--worker',type=int,default=0);ap.add_argument('--workers',type=int,default=1);args=ap.parse_args()
    inventory=pd.read_csv(OUT/('model_inventory.csv' if args.stage=='screen' else 'confirmation_model_inventory.csv'))
    if args.stage=='confirmation':inventory=pd.concat([pd.read_csv(OUT/'model_inventory.csv').query("method == 'reference'"),inventory],ignore_index=True)
    if args.model:inventory=inventory[inventory.model==args.model]
    inventory=inventory.iloc[args.worker::args.workers]
    for row in inventory.itertuples():
        if not row.valid:continue
        if sha(MODELS/(row.model+'.npz'))!=row.sha256:raise ValueError('Fixed model artifact changed')
        run_model(MODELS/(row.model+'.npz'),args.stage)

if __name__=='__main__':main()
