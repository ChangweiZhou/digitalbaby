"""Technical score tables and fixed future gates. Never launches trajectories."""
import numpy as np
from audit import countermodels
def metrics(receipt):
    out={'world':receipt['world'],'arm':receipt['arm'],'dose':receipt['dose'],'checkpoints':{},'cpu':{},'state_bytes':0,'diagnostics':{'closed_kcs':0,'attenuated_kcs':0,'full_gate_kcs':0,'commits':0,'content_l1':0.,'auxiliary_l1':0.,'attenuation_l1':0.,'transfer_l1':0.,'clipped':0}}
    for cp in ('OLD_END','DAY1','NEW_END','FINAL'):
        out['checkpoints'][cp]={b:receipt['branches'][b]['probes'][cp]['groups'] for b in ('W','N_OLD','SHUFFLED')}
    for branch,o in receipt['branches'].items():
        out['cpu'][branch]=sum(c['core_cpu_seconds'] for c in o['costs'].values())
        out['state_bytes']=o['states']['FINAL']['payload_bytes']
        if branch=='W':
            for traces in o['training'].values():
                for rec in traces:
                    for row in rec['trace']:
                        u=row['update']
                        if u:
                            for k in ('commits','content_l1','auxiliary_l1','attenuation_l1','transfer_l1','clipped'):out['diagnostics'][k]+=u[k]
                            for k,v in zip(('closed_kcs','attenuated_kcs','full_gate_kcs'),u['gate_counts']):out['diagnostics'][k]+=v
    out['countermodels']=countermodels(receipt['master'],receipt['dose'])
    return out
def full_guards(rows,candidate):
    c=[r for r in rows if r['arm']==candidate and r['dose']==40];b=[r for r in rows if r['arm']=='STABLE' and r['dose']==40]
    if not c or len(b)!=len(c):raise ValueError('paired complete dose 40 required')
    means=lambda rr,key,branch='W':float(np.mean([r['checkpoints']['FINAL'][branch][key] for r in rr]))
    guards={}
    for key in ('old','new','held'):
        guards[key+'_absolute']=means(c,key)>=.75
        guards[key+'_vs_stable']=means(c,key)>=means(b,key)-.05-1e-12
    guards['no_old_gain']=means(c,'held')-means(c,'held','N_OLD')>=.2-1e-12
    guards['shuffled_gain']=means(c,'held')-means(c,'held','SHUFFLED')>=.2-1e-12
    shortcuts=float(np.mean([max(r['countermodels'].values()) for r in c]))
    guards['shortcut_gap']=means(c,'held')-shortcuts>=.1-1e-12
    return guards
