"""Retrospective V66 design checks on V65 screening cues; no confirmatory run.

Uses frozen parent code, saves only new review outputs, and never changes inputs.
"""
from pathlib import Path
import sys, json, hashlib, time, csv
ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
sys.path.insert(0, str(BASE / 'iteration10/src'))
from common import np, train, events, state_error, RetentionLearner
from smallfly import SmallFly, FrozenReader
from experiment import measure_pair
from scipy.stats import spearmanr

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def state(m):
    return np.column_stack((m.fast, m.slow))

def cosine(a, b):
    den = np.linalg.norm(a, axis=0)*np.linalg.norm(b, axis=0)
    return np.divide((a*b).sum(0), den, out=np.full(3, np.nan), where=den>0)

def rest(m, h):
    n = m.clone()
    if h: n.step(float(h))
    return n

def probe_parts(m, p):
    x = m.encode(p)
    end = 1-(1-m.adapt*np.exp(-.05*5*x))*np.exp(-5/m.ta)
    dx = .5*(m.adapt+end)*x
    q = dx[:,None]*m.Q
    gamma_in = (q[:,:2]*(m.k0[3]+m.fast[:,0,None])).sum(0)
    gamma = np.clip(gamma_in, -35.2, 36.46)
    alpha_in = (q[:,2:]*(m.k0[5]+(m.fast[:,1]+m.slow)[:,None])).sum(0)+gamma@m.MM
    alpha = np.clip(alpha_in, -11.2, 19.96)
    return q, gamma_in, alpha_in, alpha

def sensitivity(m, A, B, role):
    """Gradient at the no-write state, including both clip levels."""
    coeff=[]
    for p in (A,B):
        q, gi, ai, _ = probe_parts(m,p)
        ga=((gi>-35.2)&(gi<36.46)).astype(float)
        aa=((ai>-11.2)&(ai<19.96)).astype(float)
        cg=q[:,:2]@(ga*(m.MM@aa)/2)
        ca=q[:,2:]@(aa/2)
        coeff.append(np.column_stack((cg,ca,ca)))
    return (2*role-1)*(coeff[0]-coeff[1])

def linear_projection(m, A, B, role, delta):
    vals=[]
    for p in (A,B):
        q,_,_,_=probe_parts(m,p)
        dg=(q[:,:2]*delta[:,0,None]).sum(0)
        da=(q[:,2:]*(delta[:,1]+delta[:,2])[:,None]).sum(0)+dg@m.MM
        vals.append(da.mean())
    return float((2*role-1)*(vals[0]-vals[1]))

def write_csv(name, rows):
    with (ROOT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def main():
    start=time.perf_counter()
    modelpath=BASE/'iteration10/models/reference.npz'
    headpath=BASE/'iteration8/models/current_RBF64.npz'
    inputpath=BASE/'iteration10/data/input_banks.npz'
    sources=[modelpath,headpath,inputpath,BASE/'iteration5/src/hybrid.py',BASE/'iteration10/src/smallfly.py',
             BASE/'iteration10/src/experiment.py',BASE/'iteration10/data/screen/reference.csv',
             BASE/'iteration9/results/focal_choice_and_distortion.csv',BASE/'iteration4/results/fitted_kernels.json']
    before={str(p.relative_to(BASE)):sha(p) for p in sources}
    banks=np.load(inputpath,allow_pickle=False)
    template=SmallFly(modelpath); reader=FrozenReader(template.Q)
    config=json.loads((BASE/'iteration10/config.json').read_text())
    check=dict(adapt_max_error=0., counters_max_error=0., reader_actual_effect_max_error=0.,
               off_support_effect_max_error=0., on_support_reconstruction_max_error=0.,
               expression_reconstruction_max_error=0., parent_state_max_error=0., parent_rate_max_error=0.,
               decay_reconstruction_max_error=0., frozen_result_max_error=0.)
    atlas=[]; roleswap=[]; focals=[]
    for seed in config['screen_stream_seeds']:
        b=banks[f'screen_{seed}']; roles=banks[f'roles_{seed}']
        m=template.clone(); original=RetentionLearner()
        for j in range(4):
            for dt,p,u in events(b[2*j],b[2*j+1],int(roles[j])):
                v=m.step(dt,p,u); old=original.step(dt,p,u)
                check['parent_rate_max_error']=max(check['parent_rate_max_error'],float(np.max(abs(v['rates']-old['rates']))))
        check['parent_state_max_error']=max(check['parent_state_max_error'],state_error(m,original))
        checkpoint=m.clone(); footprints=[]; eligible=[]
        for j in range(4):
            omit=template.clone()
            for k in range(4):
                omit.mode='no_learning' if k==j else 'full'
                train(omit,b[2*k],b[2*k+1],int(roles[k]))
            fp=state(m)-state(omit); footprints.append(fp)
            q=measure_pair(m,b[2*j],b[2*j+1],int(roles[j]),reader)
            eligible.append(bool(q['correct'] and q['oracle_correct']))
            focals.append(dict(stream=seed,focal=j,eligible=eligible[-1],margin_Hz=q['oracle_margin_Hz']))
        # The first 32 already-used later cue pairs, both orientations, checkpoint fixed.
        for k in range(4,36):
            pair=[]
            for role in (0,1):
                w=checkpoint.clone();n=checkpoint.clone();n.mode='no_learning'
                train(w,b[2*k],b[2*k+1],role);train(n,b[2*k],b[2*k+1],role)
                ds=state(w)-state(n);wh=rest(w,86400);nh=rest(n,86400)
                effects=[];cs=[]
                for j in range(4):
                    mw=measure_pair(wh,b[2*j],b[2*j+1],int(roles[j]),reader)
                    mn=measure_pair(nh,b[2*j],b[2*j+1],int(roles[j]),reader)
                    c=cosine(footprints[j],ds)
                    row=dict(stream=seed,candidate=k,focal=j,role=role,eligible=eligible[j],
                             cosine_fast0=c[0],cosine_fast1=c[1],cosine_slow=c[2],
                             cosine_mean=float(np.nanmean(c)),E24_Hz=mw['oracle_margin_Hz']-mn['oracle_margin_Hz'],
                             pregradient_Hz=float((sensitivity(nh,b[2*j],b[2*j+1],int(roles[j]))*(state(wh)-state(nh))).sum()),
                             write_norm=float(np.linalg.norm(ds)))
                    roleswap.append(row)
        # Ordinary saved V65 stream, with the proposed per-association twin audit.
        for k in range(4,96):
            w=m.clone();n=m.clone();n.mode='no_learning'
            train(w,b[2*k],b[2*k+1],int(roles[k]));train(n,b[2*k],b[2*k+1],int(roles[k]))
            ds=state(w)-state(n)
            check['adapt_max_error']=max(check['adapt_max_error'],float(np.max(abs(w.adapt-n.adapt))))
            check['counters_max_error']=max(check['counters_max_error'],*[float(abs(int(getattr(w,v))-int(getattr(n,v)))) for v in ('elapsed','event_count','presentation_count')])
            decay=np.exp(-86400/np.array([m.tg,m.kernel['fast_tau'],m.kernel['slow_tau']]))
            for h in (0,86400):
                wh=rest(w,h);nh=rest(n,h);dsh=state(wh)-state(nh)
                if h:check['decay_reconstruction_max_error']=max(check['decay_reconstruction_max_error'],float(np.max(abs(dsh-ds*decay))))
                for j in range(4):
                    A,B=b[2*j:2*j+2];r=int(roles[j])
                    mw=measure_pair(wh,A,B,r,reader);mn=measure_pair(nh,A,B,r,reader)
                    effect=mw['oracle_margin_Hz']-mn['oracle_margin_Hz']
                    check['reader_actual_effect_max_error']=max(check['reader_actual_effect_max_error'],abs(effect-(mw['margin_Hz']-mn['margin_Hz'])))
                    support=(m.encode(A)>0)|(m.encode(B)>0)
                    outside=nh.clone();outside.fast[~support]=wh.fast[~support];outside.slow[~support]=wh.slow[~support]
                    inside=nh.clone();inside.fast[support]=wh.fast[support];inside.slow[support]=wh.slow[support]
                    mo=measure_pair(outside,A,B,r,reader);mi=measure_pair(inside,A,B,r,reader)
                    check['off_support_effect_max_error']=max(check['off_support_effect_max_error'],abs(mo['oracle_margin_Hz']-mn['oracle_margin_Hz']))
                    check['on_support_reconstruction_max_error']=max(check['on_support_reconstruction_max_error'],abs(mi['oracle_margin_Hz']-mw['oracle_margin_Hz']))
                    reconstructed=(2*r-1)*((probe_parts(wh,A)[3]-probe_parts(nh,A)[3]).mean()-(probe_parts(wh,B)[3]-probe_parts(nh,B)[3]).mean())
                    check['expression_reconstruction_max_error']=max(check['expression_reconstruction_max_error'],abs(effect-reconstructed))
                    c=cosine(footprints[j],ds)
                    atlas.append(dict(stream=seed,focal=j,later=k+1,horizon=h,eligible=eligible[j],
                        E_Hz=effect,cosine_fast0=c[0],cosine_fast1=c[1],cosine_slow=c[2],
                        cosine_mean=float(np.nanmean(c)),pregradient_Hz=float((sensitivity(nh,A,B,r)*dsh).sum()),
                        unclipped_projection_Hz=linear_projection(nh,A,B,r,dsh),
                        write_norm=float(np.linalg.norm(ds)),margin_no_write_Hz=mn['oracle_margin_Hz']))
            m=w
        # Match the original saved +24h snapshot at load 96, with no intervening diagnostics.
        loaded=rest(m,86400)
        with (BASE/'iteration10/data/screen/reference.csv').open() as f:
            fixture=[row for row in csv.DictReader(f) if row['stream']==str(seed) and row['task']=='load' and row['load']=='96' and row['delay']=='86400' and row['schedule']=='snapshot']
        assert len(fixture)==96
        for row in fixture:
            j=int(row['pair']);now=measure_pair(loaded,b[2*j],b[2*j+1],int(roles[j]),reader)
            check['frozen_result_max_error']=max(check['frozen_result_max_error'],abs(now['oracle_margin_Hz']-float(row['oracle_margin_Hz'])))
        print(f'checked retrospective stream {seed}',flush=True)
    write_csv('retrospective_atlas.csv',atlas);write_csv('role_swap_feasibility.csv',roleswap);write_csv('focal_eligibility.csv',focals)
    delayed=[r for r in atlas if r['horizon']==86400 and r['eligible']]
    swaps=[r for r in roleswap if r['eligible']]
    grouped={}
    for r in swaps:grouped.setdefault((r['stream'],r['candidate'],r['focal']),[]).append(r)
    straddles={k:sum(v[0][k]*v[1][k]<0 for v in grouped.values()) for k in ('cosine_fast0','cosine_fast1','cosine_slow','cosine_mean','pregradient_Hz')}
    correlations=[]
    for seed in config['screen_stream_seeds']:
        rr=[r for r in delayed if r['stream']==seed]
        correlations.append(dict(stream=seed,**{feature:float(spearmanr([r[feature] for r in rr],[r['E_Hz'] for r in rr]).statistic) for feature in ('cosine_mean','cosine_slow','pregradient_Hz','unclipped_projection_Hz','write_norm')}))
    summary=dict(status='retrospective_design_feasibility_only',new_confirmatory_streams=0,
        streams=8,eligible_focals=sum(r['eligible'] for r in focals),total_focals=len(focals),atlas_rows=len(atlas),
        eligible_delayed_rows=len(delayed),damaging_delayed_rows=sum(r['E_Hz']<-.1 for r in delayed),
        reinforcing_delayed_rows=sum(r['E_Hz']>.1 for r in delayed),
        damaging_with_positive_mean_cosine=sum(r['E_Hz']<-.1 and r['cosine_mean']>0 for r in delayed),
        candidate_pairs_per_stream=32,eligible_candidate_focal_pairs=len(grouped),role_swap_straddles=straddles,
        delayed_E_quantiles=np.quantile([r['E_Hz'] for r in delayed],[0,.05,.25,.5,.75,.95,1]).tolist(),
        mean_cosine_range=[min(r['cosine_mean'] for r in swaps),max(r['cosine_mean'] for r in swaps)],
        correlations_by_stream=correlations,checks=check,
        kernel=dict(template.kernel,tg=template.tg,adapt_tau=template.ta),
        day_decay_factors=np.exp(-86400/np.array([template.tg,template.kernel['fast_tau'],template.kernel['slow_tau']])).tolist(),
        sources=before,sources_unchanged=before=={str(p.relative_to(BASE)):sha(p) for p in sources},
        seconds=time.perf_counter()-start)
    encoded=json.dumps(summary,indent=2,allow_nan=False,default=lambda x:x.item())
    (ROOT/'feasibility_summary.json').write_text(encoded+'\n')
    print(encoded)
    assert summary['sources_unchanged']
    assert max(check.values())<1e-9

if __name__=='__main__':main()
