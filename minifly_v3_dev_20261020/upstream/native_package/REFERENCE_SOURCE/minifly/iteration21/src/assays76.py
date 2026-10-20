"""Frozen tasks and evaluator-only outcomes. No evaluation labels enter updates."""
from common76 import *
from model76 import PairLearner

def stream(seed,task):
    rng=np.random.default_rng(seed+({'ANOMALY':110000,'REVERSAL':220000}[task]));base=rng.integers(0,2,64)
    out=[]
    for t in range(CFG['trials']):
        idx=int(rng.integers(64));clean=int(base[idx])
        if task=='REVERSAL':clean^=(t//256)%2
        label=clean
        if task=='ANOMALY' and rng.random()<.1:label=1-label
        out.append((idx,label,clean))
    final=base.copy()
    if task=='REVERSAL':final^=((CFG['trials']-1)//256)%2
    return out,final

def measured(l,codes,roles,h=0):
    a,r,_=l.read(codes,h);d=2*np.asarray(roles)-1
    a=a*d;r=r*d;b=(a>=1)&(r>=l.reader.pair_threshold)
    aa,rr,_=l.read(codes,h,erase_alpha_fast=True)
    erased=(d*aa>=1)&(d*rr>=l.reader.pair_threshold)
    return a,r,b,erased

def row_metrics(a,r,b,erased):
    return dict(joint_accuracy=float(b.mean()),reader_margin=float(r.mean()),actual_margin=float(a.mean()),
        reader_threshold_pass=float((r>=1).mean()),actual_threshold_pass=float((a>=1).mean()),
        below_1Hz=float((abs(r)<1).mean()),erase_alpha_fast_joint=float(erased.mean()),
        alpha_fast_dependent=float((b&~erased).mean()))

def run_bridge(spec,folder):
    seed=spec['seed'];pn,roles=bank(seed);template=Fly();codes=np.array([template.encode(p) for p in pn])
    rows=[];epochs=[];ends=[];diags=[];vectors={};checks=[]
    for task in CFG['tasks']:
        events,final_roles=stream(seed,task)
        for arm in CFG['arms']:
            l=PairLearner(arm);online=[]
            for t,(idx,label,clean) in enumerate(events):
                cc=codes[2*idx:2*idx+2];a,r,_=l.read(cc)
                if t>=CFG['warmup']:
                    d=2*clean-1;aa=float(a[0]*d);rr=float(r[0]*d)
                    online.append(dict(epoch=t//256,force_accuracy=float((r[0]>=0)==bool(clean)),
                        joint_accuracy=float(aa>=1 and rr>=l.reader.pair_threshold),
                        actual_margin=aa,reader_margin=rr,below_1Hz=float(abs(rr)<1)))
                l.learn_pair(cc,label)
            tab=pd.DataFrame(online);vals=tab.drop(columns='epoch').mean().to_dict()
            rows.append(dict(seed=seed,task=task,arm=arm,n_scored=len(tab),**vals))
            for epoch,g in tab.groupby('epoch'):epochs.append(dict(seed=seed,task=task,arm=arm,epoch=int(epoch),**g.drop(columns='epoch').mean().to_dict()))
            diags.append(dict(seed=seed,task=task,arm=arm,**l.diagnostic()))
            native_end=float(l.m.elapsed);before=state_arrays(l.m).copy()
            for h in CFG['horizons']:
                a,r,b,e=measured(l,codes[:128],final_roles,h)
                ends.append(dict(seed=seed,task=task,arm=arm,phase='END',horizon=h,**row_metrics(a,r,b,e)))
                vectors[f'{task}_{arm}_END_{h}']=np.stack([a,r,b,e])
            assert np.array_equal(before,state_arrays(l.m)) and l.m.elapsed==native_end
            no=l.clone();mixed=l.clone()
            for k in range(108,120):
                for _ in range(CFG['acquisition_bouts']):
                    mixed.learn_pair(codes[2*k:2*k+2],int(roles[k]));no.learn_pair(codes[2*k:2*k+2],int(roles[k]),False)
            for h in CFG['horizons']:
                a,r,b,e=measured(mixed,codes[:128],final_roles,h);na,nr,nb,_=measured(no,codes[:128],final_roles,h)
                damage=np.maximum(na-a,0);new=measured(mixed,codes[216:240],roles[108:120],h)[2]
                ends.append(dict(seed=seed,task=task,arm=arm,phase='MIXED',horizon=h,**row_metrics(a,r,b,e),
                    mixed_damage=float(damage.mean()),mixed_worst=float(damage.max()),new_accuracy=float(new.mean())))
                vectors[f'{task}_{arm}_MIXED_{h}']=np.stack([a,r,b,e,na,nr,nb])
            checks.append(dict(task=task,arm=arm,elapsed_end=native_end,elapsed_mixed=float(mixed.m.elapsed),
                identical_no_write_adaptation=bool(np.array_equal(mixed.m.adapt,no.m.adapt))))
            assert native_end==CFG['trials']*330
            assert mixed.m.elapsed==native_end+12*CFG['acquisition_bouts']*330
    pd.DataFrame(rows).to_csv(folder/'bridge_online.csv',index=False);pd.DataFrame(epochs).to_csv(folder/'bridge_epochs.csv',index=False)
    pd.DataFrame(ends).to_csv(folder/'bridge_endpoints.csv',index=False);pd.DataFrame(diags).to_csv(folder/'diagnostics.csv',index=False)
    np.savez_compressed(folder/'bridge_vectors.npz',**vectors);atomic_json(folder/'checks.json',checks)
    return dict(seed=seed,online_rows=len(rows),endpoint_rows=len(ends))

def labels_for(role,case,bouts):
    goal=1-role if case=='REV' else role
    labels=[goal]*bouts
    if case=='ANOMALY1':labels[0]=1-role
    return labels,goal

def run_transfer(spec,folder):
    seed=spec['seed'];pn,roles=bank(seed);parent=PairLearner('REF');codes=np.array([parent.m.encode(p) for p in pn])
    rows=[];caps=[];vectors={};checks=[];diags=[]
    for j in range(96):
        for _ in range(CFG['acquisition_bouts']):parent.learn_pair(codes[2*j:2*j+2],int(roles[j]))
        load=j+1
        if load not in CFG['loads']:continue
        parent.m.save(folder/f'prestate_{load}.npz')
        for h in CFG['horizons']:
            a,r,b,e=measured(parent,codes[:2*load],roles[:load],h)
            caps.append(dict(seed=seed,load=load,horizon=h,**row_metrics(a,r,b,e)))
        initial=measured(parent,codes[:2*load],roles[:load])[2]
        for focal in [load//2-1,load-1]:
            mask=np.arange(load)!=focal;refs={}
            for case in CFG['cases']:
                variants=[(a,CFG['challenge_bouts'],a) for a in CFG['transfer_arms']]+[('REF',CFG['native_dose_anchor'],'REF18')]
                for arm,bouts,name in variants:
                    l=PairLearner(arm,prestate=parent.m);no=PairLearner('REF',prestate=parent.m)
                    before=sha(folder/f'prestate_{load}.npz')
                    labels,goal=labels_for(int(roles[focal]),case,bouts);pr=roles[:load].copy();pr[focal]=goal
                    for label in labels:
                        l.learn_pair(codes[2*focal:2*focal+2],label);no.learn_pair(codes[2*focal:2*focal+2],label,False)
                    branches=[('LAST',l.clone(),no.clone())]
                    l.rest((CFG['native_dose_anchor']-bouts)*330);no.rest((CFG['native_dose_anchor']-bouts)*330)
                    branches.append(('COMMON',l.clone(),no.clone()))
                    diags.append(dict(seed=seed,load=load,focal=focal,case=case,arm=name,bouts=bouts,phase='CHALLENGE',**l.diagnostic()))
                    for k in range(108,120):
                        for _ in range(CFG['acquisition_bouts']):
                            l.learn_pair(codes[2*k:2*k+2],int(roles[k]));no.learn_pair(codes[2*k:2*k+2],int(roles[k]),False)
                    branches.append(('MIXED',l,no))
                    for phase,q,z in branches:
                        assert np.array_equal(q.m.adapt,z.m.adapt)
                        expected=parent.m.elapsed+(bouts if phase=='LAST' else CFG['native_dose_anchor'])*330
                        if phase=='MIXED':expected+=12*CFG['acquisition_bouts']*330
                        assert q.m.elapsed==z.m.elapsed==expected
                        for h in CFG['horizons']:
                            a,r,b,e=measured(q,codes[:2*load],pr,h);na,nr,nb,_=measured(z,codes[:2*load],pr,h)
                            key=(case,phase,h)
                            if name=='REF':refs[key]=(a.copy(),b.copy())
                            ra,rb=refs[key];damage=np.maximum(na[mask]-a[mask],0);direct=np.maximum(ra[mask]-a[mask],0)
                            row=dict(seed=seed,load=load,focal=focal,age='NEWEST' if focal==load-1 else 'MIDDLE',case=case,arm=name,bouts=bouts,
                                phase=phase,horizon=h,elapsed=float(q.m.elapsed+h),target_actual=float(a[focal]),target_reader=float(r[focal]),
                                target_choice=float(b[focal]),target_without_alpha_fast=float(e[focal]),
                                old_accuracy=float(b[mask].mean()),old_loss=float((initial[mask]&~b[mask]).mean()),
                                old_damage=float(damage.mean()),old_worst=float(damage.max()),direct_worst_vs_REF12=float(direct.max()),
                                direct_mean_vs_REF12=float(direct.mean()),old_accuracy_REF12=float(rb[mask].mean()),
                                new_accuracy=float(measured(q,codes[216:240],roles[108:120],h)[2].mean()) if phase=='MIXED' else np.nan)
                            rows.append(row);vk=f'{load}_{focal}_{case}_{name}_{phase}_{h}'
                            vectors[vk]=np.stack([a,r,b,e,na,nr,nb,ra,rb,initial])
                    assert sha(folder/f'prestate_{load}.npz')==before
                    checks.append(dict(load=load,focal=focal,case=case,arm=name,bouts=bouts,
                        shared_prestate_sha=before,common_elapsed=float(parent.m.elapsed+CFG['native_dose_anchor']*330),
                        mixed_elapsed=float(l.m.elapsed)))
    pd.DataFrame(rows).to_csv(folder/'transfer.csv',index=False);pd.DataFrame(caps).to_csv(folder/'capacity.csv',index=False)
    pd.DataFrame(diags).to_csv(folder/'diagnostics.csv',index=False);np.savez_compressed(folder/'transfer_vectors.npz',**vectors)
    atomic_json(folder/'checks.json',checks)
    return dict(seed=seed,rows=len(rows),expected_rows=2*2*3*9*3*2)
