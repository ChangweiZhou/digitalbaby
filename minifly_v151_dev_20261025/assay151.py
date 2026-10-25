"""Paid, discard-only counterfactuals on pinned native states. No native backpropagation."""
import sys, pickle, time
import numpy as np
from support151 import H,R,D,COUNTS,TIMES,save,load,sha,clone,digest,prompt,values,prepared,centered,intervene,query_vectors,budget,ledger
sys.path.insert(0,str(R/'minifly_v150_dev_20261025'))
import native
from core import gradients, probabilities, project, warmup
sys.path.insert(0,str(R/'minifly_v150_dev_20261025/vendor'))
import corpus

ASSETS=load(D/'ASSETS.json')['anchors']
SITES=load(D/'SITES.json')['texts']
RECORDS={name:corpus.records(corpus.load(name))[:z['training_record_count']] for name,z in SITES.items()}

def get_anchor(info):
    if info['kind']=='reconstructed_canonical_birth': m=native.Native('A',.1)
    else:
        p=R/info['checkpoint']; assert sha(p)==info['sha256']; m=pickle.loads(p.read_bytes())
    assert m.target_n==info['training_targets'] and not m.homeo and m.fresh_adapt
    assert not np.any(m.theta) and m.front=='perm'
    assert np.max(np.linalg.norm(m.C,axis=1))<=4+1e-12
    return m

def active_count(pre):
    f=pre.fe.clone(); f.advance(pre.time+native.fastmem.DELTA)
    return int(pre.encode(f.read())[0].sum())

def source(anchor,info,site):
    e=site['source'];rec=RECORDS[info['text']][e['record_index']];prefix=rec[:e['target_position']]
    import hashlib
    assert len(prefix)==e['prefix_length'] and hashlib.sha256(prefix).hexdigest()==e['prefix_sha256']
    assert e['target']==(rec[e['target_position']] if e['target_position']<len(rec) else 256)
    return prepared(anchor,prefix)

def teaching(pre,v,y):
    p,_=probabilities(pre.C,v); residual=p.copy(); residual[y]-=1
    g=centered(pre.C.T@residual/4);gc=np.outer(residual,centered(v)/4)
    ea=(np.tanh(centered(v)/2)-pre.C[y])/2
    scale=.1*warmup(pre.target_n)
    return {'A_COMMON':scale*ea,'B_COMMON':scale*g}, -(scale/16)*gc

def describe(C,vs,ys):
    rows=[]
    for v,y in zip(vs,ys):
        p,lp=probabilities(C,v);z=C@centered(v)/4;span=float(np.ptp(z));r=float(np.sqrt(np.mean(centered(v)**2)))
        rows.append({'BPS':float(-lp[y]/np.log(2)),'own_class':int(p.argmax()),'target':y,
                     'p_target':float(p[y]),'p_max':float(p.max()),'centered_RMS':r,'logit_span':span,
                     'pmax_bound_from_span':float(1/(1+256*np.exp(-span))),
                     'pmax_bound_from_RMS':float(1/(1+256*np.exp(-8*r)))})
    return rows

def query_targets(site):
    return [site['source']['target'],site['source']['target'],site['peers']['same_target']['target'],site['peers']['different_target']['target']]

def operator_site(anchor,info,site,phase):
    budget(60); t=time.process_time();pre=source(anchor,info,site);before=digest(pre);vpre=values(pre)
    recs=RECORDS[info['text']]; zero=np.zeros(16)
    z=intervene(pre,zero,ident=info['id']+'|'+site['site_id']+'|ZERO')
    vz,keys=query_vectors(z,site,recs,with_keys=True)
    zr=intervene(pre,zero,ident=info['id']+'|'+site['site_id']+'|ZERO_REPEAT')
    vr=query_vectors(zr,site,recs)
    pa=intervene(pre,zero,passive=True,ident=info['id']+'|'+site['site_id']+'|PASSIVE')
    vp=query_vectors(pa,site,recs)
    repeat_error=float(np.max(abs(vr-vz))); passive_error=float(np.max(abs(vp-vz)))
    assert repeat_error<=1e-12 and passive_error<=1e-12
    functional=functional_error_local(z,pa)
    assert functional<=1e-12
    K=[]; pulse_rows=[]
    for h in [.001,.0005]:
        mat=np.zeros((4,16,16))
        for j in range(16):
            vecs=[]
            for sign in [1,-1]:
                a=np.zeros(16);a[j]=sign*h
                q=intervene(pre,a,ident=f"{info['id']}|{site['site_id']}|PULSE_{h}_{j}_{sign}")
                got=query_vectors(q,site,recs);vecs.append(got)
                pulse_rows.append({'h':h,'j':j,'sign':sign,'vectors':got})
                COUNTS['partial_coordinate_pulse_branches']+=1
            mat[:,:,j]=(vecs[0]-vecs[1])/(2*h)
        K.append(mat)
    dirs,_=teaching(pre,vpre,site['source']['target'])
    ys=query_targets(site)
    # Calibration native vectors are saved without computing any calibrated loss objective.
    row={'id':info['id']+'|'+site['site_id'],'anchor':info['id'],'site':site['site_id'],'phase':phase,
         'pre_values':vpre,'zero_values':vz,'zero_repeat_values':vr,'passive_values':vp,
         'KC_active_counts':[int(x.sum()) for x in keys],
         'KC_Jaccard_with_immediate_source':[float(np.minimum(keys[0],x).sum()/np.maximum(keys[0],x).sum()) if np.maximum(keys[0],x).sum()>0 else None for x in keys],'zero_repeat_error':repeat_error,'passive_error':passive_error,
         'passive_functional_error':functional,'K_h':K[0],'K_half':K[1],
         'scale_difference_max':float(np.max(abs(K[0]-K[1]))),
         'offdiagonal_max':float(max(np.max(abs(mat-np.stack([np.diag(np.diag(a)) for a in mat]))) for mat in K)),
         'n':pre.target_n,'warmup':warmup(pre.target_n),'directions':dirs,
         'active_KCs':active_count(pre),'prepared_state_hash':before,
         'pulse_rows':pulse_rows,'CPU':time.process_time()-t}
    assert digest(pre)==before
    save('OP_'+info['id']+'__'+site['site_id']+'.json',row)
    ledger('OPERATOR_SITE_COMMIT',id=row['id'],phase=phase,CPU_site=row['CPU'])
    return pre,z,row

def functional_error_local(a,b):
    from support151 import functional_error
    return functional_error(a,b)

def rule_site(pre,z,op,info,site,phase,gain=None):
    recs=RECORDS[info['text']];dirs={k:np.array(v) for k,v in op['directions'].items()}
    if phase=='SCORED' and gain is not None: dirs['B_MATCHED']=gain*dirs['B_COMMON']
    ys=query_targets(site);vz=np.asarray(op['zero_values']);C=pre.C
    out=[]
    for arm,a in dirs.items():
        if np.max(abs(a))>1:
            out.append({'arm':arm,'status':'MATCH_UNIDENTIFIABLE_COEFFICIENT'});continue
        q=intervene(pre,a,observed=True,ident=info['id']+'|'+site['site_id']+'|'+arm)
        got=query_vectors(q,site,recs,immediate_only=phase=='CALIBRATION')
        dv=got-vz[:len(got)]
        plastic=sum(float(np.sum((s.fly.m.fast-t.fly.m.fast)**2)+np.sum((s.fly.m.slow-t.fly.m.slow)**2)) for s,t in zip(q.bank,z.bank))
        r={'arm':arm,'status':'COMPLETE','coefficient':a,'vectors':got,'delta_v':dv,
           'centered_displacement_norms':np.linalg.norm(centered(dv),axis=1),
           'coefficient_energy':float(a@a),'plastic_L2_squared':plastic}
        if phase=='SCORED':
            assert (H/'GAIN_LOCK.json').is_file()
            base=describe(C,vz,ys);after=describe(C,got,ys)
            prediction=np.asarray(op['K_half'])@a
            r.update(baseline=base,after=after,
                     delta_BPS=[b['BPS']-a0['BPS'] for a0,b in zip(base,after)],
                     predicted_delta_v=prediction,
                     delta_v_linear_residual_max=float(np.max(abs(dv-prediction))),
                     delta_BPS_first_order=[float(gradients(C,v,y)[0]@d/np.log(2)) for v,y,d in zip(vz,ys,prediction)])
            r['delta_BPS_linear_residual']=[x-y for x,y in zip(r['delta_BPS'],r['delta_BPS_first_order'])]
            r['delta_logit_prediction_residual_max']=float(np.max(abs((centered(dv-prediction)/4)@C.T)))
            r['descent_alignment']=[float(-gradients(C,v,y)[0]@d/(np.linalg.norm(gradients(C,v,y)[0])*np.linalg.norm(d))) if np.linalg.norm(gradients(C,v,y)[0])*np.linalg.norm(d)>0 else None for v,y,d in zip(vz,ys,dv)]
        out.append(r)
    if phase=='SCORED' and gain is None:out.append({'arm':'B_MATCHED','status':'MATCH_UNIDENTIFIABLE_CALIBRATION'})
    row={'anchor':info['id'],'site':site['site_id'],'phase':phase,'conditions':out,'C_hash':__import__('hashlib').sha256(C.tobytes()).hexdigest()}
    save('RULE_'+info['id']+'__'+site['site_id']+'.json',row)
    ledger('RULE_SITE_COMMIT',id=info['id']+'|'+site['site_id'],phase=phase)
    return row

def gains(cal_rows):
    out=[]
    for info in ASSETS:
        if not info['primary']: continue
        rows=[r for r in cal_rows if r['anchor']==info['id']]
        A=sum(float(np.asarray(r['conditions'][0]['centered_displacement_norms'])[0])**2 for r in rows)
        B=sum(float(np.asarray(r['conditions'][1]['centered_displacement_norms'])[0])**2 for r in rows)
        # Bound numerical floor using repeated-zero error and float64 scale, fixed before scored outcomes.
        floor2=0.;nonnull=0
        for r in rows:
            op=load(H/('OP_'+r['anchor']+'__'+r['site']+'.json'))
            v=np.asarray(op['zero_values'])[0]
            floor=max(4*np.sqrt(16)*op['zero_repeat_error'],100*np.finfo(float).eps*max(1,float(np.linalg.norm(v))))
            floor2+=floor**2
            nonnull+=float(r['conditions'][1]['centered_displacement_norms'][0])>floor
        lam=np.sqrt(A/B) if B>floor2 and A>floor2 else None
        valid=lam is not None and np.isfinite(lam) and .25<=lam<=4
        out.append({'anchor':info['id'],'lambda':float(lam) if valid else None,'status':'LOCKED' if valid else 'MATCH_UNIDENTIFIABLE',
                    'A_displacement_energy':A,'B_displacement_energy':B,'floor_energy':floor2,'nonnull_calibration_sources':nonnull})
    save('GAIN_LOCK.json',{'version':'V151','status':'LOCKED_BEFORE_SCORED_INTERVENTIONS','rule':'calibration immediate centered displacement only','anchors':out,'scored_branches_executed':0,'CPU_at_lock':__import__('support151').cpu()})
    ledger('GAIN_LOCK',sha256=sha(H/'GAIN_LOCK.json'))
    return {r['anchor']:r['lambda'] for r in out}

def code_site(anchor,info,site):
    budget(30);pre=source(anchor,info,site);before=digest(pre);vpre=values(pre);y=site['source']['target']
    dirs,raw_delta=teaching(pre,vpre,y);dc=project(pre.C+raw_delta)-pre.C
    recs=RECORDS[info['text']];zero=np.zeros(16);ys=query_targets(site)
    branches={'ZERO':(zero,None,False),'V_ONLY':(dirs['B_COMMON'],None,True),'C_ONLY':(zero,dc,True),'JOINT':(dirs['B_COMMON'],dc,True)}
    rows=[];baseline=None;vzero=None
    for name,(coef,delta,obs) in branches.items():
        q=intervene(pre,coef,observed=obs,delta_C=delta,ident=info['id']+'|'+site['site_id']+'|'+name)
        vs=query_vectors(q,site,recs);desc=describe(q.C,vs,ys)
        if name=='ZERO':baseline=desc;vzero=vs
        diff=vs-vzero
        dc_actual=q.C-pre.C
        first=[float((gradients(pre.C,v,yy)[0]@d+np.sum(gradients(pre.C,v,yy)[1]*dc_actual))/np.log(2)) for v,yy,d in zip(vzero,ys,diff)]
        rows.append({'condition':name,'vectors':vs,'description':desc,'delta_v':diff,'delta_BPS':[b['BPS']-a['BPS'] for a,b in zip(baseline,desc)],
                     'delta_BPS_first_order':first,'code_step_norm':float(np.linalg.norm(dc_actual)),'delta_C':dc_actual})
    row={'anchor':info['id'],'site':site['site_id'],'branches':rows,'raw_code_step_norm':float(np.linalg.norm(raw_delta)),
         'projected_code_step_norm':float(np.linalg.norm(dc)),'code_before':pre.C,'pre_values':vpre,'n':pre.target_n,'projection_difference_norm':float(np.linalg.norm(dc-raw_delta)),
         'interaction_delta_BPS':np.array(rows[3]['delta_BPS'])-np.array(rows[1]['delta_BPS'])-np.array(rows[2]['delta_BPS'])}
    assert digest(pre)==before
    save('CODE_'+info['id']+'__'+site['site_id']+'.json',row)
    ledger('CODE_SITE_COMMIT',id=info['id']+'|'+site['site_id'])
    return row
