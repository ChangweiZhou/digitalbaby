"""Independent reduction of saved vectors and event ledger; no native model queries or teaching."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import json,time,hashlib,math,copy
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;R=H.parent;D=R/'minifly_v151_design_20261025'
load=lambda p:json.loads(Path(p).read_text())
def save(n,x):(H/n).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def center(v):return v-v.mean(axis=-1,keepdims=True)
def grad(C,v,y):
    z=C@center(v)/4;z-=z.max();p=np.exp(z)/np.exp(z).sum();r=p.copy();r[y]-=1
    return center(C.T@r/4),np.outer(r,center(v)/4)
def bps(C,v,y):
    z=C@center(v)/4;lp=z-z.max()-np.log(np.exp(z-z.max()).sum())
    return float(-lp[y]/np.log(2)),int(np.argmax(lp))
def project(C):
    C=center(C);return C/np.maximum(1,np.linalg.norm(C,axis=1,keepdims=True)/4)
def warmup(n):return 1/16+15/16*min(n/1024,1)
def close(a,b):
    err=float(np.max(abs(np.asarray(a)-np.asarray(b))))
    assert err<1e-10,err
    return err

def main():
    begin=time.process_time();spec=load(D/'SPEC.json');assets=load(D/'ASSETS.json')['anchors'];sites=load(D/'SITES.json')['texts'];execution=load(H/'EXECUTION_SUMMARY.json');pref=load(H/'CYCLE1_PREFLIGHT.json');gain=load(H/'GAIN_LOCK.json')
    expected={(a['id'],s['site_id']) for a in assets if a['primary'] for s in sites[a['text']]['sites']}
    ops=[load(p) for p in H.glob('OP_*.json')];rules=[load(p) for p in H.glob('RULE_*.json')];codes=[load(p) for p in H.glob('CODE_*.json')]
    def valid_roster(rows):
        try:return len(rows)==36 and len({(x['anchor'],x['site']) for x in rows})==36 and {(x['anchor'],x['site']) for x in rows}==expected and all(np.asarray(x['K_half']).shape==(4,16,16) and np.isfinite(x['K_half']).all() for x in rows)
        except (KeyError,ValueError,TypeError):return False
    assert valid_roster(ops)
    bad=[ops[:-1],ops+[ops[0]]]
    b=copy.deepcopy(ops);b[0]['site']='UNKNOWN';bad.append(b)
    b=copy.deepcopy(ops);b[0]['K_half'][0][0][0]=float('nan');bad.append(b)
    b=copy.deepcopy(ops);b[0]['K_half']=[0];bad.append(b)
    assert not any(valid_roster(x) for x in bad)
    assert len(rules)==36 and {(x['anchor'],x['site']) for x in rules}==expected
    assert len(codes)==8
    words=load(R/'minifly_cls_dev_20261017/OUTPUT_CODEBOOK.json')['words_by_class'];C=np.array([[1. if w&(1<<j) else -1. for j in range(16)] for w in words])
    maxerr=0.;opmap={(x['anchor'],x['site']):x for x in ops};amap={x['id']:x for x in assets};smap={s['site_id']:s for z in sites.values() for s in z['sites']}
    for o in ops:
        z=np.array(o['zero_values']);assert z.shape==(4,16) and len(o['pulse_rows'])==64
        close(o['zero_repeat_values'],z);close(o['passive_values'],z)
        for h,key in [(.001,'K_h'),(.0005,'K_half')]:
            est=np.zeros((4,16,16))
            for j in range(16):
                pp=next(q for q in o['pulse_rows'] if q['h']==h and q['j']==j and q['sign']==1)
                mm=next(q for q in o['pulse_rows'] if q['h']==h and q['j']==j and q['sign']==-1)
                est[:,:,j]=(np.array(pp['vectors'])-np.array(mm['vectors']))/(2*h)
            maxerr=max(maxerr,close(o[key],est))
        n=amap[o['anchor']]['training_targets'];assert o['n']==n and o['warmup']==warmup(n)
        v=np.array(o['pre_values']);y=smap[o['site']]['source']['target'];scale=.1*warmup(n)
        maxerr=max(maxerr,close(o['directions']['A_COMMON'],scale*(np.tanh(center(v)/2)-C[y])/2),close(o['directions']['B_COMMON'],scale*grad(C,v,y)[0]))
    means=[];saved_conditions=[]
    for a in [x for x in assets if x['primary']]:
        rr=[x for x in rules if x['anchor']==a['id']];cal=[x for x in rr if x['phase']=='CALIBRATION'];scored=[x for x in rr if x['phase']=='SCORED'];assert len(cal)==2 and len(scored)==4
        ag=next(x for x in gain['anchors'] if x['anchor']==a['id']);ae=be=0.
        for q in cal:
            o=opmap[(q['anchor'],q['site'])];z=np.array(o['zero_values'])[:1]
            assert [r['arm'] for r in q['conditions']]==['A_COMMON','B_COMMON']
            for r in q['conditions']:
                assert 'delta_BPS' not in r and 'after' not in r
                dv=np.array(r['vectors'])-z;close(dv,r['delta_v']);close(np.linalg.norm(center(dv),axis=1),r['centered_displacement_norms'])
                if r['arm']=='A_COMMON':ae+=float(np.linalg.norm(center(dv)))**2
                else:be+=float(np.linalg.norm(center(dv)))**2
        close(ae,ag['A_displacement_energy']);close(be,ag['B_displacement_energy'])
        if ag['lambda'] is not None:close(ag['lambda'],np.sqrt(ae/be))
        for q in scored:
            o=opmap[(q['anchor'],q['site'])];site=smap[q['site']];ys=[site['source']['target'],site['source']['target'],site['peers']['same_target']['target'],site['peers']['different_target']['target']];z=np.array(o['zero_values'])
            assert [r['arm'] for r in q['conditions']]==['A_COMMON','B_COMMON','B_MATCHED']
            for r in q['conditions']:
                if r['status']!='COMPLETE':continue
                vv=np.array(r['vectors']);dv=vv-z;coef=np.array(r['coefficient']);predict=np.array(o['K_half'])@coef
                expectcoef=np.array(o['directions']['A_COMMON' if r['arm']=='A_COMMON' else 'B_COMMON'])*(ag['lambda'] if r['arm']=='B_MATCHED' else 1)
                maxerr=max(maxerr,close(expectcoef,coef),close(dv,r['delta_v']),close(predict,r['predicted_delta_v']))
                for i,(v,y) in enumerate(zip(vv,ys)):
                    b,choice=bps(C,v,y);base,_=bps(C,z[i],y)
                    maxerr=max(maxerr,close(b-base,r['delta_BPS'][i]),close(float(grad(C,z[i],y)[0]@predict[i]/np.log(2)),r['delta_BPS_first_order'][i]))
                    assert choice==r['after'][i]['own_class']
                saved_conditions.append({'anchor':a['id'],'site':q['site'],'arm':r['arm'],'delta_BPS':r['delta_BPS'],
                     'displacement_norms':r['centered_displacement_norms'],'coefficient_energy':r['coefficient_energy'],
                     'prediction_residual_max':r['delta_v_linear_residual_max'],'loss_prediction_residual':r['delta_BPS_linear_residual'],
                     'KC_Jaccard':o['KC_Jaccard_with_immediate_source'],'active_KCs':o['active_KCs']})
        for arm in ['A_COMMON','B_COMMON','B_MATCHED']:
            rows=[r for r in saved_conditions if r['anchor']==a['id'] and r['arm']==arm]
            if rows:means.append({'anchor':a['id'],'arm':arm,'sites':len(rows),'mean_delta_BPS':np.mean([r['delta_BPS'] for r in rows],axis=0).tolist(),
                     'self_improved_sites':sum(r['delta_BPS'][0]<0 for r in rows),'mean_displacement':np.mean([r['displacement_norms'] for r in rows],axis=0).tolist(),
                     'mean_coefficient_energy':float(np.mean([r['coefficient_energy'] for r in rows])),
                     'max_value_prediction_residual':max(r['prediction_residual_max'] for r in rows)})
    cs=[]
    for q in codes:
        c=np.array(q['code_before']);pre=np.array(q['pre_values']);s=smap[q['site']];y=s['source']['target'];ys=[y,y,s['peers']['same_target']['target'],s['peers']['different_target']['target']]
        dc=project(c-.1/16*warmup(q['n'])*grad(c,pre,y)[1])-c
        base=np.array(q['branches'][0]['vectors'])
        for r in q['branches']:
            d=np.array(r['delta_C']);vv=np.array(r['vectors']);expect=dc if r['condition'] in ['C_ONLY','JOINT'] else np.zeros_like(dc)
            maxerr=max(maxerr,close(d,expect))
            for i,(v,yy) in enumerate(zip(vv,ys)):
                new,_=bps(c+d,v,yy);old,_=bps(c,base[i],yy)
                predicted=(grad(c,base[i],yy)[0]@(v-base[i])+np.sum(grad(c,base[i],yy)[1]*d))/np.log(2)
                maxerr=max(maxerr,close(new-old,r['delta_BPS'][i]),close(predicted,r['delta_BPS_first_order'][i]))
        cs.append({'anchor':q['anchor'],'site':q['site'],'code_step_norm':q['projected_code_step_norm'],
                   'projection_difference_norm':q['projection_difference_norm'],'interaction_delta_BPS':q['interaction_delta_BPS'],
                   'effects':{r['condition']:r['delta_BPS'] for r in q['branches']}})
    events=[]
    for name in ['LEDGER.jsonl','TECH_CHILD_LEDGER.jsonl']:
        events.extend(json.loads(line) for line in (H/name).read_text().splitlines())
    branches=[x for x in events if x['event']=='BRANCH'];assert len({x['id'] for x in branches})==len(branches)
    counter=execution['counts'];assert len(branches)==counter['branch_events']
    assert sum(x['observed_target'] for x in branches)==counter['target_applications']
    assert sum(x['nonzero_compartments'] for x in branches)==counter['nonzero_compartment_increments']
    assert sum(x['signed_API_calls'] for x in branches)==counter['signed_API_calls']
    counts=spec['counts_caps']
    for key,cap in [('branch_events','branch_events_including_passive_max'),('target_applications','total_observed_target_applications'),('signed_API_calls','signed_API_calls_including_zero_max'),('nonzero_compartment_increments','total_nonzero_compartment_write_increments_max'),('native_vector_readouts','native_vector_readouts_max')]:assert counter[key]<=counts[cap]
    assert counter['partial_coordinate_pulse_branches']==2304
    assert counter['native_vector_readouts']==36*67*4+24+sum(x['status']=='COMPLETE' for r in rules if r['phase']=='SCORED' for x in r['conditions'])*4+8*4*4+44+pref['preflight_readout_vectors']
    evmain=[json.loads(x) for x in (H/'LEDGER.jsonl').read_text().splitlines()]
    locked=next(i for i,x in enumerate(evmain) if x['event']=='GAIN_LOCK')
    assert not any('_score_' in x.get('id','') for x in evmain[:locked])
    assert sha(H/'GAIN_LOCK.json')==execution['gain_lock_sha256']
    for path,expected_sha in load(H/'SOURCE_LOCK.json').items():assert sha(R/path)==expected_sha,path
    for a in assets:
        if 'checkpoint' in a:assert sha(R/a['checkpoint'])==a['sha256']
    compact={'version':'V151','query_order':['immediate_self','reset_self','same_target_peer','different_target_peer'],'anchor_means':means,
             'per_site_conditions':saved_conditions,'C_factorial':cs,'operator_summary':{'sites':36,'max_scale_disagreement':max(x['scale_difference_max'] for x in ops),'max_offdiagonal':max(x['offdiagonal_max'] for x in ops),'max_zero_repeat_error':max(x['zero_repeat_error'] for x in ops),'max_zero_passive_error':max(x['passive_error'] for x in ops)}}
    save('COMPACT.json',compact)
    audit={'version':'V151','status':'THREE_CYCLES_COMPLETE','exact_operator_sites':36,'exact_rule_sites':36,'exact_C_factorial_sites':8,
           'max_reduction_error':maxerr,'five_corrupt_rosters_rejected':True,'source_and_anchor_hashes_match':True,
           'gain_locked_before_scored_interventions':True,'calibration_loss_not_computed_for_fitting':True,'no_trajectory_replay':True,
           'preflight_resume_equal':pref['cross_process_resume_equal'],'counts':counter,'full_training_histories':0,'addition_records':0,
           'new_core_adopted':False,'addition_efficacy_established':False,'CPU':time.process_time()-begin}
    save('FINAL_AUDIT.json',audit)
    print(json.dumps({'status':'THREE_CYCLES_COMPLETE','max_error':maxerr,'means':means,'operator':compact['operator_summary']},indent=2))
if __name__=='__main__':main()
