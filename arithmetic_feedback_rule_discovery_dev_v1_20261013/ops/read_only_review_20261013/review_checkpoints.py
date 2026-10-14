"""Inference/algebra only on trusted saved technical checkpoints; no updates."""
import json,gzip,sys
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from core import Core,EOS
from interface import ReadOnly

WORLD=202610130103
RECEIPTS=ROOT/'cycles/cycle3/final_receipts'
OUT=Path(__file__).parent
CONFIGS=('R1-FULL','R4-FULL','F1-FULL','F4-FULL','R1-DIAG','R4-DIAG')

def rows(core,queries):
    found=[]
    digest=core.learning_digest()
    with ReadOnly(core):
        for item in queries:
            core.begin()
            for b in bytes.fromhex(item['prompt_hex']):core.observe(b,False)
            core.start_reply()
            own,status=core.generate(16)
            events=[]
            for t in list(bytes.fromhex(item['answer_hex']))+[EOS]:
                choice,p,z=core.predict();u,_=core.features()
                other=z.copy();other[t]=-np.inf
                events.append({'target':t,'choice':choice,'margin':float(z[t]-np.max(other)),
                    'logit':float(z[t]),'target_probability':float(p[t]),'eos_probability':float(p[EOS]),
                    'u':u,'step_size':float(u@(core.P@u if core.config.full else core.P*u))})
                core.observe(t,False)
            found.append({'item':item,'own_hex':own.hex(),'status':status,'events':events})
    assert core.learning_digest()==digest
    return found

def summary(core,queries):
    r=rows(core,queries);ev=[e for x in r for e in x['events']]
    first=[x['events'][0] for x in r];payload=[e for e in ev if e['target']!=EOS];ends=[e for e in ev if e['target']==EOS]
    U=np.array([e['u'] for e in first]);s=np.linalg.svd(U-U.mean(axis=0),compute_uv=False)
    # Counts describe inference, never feed the evaluator metadata into the core.
    def rate(z):return sum(e['choice']==e['target'] for e in z)/len(z)
    dist=[];posdist={str(i):[] for i in range(4)}
    for i in range(len(r)):
        a=bytes.fromhex(r[i]['item']['prompt_hex'])
        for j in range(i):
            b=bytes.fromhex(r[j]['item']['prompt_hex']);d=float(np.linalg.norm(U[i]-U[j]));dist.append(d)
            if len(a)==len(b)==4:
                diff=[k for k in range(4) if a[k]!=b[k]]
                if len(diff)==1:posdist[str(diff[0])].append(d)
    eig=np.linalg.eigvalsh(core.P) if core.config.full else core.P
    earray=np.array([e['step_size'] for e in ev])
    return {'queries':len(r),'own_exact':sum(x['own_hex']==x['item']['answer_hex'] and x['status']=='EOS' for x in r),
        'teacher_forced_all_steps_exact':sum(all(e['choice']==e['target'] for e in x['events']) for x in r),
        'teacher_forced_byte_accuracy':rate(ev),'teacher_forced_payload_accuracy':rate(payload),
        'teacher_forced_eos_accuracy':rate(ends),'first_byte_accuracy':rate(first),
        'empty_outputs':sum(x['own_hex']=='' for x in r),'truncated_outputs':sum(x['status']!='EOS' for x in r),
        'first_step_eos_choices':sum(e['choice']==EOS for e in first),
        'first_step_mean_eos_probability':float(np.mean([e['eos_probability'] for e in first])),
        'first_step_mean_target_probability':float(np.mean([e['target_probability'] for e in first])),
        'mean_feature_cosine':float(np.mean(U@U.T)),'mean_pair_distance':float(np.mean(dist)),
        'single_position_changed_feature_distance':{k:float(np.mean(v)) if v else None for k,v in posdist.items()},
        'feature_centered_singular_values':s.tolist(),'feature_effective_rank':float((s*s).sum()**2/((s**4).sum()+1e-30)),
        'confidence_eigen_min':float(eig.min()),'confidence_eigen_max':float(eig.max()),'confidence_trace':float(eig.sum()),
        'prediction_feature_uncertainty_median':float(np.median(earray)),
        'prediction_feature_uncertainty_min':float(earray.min()),'prediction_feature_uncertainty_max':float(earray.max()),
        'first_byte_true_minus_runner_up_median':float(np.median([e['margin'] for e in first])),
        'teacher_forced_event_loss_mean':float(np.mean([.5*sum((np.eye(257)[e['target']]-e['u']@core.V)**2) for e in ev])),
        'teacher_forced_confusion':{str(t):{str(c):sum(e['target']==t and e['choice']==c for e in ev) for c in sorted(set(e['choice'] for e in ev if e['target']==t))} for t in sorted(set(e['target'] for e in ev))}},r

def feature_input_sensitivity(core,queries):
    """Analytical local feature Jacobians wrt actual byte embeddings, no updates."""
    def jac(cache):
        h,e,z,r,g,rh=cache;t=core.theta
        dz=z*(1-z);dr=r*(1-r);dg=1-g*g
        Ah=np.diag(1-z)+(g-h)[:,None]*dz[:,None]*t['Uz']+z[:,None]*dg[:,None]*(t['Ug']@(np.diag(r)+h[:,None]*dr[:,None]*t['Ur']))
        Ae=(g-h)[:,None]*dz[:,None]*t['Wz']+z[:,None]*dg[:,None]*(t['Wg']+t['Ug']@(h[:,None]*dr[:,None]*t['Wr']))
        return Ah,Ae
    norms=[]
    for item in queries:
        if len(bytes.fromhex(item['prompt_hex']))!=4:continue
        core.begin()
        for b in bytes.fromhex(item['prompt_hex']):core.observe(b,False)
        core.start_reply()
        u,processing=core.features();v=np.r_[core._processed_h(processing),1.]
        downstream=((np.eye(65)-np.outer(u,u))/np.linalg.norm(v))[:,:64]
        for cache in reversed(processing):downstream=downstream@jac(cache)[0]
        h=np.zeros(64);caches=[]
        for token in [257]+list(bytes.fromhex(item['prompt_hex']))+[258]:
            h,cache=core.transition(h,core.theta['E'][token]);caches.append(cache)
        local=[]
        for cache in reversed(caches):
            Ah,Ae=jac(cache);local.append(float(np.linalg.norm(downstream@Ae)));downstream=downstream@Ah
        norms.append(list(reversed(local)))
    return np.array(norms).mean(axis=0).tolist()

def main():
    report={'schema':'RAW_RULE_READ_ONLY_CHECKPOINT_REVIEW_V1','scope':'one reserved final technical world; inference and algebra only; no training or causal mechanism ablation','world':WORLD,'budgets':{}}
    for name in CONFIGS:
        receipt=json.loads(gzip.open(RECEIPTS/f'{WORLD}_{name}.json.gz','rt').read())
        hist=receipt['histories']['W'];curve=[]
        for ck in hist['checkpoints']:
            n=ck['n_per_operation'];c=Core.load(RECEIPTS/f'{WORLD}_{name}_W_{n}_root.pkl')
            result,queried=summary(c,ck['taught_pre']);result['n_per_operation']=n
            result['median_training_encoder_update_norm']=float(np.median([r['encoder_update']['encoder_update_norm'] for r in hist['root_records'][:ck['root_records']] if r['encoder_update'] is not None])) if name.startswith('R') else 0.
            curve.append(result)
            if n==128:
                allrows=[dict(prompt_hex=r['prompt_hex'],answer_hex=r['answer_hex']) for r in hist['root_records']]
                observed=rows(c,allrows);events=[e for x in observed for e in x['events']]
                X=np.array([e['u'] for e in events]);Y=np.zeros((len(events),257));Y[np.arange(len(events)),[e['target'] for e in events]]=1
                A=.1*np.eye(65)+X.T@X;B=X.T@Y
                xeig=np.linalg.eigvalsh(X.T@X)
                result['current_basis_observation_ridge_degrees_freedom']=float(np.sum(xeig/(xeig+.1)))
                result['current_basis_observation_gram_eigenvalues']=xeig.tolist()
                result['fixed_current_basis_normal_equation_relative_residual']=float(np.linalg.norm(A@c.V-B)/(np.linalg.norm(B)+1e-30))
                if c.config.full:
                    result['fixed_current_basis_confidence_inverse_relative_residual']=float(np.linalg.norm(A@c.P-np.eye(65))/np.sqrt(65))
                # Current-state cue retention across time at same key; no relabeling/fitting.
                initial=Core.load(RECEIPTS/f'{WORLD}_{name}_N_OLD_final.pkl')
                birthrows=rows(initial,ck['taught_pre'])
                U0=np.array([r['events'][0]['u'] for r in birthrows]);Ut=np.array([r['events'][0]['u'] for r in queried])
                result['mean_cue_feature_displacement_from_birth']=float(np.linalg.norm(Ut-U0,axis=1).mean())
                result['encoder_relative_parameter_displacement_from_birth']=float(np.sqrt(sum(((c.theta[k]-initial.theta[k])**2).sum() for k in c.theta)/sum((initial.theta[k]**2).sum() for k in c.theta)))
                result['mean_feature_embedding_jacobian_norm_by_token_position']=feature_input_sensitivity(c,ck['taught_pre'])
        report['budgets'][name]=curve
    (OUT/'CHECKPOINT_REVIEW.json').write_text(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    with threadpool_limits(1):report=main()
    for name,curve in report['budgets'].items():
        end=curve[-1]
        print(name,json.dumps({k:end[k] for k in ('queries','own_exact','teacher_forced_all_steps_exact','teacher_forced_payload_accuracy','teacher_forced_eos_accuracy','first_byte_accuracy','empty_outputs','mean_feature_cosine','feature_effective_rank','single_position_changed_feature_distance','confidence_trace','prediction_feature_uncertainty_median','mean_cue_feature_displacement_from_birth','fixed_current_basis_normal_equation_relative_residual')}))
