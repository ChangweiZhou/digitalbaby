"""Post-hoc label-oracle least-squares diagnostics, not candidate learning/science."""
import json,sys,gzip
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from core import Core,EOS
from interface import ReadOnly
sys.path.insert(0,str(Path(__file__).parent))
from review_checkpoints import rows
OUT=Path(__file__).parent;R=ROOT/'cycles/cycle3/final_receipts';WORLD=202610130103

def fit(X,target,group,first,regularization):
    Y=np.zeros((len(target),257));Y[np.arange(len(target)),target]=1
    if regularization:
        V=np.linalg.solve(X.T@X+regularization*np.eye(X.shape[1]),X.T@Y)
    else:V=np.linalg.lstsq(X,Y,rcond=None)[0]
    prediction=np.argmax(X@V,axis=1);correct=prediction==target
    return {'oracle_ridge_lambda':regularization,'all_event_accuracy':float(correct.mean()),
            'first_byte_accuracy':float(correct[first].mean()),'eos_accuracy':float(correct[target==EOS].mean()) if np.any(target==EOS) else None,
            'all_teacher_forced_steps_exact':sum(bool(correct[np.array(group)==i].all()) for i in set(group)),
            'queries':len(set(group)),'rows':len(X),'feature_rank':int(np.linalg.matrix_rank(X)),
            'condition_number':float(np.linalg.cond(X)),'readout_norm':float(np.linalg.norm(V)),
            'mean_squared_residual':float(np.mean((X@V-Y)**2))}

def head_diagnostics(c,queries):
    observed=rows(c,queries);events=[e for q in observed for e in q['events']]
    X=np.array([e['u'] for e in events]);t=np.array([e['target'] for e in events]);g=np.array([i for i,q in enumerate(observed) for e in q['events']]);first=np.array([j==0 for q in observed for j,e in enumerate(q['events'])])
    out={'first_only':[],'joint_payload_and_eos':[]}
    for reg in (.1,0.):
        out['first_only'].append(fit(X[first],t[first],g[first],np.ones(sum(first),dtype=bool),reg))
        out['joint_payload_and_eos'].append(fit(X,t,g,first,reg))
    # Pair distances to another cue that requires a different first byte.
    F=X[first];labels=t[first];d=np.linalg.norm(F[:,None,:]-F[None,:,:],axis=-1);d[labels[:,None]==labels[None,:]]=np.inf
    out['nearest_conflicting_first_byte_feature_distance']={'minimum':float(d.min()),'median':float(np.median(d.min(axis=1))),'mean':float(np.mean(d.min(axis=1)))}
    return out

def prefix_features(c,queries):
    stages={'before_equal':[],'after_equal':[],'after_reply':[]}
    before=c.learning_digest()
    with ReadOnly(c):
        for q in queries:
            payload=bytes.fromhex(q['prompt_hex']);c.begin()
            for b in payload[:-1]:c.observe(b,False)
            stages['before_equal'].append(c.features()[0])
            c.observe(payload[-1],False);stages['after_equal'].append(c.features()[0])
            c.start_reply();stages['after_reply'].append(c.features()[0])
    assert c.learning_digest()==before
    result={};labels=np.array([bytes.fromhex(q['answer_hex'])[0] for q in queries]);groups=np.arange(len(queries))
    for name,features in stages.items():
        X=np.array(features);center=X-X.mean(0);s=np.linalg.svd(center,compute_uv=False)
        result[name]={'mean_cosine':float(np.mean(X@X.T)),'centered_effective_rank':float((s*s).sum()**2/(s**4).sum()),
            'oracle_first_only_ridge':fit(X,labels,groups,np.ones(len(X),dtype=bool),.1),
            'oracle_first_only_unregularized':fit(X,labels,groups,np.ones(len(X),dtype=bool),0.)}
    return result

def main():
    result={'scope':'post-hoc evaluator-label oracle diagnosis on saved checkpoints; no candidate/core mutation or scientific selection','world':WORLD,'configs':{}}
    for name in ('R1-FULL','R4-FULL','F1-FULL','F4-FULL'):
        c=Core.load(R/f'{WORLD}_{name}_W_128_root.pkl');receipt=json.loads(gzip.open(R/f'{WORLD}_{name}.json.gz','rt').read());ck=receipt['histories']['W']['checkpoints'][-1]
        result['configs'][name]={'unique_taught_contexts':head_diagnostics(c,ck['taught_pre']),
            'occurrence_weighted_training_contexts':head_diagnostics(c,[dict(prompt_hex=r['prompt_hex'],answer_hex=r['answer_hex']) for r in receipt['histories']['W']['root_records']]),
            'prefix_stages':prefix_features(c,ck['taught_pre'])}
        print(name,json.dumps(result['configs'][name]['unique_taught_contexts']))
    (OUT/'ANALYTIC_HEAD_DIAGNOSTIC.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':
    with threadpool_limits(1):main()
