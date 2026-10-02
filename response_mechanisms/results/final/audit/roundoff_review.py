"""Supplemental read-only diagnostic; does not change the locked analysis or auditor."""
from pathlib import Path
import importlib.util,json,math,hashlib
import numpy as np
R=Path(__file__).resolve().parents[2]
p=R/'scratch/final_audit/verify_complete.py'
spec=importlib.util.spec_from_file_location('independent_auditor',p);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
lock=a.read(R/'SOURCE_LOCK.json');out=a.read(R/'results/final/FINAL_METRICS.json')
docs={arm:[a.receipt(R/f'results/final/{arm}/{w}.json.gz') for w in lock['worlds']] for arm in lock['arms']}
failures=[]
def equal(x,y,path='root'):
    if isinstance(x,dict):
        assert set(x)==set(y),(path,set(x),set(y))
        for k in x: equal(x[k],y[k],path+'/'+str(k))
    elif isinstance(x,(list,tuple)):
        assert len(x)==len(y),(path,len(x),len(y))
        for k,(i,j) in enumerate(zip(x,y)): equal(i,j,path+'/'+str(k))
    elif x is None or isinstance(x,(str,bool)):
        if x!=y: failures.append(dict(path=path,locked=x,independent=y))
    else:
        assert math.isfinite(float(x)) and math.isfinite(float(y)),('nonfinite',path)
        if not math.isclose(float(x),float(y),rel_tol=1e-10,abs_tol=1e-10):failures.append(dict(path=path,locked=x,independent=y,abs_difference=abs(float(x)-float(y))))
a.equal=equal
original_result=a.numerical(docs,out);original_failures=failures.copy()
print('Original full numerical failures:',json.dumps(original_failures),flush=True)
computed={arm:[a.independent_metrics(d) for d in rows] for arm,rows in docs.items()}
metrics={}
for name in ('g1','g2','g12','g12_g1','g12_g2','H_projection','bias','benefit'):
    x=np.array([r['metrics'][name] for r in out['world_level']['H']]);y=np.array([r['metrics'][name] for r in out['world_level']['FE0']])
    xx=np.array([r[name] for r in computed['H']]);yy=np.array([r[name] for r in computed['FE0']])
    metrics[name]=dict(locked=a.summary(x-y),alternative_order=a.summary(xx-yy),locked_max_abs_contrast=float(np.max(np.abs(x-y))),alternative_max_abs_contrast=float(np.max(np.abs(xx-yy))),max_abs_metric_reconstruction_error=float(max(np.max(np.abs(xx-x)),np.max(np.abs(yy-y)))),mean_absolute_arm_magnitude=float(np.mean(np.abs(np.r_[x,y]))),locked_values=(x-y).tolist(),alternative_values=(xx-yy).tolist())
raw_identical=True;max_offset_error=0.;max_raw=0.;max_offsets=0.
for h,f in zip(docs['H'],docs['FE0']):
    for phase,branches in h['probes'].items():
        for branch in ('W','N_old'):
            for group,probe in branches[branch].items():
                peer=f['probes'][phase][branch][group]
                raw=np.array(probe['raw_values']);rawf=np.array(peer['raw_values'])
                raw_identical &= np.array_equal(raw,rawf)
                offset=np.array(probe['homeostasis']);actual=np.array(probe['values']);expected=raw-offset
                max_offset_error=max(max_offset_error,float(np.max(np.abs(actual-expected))))
                max_raw=max(max_raw,float(np.max(np.abs(raw))))
                max_offsets=max(max_offsets,float(np.max(np.abs(offset))))

def locked_order_components(values):
    v=np.asarray(values,dtype=float).reshape(4,4,4)
    b=np.average(v,axis=(0,1));f=np.average(v,axis=1)-b;g=np.average(v,axis=0)-b
    h=v-b-f[:,None,:]-g[None,:,:]
    return v,b,f,g,h

a.components=locked_order_components;failures.clear()
locked_order_result=a.numerical(docs,out);locked_order_failures=failures.copy()
report=dict(purpose=__doc__,source_lock_sha256=a.sha(R/'SOURCE_LOCK.json'),original_auditor_sha256=a.sha(p),locked_metrics_sha256=a.sha(R/'results/final/FINAL_METRICS.json'),original_full_numerical_comparison_failures=original_failures,locked_operation_order_full_numerical_comparison_failures=locked_order_failures,locked_operation_order_independent_result=locked_order_result,original_operation_order_independent_result=original_result,H_FE0_contrasts=metrics,H_and_FE0_all_probe_raw_values_bitwise_identical=bool(raw_identical),H_output_raw_minus_saved_homeostasis_max_error=max_offset_error,max_absolute_H_FE0_raw_value=max_raw,max_absolute_H_homeostasis=max_offsets,diagnostic_change='In-memory only: interaction=v-base-first[:,None,:]-second[None,:,:], preserving locked arithmetic order. No source, receipt, metrics, threshold or original auditor edits.')
target=R/'scratch/final_audit/ROUNDOFF_REVIEW.json'
with target.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('H_FE0_contrasts','locked_operation_order_independent_result','original_operation_order_independent_result')},indent=2),flush=True)
print('Contrast statistics:',json.dumps({k:{n:v for n,v in data.items() if not n.endswith('_values')} for k,data in metrics.items()},indent=2),flush=True)
