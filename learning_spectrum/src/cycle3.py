import gzip,json,time,resource
import numpy as np
from scipy.stats import t as student
from geometry import ROOT,source_verify
from calibration import Observation,predict_baseline

def center(x):return x-np.mean(x,axis=-1,keepdims=True)

def main():
    start=time.monotonic();source_verify();prior=ROOT.parent.parent/'minifly-response/response_mechanisms/results/final';rows=[]
    for world in range(300001,300033):
        d=json.loads(gzip.decompress((prior/'H'/f'{world}.json.gz').read_bytes()))
        obs=[Observation(bytes.fromhex(r['cue_hex']),r['teacher_at'],tuple(r['raw_values'])) for r in d['records'] if r['branch']=='W']
        p=d['probes']['final']['W']['old'];v=np.array(p['raw_values']);h=np.array(p['values']);y=np.array(d['fixture']['old_fact']['labels']);at=p['response_at']
        cues=[bytes.fromhex(x) for x in d['fixture']['old_fact']['cues_hex']]
        estimates=np.array([predict_baseline(obs,cue,at) for cue in cues]);corrected=v-estimates
        r=v.argmax(1)==y;c=corrected.argmax(1)==y
        before=v[np.arange(16),y][:,None]-v;after=corrected[np.arange(16),y][:,None]-corrected
        predicted=before-estimates[np.arange(16),y][:,None]+estimates
        error=float(np.max(np.abs(after-predicted)))
        if error>1e-12:raise AssertionError('margin algebra mismatch')
        mask=np.eye(4,dtype=bool)[y];mb=np.min(np.where(mask,np.inf,before),axis=1);ma=np.min(np.where(mask,np.inf,after),axis=1)
        oracle=v.mean(0);hm=np.array(p['homeostasis']);mseb=float(np.mean((center(estimates)-center(oracle))**2));mseh=float(np.mean((center(hm)-center(oracle))**2))
        rows.append({'world':world,'raw_accuracy':float(r.mean()),'H_accuracy':float((h.argmax(1)==y).mean()),'B_accuracy':float(c.mean()),'wrong_to_right':int((~r&c).sum()),'right_to_wrong':int((r&~c).sum()),'margin_shift_mean':float(np.mean(ma-mb)),'decision_baseline_MSE_B':mseb,'decision_baseline_MSE_H':mseh,'baseline_predictions':estimates.tolist(),'true_labels_evaluator_only':y.tolist(),'raw_values':v.tolist(),'corrected_values':corrected.tolist(),'pairwise_margins_before':before.tolist(),'pairwise_margins_after':after.tolist(),'margin_algebra_error':error})
    differences=np.array([x['B_accuracy']-x['raw_accuracy'] for x in rows]);mean=float(differences.mean());se=float(differences.std(ddof=1)/np.sqrt(32));ci=[mean-float(student.ppf(.975,31))*se,mean+float(student.ppf(.975,31))*se]
    repairs=sum(r['wrong_to_right'] for r in rows);breaks=sum(r['right_to_wrong'] for r in rows);mseb=float(np.mean([r['decision_baseline_MSE_B'] for r in rows]));mseh=float(np.mean([r['decision_baseline_MSE_H'] for r in rows]))
    summary={'raw_accuracy':float(np.mean([r['raw_accuracy'] for r in rows])),'H_accuracy':float(np.mean([r['H_accuracy'] for r in rows])),'B_accuracy':float(np.mean([r['B_accuracy'] for r in rows])),'accuracy_delta':mean,'paired_ci95':ci,'wrong_to_right':repairs,'right_to_wrong':breaks,'mean_margin_shift':float(np.mean([r['margin_shift_mean'] for r in rows])),'baseline_MSE_B':mseb,'baseline_MSE_H':mseh,'development_gate_pass':bool(repairs>breaks and ci[0]>0 and mseb<mseh),'fresh_world_efficacy_tested':False}
    out={'schema':'LEARNING-SPECTRUM-CYCLE3-v1','summary':summary,'worlds':rows,'resources':{'seconds':time.monotonic()-start,'max_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}}
    with (ROOT/'results/CYCLE3_RESULT.json').open('x') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps(summary,indent=2));print(out['resources'])
if __name__=='__main__':main()
