"""Exploratory paired DEV reanalysis; reads accepted compact metrics, runs no training."""
from pathlib import Path
import gzip, hashlib, itertools, json
import numpy as np
from scipy import stats
ROOT=Path(__file__).resolve().parents[2]
INPUT=ROOT/'results/dev/WORLD_METRICS.json.gz'

def paired(d):
    x=np.asarray(d,float);n=len(x);mean=float(x.mean());se=float(x.std(ddof=1)/np.sqrt(n))
    ci=[float(mean-stats.t.ppf(.975,n-1)*se)*100,float(mean+stats.t.ppf(.975,n-1)*se)*100] if se>0 else None
    p=float(2*stats.t.sf(abs(mean/se),n-1)) if se>0 else None
    samples=[float(np.mean(x*np.array(s))) for s in itertools.product((-1,1),repeat=n)]
    nz=x[abs(x)>1e-12]
    return dict(worlds=n,mean_difference_pp=mean*100,paired_student_two_sided_95_CI_pp=ci,
        paired_student_two_sided_p=p,exact_sign_flip_two_sided_p=sum(abs(v)>=abs(mean)-1e-12 for v in samples)/len(samples),
        sign_test_two_sided_p=float(stats.binomtest(int(sum(nz>0)),len(nz),.5).pvalue) if len(nz) else 1.,
        positive=int(sum(x>1e-12)),negative=int(sum(x< -1e-12)),ties=int(sum(abs(x)<=1e-12)),world_differences_pp=(x*100).tolist())

def stratified(rows,field):
    groups=[np.array([r['COV-SUPPORT']['W'][field]-r['COV-R']['W'][field] for r in rows if r['family']==f]) for f in ('PATH_LOOP','STAR_LOOP')]
    parts=[.25*g.var(ddof=1)/len(g) for g in groups];v=sum(parts)
    if not v:return {'worlds':len(rows),'uncertainty':'not estimated; no observed between-world variation'}
    df=v*v/sum(a*a/(len(g)-1) for a,g in zip(parts,groups));se=v**.5;mean=sum(g.mean()/2 for g in groups)
    return dict(worlds=len(rows),mean_difference_pp=float(mean*100),df=float(df),
        two_sided_95_CI_pp=[float((mean-stats.t.ppf(.975,df)*se)*100),float((mean+stats.t.ppf(.975,df)*se)*100)],
        two_sided_p=float(2*stats.t.sf(abs(mean/se),df)))

with gzip.open(INPUT,'rt') as f:rows=json.load(f)
result={'classification':'EXPLORATORY_POSTHOC_DEV_REANALYSIS_NOT_CONFIRMATION',
 'source':str(INPUT),'source_sha256':hashlib.sha256(INPUT.read_bytes()).hexdigest(),
 'new_training_records':0,'frozen_verdict_unchanged':'NO_ADVANCE_RETAIN_COV_R',
 'unit':'independent paired world, six per noise stratum; probes and trajectories are not independent samples',
 'limitations':['All p values and intervals are unadjusted exploratory summaries; no new adoption criterion.',
 'Student intervals use small-sample distribution assumptions; stratified sensitivity preserves the two task geometries.',
 'Sign-flip test assumes sign exchangeability under the null; it is not a randomized assignment test.',
 'No significant decline is not evidence of equivalence or proven benefit.',
 'All-zero observed differences do not justify a zero-width population uncertainty interval.',
 'DEV cannot establish performance on the unrun TRIANGLE_TAIL confirmation geometry.'], 'strata':{}}
for noise in ('BLURRY','STANDARD'):
    subset=[r for r in rows if r['noise']==noise]
    summary={k:paired([r['COV-SUPPORT']['W'][k]-r['COV-R']['W'][k] for r in subset]) for k in ('J','old_final','old_score','new_final')}
    summary['J_stratified_sensitivity']=stratified(subset,'J')
    summary['new_curve_by_dose']={str(dose):paired([r['COV-SUPPORT']['W']['new_curve'][i]-r['COV-R']['W']['new_curve'][i] for r in subset]) for i,dose in enumerate((20,40,80))}
    result['strata'][noise]=summary
out=Path(__file__).with_name('REANALYSIS.json');out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({'artifact':str(out),'BLURRY_J':result['strata']['BLURRY']['J'],'BLURRY_stratified':result['strata']['BLURRY']['J_stratified_sensitivity'],'new_training_records':0},indent=2))
