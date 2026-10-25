import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
import sys,time,json,pickle,traceback,resource
from pathlib import Path
from support151 import H,R,D,COUNTS,TIMES,save,load,sha,digest,ledger,cpu,budget
import preflight151
if len(sys.argv)>1 and sys.argv[1]=='resume_child':
    preflight151.child();sys.exit(0)
from assay151 import ASSETS,SITES,get_anchor,operator_site,rule_site,gains,code_site

def main():
    for path,expected in load(H/'SOURCE_LOCK.json').items():assert sha(R/path)==expected,path
    ledger('SOURCE_LOCK_VERIFIED',sha256=sha(H/'SOURCE_LOCK.json'))
    preflight151.run()
    cal=[]
    for info in ASSETS:
        if not info['primary']:continue
        anchor=get_anchor(info);before=digest(anchor)
        for site in SITES[info['text']]['sites']:
            if site['role']!='calibration':continue
            save('ACTIVE.json',{'phase':'CALIBRATION','anchor':info['id'],'site':site['site_id'],'CPU':cpu(),'targets':COUNTS['target_applications']})
            pre,z,op=operator_site(anchor,info,site,'CALIBRATION')
            cal.append(rule_site(pre,z,op,info,site,'CALIBRATION'))
        assert digest(anchor)==before
        ledger('CALIBRATION_ANCHOR_COMMIT',anchor=info['id'])
        del anchor
    lock=gains(cal);locksha=sha(H/'GAIN_LOCK.json');save('CYCLE2_CALIBRATION.json',{'status':'GAIN_LOCK_COMPLETE','sites':12,'target_branches':24,'loss_used_for_fitting':False,'gain_lock_sha256':locksha})
    results=[]
    for info in ASSETS:
        if not info['primary']:continue
        anchor=get_anchor(info);before=digest(anchor)
        for site in SITES[info['text']]['sites']:
            if site['role']!='scored':continue
            assert sha(H/'GAIN_LOCK.json')==locksha
            save('ACTIVE.json',{'phase':'SCORED','anchor':info['id'],'site':site['site_id'],'CPU':cpu(),'targets':COUNTS['target_applications']})
            pre,z,op=operator_site(anchor,info,site,'SCORED')
            results.append(rule_site(pre,z,op,info,site,'SCORED',lock[info['id']]))
        assert digest(anchor)==before
        ledger('SCORED_ANCHOR_COMMIT',anchor=info['id'])
        del anchor
    codes=[]
    for info in ASSETS:
        if info['primary']:continue
        anchor=get_anchor(info);before=digest(anchor)
        for site in SITES[info['text']]['sites']:
            if site['role']!='scored':continue
            save('ACTIVE.json',{'phase':'CODE_FACTORIAL','anchor':info['id'],'site':site['site_id'],'CPU':cpu(),'targets':COUNTS['target_applications']})
            codes.append(code_site(anchor,info,site))
        assert digest(anchor)==before
        del anchor
    assert sha(H/'GAIN_LOCK.json')==locksha
    for a in ASSETS:
        if 'checkpoint' in a:assert sha(R/a['checkpoint'])==a['sha256']
    budget()
    save('EXECUTION_SUMMARY.json',{'version':'V151','status':'EXACT_ROSTER_COMPLETE','scored_sites':len(results),'C_factorial_sites':len(codes),
         'counts':dict(COUNTS),'CPU_including_waited_children':cpu(),'timings':dict(TIMES),'gain_lock_sha256':locksha,
         'RSS_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'addition_records':0,'full_training_histories':0})
    ledger('EXECUTION_COMPLETE',counts=dict(COUNTS))
    save('STATUS.json',{'version':'V151','status':'EXECUTION_COMPLETE_AWAITING_CYCLE3_AUDIT','targets':COUNTS['target_applications'],'scientific_addition_records':0,'new_core_adopted':False})
try:main()
except BaseException as e:
    name='PAUSED_SESSION_LIMIT' if str(e).startswith('PAUSED_') else 'FAILED'
    save('FAILURE.json',{'status':name,'error':repr(e),'traceback':traceback.format_exc(),'counts':dict(COUNTS),'CPU':cpu(),'no_retry':True})
    save('STATUS.json',{'version':'V151','status':name,'error':repr(e),'counts':dict(COUNTS)})
    raise
