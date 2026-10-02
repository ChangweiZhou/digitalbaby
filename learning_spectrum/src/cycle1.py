"""Development-only reconstruction and fixed-geometry preflight."""
import gzip,json,sys,time,resource
from pathlib import Path
import numpy as np
from geometry import *

def main():
    start=time.monotonic();verified=source_verify();results={}
    # No Full151 model is instantiated. Only immutable byte receptor source used.
    for world in (300001,300002):
        d,records=history(world);arm_results={}
        for name,centered in [('original_J',False),('candidate_A',True)]:
            g=Geometry(centered);snap={}
            for r in records:
                g.feed_cue(bytes.fromhex(r['cue_hex']),r['at'])
                if r['index'] in (95,191):
                    checkpoint='old_end' if r['index']==95 else 'new_end'
                    at=r['at']+RECORD_SECONDS
                    snap[checkpoint]={s:spectrum(g.panel([bytes.fromhex(c) for c in d[s+'_fact']['cues_hex']],at)) for s in ('old','new')}
            arm_results[name]=snap
        results[str(world)]=arm_results
    prior=ROOT.parent.parent/'minifly-response/response_mechanisms/results/final'
    counts={k:0 for k in ('n','raw_correct','online_correct','oracle_correct','wrong_to_right','right_to_wrong')};rms=[];source=[];told=[]
    for world in range(300001,300033):
        hp=prior/'H'/f'{world}.json.gz';jp=prior/'J'/f'{world}.json.gz'
        h=json.loads(gzip.decompress(hp.read_bytes()));j=json.loads(gzip.decompress(jp.read_bytes()))
        p=h['probes']['final']['W']['old'];v=np.array(p['raw_values']);u=np.array(p['values']);y=np.array(h['fixture']['old_fact']['labels'])
        r=v.argmax(1)==y;c=u.argmax(1)==y;o=(v-v.mean(0)).argmax(1)==y
        counts['n']+=len(y);counts['raw_correct']+=int(r.sum());counts['online_correct']+=int(c.sum());counts['oracle_correct']+=int(o.sum());counts['wrong_to_right']+=int((~r&c).sum());counts['right_to_wrong']+=int((r&~c).sum())
        delta=np.array(j['probes']['final']['W']['old']['extra_values'])-np.array(j['probes']['final']['N_old']['old']['extra_values'])
        rms.append(float(np.sqrt(np.mean(components(delta)[3]**2))))
        source.extend([{'path':str(p.relative_to(prior)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in (hp,jp)])
        td=json.loads(gzip.decompress((prior/'T'/f'{world}.json.gz').read_bytes()));fd=json.loads(gzip.decompress((prior/'T_OFF'/f'{world}.json.gz').read_bytes()))
        told.append(td['probes']['old_end']['W']['old']['accuracy']-fd['probes']['old_end']['W']['old']['accuracy'])
    counts['percentages']={k:100*counts[k]/counts['n'] for k in ('raw_correct','online_correct','oracle_correct')}
    out={'schema':'LEARNING-SPECTRUM-CYCLE1-v1','source_files_verified':len(verified),'geometry':results,'historical_H':counts,'historical_J_added_old_write_final_interaction_rms_mean':float(np.mean(rms)),'historical_T_old_end_vs_T_OFF_pp':float(100*np.mean(told)),'historical_source_receipts':source,'resources':{'seconds':time.monotonic()-start,'max_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}}
    path=ROOT/'results/CYCLE1_RESULT.json'
    with path.open('x') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in out.items() if k not in ('geometry','historical_source_receipts')},indent=2))
if __name__=='__main__':main()
