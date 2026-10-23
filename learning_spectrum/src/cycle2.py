import json,time,resource
from geometry import *
from finite_horizon import *
from fixture import relation_cue

def main():
    start=time.monotonic();source_verify();results={}
    cues=[relation_cue(a,b) for a in b'ABCDEF' for b in b'ABCDEF' if a!=b]
    native,birth=native_panel(cues,0.)
    for world in (300001,300002):
        results[str(world)]={}
        for arm,centered in [('original_J',False),('candidate_A',True)]:
            dynamics,g=finite(world,centered)
            d,rr=history(world);sensor=Geometry(centered);shared={}
            for r in rr:
                sensor.feed_cue(bytes.fromhex(r['cue_hex']),r['at'])
                if r['index'] in (95,191):
                    at=r['at']+RECORD_SECONDS;name='old_end' if r['index']==95 else 'new_end'
                    shared[name]=kernel_metrics(native,sensor.panel(cues,at))
            results[str(world)][arm]={'dynamics':dynamics,'shared':shared,'formation_gate':dynamics['old_end']['symmetric_interaction_min']>=.20,'final_gain_gate':dynamics['final']['symmetric_interaction_min']>=.02}
    out={'schema':'LEARNING-SPECTRUM-CYCLE2-v1','results':results,'native_birth':birth,'identity_negative_control':kernel_metrics(native,np.eye(30)),'prospective_total_score_prediction':'NO QUANTITATIVE THEORY PREDICTION','full_run_authorized_by_gates':False,'resources':{'seconds':time.monotonic()-start,'max_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}}
    with (ROOT/'results/CYCLE2_RESULT.json').open('x') as f:json.dump(out,f,indent=2,allow_nan=False)
    for w,aa in results.items():
        for arm,r in aa.items():print(w,arm,'old gain',r['dynamics']['old_end']['symmetric_interaction_min'],'final gain',r['dynamics']['final']['symmetric_interaction_min'],'shared',r['shared'],'replay max',max(x['direct_replay_max_error'] for x in r['dynamics'].values()))
    print('resources',out['resources'])
if __name__=='__main__':main()
