"""Synthetic selection/CPU/shortcut checks. No model execution or science data."""
import copy,math
import bootstrap
import campaign_analysis as a
import campaign_common as c
import bind_evaluator as e
import protocol

def run():
    metric=lambda value:dict(G=value,Q=value,HELD_FINAL=value,OLD_END=.5,OLD_REST=.5,OLD_FINAL=.5,NEW_END=.5,NEW_REST=.5)
    worlds={str(i):dict(arms={},shortcut=dict(G=0.,HELD_FINAL=0.,OLD_MODE=0.,OLD_FINAL_MODE=0.,NEW_MODE=0.)) for i in range(8)}
    for w in worlds.values():
        for arm in protocol.ARMS:
            value=.1 if arm=='NATIVE' else .2 if arm=='F000' else .15
            w['arms'][arm]={'W':{'MEM':metric(value)},'N_OLD':{'MEM':{k:0. for k in metric(0)}}}
            if arm=='NATIVE':w['arms'][arm]['W']['LEGACY']=metric(.12)
    assert a.choose(worlds)['selected']=='F000'
    for w in worlds.values():w['arms']['F111']['W']['MEM']=metric(.4)
    assert a.choose(worlds)['selected']=='F111'
    for w in worlds.values():w['arms']['F001']['W']['MEM']=metric(.4)
    assert a.choose(worlds)['selected']=='F001'
    for w in worlds.values():
        for arm in protocol.FACTORIAL:w['arms'][arm]['N_OLD']['MEM']=copy.deepcopy(w['arms'][arm]['W']['MEM'])
    assert a.choose(worlds)['selected'] is None
    points=[dict(records=n,tag=tag,CPU_MEM=cpu,readings={'MEM':{'accuracy':value}}) for n,tag,cpu,value in [(0,'SAMPLE',1,.1),(4,'SAMPLE',3,.2),(4,'OLD_REST',4,0.),(8,'SAMPLE',6,.3),(8,'NEW_REST',7,.2)]]
    boundaries=[dict(phase=p,tag=t,readings={'MEM':{'old':{'accuracy':0.},'new':{'accuracy':0.}}}) for p,t in [('BIRTH','BIRTH'),('OLD','BEFORE_REST'),('OLD','AFTER_REST'),('NEW','BEFORE_REST'),('NEW','AFTER_REST')]]
    result=a.history_metrics(dict(points=points,boundaries=boundaries),'MEM',8)
    assert abs(result['Q']-.1125)<1e-12
    assert result['G']==e.sample_auc([(0,.1),(4,.2),(8,.3)])
    assert a.shortcut_scores([],[(b'unknown',b'value')])['envelope']==0.
    suffix=e.SuffixBackoff();suffix.observe(b'\xff\x00',b'\x12')
    assert suffix.generate(b'\xff\x00')==(b'\x12','EOS')
    checks=dict(root_selectable=True,nonroot_requires_root_improvement=True,tie_fewer_bits=True,causal_failure_rejects=True,
        CPU_rest_state_included=True,CPU_initialization_not_extrapolated_to_zero=True,G_rest_excluded=True,arbitrary_byte_suffix=True)
    c.write(c.OPS/'ANALYSIS_QUALIFICATION.json',dict(passed=True,checks=checks,synthetic_only=True,science_records=0,science_jobs=0))
    return checks
if __name__=='__main__':print(run())
