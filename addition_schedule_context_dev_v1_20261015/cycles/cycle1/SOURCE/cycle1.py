import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import ast,copy,json,time
import bootstrap,config,data,models,engine,audits,probes
import numpy as np
from ledger import apply
from artifacts import write,now


def run():
    checks=[];world=config.TECH_WORLDS[0];source=data.make_addition(world,technical=True)
    for w in config.SCIENCE_WORLDS:
        s=data.make_addition(w)
        assert [len(s['groups'][k]) for k in ('old','new','held')]==[60,20,20]
        for h in config.HISTORIES:
            blocked=data.ordered(s,'BLOCKED',h);mixed=data.ordered(s,'INTERLEAVED',h)
            assert sorted(blocked,key=lambda r:r['id'])==sorted(mixed,key=lambda r:r['id'])
            assert [r['id'] for r in mixed if r['kind']=='OLD']==[r['id'] for r in blocked if r['kind']=='OLD']
            assert all(r['prompt'] not in {data.example(p)[0].hex() for p in s['groups']['held']} for r in mixed)
        assert 'NEW:0' in data.checkpoint_plan(s,'BLOCKED')[16384]
        assert 'NEW:0' in data.checkpoint_plan(s,'INTERLEAVED')[16]
    checks.append('all_prospective_data_multisets_orbits_and_zero_dose_locations')
    for w in (*config.NATIVE_MAPS,*config.TECH_WORLDS):
        f=data.native_fixture(w);aliases=data.native_aliases(f)
        assert len(f)==16 and len(aliases)==4
        assert all(all(r['stratum']=='collision' and len(bytes.fromhex(r['prefix']))==1 for r in a['rows']) for a in aliases)
        assert sorted(len(x['answer']) for x in f if x['stratum']=='collision')==sorted(len(x['answer']) for x in f if x['stratum']=='matched')
    f=data.native_fixture(world)
    for item in f:
        fronts=[models.Front(v) for v in config.NATIVE_VARIANTS]
        old=models.stores.content_model.VisibleContextFE()
        for i,b in enumerate(item['cue']):
            old.feed(b,i*models.PARENT.DELTA)
            for fe in fronts:fe.feed(b,i*models.PARENT.DELTA)
        for fe in fronts:fe.freeze_prompt()
        np.testing.assert_array_equal(fronts[0].read(),old.read());np.testing.assert_array_equal(fronts[0].read(),fronts[1].read())
        for fe in fronts:fe.feed(item['answer'][0],13*models.PARENT.DELTA)
        assert fronts[1].q==item['cue'] and fronts[1].prefix==item['answer'][:1]
        assert len(np.flatnonzero(fronts[1].read()))==24
    checks.append('matched_native_fixtures_and_first_position_exact_PN_parity')
    technical={}
    for h in config.HISTORIES:
        for schedule in config.SCHEDULES:
            model=models.Keep(data.seed(world,'model'));rows=[];new_states=[]
            for r in data.ordered(source,schedule,h):
                rows.append(apply(1,f'{h}/{schedule}/{r["id"]}',lambda r=r:engine.apply_addition(model,r)))
                if r['kind']=='NEW':new_states.append(model.digest())
            audits.audit_addition_rows(source,schedule,h,rows)
            technical[h,schedule]=(model,rows,new_states)
    assert technical['N_OLD','BLOCKED'][2]==technical['N_OLD','INTERLEAVED'][2]
    checks.append('six_actual_technical_histories_and_N_OLD_matched_NEW_state')
    adapted=models.Keep(71);parent=models.PARENT.Recurrent('KEEP-C16-CE',71)
    for i,(p,y) in enumerate([(b'2+3=',b'5'),(b'1+7=',b'8'),(b'3+6=',b'9')]):
        a=apply(1,f'parity/adapted/{i}',lambda:adapted.train(p,y,True,capture=True))
        b=apply(1,f'parity/parent/{i}',lambda:parent.train(p,y,True))
        assert a['loss']==b['loss'] and adapted.digest()==parent.digest()
    assert adapted.last_capture['metrics']['E']['raw_gradient_l2']>0
    checks.append('multi_step_nonzero_internal_gradient_exact_parent_update_parity')
    for variant in config.NATIVE_VARIANTS:
        native=models.Native(variant)
        assert len({id(s.fly) for bank in native.banks for s in bank})==32
        item=f[0]
        for allowed in (True,False):
            row=apply(1,f'{variant}/{allowed}',lambda:engine.apply_native(native,item,allowed))
            audits.audit_native_record(row,item,allowed)
        before=native.digest();cached=(native.fronts[1].cache_hits,native.fronts[1].cache_misses,len(native.fronts[1].cache))
        result=probes.native(native,f[:1],cap=2)
        assert before==native.digest() and cached==(native.fronts[1].cache_hits,native.fronts[1].cache_misses,len(native.fronts[1].cache))
    checks.append('both_native_variants_actual_nonzero_NO_WRITE_and_query_cache_isolation')
    return dict(accepted=True,checks=checks,technical_only=True,behavior_at_full_dose_not_tested=True)

if __name__=='__main__':
    import traceback
    started=time.monotonic()
    try:r=run()
    except BaseException as e:
        write(config.HERE/'cycles/cycle1/RESULT.json',dict(accepted=False,error=repr(e),traceback=traceback.format_exc(),utc=now()));raise
    r.update(utc=now(),elapsed=time.monotonic()-started)
    write(config.HERE/'cycles/cycle1/RESULT.json',r);print(r)
