"""V186 prelaunch census. No candidate learns from the locked documents."""
import ast
from collections import defaultdict, Counter
import hashlib
import json
import math
from pathlib import Path
import resource
import time

from qualify import DecaySensor, PARENT, Learner, np, T
from lock_roster import select, sha, corrected
from generation import generate

def load_definitions():
    root = PARENT.parent / 'minifly_v179_continuous_local_life_dev_20261102/local/source'
    env = dict(EOS=256,V=257,np=np,math=math,defaultdict=defaultdict,Counter=Counter)
    for path,names in [('minifly_v148_pretrain_probe_dev_20261025/ladder.py',('stream','Unigram')),
                       ('minifly_v169_native_dev_v1_20261027/v169_count.py',('IKN','line_bits'))]:
        file = root/path
        tree = ast.parse(file.read_text())
        definitions = [n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
        assert len(definitions)==len(names)
        exec(compile(ast.Module(body=definitions,type_ignores=[]),str(file),'exec'),env)
    return env

def main():
    started = time.process_time()
    _, worlds, _ = select()
    lock = json.loads(Path('FRESH_ROSTER.json').read_text())
    assert len(worlds)==2
    env = load_definitions()
    sensor = DecaySensor()
    schedules = {'A':2,'B':3,'EDIT':2,'NOEDIT':2}
    totals = dict(records=0,targets=0,updates=0,score_sites=0,generation_sites_max=0,cache_bytes_max=0)
    fits, fit_targets, baselines = 0,0,{}
    sensor_seconds = 0.
    for world,(_,_,splits,meta) in worlds.items():
        assert {k:[sha(r) for r in rs] for k,rs in splits.items()}==lock['worlds'][world]['splits']
        rule = bytes.fromhex(meta['context_hex']),meta['old'],meta['new']
        for role in ('calib','held'):
            splits['EDIT_'+role] = [corrected(r,rule) for r in splits['A_'+role]]
        hist = []
        for stage, phase in [('A2',('A',)),('B3',('A','B')),('EDIT2',('A','B','EDIT')),('NOEDIT2',('A','B','NOEDIT'))]:
            hist = [r for p in phase for _ in range(schedules[p]) for r in splits[p+'_train']]
            cal = splits[('EDIT' if stage=='EDIT2' else 'A')+'_calib'] + splits['B_calib']
            grid=[]
            for order in range(1,6):
                for discount in (.5,.75,.9):
                    m = env['IKN'](order,discount).fit(hist)
                    loss = float(env['line_bits'](m,cal).sum()/sum(len(r)+1 for r in cal))
                    assert np.isfinite(loss) and loss>=0
                    grid.append([loss,order,discount])
            uni = env['Unigram']().fit(hist)
            assert np.isfinite(env['line_bits'](uni,cal)).all()
            fits += 16
            fit_targets += 16*sum(len(r)+1 for r in hist)
            baselines[world+'_'+stage]={'IKN_best':min(grid),'bigram_best':min(row for row in grid if row[1]==2),'grid':grid}
        n = {p:sum(len(r)+1 for r in splits[p+'_train']) for p in schedules}
        assert max(len(r)+1 for p in schedules for r in splits[p+'_train'])<=128
        totals['records']+=4*sum(len(splits[p+'_train'])*schedules[p] for p in schedules)
        totals['targets']+=4*sum(n[p]*schedules[p] for p in schedules)
        totals['updates']+=2*sum(math.ceil(n[p]/batch)*schedules[p] for p in schedules for batch in (32,256))
        totals['score_sites']+=4*10*sum(len(r)+1 for rs in splits.values() for r in rs)
        totals['generation_sites_max']+=4*3*32*64
        for decay in (.95,.5):
            t=time.process_time()
            packed=[[np.flatnonzero(row).astype(np.uint16) for row in sensor.codes(r,decay)]
                    for rs in splits.values() for r in rs]
            sensor_seconds+=time.process_time()-t
            totals['cache_bytes_max']=max(totals['cache_bytes_max'],sum(row.nbytes for rec in packed for row in rec))
            del packed
    assert totals['records']==4480 and totals['targets']+1264<=600000
    Path('BASELINE_PREFLIGHT.json').write_text(json.dumps({'status':'TRAIN_CALIBRATION_PASS','fits':fits,
        'fit_targets':fit_targets,'panels':baselines,'held_baseline_scores_computed':False,
        'exact_prefix_control':'PENDING_INTEGRATION'},indent=2)+'\n')
    Path('ROSTER_PROFILE.json').write_text(json.dumps({'status':'PARTIAL_RESOURCE_CENSUS',
        'roster':totals,'actual_full_roster_sensor_pack_cpu_seconds':sensor_seconds,
        'profile_cpu_seconds':time.process_time()-started,
        'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'checkpoint_payload_projection_bytes':80*458410,
        'pending':['integrated evaluator and exact-prefix control','timed batch-32 and batch-256 updates',
                   'full generation and archive resource projection']},indent=2)+'\n')
if __name__=='__main__':
    main()
