"""Locked three-world acquisition screen."""
import copy
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import time

from learner import Learner
from worlds import addition, exposure_digest, fact, make_world, tokens
from qualify import score

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'official'
SEEDS = (66101, 66202, 66203)
DOSES = (1, 4, 16, 64, 256)
BASE = [('dense64',64,'dense','direct'), ('connected32',32,'connected','direct'),
        ('connected64',64,'connected','direct')]


def put(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def source_hashes(names):
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}


def task(seed, arm):
    import torch
    torch.set_num_threads(1)
    name, width, topology, route = arm
    path = OUT/f'screen_{name}_{seed}.json'
    if path.exists():
        return json.loads(path.read_text())
    world = make_world(seed)
    panel = [addition(*ab) for ab in world['taught'][:4]] + [fact(*item) for item in world['assignments'][0][:4]]
    rng = random.Random(seed*100+1)
    records = []
    for _ in range(256):
        batch = panel.copy()
        rng.shuffle(batch)
        records.extend(batch)
    stream = tokens(records)
    model = Learner(seed,width,topology,route)
    clock = time.process_time()
    curve = []
    for dose in DOSES:
        stop = len(tokens(records[:dose*8]))
        while model.sites+32 <= stop:
            model.train(stream[model.sites:model.sites+32], 'bp')
        probe = copy.deepcopy(model)
        for y in stream[model.sites:stop]:
            probe.transition(probe.feature(y))
        rows = score(probe,panel)
        curve.append({'dose':dose,'symbols':stop,'outputs':rows,'facts':sum(r['exact'] for r in rows[4:]),
                      'arithmetic':sum(r['exact'] for r in rows[:4])})
    result = {'seed':seed,'arm':name,'curve':curve,'cpu_seconds':time.process_time()-clock,
              'stream_sha256':exposure_digest(records),'state_sha256':model.digest()}
    put(path,result)
    print('screen',name,seed,curve[-1]['facts'],curve[-1]['arithmetic'],flush=True)
    return result


def screen():
    OUT.mkdir(exist_ok=True)
    plan = {'seeds':SEEDS,'doses':DOSES,'arms':BASE,'fact_floor_each_seed':3,
            'selection_order':['connected32','connected64','dense64'],
            'source':source_hashes(['learner.py','reference.py','qualify.py','screen.py','DESIGN.md'])}
    lock = OUT/'SCREEN_LOCK.json'
    if lock.exists():
        assert json.loads(lock.read_text()) == json.loads(json.dumps(plan))
    else:
        put(lock,plan)
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(task,seed,arm) for arm in BASE for seed in SEEDS]
        rows = [f.result() for f in futures]
        name = next((n for n in plan['selection_order'] if all(r['curve'][-1]['facts']>=3
                     for r in rows if r['arm']==n)),None)
        if name is None:
            put(OUT/'SELECTION.json', {'status':'ACQUISITION_FLOOR_MISSED','rows':rows})
            return
        chosen = next(a for a in BASE if a[0]==name)
        _, width, topology, _ = chosen
        extra = [('kc',width,topology,'kc')]
        if topology == 'connected':
            extra.append(('isolated',width,'isolated','direct'))
        futures = [pool.submit(task,seed,arm) for arm in extra for seed in SEEDS]
        rows += [f.result() for f in futures]
    kc = [r for r in rows if r['arm']=='kc']
    direct = {r['seed']:r for r in rows if r['arm']==name}
    eligible = all(r['curve'][-1]['facts']>=max(3,direct[r['seed']]['curve'][-1]['facts']) for r in kc)
    # Same engineering CPU envelope for the input comparison.
    matched_cost = sum(r['cpu_seconds'] for r in kc) <= 1.25*sum(r['cpu_seconds'] for r in direct.values())
    route = 'kc' if eligible and matched_cost else 'direct'
    put(OUT/'SELECTION.json', {'status':'SELECTED','name':name,'width':width,'topology':topology,'route':route,
                              'kc_acquisition_eligible':eligible,'kc_within_125pct_screen_cpu':matched_cost,'rows':rows})
    print('selected',name,route,flush=True)


if __name__=='__main__':
    screen()
