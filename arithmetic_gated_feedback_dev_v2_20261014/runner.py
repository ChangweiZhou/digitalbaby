"""Fixed technical world-core jobs; no DEV/CONFIRM launch path."""
import argparse
import copy
import gzip
import json
import os
import resource
import time
from core import Learner, MODELS
from protocol import ROOT, D, atomic, seed, show, plans, identity
from runner_helpers import checkpoint


def segment(c, rows, phase, world, offset, name, recipe, allowed, feedback, hb, checks, w):
    before, before_resources = c.digest(), dict(c.resource)
    ledger, curve = [], []
    for i, row in enumerate(rows):
        index = offset + i
        observable = row['reply'] if recipe == 'CE_ONLY' or show(world, index, phase) else None
        credit = allowed and phase != 'WARM' and recipe != 'CE_ONLY'
        answer, status, cache = c.act(bytes.fromhex(row['prompt']), seed(world, index), credit)
        reward = int(answer.hex() == row['truth'] and status == 'EOS')
        baseline = float(cache['baseline'])
        actions = list(cache['actions'])
        update = c.update(bytes.fromhex(row['prompt']), None if observable is None else bytes.fromhex(observable),
                          reward, cache, phase, recipe, allowed)
        recorded = dict(index=index, prompt=row['prompt'], expected=row['truth'], observed=observable,
                        operation=row['operation'], group=row['group'], actual=answer.hex(), status=status,
                        reward=reward, actions=actions, baseline=baseline, allowed=allowed,
                        phase=phase, recipe=recipe, update=update)
        ledger.append(recorded)
        feedback.append(recorded)
        if i + 1 in checks:
            curve.append(checkpoint(c, index + 1, w, feedback))
        if i % 64 == 0 or i + 1 == len(rows):
            hb(dict(phase=phase, branch=name, completed_records=i + 1,
                    segment_records=len(rows), partial=True))
    after = c.digest()
    if not allowed:
        assert before == after, 'no-write learning state changed'
    return dict(phase=phase, kind=name, recipe=recipe, offset=offset, allowed=allowed,
                before=before, after=after, resources_before=before_resources,
                resources_after=dict(c.resource), records=ledger, curve=curve)


def job(world, model, nb, nm, stages, out, hb):
    start, wall = time.process_time(), time.monotonic()
    w, warm, shuffled, tails = plans(world, nb, nm, stages)
    for original, false in zip(warm, shuffled):
        original['truth'] = false['truth'] = original['reply']
    for rows in tails.values():
        for row in rows:
            row['truth'] = row['reply']
    born = Learner(model, seed(world, 0, 'model'))
    states = out / 'states' / f'{world}_{model}'
    states.mkdir(parents=True, exist_ok=False)
    born.save(states / 'BIRTH.npz')
    r = dict(schema='GATED_FEEDBACK_TECHNICAL_JOB_V2', technical=True, world=world, model=model,
             binary=nb, mixed=nm, stages=stages, data_digest=w['data_digest'], source=identity(),
             initial=born.digest(), model_seed=seed(world, 0, 'model'), prefixes={}, branches={},
             parameters={k: p.numel() for k, p in born.p.items()},
             learning_bytes=sum(p.numel() * p.element_size() for p in born.p.values()) * 2,
             completed=False)
    parents, fw = {}, nb + nm
    for name, stream, allowed in (('TRUE', warm, True), ('N_WARM', warm, False), ('SH_WARM', shuffled, True)):
        c, feedback = born.clone(), []
        birth_curve = checkpoint(c, 0, w, feedback)
        seg = segment(c, stream, 'WARM', world, 0, name, 'OLD', allowed, feedback, hb,
                      {fw * i // 4 for i in range(1, 5)}, w)
        c.save(states / (name + '_WARM.npz'))
        r['prefixes'][name] = dict(segment=seg, curve=[birth_curve] + seg['curve'],
                                  taught=checkpoint(c, fw, w, feedback, True), state=c.digest())
        parents[name] = (c, feedback)
    scalar_parents = {}
    roster = (('CE_ONLY', 'TRUE', 'CE_ONLY'), ('OLD', 'TRUE', 'OLD'), ('SAFE', 'TRUE', 'SAFE'),
              ('N_WARM_SAFE', 'N_WARM', 'SAFE'), ('SH_WARM_SAFE', 'SH_WARM', 'SAFE'))
    for name, parent, recipe in roster:
        c, feedback = parents[parent][0].clone(), list(parents[parent][1])
        segments, curve = [], list(r['prefixes'][parent]['curve'])
        offset = fw
        for phase, n in zip(('HALF', 'SPARSE', 'SCALAR'), stages):
            if phase == 'SCALAR':
                c.save(states / (name + '_SCALAR_START.npz'))
                if name in ('OLD', 'SAFE'):
                    scalar_parents[name] = (c.clone(), list(feedback), list(curve))
            seg = segment(c, tails[phase], phase, world, offset, name, recipe, True,
                          feedback, hb, {n} if phase != 'SCALAR' else {n // 2, n}, w)
            segments.append(seg)
            curve.extend(seg['curve'])
            offset += n
        c.save(states / (name + '_FINAL.npz'))
        r['branches'][name] = dict(prefix=parent, recipe=recipe, segments=segments, curve=curve,
                                  final=c.digest(), resource=dict(c.resource),
                                  taught=checkpoint(c, offset, w, feedback, True))
    for parent in ('OLD', 'SAFE'):
        name = 'FREEZE_' + parent
        c, feedback, curve = scalar_parents[parent]
        fork_state = c.digest()
        offset, n = fw + sum(stages[:2]), stages[-1]
        seg = segment(c, tails['SCALAR'], 'SCALAR', world, offset, name, parent,
                      False, feedback, hb, {n // 2, n}, w)
        curve.extend(seg['curve'])
        c.save(states / (name + '_FINAL.npz'))
        r['branches'][name] = dict(prefix='TRUE', recipe=parent, curve_parent=parent,
                                  segments=[seg], curve=curve, fork_state=fork_state,
                                  final=c.digest(), resource=dict(c.resource),
                                  taught=checkpoint(c, offset + n, w, feedback, True))
        assert fork_state == r['branches'][parent]['segments'][-1]['before'] == c.digest()
    r['executed_records'] = 3 * fw + 5 * sum(stages[:2]) + 7 * stages[-1]
    r['correlated_final_histories'] = 7
    r['total_cpu'], r['wall'] = time.process_time() - start, time.monotonic() - wall
    r['peak_rss_bytes'] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    r['archives'] = {p.name: str(p.relative_to(ROOT)) for p in states.glob('*.npz')}
    assert len(r['archives']) == 16
    r['completed'] = True
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=('cycle2', 'cycle3'), required=True)
    p.add_argument('--world', type=int, required=True)
    p.add_argument('--model', choices=MODELS, required=True)
    a = p.parse_args()
    config = D['technical'][a.stage]
    if a.world not in config['worlds']:
        raise ValueError('only fixed technical worlds authorized')
    source = identity()
    lock = json.loads((ROOT / 'SOURCE_LOCK.json').read_text())
    assert source == lock['source']
    out = ROOT / 'cycles' / a.stage
    target = out / 'receipts' / f'{a.world}_{a.model}.json.gz'
    if target.exists() or (out / 'states' / f'{a.world}_{a.model}').exists():
        raise RuntimeError('native trajectory or committed job already exists; no repeat')
    active = out / 'active' / f'{a.world}_{a.model}.json'
    def heartbeat(x):
        atomic(active, dict(x, world=a.world, model=a.model, pid=os.getpid(), heartbeat=time.time()))
    r = job(a.world, a.model, config['binary'], config['mixed'], config['stages'], out, heartbeat)
    from auditor import audit
    t = time.process_time()
    certificate = audit(r, source)
    certificate['audit_cpu'] = time.process_time() - t
    assert identity() == source
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + '.pending')
    with gzip.open(temp, 'wt') as f:
        json.dump(r, f, allow_nan=False)
    temp.replace(target)
    atomic(target.with_suffix('.audit.json'), certificate)
    atomic(active, dict(committed=True, world=a.world, model=a.model, pid=os.getpid(),
                        heartbeat=time.time(), records=r['executed_records']))
    print(json.dumps(dict(world=a.world, model=a.model, records=r['executed_records'],
                          cpu=r['total_cpu'], wall=r['wall'], audit_cpu=certificate['audit_cpu'])), flush=True)


if __name__ == '__main__':
    main()
