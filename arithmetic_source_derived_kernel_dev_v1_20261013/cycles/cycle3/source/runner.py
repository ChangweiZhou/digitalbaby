"""Only fixed technical worlds are admitted. No DEV or confirmation launch."""
import argparse
import collections
import gzip
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path
import torch
from core import Learner, ARMS
from worlds import make_world, schedule, shuffle_targets, probes

ROOT = Path(__file__).resolve().parent


def identity():
    local = ('core.py','worlds.py','runner.py','auditor.py','reducer.py','tests.py','reference_oracles.py','DESIGN.json','operations.py','supplement_tests.py','inference.py')
    result = {f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in local}
    lineage = json.loads((ROOT/'SOURCE_LINEAGE.json').read_text())
    for x in lineage['files']:
        result['external:'+x['path']] = hashlib.sha256((ROOT.parent/x['path']).read_bytes()).hexdigest()
    return result


class Shortcuts:
    def __init__(self, records):
        self.counts = {name: collections.defaultdict(collections.Counter) for name in ('exact','canonical','length','suffix','bag')}
        self.global_count = collections.Counter()
        for r in records:
            b = bytes.fromhex(r['prompt'])
            keys = dict(exact=r['prompt'], canonical=r['group'], length=len(b), suffix=b[-2:], bag=tuple(sorted(b)))
            self.global_count[r['reply']] += 1
            for name, key in keys.items(): self.counts[name][key][r['reply']] += 1

    @staticmethod
    def mode(c):
        return min(c, key=lambda a: (-c[a], a)) if c else None

    def predict(self, r):
        b = bytes.fromhex(r['prompt'])
        keys = dict(exact=r['prompt'], canonical=r['group'], length=len(b), suffix=b[-2:], bag=tuple(sorted(b)))
        out = {name: self.mode(self.counts[name].get(key, {})) for name, key in keys.items()}
        out['global'] = self.mode(self.global_count)
        out['length'] = out['length'] or out['global']
        out['suffix'] = out['suffix'] or out['global']
        out['bag'] = out['bag'] or out['length']
        return out


def score(core, rows, label, counter):
    before = core.digest()
    generated = core.generate_batch([bytes.fromhex(r['prompt']) for r in rows])
    out = []
    for r, (answer, status) in zip(rows, generated):
        out.append(dict(prompt=r['prompt'], expected=r['reply'], operation=r['operation'], group=r['group'],
                        actual=answer.hex(), status=status,
                        exact=answer.hex()==r['reply'] and status=='EOS', shortcuts=counter.predict(r)))
    after = core.digest()
    assert before == after, 'probe mutated learning state'
    return dict(label=label, before=before, after=after, rows=out)


def checkpoint(c, n, held, feedback):
    counter = Shortcuts(feedback)
    scores = [score(c, held[0], 'old_held', counter), score(c, held[1], 'new_held', counter)]
    return dict(external_W_coordinates=n, probes=scores, resource=dict(c.resource),
                production_cpu=c.resource['training_cpu']+c.resource['generation_cpu'])


def train(c, stream, allowed, label, checkpoint_fn, heartbeat, branch):
    before = c.digest(); counters = dict(c.resource); ledger = []
    for i, r in enumerate(stream):
        old_steps = c.steps
        pred = c.observe(bytes.fromhex(r['prompt']), bytes.fromhex(r['reply']), allowed)
        ledger.append(dict(index=i, prompt=r['prompt'], reply=r['reply'], operation=r['operation'],
                           group=r['group'], allowed=allowed, predictions=pred,
                           optimizer_steps=c.steps-old_steps))
        checkpoint_fn(i+1)
        if i % 64 == 0 or i+1 == len(stream):
            heartbeat(dict(branch=branch, phase=label, completed_segment_records=i+1,
                           segment_records=len(stream), partial=True))
    after = c.digest()
    if not allowed: assert before == after, 'whole no-write failed'
    return dict(label=label, allowed=allowed, before=before, after=after,
                counts_before=counters, counts_after=dict(c.resource), records=ledger)


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.pending-'+str(os.getpid()))
    temp.write_text(json.dumps(value, indent=2)+'\n'); os.replace(temp, path)


def job(world_id, arm, binary_n, mixed_n, heartbeat=lambda _: None):
    started = time.process_time(); wall = time.monotonic()
    rss=lambda: int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024)
    import_peak=rss()
    world = make_world(world_id)
    old = schedule(world, 'binary', binary_n); new = schedule(world, 'mixed', mixed_n)
    shuffled = shuffle_targets(old, world_id+19)
    family = 'ELM' if arm.startswith('E') else arm
    seed = int.from_bytes(hashlib.sha256(f'SOURCE_CELL|{world_id}|{family}'.encode()).digest()[:8], 'big') % (2**63)
    birth = Learner(arm, seed); initial = birth.digest()
    held = (probes(world['binary_held'], None), probes(world['mixed_held'], None))
    r = dict(schema='SOURCE_CELL_TECHNICAL_JOB', technical=True, completed=False,
             world=world_id, arm=arm, binary_n=binary_n, mixed_n=mixed_n,
             data_digest=world['data_digest'], source=identity(), initial_digest=initial,
             model_seed=seed, framework=torch.__version__,
             trainable_parameters=sum(p.numel() for p in birth.net.parameters() if p.requires_grad),
             branches={}, counted_queries='all128oldheld and256mixedheld each declared checkpoint; diagnostic taught endpoints separate')
    prefixes = {}
    for name, stream, allowed in [('W', old, True), ('N_BINARY', old, False), ('SH_BINARY', shuffled, True)]:
        c = birth.clone(); assert c.digest() == initial
        curve = [checkpoint(c, 0, held, [])]
        def old_checkpoint(n):
            if n in (256,512,1024,2048,4096,8192):
                curve.append(checkpoint(c, n, held, stream[:n] if allowed else []))
        seg = train(c, stream, allowed, 'binary', old_checkpoint, heartbeat, name)
        prefixes[name] = c
        r['branches'][name] = dict(segments=[seg], binary_curve=curve)
    w_prefix = prefixes['W'].clone()
    for name in ('W','N_BINARY','SH_BINARY','N_MIXED'):
        c = w_prefix.clone() if name == 'N_MIXED' else prefixes[name]
        old_feedback = old if name in ('W','N_MIXED') else shuffled if name == 'SH_BINARY' else []
        curve = []
        def mixed_checkpoint(n):
            if n in (64,128,256,512,1024,2048):
                feedback = old_feedback+(new[:n] if name != 'N_MIXED' else [])
                curve.append(checkpoint(c, binary_n+n, held, feedback))
        seg = train(c, new, name != 'N_MIXED', 'mixed', mixed_checkpoint, heartbeat, name)
        branch = r['branches'].setdefault(name, dict(segments=[], binary_curve_parent='W'))
        branch['segments'].append(seg); branch['mixed_curve'] = curve
        branch['final_state'] = c.digest()
        branch['state_bytes'] = c.state_bytes()
        branch['last_gradient'] = dict(c.last_gradient); branch['last_update'] = dict(c.last_update)
        feedback = old_feedback+(new if name != 'N_MIXED' else [])
        counter = Shortcuts(feedback)
        # Actual truth rows from W, including SH_BINARY diagnostics. Never use
        # shuffled replies as the expected answers of taught-cue probes.
        branch['taught_final'] = [score(c, probes(old,None), 'old_taught',counter),
                                 score(c, probes(new,None), 'new_taught',counter)]
        branch['resource'] = dict(c.resource)
    r['completed'] = True
    r['executed_external_records'] = 3*binary_n+4*mixed_n
    r['permitted_optimizer_steps'] = 2*binary_n+3*mixed_n
    r['correlated_final_histories'] = 4
    r['total_process_cpu'] = time.process_time()-started
    r['total_wall'] = time.monotonic()-wall
    r['peak_process_rss_job_bytes']=rss()
    r['import_peak_process_rss_bytes']=import_peak
    r['memory_scope']='process peak includes runtime, data, curves, receipt and native model; not isolated tensor allocator peak'
    return r


def main():
    p = argparse.ArgumentParser(); p.add_argument('--cycle', type=int, required=True)
    p.add_argument('--arm', choices=ARMS, required=True); a = p.parse_args()
    if a.cycle not in (2,3): raise RuntimeError('only fixed technical cycles admitted')
    world, binary, mixed = (202610138902,1024,256) if a.cycle == 2 else (202610138903,8192,2048)
    out = ROOT/f'cycles/cycle{a.cycle}'
    target = out/f'receipts/{world}_{a.arm}.json.gz'
    if target.exists(): raise RuntimeError('committed technical job already exists; no repeat')
    active = out/f'active/{a.arm}.json'
    started = time.time()
    def hb(progress):
        atomic_json(active, dict(progress, arm=a.arm, world=world, pid=os.getpid(), heartbeat=time.time(), started=started))
    r = job(world,a.arm,binary,mixed,hb)
    from auditor import audit
    audit_started = time.process_time(); certificate = audit(r,identity())
    certificate['independent_audit_cpu'] = time.process_time()-audit_started
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name+'.pending-'+str(os.getpid()))
    with gzip.open(temp,'wt') as f: json.dump(r,f)
    os.replace(temp,target)
    atomic_json(target.with_suffix('.audit.json'),certificate)
    atomic_json(active,dict(arm=a.arm,world=world,pid=os.getpid(),heartbeat=time.time(),committed=True,records=r['executed_external_records']))
    print(json.dumps(dict(cycle=a.cycle,arm=a.arm,committed_records=r['executed_external_records'],cpu=r['total_process_cpu'],audit_cpu=certificate['independent_audit_cpu'])),flush=True)


if __name__ == '__main__': main()
