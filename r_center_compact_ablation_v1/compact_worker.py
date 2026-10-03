"""One exact world/arm job. Completed receipts are never rerun or overwritten."""
import argparse
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path
from compact_bridge import ROOT, make_core
from compact_fixture import make_world, BRANCHES, SCIENCE_WORLDS, DEVELOPMENT_WORLDS
from compact_experiment import instrument, record, probe, checkpoints_after
from compact_checkpoint import save, load
from compact_integrity import verify
from compact_storage import atomic_json, commit_receipt
from compact_audit import audit_receipt, read


def run_job(world_id, arm, folder, identity, *, resume=False, stop_after=None, progress_callback=None,
            record_limit=864, branch_limit=4):
    began = time.monotonic()
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    import fcntl
    lockdir = folder / 'locks'; lockdir.mkdir(exist_ok=True)
    mutex = (lockdir / f'{world_id}_{arm}.lock').open('a')
    fcntl.flock(mutex, fcntl.LOCK_EX | fcntl.LOCK_NB)
    receipt_path = folder / 'receipts' / f'{world_id}_{arm}.json.gz'
    cp = folder / 'checkpoints' / f'{world_id}_{arm}.npz'
    active = folder / 'active' / f'{world_id}_{arm}.json'
    complete = record_limit == 864 and branch_limit == 4
    if not complete and world_id not in DEVELOPMENT_WORLDS: raise ValueError('short jobs are engineering-only')
    if receipt_path.exists():
        doc = read(receipt_path); audit_receipt(doc, identity, complete=complete)
        return doc, {'already_completed': True}
    world = make_world(world_id)
    if cp.exists():
        if not resume: raise ValueError('unfinished checkpoint requires explicit resume')
        core, cursor = load(cp, arm, identity)
        if (cursor['world'] != world_id or cursor['arm'] != arm or cursor['fixture_sha256'] != world['sha256']
                or cursor['limits'] != [record_limit, branch_limit]):
            raise ValueError('checkpoint job mismatch')
    else:
        core = make_core(arm)
        cursor = {'world': world_id, 'arm': arm, 'fixture_sha256': world['sha256'], 'branch_index': 0,
                  'next_record': 0, 'branches': {}, 'births': core.births, 'worker_s_prior': 0.,
                  'records_this_attempt': 0, 'limits': [record_limit, branch_limit]}
    loaded_worker = cursor['worker_s_prior']
    trace = instrument(core)
    for bi in range(cursor['branch_index'], branch_limit):
        branch = BRANCHES[bi]
        cursor['branches'].setdefault(branch, {'records': [], 'probes': [], 'births': core.births})
        bdoc = cursor['branches'][branch]
        for i in range(cursor['next_record'], record_limit):
            bdoc['records'].append(record(core, world['events'][i], branch, trace))
            for name in checkpoints_after(i + 1):
                core.flush(world['clocks'][name])
                bdoc['probes'].append(probe(core, world, world['clocks'][name], name))
            cursor['next_record'] = i + 1
            cursor['records_this_attempt'] += 1
            if (i + 1) % 32 == 0:
                cursor['worker_s_prior'] = loaded_worker + time.monotonic() - began
                save(core, cp, cursor, identity)
                atomic_json(active, {'pid': os.getpid(), 'world': world_id, 'arm': arm, 'branch': branch,
                                     'committed_record_cursor': i + 1, 'heartbeat_unix': time.time()})
                if progress_callback: progress_callback(branch, i + 1)
                if stop_after is not None and cursor['records_this_attempt'] >= stop_after:
                    return None, {'deliberate_technical_stop': True, 'next_record': i + 1}
        cursor['branch_index'] = bi + 1; cursor['next_record'] = 0
        bdoc['final_private_digest'] = [m.state_digest() for m in core.private]
        if bi + 1 < branch_limit:
            # Branches start from identically born, untrained state; each physical store still has its own birth call.
            core = make_core(arm); trace = instrument(core)
        cursor['worker_s_prior'] = loaded_worker + time.monotonic() - began
        save(core, cp, cursor, identity)
    doc = {k: cursor[k] for k in ('world', 'arm', 'fixture_sha256', 'branches', 'births')}
    doc.update(identity=identity, schema='COMPACT_SCIENCE_RECEIPT_V1',
               worker_s=loaded_worker + time.monotonic() - began,
               peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == 'Darwin' else 1024),
               runtime={'python': platform.python_version()}, logical_lives=branch_limit,
               engineering_only=not complete or world_id in DEVELOPMENT_WORLDS, completed_unix=time.time())
    audit_receipt(doc, identity, complete=complete)
    commit_receipt(receipt_path, doc)
    if active.exists(): active.unlink()
    return doc, {'already_completed': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world', type=int, required=True); parser.add_argument('--arm', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.world not in SCIENCE_WORLDS: raise ValueError('not a locked science world')
    identity = verify()
    qualification = json.loads((ROOT / 'results/QUALIFICATION.json').read_text())
    if qualification['verdict'] != 'PASS' or qualification['identity'] != identity:
        raise ValueError('science requires final qualified source')
    def update(branch, cursor):
        print(json.dumps({'world': args.world, 'arm': args.arm, 'branch': branch, 'cursor': cursor}), flush=True)
    doc, state = run_job(args.world, args.arm, ROOT / 'results/science', identity,
                         resume=args.resume, progress_callback=update)
    print(json.dumps({'complete': True, 'world': args.world, 'arm': args.arm, **state,
                      'worker_s': doc['worker_s']}), flush=True)


if __name__ == '__main__': main()
