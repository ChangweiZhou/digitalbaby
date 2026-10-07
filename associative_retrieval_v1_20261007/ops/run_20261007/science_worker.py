"""Formal-world adapter; native event/query functions are imported unchanged."""
import argparse
import fcntl
import json
import os
import resource
import time
import traceback
from pathlib import Path
import bootstrap_ops as boot
from contracts import require_sources, require_scope, RSS_CAP
from assays import make_world
from bridge_core import RetrievalCore
import bridge_checkpoint
from dev_worker import record, probe
from io_utils import atomic_json, read_receipt, write_receipt
from science_audit import audit_science

def check_pause(stop_file, deadline):
    return (stop_file is not None and Path(stop_file).exists()) or (deadline is not None and time.time() >= deadline)

def branch_receipt(c, cur, w, lock, stage, branch, qualification, cpu, wall):
    bd = {'births': cur['births'], 'records': cur['records'], 'probes': cur['probes'],
          'final_state_digest': c.state_digest(), 'native_final_digest': c.native_digest(),
          'association_final_digest': c.associations.digest(), 'final_time': c.last_time,
          'association_mutable_bytes': c.associations.mutable_bytes(),
          'fixed_permutation_bytes': c.permutation.nbytes}
    return {'schema': 'LINK_SCIENCE_V1', 'development': False, 'qualification': qualification,
            'world': w['world'], 'assay': w['assay'], 'stage': stage, 'branch': branch,
            'source_identity': lock['native_identity'], 'operations_identity': lock['operations_identity'],
            'fixture_sha256': w['sha256'], 'limit': cur['limit'], 'complete': False,
            'complete_branch': not qualification, 'branches': {branch: bd},
            'cpu_s': cur['prior_cpu_s'] + time.process_time()-cpu,
            'worker_s': cur['prior_worker_s'] + time.monotonic()-wall,
            'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}

def run(args):
    require_scope(args.world, args.assay, args.stage, args.qualification)
    lock = require_sources()
    w = make_world(args.world, args.assay)
    if args.qualification:
        if args.branch not in w['branches'] or not args.limit or not 0 < args.limit <= 16:
            raise ValueError('short qualification guard')
        branches = [args.branch]
        limit = args.limit
    else:
        if args.branch is not None or args.limit is not None or args.stop_after is not None:
            raise ValueError('cannot reduce or interrupt science through qualification options')
        branches = w['branches']
        limit = len(w['events'])
    folder = Path(args.folder)
    for name in ('receipts', 'branches', 'checkpoints', 'active', 'logs', 'job_audits', 'failures'):
        (folder/name).mkdir(parents=True, exist_ok=True)
    key = f'{args.world}_{args.assay}'
    active = folder/'active'/f'{key}.json'
    committed = folder/'receipts'/f'{key}.json.gz'
    if committed.exists():
        audit_science(read_receipt(committed), lock)
        return {'status': 'SKIPPED_COMMITTED', 'job': key}
    # A process lock is released by the kernel, even after a crash. Scientific
    # recovery is still explicit: an orphan checkpoint never implies permission.
    with open(folder/'checkpoints'/f'{key}.lock', 'a+') as job_lock:
        fcntl.flock(job_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        committed_branches = []
        def heartbeat(branch, records, status):
            atomic_json(active, {'pid': os.getpid(), 'world': args.world, 'assay': args.assay,
                                 'stage': args.stage, 'branch': branch, 'completed_records_current_life': records,
                                 'total_records_current_life': limit,
                                 'committed_branch_lives': len(committed_branches),
                                 'status': status, 'updated_unix': time.time(),
                                 'worker_elapsed_s': time.monotonic()-started})
        heartbeat(None, 0, 'BIRTH_PENDING')
        docs = []
        for branch in branches:
            bp = folder/'branches'/f'{key}_{branch}.json.gz'
            cp = folder/'checkpoints'/f'{key}_{branch}.npz'
            if bp.exists():
                d = read_receipt(bp)
                audit_science(d, lock, branch_only=True)
                docs.append(d)
                committed_branches.append(branch)
                cp.unlink(missing_ok=True)
                continue
            if cp.exists() and not args.resume:
                raise ValueError('existing checkpoint requires explicit authorized resume; no automatic replay')
            if check_pause(args.stop_file, args.deadline):
                heartbeat(branch, 0, 'PAUSED_SAFE_BOUNDARY')
                return {'status': 'PAUSED_SAFE_BOUNDARY', 'job': key, 'lives': len(committed_branches)}
            cpu, wall = time.process_time(), time.monotonic()
            if cp.exists():
                c, cur = bridge_checkpoint.load(cp, lock['native_identity'])
                if (cur['world'], cur['assay'], cur['branch'], cur['limit'], cur['operations_identity']) != (
                        args.world, args.assay, branch, limit, lock['operations_identity']):
                    raise ValueError('checkpoint operative identity/job mismatch')
            else:
                c = RetrievalCore()
                cur = {'world': args.world, 'assay': args.assay, 'branch': branch,
                       'limit': limit, 'fixture_sha256': w['sha256'], 'next_record': 0,
                       'records': [], 'probes': [], 'births': c.births,
                       'prior_cpu_s': 0., 'prior_worker_s': 0., 'operations_identity': lock['operations_identity']}
            prior_cpu, prior_wall = cur['prior_cpu_s'], cur['prior_worker_s']
            last_beat = 0.
            for i in range(cur['next_record'], limit):
                cur['records'].append(record(c, w['events'][i], branch))
                cur['next_record'] = i+1
                if not args.qualification:
                    for name in w['boundaries'].get(str(i+1), []):
                        c.flush(w['clocks'][name])
                        cur['probes'].append(probe(c, w, w['clocks'][name], name, i+1))
                pausing = check_pause(args.stop_file, args.deadline) or args.stop_after == i+1
                if (i+1) % 48 == 0 or pausing or i+1 == limit:
                    cur['prior_cpu_s'] = prior_cpu + time.process_time()-cpu
                    cur['prior_worker_s'] = prior_wall + time.monotonic()-wall
                    bridge_checkpoint.save(c, cp, cur, lock['native_identity'])
                if time.monotonic()-last_beat >= 3 or pausing or i+1 == limit:
                    heartbeat(branch, i+1, 'PAUSED_SAFE_BOUNDARY' if pausing else 'EXECUTING')
                    last_beat = time.monotonic()
                if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > RSS_CAP:
                    raise MemoryError('worker exceeded fixed 1 GiB measured RSS cap')
                if pausing:
                    return {'status': 'PAUSED_SAFE_BOUNDARY', 'job': key,
                            'records': i+1, 'lives': len(committed_branches)}
            if args.qualification:
                cur['probes'].append(probe(c, w, c.last_time, 'short_end', limit))
            # cur prior charges already include this segment at its last save;
            # use segment-start totals to avoid charging the same segment twice.
            cur['prior_cpu_s'], cur['prior_worker_s'] = prior_cpu, prior_wall
            d = branch_receipt(c, cur, w, lock, args.stage, branch, args.qualification, cpu, wall)
            audit_science(d, lock, branch_only=True)
            require_sources()
            write_receipt(bp, d)
            docs.append(d)
            committed_branches.append(branch)
            cp.unlink(missing_ok=True)
            heartbeat(branch, limit, 'LIFE_COMMITTED')
            print(json.dumps({'status': 'LIFE_COMMITTED', 'job': key, 'branch': branch}), flush=True)
        merged = dict(docs[0])
        merged['branches'] = {b: bd for d in docs for b, bd in d['branches'].items()}
        merged['complete'] = not args.qualification
        merged['cpu_s'] = sum(d['cpu_s'] for d in docs)
        merged['worker_s'] = sum(d['worker_s'] for d in docs)
        merged['peak_rss_bytes'] = max(d['peak_rss_bytes'] for d in docs)
        a = audit_science(merged, lock)
        require_sources()
        write_receipt(committed, merged)
        atomic_json(folder/'job_audits'/f'{key}.json', a)
        active.unlink(missing_ok=True)
        return {'status': 'COMMITTED', 'job': key, 'lives': len(branches),
                'records': limit*len(branches), 'receipt': str(committed)}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--world', type=int, required=True)
    p.add_argument('--assay', choices=('lifetime', 'reuse'), required=True)
    p.add_argument('--stage', choices=('screen', 'confirm', 'qualification'), required=True)
    p.add_argument('--folder', required=True)
    p.add_argument('--stop-file')
    p.add_argument('--deadline', type=float)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--qualification', action='store_true')
    p.add_argument('--branch')
    p.add_argument('--limit', type=int)
    p.add_argument('--stop-after', type=int)
    args = p.parse_args()
    try:
        result = run(args)
        print(json.dumps(result), flush=True)
        return 75 if result['status'] == 'PAUSED_SAFE_BOUNDARY' else 0
    except Exception as e:
        doc = {'error': repr(e), 'traceback': traceback.format_exc(), 'pid': os.getpid(),
               'world': args.world, 'assay': args.assay, 'stage': args.stage, 'at_unix': time.time()}
        atomic_json(Path(args.folder)/'failures'/f'{args.world}_{args.assay}.json', doc)
        print(doc['traceback'], flush=True)
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
