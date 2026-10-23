"""Prepare a scoped, validated publication snapshot without blocking computation.

A separate connector-based publisher uploads manifest blobs, creates a normal
fast-forward commit, updates the branch, and verifies that exact remote commit.
The private Git index here does not change a running checkout or its main index.
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; REPO = ROOT.parent
sys.path.insert(0, str(ROOT/'src'))
from locked_run import verify_lock, atomic
from audit_receipts import load, validate


def prepare(base, world, receipts_only=False):
    lock, digest = verify_lock(); assert world in lock['worlds']
    dest = ROOT/'results/final'; spool = ROOT/'scratch/recovery_publication'; spool.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, GIT_INDEX_FILE=str(spool/'index'))
    def git(*args):
        return subprocess.check_output(['git',*args],cwd=REPO,env=env,text=True).strip()
    assert re.fullmatch('[0-9a-f]{40}',base)
    git('read-tree',base)
    done = lock['worlds'][:lock['worlds'].index(world)+1]
    if world != 300001:
        queue = json.loads((dest/'publication_queue'/f'{world}.json').read_text())
        assert queue['lock_sha256']==digest and queue['completed']==len(done)*7
    paths = [dest/a/f'{w}.json.gz' for w in done for a in lock['arms']]
    paths += [dest/'replays'/a/f'{w}.json.gz' for w,a in lock['replay_jobs']]
    assert all(p.exists() for p in paths)
    receipt_paths = list(paths)
    paths += [ROOT/'RECOVERY_AMENDMENT_20261001.md',ROOT/'ops',dest/'RUN_LEDGER.json',
        dest/'RUN_STATUS.json',dest/'REPLAY_AUDIT.json',dest/'RECOVERY_LAUNCH_AUDIT.md']
    paths += [p for p in (dest/'operations').glob('*.json')]
    paths += [p for w in done for p in (dest/'operations').glob(f'{w}-*.txt')]
    paths += [p for w in done for p in (dest/'publication_queue').glob(f'{w}.json')]
    paths += [p for p in (dest/'publication').glob('*.json')]
    paths += [p for p in (dest/'publication_inspection').glob('*.json')]
    if world==lock['worlds'][-1]:
        paths += [p for n in ('FINAL_METRICS.json','REPORT.md','FINAL_TESTS.txt','ANALYSIS_LOG.txt',
                    'OPERATIONAL_AUDIT.json','OPERATIONAL_AUDIT.md','INDEPENDENT_AUDIT.md') if (p:=dest/n).exists()]
    if receipts_only:
        paths = receipt_paths
    paths = [str(p.relative_to(REPO)) for p in paths if p.exists()]
    git('add','--',*paths)
    changed=git('diff','--cached',base,'--name-status').splitlines()
    inspections=[]
    allowed=set(lock['arms'])|{'final','W','N_old','old','new','fact','RESPONSE-MECHANISMS-RECEIPT-v1',
        'RESPONSE-FACT-LIFE-v1','0.61.2','2.2.6','2.2.3','3.11.15','1.14.1','1','none','timing','homeostasis',
        'added_output_bank','Array-state inventory, excludes Python objects and unchanged native stores; RSS measured separately'}
    for line in changed:
        status,path=line.split('\t',1)
        assert status in ('A','M') and path.startswith('response_mechanisms/')
        if not path.endswith('.json.gz'):continue
        assert status=='A', 'Existing receipt changed'
        p=REPO/path;d=load(p)
        assert d['world'] in done or ('/replays/' in path and d['world']==300001)
        validate(d,expected_world=d['world'],expected_arm=p.parent.name,expected_bouts=lock['bouts'],
            expected_sources=lock['source_hashes'],expected_params=lock['params'],
            expected_runtime=lock['expected_receipt_runtime'],lock_sha256=digest)
        strings=[]
        def walk(v):
            if isinstance(v,dict):
                for x in v.values():walk(x)
            elif isinstance(v,list):
                for x in v:walk(x)
            elif isinstance(v,str):strings.append(v)
        walk(d)
        free=sorted(set(s for s in strings if not re.fullmatch('[0-9a-f]{64}',s) and not re.fullmatch('[0-9a-f]{24}',s)))
        assert set(free)<=allowed, 'Unexpected free text in synthetic receipt'
        raw=p.read_bytes();sha=hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()
        assert sha==git('rev-parse',':'+path)
        inspections.append(dict(path=path,world=d['world'],arm=d['arm'],bytes=len(raw),git_blob_sha1=sha,
            sha256=hashlib.sha256(raw).hexdigest(),rows=len(d['records']),all_non_hash_non_cue_strings=free,
            inspection='Original locked source, parameters, runtime and deterministic synthetic fixture validated; exhaustive value-string scan contains only static metadata or generated cue/hash strings. Remaining values numeric, boolean or null. No personal inputs, credentials or communications.'))
    inspection=dest/'publication_inspection'/f'{"receipts" if receipts_only else "recovery"}-through-{world}.json'
    atomic(inspection,dict(up_to_world=world,primary_receipt_count=len(done)*7,lock_sha256=digest,receipts=inspections))
    checkpoint=dest/('RECEIPT_CHECKPOINT.json' if receipts_only else 'CHECKPOINT.json')
    atomic(checkpoint,dict(validated_primary_lives_in_this_and_prior_checkpoints=len(done)*7,
        expected_primary_lives=224,last_complete_world_in_checkpoint=world,validated_replays=7,
        lock_sha256=digest,scope='Synthetic receipt files and validation manifest only' if receipts_only else 'Receipts and authorized project operational records',
        meaning='Exact scope of this Git publication snapshot. Local computation may be ahead; remote durability requires verified commit.'))
    git('add','--',str(inspection.relative_to(REPO)),str(checkpoint.relative_to(REPO)))
    changed=git('diff','--cached',base,'--name-status').splitlines();entries=[]
    known=set(line.split()[2] for line in git('ls-tree','-r',base).splitlines())
    for line in changed:
        status,path=line.split('\t',1);assert status in ('A','M') and path.startswith('response_mechanisms/')
        sha=git('rev-parse',':'+path);raw=subprocess.check_output(['git','cat-file','blob',sha],cwd=REPO)
        if not path.endswith('.gz'):raw.decode('utf8')
        if path.endswith('.json'):json.loads(raw)
        entries.append(dict(path=path,sha=sha,mode='100644',type='blob',bytes=len(raw),
            encoding='base64' if path.endswith('.gz') else 'utf-8',already_in_base=sha in known))
    manifest=dict(base_commit=base,base_tree=git('rev-parse',base+'^{tree}'),tree_sha=git('write-tree'),
        branch='response-mechanisms-20261001',world=world,primary_receipts=len(done)*7,lock_sha256=digest,receipts_only=receipts_only,entries=entries)
    target=spool/f'{world}.manifest.json';atomic(target,manifest)
    print(json.dumps(dict(manifest=str(target),entries=len(entries),new_receipts=len(inspections),base_commit=base,tree_sha=manifest['tree_sha'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--world',required=True,type=int);p.add_argument('--receipts-only',action='store_true')
    a=p.parse_args();prepare(a.base,a.world,a.receipts_only)
