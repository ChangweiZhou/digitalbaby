"""Transport-only checkpoint inspection and staging. Does not change locked science."""
from __future__ import annotations
import hashlib,json,re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parent
sys.path.insert(0,str(ROOT/'src'))
from assay import ARMS
from audit_receipts import load,validate
from locked_run import verify_lock


def git(*args):return subprocess.check_output(['git',*args],cwd=REPO,text=True).strip()

def prepare(world):
    lock,locksha=verify_lock()
    request=ROOT/'scratch/publication'/f'{world}.request.json'
    r=json.loads(request.read_text());assert r['world']==world and r['lock_sha256']==locksha
    assert world in lock['worlds']
    head=git('rev-parse','HEAD');remote=git('ls-remote','origin','refs/heads/response-mechanisms-20261001').split()[0]
    if head!=remote:raise AssertionError('local/remote head mismatch; do not overwrite external changes')
    paths=[ROOT/'results/final'/a/f'{world}.json.gz' for a in ARMS]
    if world==300001:paths += [ROOT/'results/final/replays'/a/f'{world}.json.gz' for a in ARMS]
    inspections={}
    allowed=set(ARMS)|{'final','W','N_old','old','new','fact','RESPONSE-MECHANISMS-RECEIPT-v1',
       'RESPONSE-FACT-LIFE-v1','0.61.2','2.2.6','2.2.3','3.11.15','1.14.1','1','none','timing','homeostasis',
       'added_output_bank','Array-state inventory, excludes Python objects and unchanged native stores; RSS measured separately'}
    for p in paths:
        d=load(p)
        validate(d,expected_world=world,expected_arm=p.parent.name,expected_bouts=lock['bouts'],
                 expected_sources=lock['source_hashes'],expected_params=lock['params'],
                 expected_runtime=lock['expected_receipt_runtime'],lock_sha256=locksha)
        strings=[]
        def walk(v):
            if isinstance(v,dict):
                for x in v.values():walk(x)
            elif isinstance(v,list):
                for x in v:walk(x)
            elif isinstance(v,str):strings.append(v)
        walk(d)
        free=sorted(set(s for s in strings if not re.fullmatch('[0-9a-f]{64}',s) and not re.fullmatch('[0-9a-f]{24}',s)))
        if not set(free)<=allowed:raise AssertionError('unexpected free text in synthetic receipt')
        raw=p.read_bytes();relative=str(p.relative_to(REPO));h=hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()
        inspections[relative]=dict(file=relative,world=world,arm=d['arm'],schema=d['schema'],rows=len(d['records']),
            sha256=hashlib.sha256(raw).hexdigest(),git_blob_sha1=h,bytes=len(raw),
            inspection='Exact locked source/parameters/runtime and complete deterministic synthetic fixture, neural responses, tensor reductions and write ledgers validated. Every string exhaustively checked as static metadata token or generated digit cue/hash; remaining values numeric/boolean/null. No personal inputs, credentials, external documents or private communications.',
            all_non_hash_non_cue_strings=free)
    evidence=ROOT/'results/final/publication_inspection'/f'{world}.json';evidence.parent.mkdir(exist_ok=True)
    evidence.write_text(json.dumps(list(inspections.values()),indent=2)+'\n')
    checkpoint=ROOT/'results/final/CHECKPOINT.json'
    checkpoint.write_text(json.dumps(dict(validated_primary_lives=r['completed'],expected_primary_lives=224,last_complete_world=world,validated_replays=7,lock_sha256=locksha,meaning='Validated immutable data included in this checkpoint; live process state may have advanced'),indent=2)+'\n')
    git('add','--','response_mechanisms')
    changed=git('diff','--cached','--name-status').splitlines()
    entries=[]
    for line in changed:
        status,path=line.split('\t',1)
        if status not in ('A','M') or not path.startswith('response_mechanisms/'):
            raise AssertionError('unexpected staged path or action')
        if path.endswith('.json.gz') and status!='A':raise AssertionError('existing immutable receipt modified')
        raw=subprocess.check_output(['git','show',':'+path],cwd=REPO)
        sha=git('rev-parse',':'+path)
        e=dict(path=path,sha=sha,mode='100644',type='blob',bytes=len(raw),binary=path.endswith('.gz'))
        if e['binary']:
            if path not in inspections or inspections[path]['git_blob_sha1']!=sha:raise AssertionError('binary lacks exact inspected provenance')
            e['inspection']=inspections[path]
        else:e['chars']=len(raw.decode('utf8'))
        entries.append(e)
    manifest=dict(world=world,completed=r['completed'],lock_sha256=locksha,parent_sha=head,
                  tree_sha=git('write-tree'),branch='response-mechanisms-20261001',entries=entries)
    target=ROOT/'scratch/publication'/f'{world}.manifest.json';target.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(manifest=str(target),world=world,entries=len(entries),binary=sum(e['binary'] for e in entries),parent_sha=head,tree_sha=manifest['tree_sha'])))

if __name__=='__main__':prepare(int(sys.argv[1]))
