"""Operations and manifests only. No learner, task features, or learning rule."""
import datetime
import hashlib
import json
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = HERE / "results"
OPS = HERE / "ops/background_20261017"
DESIGN = ROOT / "native_context_binding_design_v2_1_20261017/DESIGN.json"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity():
    paths = list(HERE.glob("*.py")) + [HERE / "OUTPUT_CODEBOOK.json"]
    return {p.name: sha(p) for p in sorted(paths)}


def verify_lock():
    lock=read(HERE/'SOURCE_LOCK.json')
    for relative,digest in lock['operative_sources'].items():
        if sha(ROOT/relative)!=digest:raise PermissionError('Technical source lock mismatch: '+relative)
    campaign=read(HERE/'CAMPAIGN_SOURCE_LOCK.json')
    if campaign['identity']!=identity():raise PermissionError('Campaign source lock changed')
    if campaign['technical_lock_sha256']!=sha(HERE/'SOURCE_LOCK.json'):raise PermissionError('Technical lock changed')
    return campaign


def manifest(stage,selected=None):
    if stage!='DEV_A':raise PermissionError('Only DEV_A is admitted; no automatic continuation')
    spec=read(HERE/'SPEC.json')['dev']
    worlds=spec['worlds'][:4];arms=spec['roster']
    jobs=[dict(world=w,arm=a,key=f'{w}_{a}') for w in worlds for a in arms]
    return dict(schema='NATIVE_BIND_CAMPAIGN_MANIFEST_V2_1',stage=stage,science=True,
        worlds=worlds,roster=arms,jobs=jobs,old_n=spec['old_records_per_history'],
        new_n=spec['new_records_per_history'],full_jobs=len(jobs),histories=2*len(jobs),
        records=2*len(jobs)*(spec['old_records_per_history']+spec['new_records_per_history']),
        source_lock_sha256=sha(HERE/'CAMPAIGN_SOURCE_LOCK.json'),confirmation_admission_sha256=None)


def admission(stage,arm,world,authorization):
    verify_lock();m=manifest(stage);auth=read(authorization)
    if auth.get('schema')!='NATIVE_BIND_RUN_AUTHORIZATION_V2_1' or not auth.get('human_authorized'):
        raise PermissionError('Missing explicit human launch authorization')
    if auth.get('sessions')!=['DEV_A'] or stage not in auth['sessions']:
        raise PermissionError('Session is not authorized')
    if arm not in m['roster'] or world not in m['worlds']:raise PermissionError('Out of fixed roster')
    return m['old_n'],m['new_n'],[0,16,64,256,512,768,1024],None


def job_folder(stage, job):
    return RESULTS / stage / "jobs" / job["key"]


def committed():
    jobs = []
    for stage in ("DEV_A",):
        for path in sorted((RESULTS / stage / "jobs").glob("*/AUDIT.json")):
            certificate = read(path)
            receipt = path.with_name("RECEIPT.json")
            if not certificate.get("accepted") or not receipt.is_file():
                continue
            if certificate["receipt_sha256"] != sha(receipt):
                raise AssertionError("Committed receipt changed: " + str(receipt))
            jobs.append(dict(stage=stage, world=certificate["world"], arm=certificate["arm"],
                             records=certificate["records"], receipt=str(receipt),
                             receipt_sha256=certificate["receipt_sha256"]))
    keys = [(j["stage"], j["world"], j["arm"]) for j in jobs]
    if len(keys) != len(set(keys)):
        raise AssertionError("Duplicate committed science job")
    return jobs


def update_index():
    jobs = committed()
    value = dict(schema="NATIVE_BIND_COMMIT_INDEX_V1", updated_utc=now(),
                 full_jobs=len(jobs), histories=2*len(jobs),
                 records=sum(j["records"] for j in jobs), jobs=jobs)
    write(RESULTS / "COMMIT_INDEX.json", value)
    return value


def status(state, **extra):
    index = update_index()
    value = dict(schema="NATIVE_BIND_CAMPAIGN_STATUS_V1", status=state, updated_utc=now(),
                 heartbeat_epoch=time.time(), supervisor_pid=os.getpid(),
                 committed_full_jobs=index["full_jobs"], committed_histories=index["histories"],
                 committed_records=index["records"], technical_records=412,
                 science=True, **extra)
    write(RESULTS / "STATUS.json", value)
    return value
