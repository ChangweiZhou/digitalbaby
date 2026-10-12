"""Real filesystem synthetic commits/status/finish, no native training."""
import hashlib,json,threading,time,zipfile
from concurrent.futures import ThreadPoolExecutor
from runner import atomic_json

def finish_allowed(count,expected,audit,failed=False):
    return count==expected and audit is True and failed is False

def suite(folder):
    folder.mkdir(parents=True,exist_ok=False);receipts=folder/'receipts';receipts.mkdir()
    status=folder/'STATUS.json';index=folder/'COMMIT_INDEX.json'
    atomic_json(dict(stage='RUNNING',committed=0,native_records=0),status)
    finished=threading.Event();reads=[0];started=time.monotonic()
    def writer():
        items=[]
        for i in range(40):
            path=receipts/f'{i}.json';atomic_json(dict(job=i,native_records=0,synthetic=True),path)
            items.append(dict(job=i,path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            atomic_json(dict(commits=items),index)
            atomic_json(dict(stage='RUNNING',committed=len(items),native_records=0),status)
        finished.set()
    def reader():
        last=-1
        while not finished.is_set() or reads[0]<80:
            row=json.loads(status.read_text());assert row['native_records']==0 and last<=row['committed']<=40
            last=row['committed'];reads[0]+=1
            if reads[0]>100000:raise AssertionError('status not bounded')
    with ThreadPoolExecutor(max_workers=2) as pool:
        a=pool.submit(writer);b=pool.submit(reader);a.result(timeout=20);b.result(timeout=20)
    certificate=json.loads(index.read_text());assert len(certificate['commits'])==40
    for row in certificate['commits']:
        assert hashlib.sha256((receipts/row['path']).read_bytes()).hexdigest()==row['sha256']
    assert finish_allowed(40,40,True)
    assert not any((finish_allowed(39,40,True),finish_allowed(40,40,False),finish_allowed(40,40,True,True)))
    target=folder/'SYNTHETIC_BUNDLE.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        z.write(status,status.name);z.write(index,index.name)
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None and set(z.namelist())=={'STATUS.json','COMMIT_INDEX.json'}
    result=dict(verdict='PASS_FILESYSTEM_SYNTHETIC_ONLY',synthetic_commits=40,concurrent_single_status_reads=reads[0],
                monitor_receipt_decompressions=0,negative_finish_cases=3,source_or_science_trajectories_executed=0,
                wall_seconds=time.monotonic()-started,actual_science_ten_hour_dispatcher_qualified=False)
    atomic_json(result,folder/'RESULT.json');return result
