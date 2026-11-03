"""Engineering-only interruption and artifact checks for the final V42 source."""
from pathlib import Path
import fcntl,hashlib,json,os,shutil,signal,subprocess,sys,time,zipfile
root=Path(__file__).resolve().parent
script=root.parent/'elm_v42_read_authority.py'
work=root/('cli_verification_'+hashlib.sha256(script.read_bytes()).hexdigest()[:10]);work.mkdir(exist_ok=True)
checks={};logs={}

def command(out,extra=()):
    return [sys.executable,str(script),'--smoke','--out',str(out),*extra]

def run(out,extra=(),expected=0):
    p=subprocess.run(command(out,extra),capture_output=True,text=True,timeout=60)
    assert p.returncode==expected,(p.returncode,p.stderr,p.stdout)
    return p

def result(out):return (out/'completed/seed_942001.json').read_bytes()

baseline=work/'baseline';p=run(baseline,['--checkpoint-records','7'])
logs['baseline']=p.stdout;expected=result(baseline)
for sig,name in [(signal.SIGINT,'sigint'),(signal.SIGKILL,'sigkill')]:
    out=work/name
    with (work/(name+'.log')).open('w') as log:
        p=subprocess.Popen(command(out,['--checkpoint-records','1']),stdout=log,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            progress=out/'progress/seed_942001.json'
            if progress.exists():
                x=json.loads(progress.read_text())
                if 0<x['processed_units']<x['total_units']:break
            assert p.poll() is None,'Engineering process finished before interrupt was sent'
            time.sleep(.001)
        else:raise AssertionError('Engineering checkpoint not reached')
        p.send_signal(sig);rc=p.wait(timeout=30)
    assert rc==(0 if sig==signal.SIGINT else -signal.SIGKILL),rc
    assert (out/'checkpoints/seed_942001.checkpoint').exists()
    assert not (out/'completed/seed_942001.json').exists()
    resumed=run(out,['--checkpoint-records','3'])
    assert result(out)==expected,name+' changed final result'
    logs[name+'_resume']=resumed.stdout
    checks[name+'_checkpoint_resume_identical']=True

p=run(baseline);final=json.loads(p.stdout.splitlines()[-1])
assert final['new_evaluation_units']==0 and final['new_completed_jobs']==0 and final['reused_jobs']==1
assert result(baseline)==expected
checks['completed_rerun_zero_evaluation_work']=True;logs['noop']=p.stdout

out=work/'changed_plan';shutil.copytree(baseline,out,dirs_exist_ok=True)
plan=json.loads((out/'PLAN.json').read_text());plan['source_sha256']='changed'
(out/'PLAN.json').write_text(json.dumps(plan))
p=run(out,expected=1);assert 'differs from saved run' in p.stderr
checks['changed_plan_rejected']=True

out=work/'corrupt_result';shutil.copytree(baseline,out,dirs_exist_ok=True)
rp=out/'completed/seed_942001.json';x=json.loads(rp.read_text());x['data']['seed']=-1;rp.write_text(json.dumps(x))
p=run(out,expected=1);assert 'checksum mismatch' in p.stderr
checks['corrupt_completed_record_rejected']=True

out=work/'corrupt_checkpoint';run(out,['--stop-after-checkpoints','1'])
cp=out/'checkpoints/seed_942001.checkpoint';bad=cp.read_bytes()[:-3]+b'bad';cp.write_bytes(bad)
p=run(out,expected=1);assert 'Corrupt checkpoint' in p.stderr
assert cp.read_bytes()==bad
checks['corrupt_checkpoint_rejected_without_overwrite']=True

out=work/'writer_lock';out.mkdir(exist_ok=True)
with (out/'.execution.lock').open('a+') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=run(out,expected=1);assert 'Another writer' in p.stderr
checks['concurrent_writer_rejected']=True

zpath=baseline.parent/(baseline.name+'_results.zip')
with zipfile.ZipFile(zpath) as z:
    assert z.testzip() is None
    m=json.loads(z.read('ARTIFACT_HASHES.json'))
    assert set(m)==set(z.namelist())-{'ARTIFACT_HASHES.json'}
    assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in m.items())
checks['archive_crc_manifest_and_hashes']=True
report={'source_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),'checks':checks,'logs':logs,'new_scientific_policy_runs':0}
(root/'cli_verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'checks':checks,'verification_file':str(root/'cli_verification.json')}))
