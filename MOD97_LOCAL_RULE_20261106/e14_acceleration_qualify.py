"""Full-life speed candidate validation. Scientific E14 controller remains untouched."""
import os,sys,subprocess,json,time,hashlib
from pathlib import Path
from e9_campaign import read,save
R=Path(__file__).resolve().parent;O=R/'exp45'
def clean(v):
 if isinstance(v,dict):return {k:clean(x) for k,x in v.items() if k!='cpu'}
 if isinstance(v,list):return [clean(x) for x in v]
 return v

def main():
 status=R/'E14_ACCELERATION_STATUS.json'
 if status.exists():raise RuntimeError('Qualification already exists, do not duplicate')
 files={n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in ('e14_fast_inference.py','e14_acceleration_worker.py','e14_acceleration_qualify.py','E14_SPEED_BENCHMARK.json')}
 save(R/'E14_ACCELERATION_LOCK.json',dict(scope='Extra engineering qualification only, not an E14 candidate condition; active campaign unchanged',files=files,reason='Inference speed candidate changes logit arithmetic slightly; require exact complete scientific receipt parity before any deployment',unchanged='Training loss,optimizer,updates,data,seeds,store,diagnostics all original E14 settings',budget='One extra full logged C73129 life; account separately',acceptance='All scientific receipt fields exactly equal excluding CPU timing only; no tolerance relaxation or outcome tuning'))
 prefix='e14_fast_qualification_';env={**os.environ,'EXP45_HIDDEN':'512','E8_BATCH':'64','E8_LR':'.001','E8_WD':'1.0','E8_SCHEDULE':'blocked','E8_REPLAY_SHARE':'1/2','E8_BUFFER_CAP':'1024','E8_REPLAY_SAMPLING':'uniform','E8_SAVE_STAGES':'0','E8_DIAGNOSTICS':'1','E8_RUN_PREFIX':prefix}
 log=O/'logs/e14_acceleration_qualify.log';start=time.monotonic()
 with log.open('w') as f:
  p=subprocess.Popen([sys.executable,'-u','e14_acceleration_worker.py','exp8_life.py','C','73129'],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  save(status,dict(status='qualifying_full_life',pid=p.pid,controller_pid=os.getpid(),active_E14_unchanged=True,log=str(log.relative_to(R))))
  code=p.wait()
 receipt=O/f'{prefix}e8_life_C_73129.json'
 same=code==0 and clean(read(receipt))==clean(read(O/'e8_life_C_73129.json'))
 final=dict(status='FULL_LIFE_RECEIPT_PARITY_PASS_NOT_DEPLOYED' if same else 'CANDIDATE_REJECTED_KEEP_ORIGINAL',returncode=code,wall_seconds=time.monotonic()-start,scientific_receipt_equal=same,active_E14_unchanged=True,receipt=str(receipt.relative_to(R)),note='Any speed deployment needs a recorded controller handover; this qualification did not modify running or queued scientific jobs.')
 save(status,final);print(json.dumps(final),flush=True)
if __name__=='__main__':main()
