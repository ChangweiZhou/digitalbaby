"""Reserved-world operations qualification only; never admits science."""
import json,os,subprocess,sys,time
from pathlib import Path
from operations import atomic_json,committed_index
from core import CONFIGS
from runner import source_lock
from finish import finish
OUT=Path(__file__).parent

def launch(extra,log,directory):
 env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
 with log.open('wb') as f:
  p=subprocess.Popen([sys.executable,str(OUT/'dispatcher.py'),'--out',str(directory),'--workers','2',
   '--technical-world','202610129620','--estimated-job-seconds','60',*extra],stdout=f,stderr=subprocess.STDOUT,
   cwd=OUT,env=env,start_new_session=True)
  atomic_json(log.with_suffix('.launch.json'),{'pid':p.pid,'detached_session':True,'technical':True,'source_lock':source_lock()})
  p.wait(timeout=300)
  if p.returncode:raise RuntimeError(f'dispatch failed {p.returncode}: {log.read_text()[-1500:]}')

def run():
 directory=OUT/'cycles/cycle3/dispatch';directory.parent.mkdir(exist_ok=True,parents=True)
 launch(['--technical-stop','1'],directory.parent/'pause.log',directory)
 paused=json.loads((directory/'STATUS.json').read_text())
 assert paused['status']=='PAUSED_SESSION_LIMIT' and 1<=paused['committed_jobs']<5
 atomic_json(directory.parent/'PAUSED_SNAPSHOT.json',paused)
 before=committed_index(directory/'receipts',{(202610129620,c) for c in CONFIGS},True,True)
 launch(['--resume'],directory.parent/'resume.log',directory)
 after=committed_index(directory/'receipts',{(202610129620,c) for c in CONFIGS},True,True)
 assert after['committed_jobs']==5 and all(row in after['rows'] for row in before['rows'])
 assert not list((directory/'active').glob('*.json'))
 summary=finish(directory/'receipts',directory.parent/'finish',True,[202610129620])
 assert summary['family_size']==82 and summary['verdict']=='TECHNICAL_PIPELINE_ONLY' and summary['selected'] is None
 final=json.loads((directory.parent/'finish/FINAL_AUDIT.json').read_text());package=json.loads((directory.parent/'finish/PACKAGE.json').read_text())
 assert final['accepted'] and package['complete'] and final['counts']['scientific_jobs']==0
 result={'passed':True,'detached_pause_resume':True,'preserved_commits':before['committed_jobs'],
  'finished_technical_jobs':5,'scientific_jobs':0,'counts':final['counts'],'source_lock':source_lock()}
 atomic_json(directory.parent/'OPERATIONS_TEST.json',result);return result
if __name__=='__main__':print(json.dumps(run()))
