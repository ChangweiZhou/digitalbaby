import os,sys,subprocess,json,shutil,concurrent.futures
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'cycles/cycle3';out.mkdir(exist_ok=True);(out/'sources').mkdir(exist_ok=True)
for p in root.glob('*.py'):shutil.copy2(p,out/'sources'/p.name)
keeper=subprocess.Popen(['caffeinate','-i','-w',str(os.getpid())])
status={}
def run(name):
 with (out/(name+'.log')).open('x') as f:
  code=subprocess.call([sys.executable,str(root/'runner.py'),'--label','STREAM_TRANSFER_V1_20261013/TECH/CYCLE_3','--config',name,'--out',str(out/'receipts')],cwd=root,stdout=f,stderr=subprocess.STDOUT)
 return name,code
try:
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  futures=[pool.submit(run,n) for n in ('ALL-0','ALL-D','ANSWER-0','ANSWER-D')]
  for f in concurrent.futures.as_completed(futures):
   n,c=f.result();status[n]=c;(out/'EXIT_STATUS.json').write_text(json.dumps(status,indent=2)+'\n');print(n,c,flush=True)
   if c:raise RuntimeError('technical job failed: '+n)
finally:keeper.terminate()
