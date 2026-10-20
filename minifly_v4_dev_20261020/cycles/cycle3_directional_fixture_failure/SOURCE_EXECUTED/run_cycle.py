from v4_common import *
import subprocess,resource,traceback
if __name__=='__main__':
 cycle=int(sys.argv[1]);start=time.monotonic();awake=None
 try:
  awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
  save(Path('STATUS.json'),dict(status='RUNNING_TECHNICAL_CYCLE',cycle=cycle,pid=os.getpid(),caffeinate_pid=awake.pid,scientific_records=0))
  mod=__import__(f'cycle{cycle}_v4');mod.run()
  save(Path('STATUS.json'),dict(status='TECHNICAL_CYCLE_COMPLETE',cycle=cycle,wall_seconds=time.monotonic()-start,scientific_records=0))
 except BaseException as exc:
  save(Path(f'cycles/cycle{cycle}/FAILURE.json'),dict(error=str(exc),traceback=traceback.format_exc(),scientific_records=0));save(Path('STATUS.json'),dict(status='TECHNICAL_FIXTURE_FAILURE',cycle=cycle,error=str(exc),scientific_records=0));raise
 finally:
  if awake is not None:awake.terminate();awake.wait()
