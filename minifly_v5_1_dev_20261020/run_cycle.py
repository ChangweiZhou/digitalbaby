"""Foreground bounded exercise with awake lease and preserved failure evidence."""
import os,sys,json,time,traceback,subprocess,resource
from pathlib import Path
HERE=Path(__file__).resolve().parent
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
if __name__=='__main__':
    cycle=int(sys.argv[1]);started=time.time()
    lease=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
    record={'cycle':cycle,'pid':os.getpid(),'caffeinate_pid':lease.pid,'wall_start':started,'scientific_records':0}
    (HERE/'ops'/f'PROCESS_CYCLE{cycle}.json').write_text(json.dumps(record,indent=2)+'\n')
    try:
        module=__import__('cycle'+str(cycle));result=module.run()
        record.update(status='COMPLETE',wall_seconds=time.time()-started,CPU_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime)
    except BaseException as exc:
        record.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc(),wall_seconds=time.time()-started)
        tb=exc.__traceback__
        while tb is not None:
            if tb.tb_frame.f_code.co_name=='run':
                local=tb.tb_frame.f_locals
                record['completed_full_record_applications']=local.get('records',0)
                record['completed_primary_partial_native_events']=local.get('partial',0)
            tb=tb.tb_next
        (HERE/'cycles'/f'cycle{cycle}'/'FAILURE.json').write_text(json.dumps(record,indent=2)+'\n')
        raise
    finally:
        if lease.poll() is None:lease.terminate();lease.wait()
        (HERE/'ops'/f'PROCESS_CYCLE{cycle}.json').write_text(json.dumps(record,indent=2)+'\n')
