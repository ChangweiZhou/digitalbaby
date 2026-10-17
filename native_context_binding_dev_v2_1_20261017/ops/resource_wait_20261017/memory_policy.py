"""Operations-only per-configuration admission using measured native memory, never outcomes."""
import ctypes,json,math,re,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
I=HERE.parents[1]

class Usage(ctypes.Structure):
    _fields_=[('uuid',ctypes.c_uint8*16)]+[(k,ctypes.c_uint64) for k in
        ('user_time','system_time','pkg_idle_wkups','interrupt_wkups','pageins','wired_size','resident_size','phys_footprint','proc_start_abstime','proc_exit_abstime')]

def usage(pid):
    lib=ctypes.CDLL('/usr/lib/libproc.dylib',use_errno=True)
    fn=lib.proc_pid_rusage;fn.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_void_p];fn.restype=ctypes.c_int
    value=Usage()
    if fn(pid,0,ctypes.byref(value))!=0:raise OSError(ctypes.get_errno(),'Cannot inspect existing worker footprint')
    return dict(pid=pid,RSS_bytes=value.resident_size,physical_footprint_bytes=value.phys_footprint,start_abstime=value.proc_start_abstime)

def system():
    raw=subprocess.check_output(['/usr/bin/vm_stat'],text=True)
    page=int(re.search(r'page size of (\d+)',raw).group(1));v={k:int(n) for k,n in re.findall(r'^([^:\n]+):\s+(\d+)\.',raw,re.M)}
    pressure=int(subprocess.check_output(['/usr/sbin/sysctl','-n','kern.memorystatus_vm_pressure_level'],text=True))
    return dict(available_proxy_bytes=page*sum(v.get(k,0) for k in ('Pages free','Pages inactive','Pages speculative')),
        pressure=pressure,swapouts=v.get('Swapouts'),swapins=v.get('Swapins'),page_size=page)

def decide(available,pressure,reserves,active,next_arm):
    growth=sum(max(0,reserves[row['arm']]-row['RSS_bytes']) for row in active)
    required=reserves[next_arm]+growth+512*1024**2
    return dict(admit=(pressure==1 or (pressure==2 and len(active)==0)) and available>=required,required_bytes=required,
        new_worker_peak_reserve_bytes=reserves[next_arm],reserved_growth_bytes=growth,
        available_reclaimable_proxy_bytes=available,pressure=pressure,concurrency_limit=4 if pressure==1 else 1 if pressure==2 else 0,system_reserve_bytes=512*1024**2,
        running_workers=active,next_arm=next_arm,rule='Per-arm observed peak×1.20; remaining growth reserved against live RSS, matching measured peak RSS units; normal pressure:max4;warning:singleton if headroom sufficient;critical:wait')

class Policy:
    def __init__(self):
        self.config=json.loads((HERE/'RESOURCE_ADMISSION.json').read_text());self.reserves=self.config['per_arm_reserve_bytes']
    def admission(self,running,job_arm):
        rows=[]
        for pid,(proc,handle,job) in running.items():
            try:r=usage(pid)
            except OSError:
                # A certified finished job need not consume another growth reservation.
                if 'folder' in job and (Path(job['folder'])/'AUDIT.json').exists():continue
                r=dict(pid=pid,physical_footprint_bytes=0,RSS_bytes=0)
            rows.append(dict(arm=job['arm'],**r))
        reserves=self.reserves.copy()
        for row in rows:reserves[row['arm']]=max(reserves[row['arm']],math.ceil(row['RSS_bytes']*1.20))
        v=system();d=decide(v['available_proxy_bytes'],v['pressure'],reserves,rows,job_arm)
        d.update(system=v,heartbeat_epoch=time.time());return d
