"""Synthetic, non-learning GPU feasibility/latency test; no corpus/model state."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import time,json,resource
from pathlib import Path
import torch
torch.set_num_threads(1);H=Path(__file__).resolve().parent;start=time.process_time()
out=dict(torch=torch.__version__,mps_available=torch.backends.mps.is_available(),training_records=0,model_changes=False)
if out['mps_available']:
 try:torch.zeros(1,dtype=torch.float64,device='mps');out['native_float64_supported']=True
 except Exception as e:out.update(native_float64_supported=False,float64_error=str(e))
 g=torch.Generator().manual_seed(176);w=torch.randn(5177,512,generator=g)/184**.5;wg=w.to('mps');rows=[]
 for batch in (1,16,256):
  x=(torch.rand(batch,5177,generator=g)<184/5177).float();xg=x.to('mps')
  for _ in range(3):ref=(x@w).relu();pred=(xg@wg).relu()
  torch.mps.synchronize();t=time.perf_counter()
  for _ in range(20):ref=(x@w).relu()
  cpu=(time.perf_counter()-t)/20
  torch.mps.synchronize();t=time.perf_counter()
  for _ in range(20):pred=(xg@wg).relu()
  torch.mps.synchronize();gpu=(time.perf_counter()-t)/20
  diff=float((ref-pred.cpu()).abs().max());rows.append(dict(batch=batch,CPU_wall_seconds=cpu,GPU_wall_seconds=gpu,speedup=cpu/gpu,max_float32_difference=diff,bitwise_equal=bool(torch.equal(ref,pred.cpu()))))
 out.update(benchmarks=rows,device_allocated_bytes=torch.mps.current_allocated_memory(),resident_operands=True,host_transfers_excluded=True)
out.update(CPU=time.process_time()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,verdict='GPU_NOT_SELECTED_FOR_FLOAT64_NATIVE; dense feature accelerator only, requiring separate numerical qualification')
(H/'GPU_FEASIBILITY.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
