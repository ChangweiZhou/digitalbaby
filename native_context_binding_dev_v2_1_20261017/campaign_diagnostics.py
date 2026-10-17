import time
import torch

def measure(model,items,theta_views=None):
    start=time.process_time();before=model.digest();rows=[]
    for prompt,answer in items:
        views={}
        candidates={'NORMAL':None,**(theta_views or {})}
        for name,theta in candidates.items():
            if theta is not None and not model.has_context:continue
            view=model.clone()
            with torch.no_grad():
                if theta is not None:
                    for k,p in view.p.items():p.copy_(theta[k])
            emitted,status,_=view.generate(prompt)
            views[name]=dict(emitted=emitted.hex(),status=status,correct=status=='EOS' and emitted==answer)
        rows.append(dict(prompt=prompt.hex(),views=views))
    if model.digest()!=before:raise AssertionError('Diagnostics wrote production state')
    return dict(rows=rows,CPU_seconds=time.process_time()-start,training_records=0,
        interpretation='Read-only parameter restoration; descriptive coordinate intervention, not efficacy proof')
