"""Read-only optimizer observers. True gradients are discarded; no diagnostic feeds training."""
import json
import torch as T
import exp45 as X

def cosine(a,b):
    a,b=a.detach().double().reshape(-1),b.detach().double().reshape(-1)
    den=a.norm()*b.norm()
    return float(T.dot(a,b)/den) if float(den)>0 else None

class Logger:
    def __init__(self,path,interval=512):
        self.path=path;self.interval=interval;self.steps=0
        if path.exists():raise FileExistsError(path)
        path.write_text('')
    def update(self,m,opt,items,local,store,stage,stage_update):
        self.steps+=1
        if self.steps%self.interval:return X.update(m,opt,items,local,store)
        named=[(f'{layer}.{kind}',getattr(getattr(m,layer),kind)) for layer in ('l1','l2','l3') for kind in ('weight','bias')]
        params=[p for _,p in named];record={};before={};actual={};true={};group={}
        def pre(optimizer,args,kwargs):
            with T.enable_grad():
                x,y,mask=X.batch_tensor(items)
                loss=T.nn.functional.cross_entropy(X.forward(m,x,False)[mask],y[mask])
                grads=T.autograd.grad(loss,params,retain_graph=False,create_graph=False)
            for (name,p),g in zip(named,grads):
                before[name]=p.detach().clone();actual[name]=p.grad.detach().clone();true[name]=g.detach()
                group[name]=next((float(g['lr']),float(g['weight_decay'])) for g in optimizer.param_groups if any(v is p for v in g['params']))
        def post(optimizer,args,kwargs):
            with T.no_grad():
                for name,p in named:
                    lr,wd=group[name];old=before[name];new=p.detach();adaptive_descent=old*(1-lr*wd)-new;total_descent=old-new
                    norm=float(old.double().norm());den=max(norm,1e-30)
                    record[name]=dict(raw_gradient_cosine_to_true=cosine(actual[name],true[name]),
                        adam_descent_cosine_to_true=cosine(adaptive_descent,true[name]),
                        total_descent_cosine_to_true=cosine(total_descent,true[name]),
                        adam_step_over_weight_norm=float(adaptive_descent.double().norm())/den,
                        total_step_over_weight_norm=float(total_descent.double().norm())/den,
                        weight_norm_before=norm,weight_norm_after=float(new.double().norm()),
                        actual_gradient_norm=float(actual[name].double().norm()),true_gradient_norm=float(true[name].double().norm()),lr=lr,wd=wd)
        h1=opt.register_step_pre_hook(pre);h2=opt.register_step_post_hook(post)
        try:loss=X.update(m,opt,items,local,store)
        finally:h1.remove();h2.remove()
        row=dict(global_update=self.steps,stage=stage,stage_update=stage_update,local=local,parameters=record)
        with self.path.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        return loss
