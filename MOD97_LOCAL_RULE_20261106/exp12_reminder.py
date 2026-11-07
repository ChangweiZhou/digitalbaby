"""E12 A: matched fresh-optimizer reminders from three saved weight states."""
import os
os.environ['E8_BATCH']='64'
os.environ['E8_SAVE_STAGES']='0'
import json, sys, time
from random import Random
import exp8_life as E
X, T = E.X, E.T

def main(seed, origin, dose):
    model=X.new_net(seed)
    if origin=='birth':
        state=T.load(E.OUT/f'e8_dev_{seed}.pt', weights_only=True)
    else:
        stage=1 if origin=='pre' else 2
        state=T.load(E.OUT/f'e12_A_e8_state_A_rp0of1_interleaved_{seed}_stage{stage}.pt',weights_only=False)['model']
    model.load_state_dict(state)
    opt=T.optim.AdamW(model.parameters(),lr=.001,weight_decay=1.,betas=(.9,.98))
    pools=[[E.render(task,a,b) for a,b in E.split(task,.25,seed)[0]] for task in E.LIFE]
    order=Random(seed+12012)
    panels={E.NAME(*task):E.panel(task,seed) for task in E.LIFE}
    start=time.process_time()
    before={k:X.score(model,v) for k,v in panels.items()}
    losses=[]
    for _ in range(dose):
        draws=[order.sample(pool,8) for pool in pools]
        batch=[row for position in zip(*draws) for row in position]
        losses.append(X.update(model,opt,batch,True))
    after={k:X.score(model,v) for k,v in panels.items()}
    row=dict(seed=seed,origin=origin,updates=dose,presentations=64*dose,
             optimizer='fresh AdamW, same in all three origins',before=before,after=after,
             loss=sum(losses)/len(losses),cpu=time.process_time()-start)
    (E.OUT/f'e12_reminder_{origin}_r{dose}_{seed}.json').write_text(json.dumps(row,indent=2))
    print(json.dumps(row),flush=True)
if __name__=='__main__': main(int(sys.argv[1]),sys.argv[2],int(sys.argv[3]))
