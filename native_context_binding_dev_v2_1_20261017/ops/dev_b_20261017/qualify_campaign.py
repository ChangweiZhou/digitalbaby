"""Synthetic adapter tests. No native learner is constructed."""
import copy,json,math,types
from pathlib import Path
import bootstrap
import torch
import campaign_common as c
import campaign_worker as w
import campaign_audit as a
import campaign_supervisor as s
import audit_technical as audit

def main():
    c.verify_lock();m=c.manifest('DEV_B');auth=c.OPS/'AUTHORIZATION.json'
    assert (m['full_jobs'],m['histories'],m['records'])==(36,72,73728)
    checks={'fixed_manifest':True}
    for name,args in [('DEV_A_rejected',('DEV_A','F111',202610177101)),('wrong_world_rejected',('DEV_B','F111',202610177101)),('wrong_arm_rejected',('DEV_B','ELM',202610177105))]:
        try:c.admission(*args,auth)
        except PermissionError:checks[name]=True
        else:raise AssertionError(name)
    fixture=c.OPS/'synthetic_qualification';fixture.mkdir(parents=True,exist_ok=False)
    (fixture/'SYNTHETIC_ONLY.txt').write_text('NO NATIVE TRAJECTORIES. Fake learner exercises serialization and auditing interfaces. Not science.\n')
    births=[dict(fly_id=str(i),canonical_B_sha256='32a3726cb25a8255c1aef5696ff40aa303138c5d061a598004d2087d4da5f964') for i in range(32)]
    class Fake:
        def __init__(self,arm,world):
            self.arm,self.world=arm,world;self.learns=arm!='NATIVE';self.has_context=self.learns
            self.p={'a':torch.tensor([0.],dtype=torch.float64)};self.actor_steps=0;self.time=0;self.last_grads={};self.births=births
            self.state=types.SimpleNamespace(fast=torch.zeros(1),slow=torch.zeros(1))
            self.state.detached=lambda *args:types.SimpleNamespace(fast=self.state.fast.clone(),slow=self.state.slow.clone())
            self.anatomy=types.SimpleNamespace(rest=lambda state,dt:state)
            self.resource={k:0 for k in ('records','input_bytes','answer_bytes','EOS_targets','native_store_events','actor_steps','raw_alpha_l1','logical_MEM_cpu','logical_LEGACY_cpu','production_cpu')}
        def clone(self):return Fake(self.arm,self.world)
        def digest(self):return str((self.resource,self.actor_steps,float(self.p['a'])))
        def theta_digest(self):return str((self.actor_steps,float(self.p['a'])))
        def checkpoint(self,path,cursor):
            assert cursor['next_record']==self.resource['records'] and cursor['next_record']%4==0
            json.dumps(cursor,allow_nan=False);c.write(Path(path).with_suffix('.SYNTHETIC.json'),dict(cursor=cursor,records=self.resource['records']))
        def rest(self,dt):self.time+=dt;self.resource['logical_MEM_cpu']+=.001
        def train_window(self,rows,allowed):
            self.last_trace=[]
            for prompt,answer in rows:
                for target in (*answer,256):
                    signal=((audit.CODE.mean(0)-audit.CODE[target])/2).tolist()
                    self.last_trace.append(dict(predicted=0,target=target,prefeedback=True,native_write=allowed,
                        logits=[0.]*257,frame_address_reused=True,signal=signal))
                self.time+=1;self.resource['records']+=1;self.resource['input_bytes']+=len(prompt);self.resource['answer_bytes']+=len(answer)
                self.resource['EOS_targets']+=1;self.resource['native_store_events']+=(2+len(prompt)+len(answer)+1)*32
            if allowed:self.actor_steps+=1;self.resource['actor_steps']+=1;self.resource['raw_alpha_l1']+=1
            self.resource['logical_MEM_cpu']+=.01;self.resource['production_cpu']+=.01
            return dict(loss=math.log(257),gradient_norm=0.,records=4,allowed=allowed,applied_update=allowed)
    def score(model,items,readout='MEM',cap=16):
        return dict(accuracy=0.,rows=[dict(prompt=p.hex(),answer=ans.hex(),emitted='ff',status='EOS',correct=False) for p,ans in items])
    original=(w.Learner,w.evaluator.score,w.campaign_diagnostics.measure)
    w.Learner=Fake;w.evaluator.score=score;w.campaign_diagnostics.measure=lambda *args:dict(synthetic=True,training_records=0)
    try:
        out=fixture/'complete_F111';result=w.run('DEV_B','F111',202610177105,out,auth)
        assert result['records']==2048 and c.read(out/'AUDIT.json')['accepted']
        r=c.read(out/'RECEIPT.json');checks['whole_job_branch_and_checkpoint_serialization']=True
        for name,mutate in [('denied_write_rejected',lambda x:x['histories']['N_OLD']['windows'][0].update(allowed=True)),
             ('fake_score_rejected',lambda x:x['histories']['W']['points'][0]['readings']['MEM'].update(accuracy=1.)),
             ('missing_sample_rejected',lambda x:x['histories']['W']['points'].pop(1))]:
            bad=copy.deepcopy(r);mutate(bad)
            try:a.audit(bad)
            except (AssertionError,ValueError):checks[name]=True
            else:raise AssertionError(name)
        w.STOP=True;out=fixture/'pause_F111';result=w.run('DEV_B','F111',202610177106,out,auth)
        assert result['status']=='PAUSED_SESSION_LIMIT'
        cp=c.read(out/'PAUSE.SYNTHETIC.json');assert cp['cursor']['next_record']==cp['records']==4
        checks['safe_window_pause_cursor']=True;w.STOP=False
    finally:w.Learner,w.evaluator.score,w.campaign_diagnostics.measure=original
    real=c.read(c.HERE/'cycles/cycle3/attempt1/F111_RECEIPT.json')
    tokens=audit.audit_job(real);checks['saved_real_trace_reducer_compatible']=tokens>0
    memory=s.memory_admission({});c.write(c.OPS/'PRELAUNCH_MEMORY.json',memory)
    checks['memory_admission_callable']=type(memory['admit']) is bool
    checks['scientific_core_unchanged']=__import__('bind_kernel').science_identity()==c.read(c.HERE/'SOURCE_LOCK.json')['model_identity']
    result=dict(passed=all(checks.values()),checks=checks,synthetic_only=True,new_native_records=0,science_jobs=0,
        no_new_training_for_qualification=True,real_saved_prediction_tokens_reduced=tokens,
        source_identity=c.identity(),memory_at_qualification=memory,scope='DEV_B finite adapter, no efficacy claim')
    c.write(c.OPS/'CAMPAIGN_QUALIFICATION.json',result);print(json.dumps(result))
if __name__=='__main__':main()
