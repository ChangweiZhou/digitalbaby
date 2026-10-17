"""Remaining cycle3 quota: checkpoint repair and detached zero-science fixture."""
import ast
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
import bootstrap
import torch
from bind_kernel import Learner,science_identity,checkpoint_metadata_digest
from bind_artifacts import Ledger,write,source_identity
from qualify import raw_records,synthetic_protocol_checks,state_close
from bind_evaluator import cpu_auc_sensitivity

HERE=bootstrap.HERE

def rejected(fn):
    try:fn()
    except (ValueError,AssertionError):return True
    return False

def method_hashes(path):
    tree=ast.parse(path.read_text());result={}
    for cls in tree.body:
        if isinstance(cls,ast.ClassDef) and cls.name=='Learner':
            for n in cls.body:
                if isinstance(n,ast.FunctionDef) and n.name not in ('checkpoint','restore','rest','__init__'):
                    result[n.name]=hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest()
    return result

def main():
    out=HERE/'cycles/cycle3/attempt2'
    if out.exists():raise ValueError('do not overwrite attempt')
    out.mkdir();(out/'source').mkdir()
    for path in HERE.glob('*.py'):shutil.copyfile(path,out/'source'/path.name)
    shutil.copyfile(HERE/'SPEC.json',out/'source/SPEC.json')
    ledger=Ledger(out,3);checks={}
    try:
        before=method_hashes(HERE/'cycles/cycle3/attempt1/source/bind_kernel.py')
        after=method_hashes(HERE/'bind_kernel.py')
        assert before==after
        assert (HERE/'bind_native.py').read_bytes()==(HERE/'cycles/cycle3/attempt1/source/bind_native.py').read_bytes()
        checks['learning_and_forward_methods_unchanged']=True
        checks['native_physics_source_unchanged']=True
        model=Learner('F111',202610177003)
        ledger.call('final-source-prefix',model,lambda:model.train_window(raw_records()))
        model.rest(86400)
        path=out/'valid.pt';cursor=dict(next_record=4,history='TECH_ONLY',stage='NEXT')
        model.checkpoint(path,cursor)
        restored,received=Learner.restore(path)
        assert received==cursor and restored.digest()==model.digest()
        assert restored.frame_version==model.frame_version
        ledger.call('final-source-next-suffix-reference',model,lambda:model.train_window(list(reversed(raw_records()))))
        ledger.call('final-source-next-suffix-restored',restored,lambda:restored.train_window(list(reversed(raw_records()))))
        assert model.digest()==restored.digest()
        checks['actual_suffix_restore_parity']=True
        denied,received=Learner.restore(path);theta=denied.theta_digest();state=denied.state.detached(True);time0=denied.time
        ledger.call('final-source-restored-denial',denied,lambda:denied.train_window(raw_records(),False))
        passive=denied.anatomy.rest(state,denied.time-time0)
        assert theta==denied.theta_digest()
        for name in ('fast','slow'):torch.testing.assert_close(getattr(denied.state,name),getattr(passive,name),atol=1e-9,rtol=1e-9)
        checks['restored_denial']=True
        original=torch.load(path,map_location='cpu',weights_only=False)
        def tamper(label,edit):
            value=copy.deepcopy(original);edit(value)
            altered=out/(label+'.pt');torch.save(value,altered)
            return rejected(lambda:Learner.restore(altered))
        tamper_results={
            'cursor':tamper('cursor',lambda v:v['cursor'].update(next_record=8)),
            'resource':tamper('resource',lambda v:v['resource'].update(records=8)),
            'frame_sequence':tamper('frame',lambda v:v.update(frame_version=1)),
            'state':tamper('state',lambda v:v['state']['fast'].add_(.001)),
        }
        def coordinated(v):
            v['cursor']['next_record']=8;v['metadata_digest']=checkpoint_metadata_digest(v)
        tamper_results['cursor_resource_disagree_even_rehashed']=tamper('coordinated',coordinated)
        assert all(tamper_results.values());checks['checkpoint_tamper_rejections']=tamper_results
        checks['selection_and_statistics']=synthetic_protocol_checks()
        q=cpu_auc_sensitivity([(0,0),(5,.5),(10,1)],8)
        assert abs(q['primary']-.1875)<1e-12
        checks['CPU_quadrature_fixture']=q
        # Harmless detached process, never a learner, supervisor or science worker.
        script="import json,os,time;from pathlib import Path;p=Path("+repr(str(out/'DETACHED_FIXTURE.json'))+");p.write_text(json.dumps({'pid':os.getpid(),'session':os.getsid(0),'phase':'RUNNING'}));time.sleep(1);p.write_text(json.dumps({'pid':os.getpid(),'session':os.getsid(0),'phase':'DONE','science_jobs':0}))"
        with (out/'detached.log').open('w') as log:
            child=subprocess.Popen([sys.executable,'-c',script],start_new_session=True,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            awake=subprocess.Popen(['caffeinate','-i','-w',str(child.pid)])
            assert child.wait(timeout=10)==0;awake.wait(timeout=10)
        fixture=json.loads((out/'DETACHED_FIXTURE.json').read_text())
        assert fixture['phase']=='DONE' and fixture['session']==fixture['pid'] and fixture['science_jobs']==0
        checks['detached_and_caffeinate_fixture']=fixture
        write(out/'RESULT.json',dict(passed=True,cycle=3,part='OPERATIONS_REPAIR',checks=checks,science_jobs=0,
              full_record_applications=16,source_identity=source_identity(),model_identity=science_identity()))
        print(json.dumps(dict(passed=True,records=16,checks=list(checks))))
    except Exception as exc:
        import traceback
        write(out/'FAILURE.json',dict(passed=False,error=repr(exc),checks=checks,traceback=traceback.format_exc()))
        raise

if __name__=='__main__':main()
