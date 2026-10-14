"""Read-only final source/environment gate; dispatches no learner or trajectory."""
import hashlib,json,platform,sys
from pathlib import Path
import torch,numpy

def verify(lock):
    root=Path(__file__).parent
    for relative,expected in lock['files'].items():
        assert hashlib.sha256((root/relative).read_bytes()).hexdigest()==expected,relative+' source changed'
    assert torch.__version__==lock['environment']['torch']
    assert numpy.__version__==lock['environment']['numpy']
    assert platform.python_version()==lock['environment']['python']
    return True

if __name__=='__main__':
    lock=json.loads((Path(__file__).parent/'SOURCE_LOCK.json').read_text());verify(lock)
    # Tamper fixture changes a copied manifest only; no locked files are altered.
    bad=json.loads(json.dumps(lock));bad['files']['core.py']='0'*64
    try:verify(bad)
    except AssertionError:pass
    else:raise AssertionError('source tamper admitted')
    print(json.dumps({'accepted':True,'source_files':len(lock['files']),'source_tamper_rejected':True,'new_training_records':0,'science_authorized':False}))
