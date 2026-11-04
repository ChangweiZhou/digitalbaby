"""Development-only check that the official QA comparison can exercise fallback."""
import importlib.util
import json
from pathlib import Path
import tempfile

from qualify import DATA, ROOT, predictor, qa, save

with tempfile.TemporaryDirectory() as temp:
    folder=Path(temp)
    source=DATA/'qa1_single-supporting-fact_train.txt'
    stories=[]
    for line in source.read_text().splitlines(keepends=True):
        if line.startswith('1 '): stories.append([])
        stories[-1].append(line)
    train=folder/'train.txt'
    train.write_text(''.join(sum(stories[:40],[])))
    assert train.read_text().count('\t')==400
    spec=importlib.util.spec_from_file_location('dev_qa_stories',ROOT.parent/'minifly-qa-generalization'/'qualify.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.materialize(24117,folder)
    checkpoint=folder/'zero.npz'
    save(predictor(),checkpoint,0)
    rows={}
    for view in ('original','verb','prefix'):
        path=folder/f'{view}.txt'
        for who,new in (('Mary','Lucy'),('John','Paul'),('Daniel','Robert'),('Sandra','Olivia')):
            path.write_text(path.read_text().replace(who,new))
        rows[view]=qa(checkpoint,train,path)
    # Each fallback call can cover at most one query. This is an upper bound on coverage.
    admitted=any(r['exact_answer_with_own_newline']<.95 and r['fallback_calls']>=.05*r['n']
                 for view,r in rows.items() if view!='original')
    result=dict(training_questions=200,development_seed=24117,admitted=admitted,rows=rows)
    (ROOT/'QA_ADMISSION.json').write_text(json.dumps(result,indent=2)+'\n')
    print({v:(r['exact_answer_with_own_newline'],r['fallback_calls']) for v,r in rows.items()},admitted)
