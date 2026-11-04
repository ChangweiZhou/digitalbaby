"""Engineering qualification for V9.2 predictor pretraining and QA transfer."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT=Path(__file__).resolve().parent
V92=ROOT.parent/'minifly_v197_long_life_factual_memory'/'v92_predictor_source.py'
QA=ROOT.parent/'minifly-v205'/'babi21.py'
DATA=ROOT.parent/'minifly-v202'/'data'
def sha(b):return hashlib.sha256(b).hexdigest()

def predictor():
    source=V92.read_text()
    env=dict(np=np,hashlib=hashlib)
    exec(source[source.index('VERSION ='):source.index('class F151ByteBrainV92')],env)
    exec('class Predictor:\n'+source[source.index('    def _features'):source.index('    def predict(')],env)
    model=env['Predictor']()
    model.w=np.zeros((env['ROWS'],256),np.float32)
    model.bias=np.zeros(256,np.float32)
    model.history=bytearray()
    return model

def digest(model):return sha(model.w.tobytes()+model.bias.tobytes())

def advance(model,raw,train):
    loss=0.
    for y in raw:
        p=model._distribution()
        loss-=math.log2(float(p[y]))
        if train:
            e=p.astype(np.float32)
            e[y]-=1
            for row,rate in model._features():
                model.w[row]-=np.float32(rate)*e
            model.bias-=np.float32(.001)*e
        model.history.append(y)
        del model.history[:-4]
    return loss

def save(model,path,position):
    np.savez(path,w=model.w,bias=model.bias,history=np.array(model.history,np.uint8),position=position)
    path.with_suffix('.sha').write_text(sha(path.read_bytes()))

def load(path):
    if sha(path.read_bytes())!=path.with_suffix('.sha').read_text():
        raise ValueError('Checkpoint digest mismatch')
    model=predictor()
    with np.load(path,allow_pickle=False) as arrays:
        model.w[:]=arrays['w']; model.bias[:]=arrays['bias']
        model.history=bytearray(arrays['history'].tobytes())
        position=int(arrays['position'])
    return model,position

def qa_code():
    code=QA.read_text()
    start=code.index('class Pred:')
    end=code.index('\ndef run(',start)
    code=code[:start]+'''class Pred:
    def __init__(self):
        self.model,_=load(Path(os.environ['CHECKPOINT']))
        self.model.history.clear()
    def p(self,h):
        self.model.history=bytearray(h[-4:])
        return self.model._distribution()
    def learn(self,h,y):
        self.model.history=bytearray(h[-4:])
        advance(self.model,bytes([y]),True)
'''+code[end:]
    code=code.replace('    def answer(maxlen=12):','    fallback=[0]\n    replies=[]\n    def answer(maxlen=12):')
    code=code.replace('else int(np.argmax(P.p(hist)))','else fallback_read(hist)')
    code=code.replace('    fallback=[0]', '''    def fallback_read(h):
        fallback[0]+=1
        return int(np.argmax(P.p(h)))
    fallback=[0]''')
    code=code.replace('                    if test:\n                        name=', '                    if test:\n                        replies.append(out.hex())\n                        name=')
    needle='    acc=np.mean'
    assert needle in code and 'replies.append' in code
    code=code.replace(needle,'''    h=hashlib.sha256(M.tobytes()+G.tobytes()+DFQ.tobytes())
    for k,v in sorted(A.items()):
        h.update(repr(k).encode()); h.update(v.tobytes())
    acc=np.mean''')
    code=code.replace("'n':len(res)","'specialist':h.hexdigest(),'fallback_calls':fallback[0],'replies':replies,'n':len(res)")
    return code

def qa(path,train,test,store='gword+vocabsent'):
    env=dict(os.environ,CHECKPOINT=str(path),TRAIN=str(train),TEST=str(test),FREEZE='1',NOGOLD='1',
             QIDF='1',QIDFADAPT='0',ENDW='1',STORE=store,SEG='1',WIN='32',BIN='1',
             HABT='0',GORD='3,4,5',QNOSTORE='1',STEPN='2',MATCHN='3',GTH='.15',UNIT='1',
             CREDIT='match',VALID='0',PYTHONHASHSEED='4',OPENBLAS_NUM_THREADS='1')
    return json.loads(subprocess.check_output([sys.executable,'-B',__file__,'qa'],env=env,text=True))

if __name__=='__main__':
    if sys.argv[1:]==['qa']:
        exec(compile(qa_code(),'v205_v92_adapter','exec'),globals())
        sys.exit()
    if len(sys.argv)>1 and sys.argv[1]=='resume':
        model,pos=load(Path(sys.argv[2]))
        loss=advance(model,Path(sys.argv[3]).read_bytes()[pos:],True)
        print(json.dumps([digest(model),loss]))
        sys.exit()
    # An Alice excerpt is an engineering fixture, separate from the proposed final corpus.
    text=Path(os.environ['MINIFLY_ALICE']).read_bytes()[5000:37000]
    train_bytes,test_bytes=text[:24000],text[24000:]
    model=predictor()
    assert abs(advance(model,b'a',False)-8)<1e-12
    model.history.clear()
    with tempfile.TemporaryDirectory() as directory:
        folder=Path(directory)
        paths={n:folder/f'{n}.npz' for n in ('zero','small','full','shuffled')}
        save(model,paths['zero'],0)
        advance(model,train_bytes[:4096],True)
        save(model,paths['small'],4096)
        tail_loss=advance(model,train_bytes[4096:],True)
        save(model,paths['full'],len(train_bytes))
        shuffled=predictor()
        advance(shuffled,np.random.default_rng(81).permutation(np.frombuffer(train_bytes,np.uint8)).tobytes(),True)
        save(shuffled,paths['shuffled'],len(train_bytes))
        frozen={}
        for name,path in paths.items():
            m,_=load(path)
            m.history.clear()
            before=digest(m)
            frozen[name]=advance(m,test_bytes,False)/len(test_bytes)
            assert before==digest(m)
        raw=folder/'raw.bin';raw.write_bytes(train_bytes)
        cmd=[sys.executable,'-B',__file__,'resume',str(paths['small']),str(raw)]
        assert json.loads(subprocess.check_output(cmd))==[digest(model),tail_loss]
        checkpoint=paths['small']
        original=checkpoint.read_bytes()
        checkpoint.write_bytes(original+b'corrupt')
        rejected=subprocess.run(cmd,capture_output=True,text=True)
        assert rejected.returncode and 'Checkpoint digest mismatch' in rejected.stderr
        checkpoint.write_bytes(original)
        q=DATA/'qa1_single-supporting-fact_train.txt'
        stories=[]
        for line in q.read_text().splitlines(keepends=True):
            if line.startswith('1 '): stories.append([])
            stories[-1].append(line)
        qa_train=folder/'train.txt';qa_train.write_text(''.join(sum(stories[:10],[])))
        qa_test=folder/'test.txt';qa_test.write_text(''.join(sum(stories[10:14],[])))
        results={k:qa(v,qa_train,qa_test) for k,v in paths.items()}
        assert len({r['specialist'] for r in results.values()})==1
        assert qa(paths['full'],qa_train,qa_test)==results['full']
        fallback_fixture={k:qa(paths[k],qa_train,qa_test,store='none') for k in ('zero','full')}
        assert all(r['fallback_calls']>=r['n'] for r in fallback_fixture.values())
        assert len({r['specialist'] for r in fallback_fixture.values()})==1
    result=dict(status='DEVELOPMENT_QUALIFICATION',source_sha256=sha(Path(__file__).read_bytes()),
                bytes={'train':len(train_bytes),'test':len(test_bytes)},frozen_bpb=frozen,qa=results,fallback_fixture=fallback_fixture,
                restart='fresh process identical predictor state and continuation loss; corrupt checkpoint rejected')
    (ROOT/'ENGINEERING.json').write_text(json.dumps(result,indent=2)+'\n')
    print(frozen, {k:(v['exact_answer_with_own_newline'],v['fallback_calls']) for k,v in results.items()})
