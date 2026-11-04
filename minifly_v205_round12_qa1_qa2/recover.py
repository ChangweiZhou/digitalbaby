"""Recover final Round 12 sources from the reviewed source-edit events."""
import builtins
import hashlib
import json
from pathlib import Path
import shutil
import sys

root=Path(__file__).resolve().parent
events=json.loads(Path(sys.argv[1]).read_text())
def file(name):return root/Path(name).name
def mapped_open(name,*args,**kwargs):return builtins.open(file(name),*args,**kwargs)
def patch(i):
    cmd=events[i]['input']['command']
    start=cmd.index("<<'EOF'") if "<<'EOF'" in cmd else cmd.index("<< 'EOF'")
    snippet=cmd[cmd.index('\n',start)+1:].split('\nEOF',1)[0]
    exec(compile(snippet,f'reviewed-event-{i}','exec'),{'open':mapped_open})
for i in (2343,2360,2451):
    d=events[i]['input'];file(d['file_path']).write_text(d['content'])
for i in (2380,2391):
    d=events[i]['input'];p=file(d['file_path'])
    p.write_text(p.read_text().replace(d['old_string'],d['new_string']))
shutil.copyfile(file('babi20.py'),file('babi21.py'))
for i in (2403,2413):patch(i)
p=file('babi21.py');s=p.read_text()
s=s.replace("if STOREMODE=='gated' and not FROZ[0]:","if MODES & {'gated','gword'} and not FROZ[0]:")
p.write_text(s)
for i in (2432,2456,2466,2471,2485,2490,2495,2520):patch(i)
p=file('babi21.py');s=p.read_text()
s=s.replace('if QIDF and not FROZ[0]:','if QIDF and (not FROZ[0] or QIDFADAPT):')
s=s.replace("QIDF=E('QIDF','0')=='1'","QIDF=E('QIDF','0')=='1'; QIDFADAPT=E('QIDFADAPT','0')=='1'")
s=s.replace("'QIDF','ENDW'","'QIDF','QIDFADAPT','ENDW'")
p.write_text(s)
for i in (2588,2598,2650,2669,2687,2720):patch(i)
for i,name in ((2515,'mkpara2.py'),(2583,'gen_codex_fresh.py')):
    c=events[i]['input']['command'];start=c.index("<<'EOF'")
    file(name).write_text(c[c.index('\n',start)+1:].split('\nEOF',1)[0]+'\n')
for name,digest in {'babi21.py':'429fd3222d4b4b63cd41c25d6d524043e228e6516b2a67d2d52aaa6ed1f3dd28',
                    'qa2.py':'87876b79f50fb7589d65e739a014e444a248a4966a349862edf304d142c19e92'}.items():
    actual=hashlib.sha256(file(name).read_bytes()).hexdigest()
    print(name,actual,flush=True)
    assert actual==digest
file('babi20.py').unlink()
