"""Recover reviewed Python heredocs and literal source edits; never execute exported shell commands."""
import ast, hashlib, json, re
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'source'
MANIFEST=[]
def command(n): return (SOURCE/f'event_{n}.txt').read_text()
def heredoc(n):
    c=command(n)
    m=re.search(r"<<'EOF'\n(.*?)\nEOF",c,re.S)
    assert m, n
    return m.group(1)+'\n'
def edits(text,n):
    tree=ast.parse(heredoc(n))
    count=0
    for node in tree.body:
        if not isinstance(node,ast.Assign) or len(node.targets)!=1: continue
        if not isinstance(node.targets[0],ast.Name) or node.targets[0].id!='s': continue
        call=node.value
        if not isinstance(call,ast.Call) or not isinstance(call.func,ast.Attribute) or call.func.attr!='replace': continue
        old,new=map(ast.literal_eval,call.args)
        hits=text.count(old)
        assert hits>0 or old==new, (n,old)
        text=text.replace(old,new); count+=1
    MANIFEST.append({'event':n,'literal_replacements':count})
    return text
out=ROOT/'recovered';out.mkdir(exist_ok=True)
s=heredoc(3344)
for n in [3349,3354,3359]: s=edits(s,n)
s+=heredoc(3364)
s=edits(s,3369)
(out/'statelaw.py').write_text(s)
t=edits(s,3398)
(out/'statelaw_t_dispersion.py').write_text(t)
# Event 3418 places the same source edits inside a for-loop. Apply those literal pairs only.
for node in ast.walk(ast.parse(heredoc(3418))):
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='replace':
        old,new=map(ast.literal_eval,node.args)
        hits=t.count(old)
        assert hits>0 or old==new,old
        t=t.replace(old,new)
(out/'statelaw_t.py').write_text(t)
g=heredoc(3033).split('class Learner:')[0]
(out/'miniworld_generator.py').write_text(g)
for name in ['MINIFLY_MEMO_ROUND23_20261104.md','MINIFLY_MEMO_ROUND24_20261104.md']:
    if not (ROOT/name).exists():
        (ROOT/name).write_bytes((Path('/Users/pencilbard/Downloads')/name).read_bytes())
attachment=Path('/Users/pencilbard/.codex/attachments/62d72a65-eeea-4f93-b48a-4cb020abec91/Pasted text.txt')
attachment_hash=hashlib.sha256(attachment.read_bytes()).hexdigest() if attachment.exists() else json.loads((ROOT/'RECOVERY.json').read_text())['attachment_sha256']
manifest={'attachment_sha256':attachment_hash,'replacements':MANIFEST,'recovered_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.py')}}
(ROOT/'RECOVERY.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
