import ast,hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent
E=json.loads(Path('/Users/pencilbard/.codex/attachments/fe22de18-d6dd-4dbb-bc64-022fee621a42/Pasted text.txt').read_text())
c=E[3535]['input']['command']
for name,code in re.findall(r"cat > /tmp/evo/(\w+\.py) <<'EOF'\n(.*?)\nEOF",c,re.S):
 (ROOT/name).write_text(code+'\n')
s=(ROOT/'inner.py').read_text()
c=E[3567]['input']['command']; body=re.search(r"<<'EOF'\n(.*?)\nEOF",c,re.S)[1]
for n in ast.walk(ast.parse(body)):
 if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='replace':
  a,b=map(ast.literal_eval,n.args);assert a in s;s=s.replace(a,b)
(ROOT/'inner2.py').write_text(s)
records=[]
for e in E[3535:3650]:
 for block in e.get('content',[]):
  if block.get('type')!='text':continue
  for line in block['text'].splitlines():
   try:d=ast.literal_eval(line)
   except (SyntaxError,ValueError):continue
   if isinstance(d,dict) and 'mu' in d and len(d['mu'])==15:records.append(d)
record=max(records,key=lambda x:x['gen']);assert record['gen']==10
(ROOT/'RECOVERED_CHECKPOINT.json').write_text(json.dumps(record,indent=2)+'\n')
(ROOT/'RECOVERY.json').write_text(json.dumps({'checkpoint_scope':'Run A generation 10, rounded to 3 decimals as exported. Final A/B/C genomes not recoverable from complete printed records.','source_hashes':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['worlds.py','inner.py','inner2.py']},'export_sha256':hashlib.sha256(Path('/Users/pencilbard/.codex/attachments/fe22de18-d6dd-4dbb-bc64-022fee621a42/Pasted text.txt').read_bytes()).hexdigest()},indent=2)+'\n')
