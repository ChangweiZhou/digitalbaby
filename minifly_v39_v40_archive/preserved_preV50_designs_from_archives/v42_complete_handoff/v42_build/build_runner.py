from pathlib import Path
import base64,hashlib,zlib
root=Path(__file__).resolve().parent
def pack(raw):return {'sha256':hashlib.sha256(raw).hexdigest(),'data':base64.b85encode(zlib.compress(raw,9)).decode()}
payload={name:pack((root/(name+'.py')).read_bytes()) for name in ('evaluator','reporting')}
reference=pack((root/'reference_inputs.json').read_bytes())
text=(root/'runner_template.py').read_text()
for key,value in {'__PROTOCOL__':(root/'DESIGN.md').read_text(),'__PAYLOADS__':payload,'__REFERENCE__':reference}.items():
    text=text.replace(key,repr(value))
out=root.parent/'elm_v42_read_authority.py'
compile(text,str(out),'exec');out.write_text(text)
print(out,len(out.read_bytes()),hashlib.sha256(out.read_bytes()).hexdigest())
