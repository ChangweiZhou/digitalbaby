from pathlib import Path
import base64,zlib,hashlib
b=Path(__file__).resolve().parent;root=b.parent
names={'legacy':root/'v43_sources/v41_legacy.py','claude':root/'v43_sources/v41_claude.py','hybrid':root/'v43_sources/v41_hybrid.py'}
names.update({n:b/(n+'.py') for n in ('engine','offline_controls','reporting','diagnostics')})
payloads={}
for name,p in names.items():
 raw=p.read_bytes();compile(raw,str(p),'exec');payloads[name]={'sha256':hashlib.sha256(raw).hexdigest(),'data':base64.b85encode(zlib.compress(raw,9)).decode()}
text=(b/'runner_template.py').read_text()
for k,v in {'__PROTOCOL__':(b/'PROTOCOL.md').read_text(),'__ORIGINAL_PROTOCOL__':(b/'ORIGINAL_DESIGN.md').read_text(),'__PAYLOADS__':payloads}.items():text=text.replace(k,repr(v))
out=root/'elm_v43_fast_slow_authority.py';compile(text,str(out),'exec');out.write_text(text)
print(out,len(out.read_bytes()),hashlib.sha256(out.read_bytes()).hexdigest())
