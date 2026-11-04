import hashlib
import json
from pathlib import Path
import sys

events=json.loads(Path(sys.argv[1]).read_text())
def command(i):return events[i]['input']['command']
s=command(2929).split("<<'EOF'\n",1)[1].split('\nEOF',1)[0]+'\n'
s=s.replace("ws=x.rstrip('.').split(); b=np.zeros(V,np.float32)","ws=x.rstrip('.').split(); b=np.zeros(V,np.float32); NS[0]+=1\n                for w in set(ws): DW[w]=DW.get(w,0)+1")
s=s.replace("sc={c: bag[c]*(H[c]+1)/(T[c]+2) for c in cand}","sc={c: bag[c]*(H[c]+1)/(T[c]+2)*(np.log((1+NS[0])/(1+DW.get(words[c],0))) if EIDF else 1) for c in cand}")
s=s.replace("DF={}; NQ=[0]","DF={}; NQ=[0]; DW={}; NS=[0]; EIDF=E('EIDF','0')=='1'")
s=s.replace("sc={c: bag[c]*(H[c]+1)/(T[c]+2)*","p0=(H.sum()+1)/(T.sum()+2); sc={c: bag[c]*((H[c]+2*p0)/(T[c]+2) if BASE else (H[c]+1)/(T[c]+2))*")
s=s.replace("EIDF=E('EIDF','0')=='1'","EIDF=E('EIDF','0')=='1'; BASE=E('BASE','1')=='1'")
Path('nobuilt.py').write_text(s)
print(hashlib.sha256(s.encode()).hexdigest())
