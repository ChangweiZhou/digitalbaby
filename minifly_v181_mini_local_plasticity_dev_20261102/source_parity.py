"""Execute literal parent FE and encode definitions, without parent constructors."""
import ast,math,copy,json,hashlib
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from sensor import Sensor,ROOT
H=Path(__file__).resolve().parent;s=Sensor()
def take(path,names):
 tree=ast.parse(path.read_text());return [n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
byte=ROOT/'minifly_a_v3/package/REFERENCE_SOURCE/BYTE_CORE_1_20260922/bytecore.py';fast=ROOT/'minifly_cls_dev_20261017/fastmem.py'
env={'np':np,'math':math,'copy':copy,'hashlib':hashlib,'IntegrityError':RuntimeError,'RECEPTOR':{'k':6,'lo':.4,'hi':1.,'seed':'BYTE-CORE-1|RECEPTOR|v1'}}
exec(compile(ast.Module(body=take(byte,['sha_seed','receptor_table','_norm','FrontEnd']),type_ignores=[]),str(byte),'exec'),env)
R=env['receptor_table']();assert np.array_equal(R,s.R);env.update(BC=NS(FrontEnd=env['FrontEnd'],R_TABLE=R,_norm=env['_norm']),DELTA=30/14,PERM=np.random.default_rng(20261017).permutation(88))
nodes=take(fast,['FEPerm']);tree=ast.parse(fast.read_text());cl=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FastMemory');encode=next(n for n in cl.body if isinstance(n,ast.FunctionDef) and n.name=='encode');nodes.append(encode)
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(fast),'exec'),env)
# Lightweight interface namespace only; zero FastMemory/Compartment/native births.
m=NS(bank=[NS(fly=NS(m=NS(B=s.B,pn_type_index=s.pn,kc_side=s.side)))],theta=np.zeros(5177),side_n=s.limits)
fixtures=[b'',bytes(range(256)),bytes(range(255,-1,-1)),b'abcx',b'abcy',b'punctuation: .?!\x00\xff\n']
positions=0
for r in fixtures:
 fe=env['FEPerm']();t=2*env['DELTA'];fe.advance(t);reference=[]
 for i in range(len(r)+1):
  view=fe.clone();view.advance(t+env['DELTA']);reference.append(env['encode'](m,view.read())[0])
  if i<len(r):t+=env['DELTA'];fe.feed(r[i],t)
 assert np.array_equal(np.array(reference),s.codes(r));positions+=len(reference)
(H/'SOURCE_PARITY.json').write_text(json.dumps({'status':'PASS','literal_parent_AST_execution':True,'positions':positions,'all_256_bytes_both_orders':True,'empty_state_and_EOS_timing':True,'native_bank_constructors_called':0,'sensor_sha256':s.digest,'parent_bytecore_sha256':hashlib.sha256(byte.read_bytes()).hexdigest(),'parent_fastmem_sha256':hashlib.sha256(fast.read_bytes()).hexdigest()},indent=2)+'\n');print('PARENT_SOURCE_PARITY_PASS',positions)
