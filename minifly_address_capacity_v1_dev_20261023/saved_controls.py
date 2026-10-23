"""Read saved tensors without importing or instantiating a native learner."""
import ast,importlib.util,json,time,sys
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent;R=OUT.parent;SOURCE=R/'minifly_stop_only_confirm_v1_20261023'
s=importlib.util.spec_from_file_location('frozen_stop_head',SOURCE/'head.py');H=importlib.util.module_from_spec(s);s.loader.exec_module(H)
ns=dict(np=np,H=H,P=SOURCE)
tree=ast.parse((SOURCE/'decode_saved.py').read_text());exec(compile(ast.Module(body=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in ('infer','metric')],type_ignores=[]),str(SOURCE/'decode_saved.py'),'exec'),ns)
def jsonout(name,v):(OUT/name).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
ns.update(dict(time=time,json=json,SOURCE=SOURCE,R=R,jsonout=jsonout))
tree=ast.parse((OUT/'exercise.py').read_text());exec(compile(ast.Module(body=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in ('feature_infer','saved_controls')],type_ignores=[]),str(OUT/'exercise.py'),'exec'),ns)
ns['saved_controls']()
