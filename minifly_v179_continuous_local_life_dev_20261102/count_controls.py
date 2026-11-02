"""Historical generic count countermodels: calibration selects; never neural output."""
import util as U
import ast,math,pickle,time
from collections import defaultdict
import numpy as np

def definitions(path,names):
 tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names];return compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec')

def main():
 src=U.H/'local/source';env=dict(EOS=256,V=257,np=np,math=math,defaultdict=defaultdict)
 exec(definitions(src/'minifly_v148_pretrain_probe_dev_20261025/ladder.py',['stream','Unigram','PPMC']),env)
 exec(definitions(src/'minifly_v169_native_dev_v1_20261027/v169_count.py',['IKN','line_bits']),env)
 records=pickle.loads((U.H/'local/records.pkl').read_bytes());result={};fit_targets=0;fit_records=0
 for w,sp in records.items():
  result[w]={}
  for role in ('A','B'):
   tr,ca,held=[sp[role+'_'+k] for k in ('train','calib','held')];n=sum(len(r)+1 for r in held);bests={};scores=[]
   uni=env['Unigram']().fit(tr);ub=env['line_bits'](uni,held);result[w][role+'_unigram']=dict(bits=float(ub.sum()/n),record_bit_sums=ub.tolist());fit_targets+=sum(len(r)+1 for r in tr);fit_records+=len(tr)
   for family,k,d in [('PPM',k,None) for k in (2,3,4,5)]+[('IKN',k,d) for k in (2,3,4,5) for d in (.5,.75,.9)]:
    m=(env['PPMC'](k) if family=='PPM' else env['IKN'](k,d)).fit(tr);cb=float(env['line_bits'](m,ca).sum()/sum(len(r)+1 for r in ca));scores.append(dict(family=family,order=k,discount=d,calibration_bits=cb));fit_targets+=sum(len(r)+1 for r in tr);fit_records+=len(tr)
    if family not in bests or cb<bests[family][0]:bests[family]=(cb,m)
   for fam,(cb,m) in bests.items():
    b=env['line_bits'](m,held);result[w][role+'_'+fam]=dict(name=m.name,bits=float(b.sum()/n),record_bit_sums=b.tolist(),calibration_bits=cb)
   result[w][role+'_calibration_grid']=scores;U.budget()
 U.save('COUNT_CONTROLS.json',dict(status='COMPLETE',worlds=result,fit_record_presentations=fit_records,fit_target_presentations=fit_targets,not_neural_output=True,selection='historical V169 grid: PPM orders2..5; IKN orders2..5 x D=.5/.75/.9; minimum calibration bits per family; no held tuning',CPU=time.process_time()));U.finish_process('count_countermodels');print('COUNT_CONTROLS_COMPLETE',flush=True)
if __name__=='__main__':main()
