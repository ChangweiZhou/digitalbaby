"""G0 feature coverage oracle, evaluator-only and never a scored learner."""
from v4_common import *
from geometry import Geometry,SegmentState,Boundary
from core_v4 import target_bits,token_from_values
from data_v4 import dataset
from scipy.sparse import csr_matrix,hstack
import resource,gc,shutil

def extract(geo,records,seed,exploratory):
 b=Boundary(seed);rows=[];cols=[];labels=[];meta=[];row=0;closures=[]
 for p,a in records:
  s=SegmentState()
  for byte in p:s.feed(byte,b.decide(byte,exploratory) if geo.segmented else False)
  for k,token in enumerate((*a,256)):
   x,_=geo.encode(s);active=np.flatnonzero(x);rows.extend([row]*len(active));cols.extend(active.tolist())
   labels.append(target_bits(token));meta.append(dict(prompt=p.hex(),position=k,want=token,is_EOS=token==256))
   if token<256:s.feed(token,b.decide(token,exploratory) if geo.segmented else False)
   row+=1
  closures.append(s.closures)
 X=csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(row,geo.n))
 # A fitted constant offset is part of the evaluator's optimistic linear diagnostic.
 X=hstack([X,csr_matrix(np.ones((row,1)))],format='csr')
 return X,np.stack(labels),meta,closures

def metrics(pred,meta):
 tokens=[token_from_values(z) for z in pred];rows=[dict(**r,got=g,correct=bool(g==r['want'])) for r,g in zip(meta,tokens)]
 digit=[r for r in rows if not r['is_EOS']];end=[r for r in rows if r['is_EOS']]
 grouped={}
 for r in rows:grouped.setdefault(r['prompt'],[]).append(r['correct'])
 return dict(correct_digit_fraction=sum(r['correct'] for r in digit)/len(digit),EOS_fraction=sum(r['correct'] for r in end)/len(end),teacher_forced_record_exact=sum(all(v) for v in grouped.values())/len(grouped),rows=rows)

def run():
 assert json.loads((HERE/'cycles/cycle1/AUDIT.json').read_text())['accepted']
 start=time.process_time();wall=time.monotonic();cfg=spec();snapshot=HERE/'cycles/cycle2/SOURCE_EXECUTED';snapshot.mkdir(parents=True,exist_ok=True)
 for f in HERE.glob('*.py'):shutil.copy2(f,snapshot/f.name)
 save(Path('cycles/cycle2/SOURCE_LOCK.json'),lock())
 train,test=dataset(cfg['worlds'][0],cfg['G0_train_records'],cfg['G0_test_records']);cases=[]
 # No held labels select hyperparameters or native model parameters.
 for n in cfg['sizes']:
  for mode,topo,explore in (('neutral_deterministic',True,False),('neutral_exploratory',True,True),('random_fan_exploratory',False,True)):
   t=time.process_time();g=Geometry(n,202610204100,topographic=topo)
   X,Y,tm,tc=extract(g,train,441,explore);H,HY,hm,hc=extract(g,test,442,explore)
   K=(X@X.T).toarray();K.flat[::len(K)+1]+=cfg['G0_ridge'];alpha=np.linalg.solve(K,Y)
   fit=metrics((X@X.T)@alpha,tm);held=metrics((H@X.T)@alpha,hm)
   r=dict(N_KC=n,mode=mode,training=fit,held=held,G0_pass=bool(held['correct_digit_fraction']>=.95),
     training_records=len(train),held_records=len(test),teacher_forced_code_extraction_record_presentations=len(train)+len(test),
     observed_target_events=len(tm)+len(hm),train_closures_mean=float(np.mean(tc)),test_closures_mean=float(np.mean(hc)),
     feature_dimension=g.dimension,fit_CPU_seconds=time.process_time()-t,
     own_output_learner=False,arithmetic_structure_in_learner=False,scientific_records=0)
   save(Path(f'cycles/cycle2/G0_{n}_{mode}.json'),r);cases.append({k:v for k,v in r.items() if k not in ('training','held')}|dict(train_digit=fit['correct_digit_fraction'],held_digit=held['correct_digit_fraction'],held_teacher_forced_exact=held['teacher_forced_record_exact']))
   print(json.dumps(cases[-1]),flush=True);del X,Y,H,HY,K,alpha,g;gc.collect()
 result=dict(accepted=True,scope='Evaluator-only G0 diagnostic; no native learning and no E3 efficacy claim',cases=cases,
  G0_any_pass=any(r['G0_pass'] for r in cases),full_record_training_applications=0,
  oracle_record_presentations=sum(r['teacher_forced_code_extraction_record_presentations'] for r in cases),
  oracle_target_events=sum(r['observed_target_events'] for r in cases),
  no_test_hyperparameter_selection=True,correct_digit_definition='Exact ordinary output-byte match at teacher-forced answer positions, excluding EOS; full own-output still required for E3',
  limitations='Readout fit is not an upper bound; neutral boundary policies have not learned; teacher forcing reveals earlier expected answer bytes. Failure stops expensive P3 under this contract, not generic boundary learning research.',
  CPU_seconds=time.process_time()-start,wall_seconds=time.monotonic()-wall,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_unchanged=lock()==json.loads((HERE/'cycles/cycle2/SOURCE_LOCK.json').read_text()),scientific_records=0)
 assert result['source_unchanged'];save(Path('cycles/cycle2/AUDIT.json'),result)
if __name__=='__main__':run()
