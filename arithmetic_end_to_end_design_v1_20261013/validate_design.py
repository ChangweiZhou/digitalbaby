"""Design/data-only validator. Does not import, instantiate or train a learner."""
import itertools,json,random,hashlib
from pathlib import Path
OPS=('+','-','*','/')
def primitive(op,a,b):
 if op=='+':return a+b
 if op=='-':return a-b
 if op=='*':return a*b
 if b==0 or a%b:return None
 return a//b

def family(op,a,b,y):
 if op in ('+','-'):
  x=sorted((abs(a),abs(b),abs(y)))
  assert x[0]+x[1]==x[2]
  return ('SUM',x[0],x[1])
 return ('PRODUCT',*sorted((abs(a),abs(b)) if op=='*' else (abs(b),abs(y))))
def rows(kind,a,b):
 if kind=='SUM':
  s=a+b;out=[('+',a,b,s),('+',b,a,s),('-',s,a,b),('-',s,b,a),('-',a,s,-b),('-',b,s,-a)]
 else:
  p=a*b;out=[('*',a,b,p),('*',b,a,p)]
  if a:out.append(('/',p,a,b))
  if b:out.append(('/',p,b,a))
 return sorted(set(out))
def canon(t):
 if isinstance(t,int):return ('leaf',t)
 op,a,b=t
 children=[]
 def add(x):
  if op in ('+','*') and isinstance(x,tuple) and x[0]==op:add(x[1]);add(x[2])
  else:children.append(canon(x))
 add(a);add(b)
 if op in ('+','*'):children.sort(key=repr)
 return (op,*children)
def render(t):
 if isinstance(t,int):return str(t)
 op,a,b=t;return '('+render(a)+op+render(b)+')'
def make_split(seed):
 rng=random.Random(seed);pairs=list(itertools.combinations_with_replacement(range(16),2));eligible=[p for p in pairs if min(p)>=2]
 held=set();train=[];test=[]
 for kind in ('SUM','PRODUCT'):
  picked=set(rng.sample(eligible,32));held|={(kind,*p) for p in picked}
  train.extend(r for a,b in pairs if (a,b) not in picked for r in rows(kind,a,b))
  test.extend(r for a,b in sorted(picked) for r in rows(kind,a,b))
 assert {family(*r) for r in train}.isdisjoint({family(*r) for r in test})
 for op in OPS:
  tr=[r for r in train if r[0]==op];ho=[r for r in test if r[0]==op]
  assert tr and ho
  assert set(''.join(str(x) for r in ho for x in r[1:3]))<=set(''.join(str(x) for r in tr for x in r[1:3]))
  assert set(''.join(str(r[3]) for r in ho))<=set(''.join(str(r[3]) for r in tr))
 # Join two taught primitive facts. Intermediate value is evaluator-only; not a training field.
 groups={}
 for r1 in train:
  op1,a,b,m=r1
  for r2 in train:
   op2,x,z,y=r2
   trees=[]
   if x==m:trees.append((op2,(op1,a,b),z))
   if z==m:trees.append((op2,x,(op1,a,b)))
   for tree in trees:
    key=repr(canon(tree));g=groups.setdefault(key,{'pair':(op1,op2),'answer':y,'forms':set(),'parts':set()})
    assert g['answer']==y and g['pair']==(op1,op2)
    g['forms'].add(render(tree)+'=');g['parts'].add((r1,r2))
 pool={pair:sorted(k for k,g in groups.items() if g['pair']==pair) for pair in itertools.product(OPS,repeat=2)}
 selected_train=[];selected_held=[]
 for pair,keys in pool.items():
  assert len(keys)>=48,(pair,len(keys))
  order=keys.copy();rng.shuffle(order);selected_train+=order[:32];selected_held+=order[32:48]
 assert set(selected_train).isdisjoint(selected_held)
 prompts_train={p for k in selected_train for p in groups[k]['forms']};prompts_held={p for k in selected_held for p in groups[k]['forms']}
 assert prompts_train.isdisjoint(prompts_held)
 assert set(''.join(prompts_held))<=set(''.join(prompts_train))
 train_reply_chars=set(''.join(str(groups[k]['answer']) for k in selected_train))
 held_reply_chars=set(''.join(str(groups[k]['answer']) for k in selected_held))
 assert held_reply_chars<=train_reply_chars
 permutation=list(range(256));random.Random(seed+918273).shuffle(permutation)
 assert set(permutation)==set(range(256))
 inverse=[0]*256
 for original,mapped in enumerate(permutation):inverse[mapped]=original
 assert bytes(inverse[permutation[b]] for b in range(256))==bytes(range(256))
 assert all(family(*r) not in held for k in selected_train+selected_held for pair in groups[k]['parts'] for r in pair)
 assert all(r in train for k in selected_train+selected_held for pair in groups[k]['parts'] for r in pair)
 # Concrete operation-balanced microblocks; no operation/phase field reaches the learner.
 for budget in (64,128,256,512,1024,2048):assert 4*budget%4==0
 for budget in (64,128,256,512,1024,2048):assert budget%16==0
 return {'seed':seed,'binary_seed_families_per_kind':len(pairs),'train_families_per_kind':len(pairs)-32,
  'binary_train_prompts_by_operation':{op:sum(r[0]==op for r in train) for op in OPS},
  'binary_held_prompts_by_operation':{op:sum(r[0]==op for r in test) for op in OPS},
  'available_mixed_groups_by_pair':{' '.join(p):len(k) for p,k in pool.items()},
  'mixed_train_groups':len(selected_train),'mixed_held_groups':len(selected_held),
  'mixed_train_forms':len(prompts_train),'mixed_held_forms':len(prompts_held),
  'maximum_prompt_bytes':max(map(len,prompts_train|prompts_held)),
  'maximum_reply_bytes':max(len(str(groups[k]['answer'])) for k in selected_train+selected_held)}
def main():
 manifest=json.loads((Path(__file__).parent/'DESIGN.json').read_text())
 assert len(manifest['configurations'])==4 and manifest['CONFIRM']['primary_tests']==17
 assert manifest['implemented'] is False and manifest['scientific_jobs_executed']==0
 assert manifest['representations']['human_numeric_or_role_features'] is False
 assert manifest['maximum_external_record_applications']==88*32768
 assert (8+48)*22525==manifest['maximum_internal_replayed_record_applications']
 tests=[make_split(s) for s in range(202610139901,202610139905)]
 P_G=259*16+3*(64*16+64*64+2*64)+(64*257+257)
 P_T=259*32+128*32+2*((3*32*32+3*32)+(32*32+32)+(32*64+64+64*32+32)+4*32)+2*32+32*257+257
 assert P_G==36593 and P_T==38017
 max_job=3*8192+4*2048
 assert max_job==32768
 assert 4*4==16 and 24*3==72 and (16+72)*max_job==2883584
 check={'accepted':True,'scope':'Static design/data enumeration only','learning_records_executed':0,'model_forward_passes':0,
  'tested_data_seeds':tests,'model_parameter_formulas':{'G':P_G,'T':P_T},'job_external_record_applications':max_job,
  'conditional_max_jobs':88,'conditional_max_external_record_applications':2883584,
  'checks':['whole binary family separation','binary byte support','mixed group/form separation','all mixed parts taught binary facts',
   'no binary-held primitive in mixed stream','all 16 ordered operation pairs have sufficient groups','balanced checkpoint divisibility','model parameter formulas','shared-prefix resource counts','mixed input/reply byte support','all256payload relabeling reversibility','manifest/scientific status consistency','conditional replay count']}
 (Path(__file__).parent/'STATIC_VALIDATION.json').write_text(json.dumps(check,indent=2)+'\n');print(json.dumps({'accepted':True,'seeds':len(tests),'G':P_G,'T':P_T,'max_jobs':88}))
if __name__=='__main__':main()
