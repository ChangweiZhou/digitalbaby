"""Lane C: existing V152 checkpoints, read-only own-prefix stopping attribution."""
import pickle
import support154 as s
import native154 as n

def run():
 source=s.ROOT/'minifly_v152_dev_20261025';index=s.load(source/'LOCAL_STATE_INDEX.json');items=[x for x in index['states'] if '_0.03' in x['group'] and x['state'] in ['OLD_W','W']];data=s.load(source/'local_states/DATA_MANIFEST.json')['202610255201'];results=[]
 if len(items)!=6 or not all((source/x['path']).exists() for x in items):s.save('LANE_C.json',dict(status='ASSET_UNAVAILABLE',new_training=0));return []
 for item in items:
  path=source/item['path'];assert s.sha(path)==item['sha256']
  with path.open('rb') as f:model=pickle.load(f)
  before=s.digest(model);questions=data['OLD'] if item['state']=='OLD_W' else data['OLD']+data['NEW'];conditions=[]
  for condition in ['FULL','NO_Q','Q_ONLY']:
   ev=n.evaluate(model,questions,'checkpoint',condition,condition=='FULL');conditions.append(dict(condition=condition,**ev))
  assert s.digest(model)==before and s.sha(path)==item['sha256'];results.append(dict(checkpoint=item,conditions=conditions))
  s.log('CHECKPOINT_COMMIT',group=item['group'],state=item['state'],questions=len(questions))
 s.save('local/LANE_C_PREDICTIONS.json',results);compact=[]
 for row in results:
  compact.append(dict(checkpoint=row['checkpoint'],conditions=[dict(condition=x['condition'],metrics=x['metrics']) for x in row['conditions']],first_payload_margins=[dict(id=r['id'],**{k:r['teacher_forced'][0][k] for k in ['context_vs_EOS','bias_vs_EOS','full_vs_EOS']}) for r in row['conditions'][0]['rows']]))
 s.save('LANE_C.json',dict(status='COMPLETE',snapshots=len(items),query_pairs=1728,new_training_records=0,rows=compact));return results
