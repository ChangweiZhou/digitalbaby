"""Single-use technical ledger: all failed attempts/reference calls remain charged."""
from pathlib import Path
import json,time,resource,sys
H=Path(__file__).resolve().parent
class Ledger:
 def __init__(self):
  self.path=H/'EXPOSURE_LEDGER.json';assert not self.path.exists(),'No uncharged rerun';self.rows=[];self.start=time.process_time();self.epoch=time.time();self.write()
 def charge(self,label,records,targets=0,updates=0,category='technical',forward_sites=0):
  self.rows.append(dict(label=label,record_views=records,supervised_target_evaluations=targets,optimizer_updates=updates,category=category,forward_sites=forward_sites));self.write()
 def write(self):
  rows=[r for r in self.rows if r['category']=='technical'];a={'technical_record_equivalents':sum(r['record_views'] for r in rows),'technical_supervised_target_evaluations':sum(r['supervised_target_evaluations'] for r in rows),'technical_optimizer_updates':sum(r['optimizer_updates'] for r in rows),'read_only_record_views':sum(r['record_views'] for r in self.rows if r['category']!='technical'),'read_only_forward_sites':sum(r['forward_sites'] for r in self.rows if r['category']!='technical')}
  assert a['technical_record_equivalents']<=32 and a['technical_supervised_target_evaluations']<=3840
  rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024);cpu=time.process_time()-self.start+15;wall=time.time()-self.epoch
  assert cpu<=900 and wall<=1800 and rss<=1073741824
  self.path.write_text(json.dumps({'totals':a,'rows':self.rows,'CPU_seconds_including_15_setup_allowance':cpu,'process_wall_seconds':wall,'peak_RSS_bytes':rss},indent=2)+'\n');return a
