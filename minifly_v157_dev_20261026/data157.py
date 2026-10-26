"""External evaluator data only; the candidate sees prompt and arriving feedback bytes."""
import hashlib
import support157 as s
import native157 as n
def make():
 out={}
 for world in [202610265701,202610265702]:
  j=n.n.s.dataset(world)
  order=sorted(j['NEW'],key=lambda r:(hashlib.sha256(f"V157|{world}|NEW|{r['id']}".encode()).digest(),r['id']))[:16]
  if world==202610265702:order=order[::-1]
  old=j['OLD'];held=j['assessment'];parts=[old,order,held];seen=set();raw=set()
  for part in parts:
   for r in part:
    key=tuple(r['orbit']);assert key not in seen and r['prompt'] not in raw;seen.add(key);raw.add(r['prompt'])
    assert len(bytes.fromhex(r['answer'])) in (1,2)
  assert [len(x) for x in parts]==[64,16,50]
  schedules={'INTERLEAVED':[r['id'] for r in order]*8,'CLUSTERED':[r['id'] for r in order for _ in range(8)]}
  assert sorted(schedules['INTERLEAVED'])==sorted(schedules['CLUSTERED'])
  out[str(world)]=dict(world=world,OLD=old,NEW=order,assessment=held,OLD_orders=j['orders']['OLD'],schedules=schedules,panel=old+order)
 return out
