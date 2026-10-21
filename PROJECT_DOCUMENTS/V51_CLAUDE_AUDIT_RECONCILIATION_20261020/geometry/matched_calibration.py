from pathlib import Path
code=Path('/tmp/v51_claude_audit_geometry_20261020/check.py').read_text()
exec(code.split('\nfor seed in (202610205101,202610205102):')[0])
result={'scope':'Encoding-only sensitivity diagnostic; perfect identity/classes fixture. Corrected prefeedback frames, exactly one threshold update per heard payload byte; no extra terminal updates. Test condition only, not a committed implementation change.','native_value_teaching_events':0,'results':[]}
for seed in (202610205101,202610205102):
 bank=fresh_bank(65536,seed);codes=Codes(seed)
 for i,b in enumerate(alpha):codes.w[i]=BITS[b];codes.classes[i]=0 if 48<=b<=57 else 1
 codes.identity_frozen=codes.context_frozen=True
 g=Geometry(65536,seed+17,bank.kc_side,bank.active_fraction,True)
 heard=0;calibrations=0;complete=0
 for p,a in records:
  s=Sequence(codes)
  for b in p:
   if heard>=2048:break
   s.feed(b);g.calibrate(s);heard+=1;calibrations+=1
  if heard>=2048:break
  s.begin_reply()
  for b in a:
   if heard>=2048:break
   g.calibrate(s);calibrations+=1
   s.feed_output(b);heard+=1
  if heard>=2048:break
  complete+=1
 row={'seed':seed,'payload_bytes':heard,'threshold_calibration_events':calibrations,'complete_records':complete,'after_causal_calibration':metrics(g,codes)}
 result['results'].append(row)
result['CPU_seconds']=time.process_time()-start
Path('/tmp/v51_claude_audit_geometry_20261020/MATCHED_CALIBRATION.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
