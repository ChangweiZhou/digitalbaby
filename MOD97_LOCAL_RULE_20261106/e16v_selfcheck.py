"""Validate the numerical checker with a deliberate one-cell error; never trains."""
import csv
import json
import shutil
import tempfile
from pathlib import Path
from e16v_crosswalk import crosswalk

root=Path(__file__).resolve().parent
with tempfile.TemporaryDirectory(prefix='e16v_checker_') as td:
    t=Path(td)
    shutil.copytree(root/'E16V_SOURCE_SNAPSHOT',t/'E16V_SOURCE_SNAPSHOT')
    _,good=crosswalk(t)
    assert len(good)==3124 and all(x['verdict']=='CONFIRMED' for x in good)
    p=t/'E16V_SOURCE_SNAPSHOT'/'MOD97_LOCAL_RULE_20261106'/'exp45'/'SUMMARY_E16.md'
    text=p.read_text();bad=text.replace('| 52.6 |','| 53.6 |',1)
    assert bad!=text
    p.write_text(bad)
    _,changed=crosswalk(t)
    errors=[x for x in changed if x['verdict']=='DISCREPANCY']
    assert len(errors)==1 and errors[0]['published']=='53.6' and abs(errors[0]['recomputed']-52.5625)<1e-10, errors
scores=list(csv.DictReader((root/'E16V_CELL.csv').open()))
assert len(scores)==168 and sum(float(x['percent'])>10 for x in scores)==18
assert max(float(x['percent']) for x in scores)==32.25
g=list(csv.DictReader((root/'E16V_GATES.csv').open()))
assert all(x['passed']=='True' for x in g if x['prediction']=='P16.4')
assert sum(x['passed']=='False' for x in g if x['prediction']=='P16D.4')==9
assert sum(x['passed']=='False' for x in g if x['prediction']=='P16D2.3')==6
result=dict(status='PASS',network_executions=0,life_reruns=0,clean_components=3124,
            deliberate_one_cell_error_detected=True,operation_count=168,operation_exceptions=18,
            original_F1_P16_4_preserved=True,P16D4_literal_misses=9,P16D2_3_literal_misses=6)
(root/'E16V_AUDIT_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
