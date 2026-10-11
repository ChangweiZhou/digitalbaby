"""Independent raw-byte, numeric, probe and commit audit; no learning calls."""
import numpy as np
from audit import audit
from equation_audit import verify_equations
def verify_probe_links(r):
    for out in r['arms'].values():
        probes=[p for phase in out['probes'].values() for p in phase.values()]+[out['erased_final']]
        for p in probes:
            previous=b''
            for q in p['rows']:
                for row in q['trace']:
                    if row['observed_target'] is not None or row['answer_update'] or row['plastic'] or any(row['write_l1']) or row['time_transform_l1'] or row['clipped_rows']:raise AssertionError('probe hidden update')
                    if bytes.fromhex(row['pre_line'])!=previous:raise AssertionError('probe sensor linkage')
                    if row['byte']==10:previous=b''
                    elif row['byte']!=32:previous+=bytes([row['byte']])
                    if bytes.fromhex(row['post_line'])!=previous:raise AssertionError('probe sensor poststate')
                actual=next(x['prediction'] for x in q['trace'] if x['prediction'])
                if actual['probabilities']!=q['probabilities'] or actual['emitted']!=q['emitted']:raise AssertionError('probe output linkage')
                if p['erase'] and abs(actual['numeric_value'])>1e-15:raise AssertionError('erase did not erase content')
    return dict(verdict='PASS')
def verify(r):
    a=audit(r);b=verify_equations(r);c=verify_probe_links(r)
    if r['training_histories']!=6 or r['training_bytes']!=24072 or a['audited_rows']!=27322:raise AssertionError('full confirmation world counts')
    return dict(verdict='PASS',basic=a,equations=b,probe=c,model_trajectory_executions=0)
