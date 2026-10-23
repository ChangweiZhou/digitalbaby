"""Static document checks only; no learner imports or scientific execution."""
from pathlib import Path
import hashlib
import json
import re

P = Path(__file__).resolve().parent
R = P.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    spec = json.loads((P / 'SPEC.json').read_text())
    evidence = json.loads((P / 'EVIDENCE.json').read_text())
    pilot = spec['native_pilot']
    combinations = len(pilot['pairs']) * pilot['lives_per_pair'] * len(pilot['arms'])
    assert combinations == 32
    assert len(set(pilot['pairs'])) == 8
    assert pilot['arms'][1]['N'] == 4 * pilot['arms'][0]['N'] == 20708
    assert pilot['base_histories'] == 3 * combinations == 96
    assert pilot['correlated_S_NEW_continuations'] == combinations
    assert pilot['main_executed_records'] == combinations * (3 * 504 + 252) == 56448
    assert pilot['main_executed_events'] == 5 * pilot['main_executed_records'] == 282240
    assert pilot['shared_prefix_logical_record_equivalents'] == combinations * 252 == 8064
    assert pilot['total_main_logical_record_equivalents'] == 56448 + 8064 == 64512
    assert pilot['qualification_partial_calibration_events'] == 8 * 9 * 2 * 2 == 288
    assert sum((length + 1) * 2 for length in pilot['lengths']) * 14 == 1260
    assert spec['native_applications_executed'] == spec['head_fits_executed'] == spec['three_cycles_executed'] == 0
    assert not any(spec[k] for k in ('implementation_authorized', 'execution_authorized', 'github_push_authorized'))
    for file, expected in evidence['files'].items():
        assert sha(R / file) == expected, file
    for file, expected in evidence['standing_requirements'].items():
        assert sha(Path(file)) == expected, file
    assert sha(P / 'RECEIVED_REVIEW.md') == evidence['received_review_sha256']
    current = json.loads((R / 'PROJECT_DOCUMENTS/CURRENT_EXTERNAL_MEMORY_EXPERIMENT_DESIGN.json').read_text())
    previous = json.loads((P / 'PREVIOUS_MEMORY_POINTER.json').read_text())
    assert {k: v for k, v in current.items() if k != 'next_design'} == previous
    assert current['next_design']['version'] == spec['version']
    review = json.loads((P / 'MECHANISM_REVIEW.json').read_text())
    assert review['full_register_entries'] == len(review['all_register_ids_consulted']) == 254
    for filename in ('DESIGN.md', 'STATIC_REVIEW.md'):
        content = (P / filename).read_text()
        for link in re.findall(r'\]\(([^)]+)\)', content):
            if '://' not in link:
                assert (P / link.split('#')[0]).exists(), link
    print(json.dumps({'static_consistency': 'PASS', 'main_planned_records': 56448,
                      'main_planned_events': 282240, 'native_applications_executed': 0,
                      'head_fits_executed': 0, 'arithmetic_pointer_unchanged': True,
                      'completed_memory_result_preserved': True}, indent=2))


if __name__ == '__main__':
    main()
