"""Document integrity and count checks only; no learner or fitting imports."""
from pathlib import Path
import hashlib
import json
import re

P = Path(__file__).resolve().parent
R = P.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    s = json.loads((P / 'SPEC.json').read_text())
    e = json.loads((P / 'EVIDENCE.json').read_text())
    c = s['controller']
    h = s['prospective_harness']
    assert c['presynaptic_dimension'] == c['input_symbol_count'] + c['H'] + 1 == 348
    assert c['weight_scalars'] == 2 * c['H'] * c['presynaptic_dimension'] == 61248
    assert c['three_array_groups_bytes_float64'] == 3 * c['weight_scalars'] * 8 == 1469952
    assert h['OLD_orbits'] == h['length_strata'] * h['OLD_per_stratum'] == 128
    assert h['NEW_orbits'] == h['length_strata'] * h['NEW_per_stratum'] == 32
    assert h['held_orbits'] == h['length_strata'] * h['held_per_stratum'] == 64
    assert h['OLD_records'] == h['OLD_orbits'] * h['OLD_passes'] == 4096
    assert h['NEW_records'] == h['NEW_orbits'] * h['NEW_passes'] == 256
    assert h['main_total_record_applications'] == len(h['worlds']) * len(h['arms']) * (4096 + 256) == 34816
    assert h['main_native_event_upper_bound'] == 16 * 34816 == 557056
    assert not any(v for k, v in s['authorization'].items() if k != 'design')
    assert not any(s['executed_by_delivery'].values())
    for file, expected in e['local_files'].items():
        assert sha(R / file) == expected, file
    for file, expected in e['standing_requirements'].items():
        assert sha(Path(file)) == expected, file
    original = (P / 'CLAUDE_V6_ORIGINAL.md').read_bytes()
    assert sha(P / 'CLAUDE_V6_ORIGINAL.md') == e['original_Claude_design_sha256']
    assert hashlib.sha1(b'blob ' + str(len(original)).encode() + b'\0' + original).hexdigest() == e['Claude_source']['blob_sha']
    m = json.loads((P / 'MECHANISM_REVIEW.json').read_text())
    assert m['register_entries'] == len(set(m['all_register_ids'])) == 254
    for name in ('DESIGN.md', 'LEARNING_RULE.md', 'REVIEW.md'):
        text = (P / name).read_text()
        assert not any(line.rstrip() != line for line in text.splitlines()), name
        for link in re.findall(r'\]\(([^)]+)\)', text):
            if '://' not in link:
                assert (P / link.split('#')[0]).exists(), (name, link)
    print(json.dumps({'static_checks': 'PASS', 'native_records_executed': 0,
                      'fits_executed': 0, 'three_cycles_executed': 0,
                      'prospective_main_records': 34816,
                      'original_Claude_blob_verified': True,
                      'arithmetic_pointer_unchanged': True,
                      'completed_memory_pointer_unchanged': True}, indent=2))


if __name__ == '__main__':
    main()
