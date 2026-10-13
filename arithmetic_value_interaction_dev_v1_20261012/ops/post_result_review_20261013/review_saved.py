"""Postrun descriptive analysis only. Reads receipts; never imports a learner."""
from collections import Counter, defaultdict
from pathlib import Path
import gzip, json, math, statistics, re

ROOT = Path(__file__).resolve().parents[2]
OPS = ('+', '-', '*', '/')
CONFIGS = ('BYTE-ADD', 'VALUE-ADD', 'BYTE-PROD', 'VALUE-PROD', 'LEGACY-B')

def mean(xs):
    return statistics.fmean(xs) if xs else None

def row_metrics(rows):
    answers = Counter(bytes.fromhex(r['payload_hex']).decode('ascii', errors='backslashreplace') for r in rows)
    valid = [r for r in rows if re.fullmatch(rb'(0|[1-9][0-9]*|-[1-9][0-9]*)', bytes.fromhex(r['payload_hex']))]
    values = [r for r in rows if r['model_predicted_value'] is not None]
    byop = {o: [r for r in rows if r['operation'] == o] for o in OPS}
    return {
        'n_probes': len(rows),
        'accuracy_pooled': mean([float(r['correct']) for r in rows]),
        'accuracy_balanced_macro': mean([mean([float(r['correct']) for r in byop[o]]) for o in OPS]),
        'ended': mean([float(r['status'] == 'ENDED') for r in rows]),
        'empty_reply': mean([float(not r['payload_hex']) for r in rows]),
        'zero_reply': mean([float(r['payload_hex'] == '30') for r in rows]),
        'canonical_integer_reply': len(valid)/len(rows),
        'first_byte_correct': mean([float(bool(r['payload_hex']) and r['payload_hex'][:2] == r['target_hex'][:2]) for r in rows]),
        'top_replies': answers.most_common(10),
        'by_operation': {o: {
            'accuracy': mean([float(r['correct']) for r in rs]),
            'zero_reply': mean([float(r['payload_hex']=='30') for r in rs]),
            'numeric_mae': mean([abs(r['model_predicted_value']-int(bytes.fromhex(r['target_hex']))) for r in rs if r['model_predicted_value'] is not None]),
            'numeric_nearest_integer_correct': mean([float(math.floor(r['model_predicted_value']+.5)==int(bytes.fromhex(r['target_hex']))) for r in rs if r['model_predicted_value'] is not None]),
        } for o,rs in byop.items()},
        'numeric_mae': mean([abs(r['model_predicted_value']-int(bytes.fromhex(r['target_hex']))) for r in values]),
        'numeric_rmse': math.sqrt(mean([(r['model_predicted_value']-int(bytes.fromhex(r['target_hex'])))**2 for r in values])) if values else None,
        'numeric_prediction_mean': mean([r['model_predicted_value'] for r in values]),
        'numeric_prediction_sd': statistics.pstdev([r['model_predicted_value'] for r in values]) if values else None,
        'numeric_nearest_integer_correct': mean([float(math.floor(r['model_predicted_value']+.5)==int(bytes.fromhex(r['target_hex']))) for r in values]),
    }

data = defaultdict(list)
for path in sorted((ROOT/'results/receipts').glob('*.json.gz')):
    with gzip.open(path, 'rt') as f:
        receipt = json.load(f)
    data[receipt['config']].append(receipt)

report = {
    'postrun_descriptive_only': True,
    'new_training_records': 0,
    'no_learner_import_or_replay': True,
    'official_verdict_unchanged': 'NO_CONFIRMED_REPLACEMENT',
    'numerical_rounding_diagnostic_is_not_own_output_score': True,
    'counts': {'receipts': sum(map(len, data.values())), 'worlds_per_configuration': {k: len(v) for k,v in data.items()}},
    'configurations': {},
}
for config in CONFIGS:
    receipts = data[config]
    histories = {}
    for history in ('W', 'N_ALL', 'SHUFFLE'):
        hs = [r['histories'][history] for r in receipts]
        final = [a for h in hs for a in h['final']]
        taught = [a for h in hs for a in h['checkpoints'][-1]['taught']]
        checkpoints = []
        for j in range(6):
            rows = [a for h in hs for a in h['checkpoints'][j]['held']]
            tr = [a for h in hs for a in h['checkpoints'][j]['taught']]
            checkpoints.append({'samples_per_operation': hs[0]['checkpoints'][j]['samples_per_operation'],
                'held': row_metrics(rows), 'taught': row_metrics(tr)})
        immediate = {}
        for op in OPS:
            last = [next(b for b in reversed(h['blocks']) if b['operation'] == op) for h in hs]
            rows = [a for b in last for a in b['held'] if a['operation']==op]
            taught_rows = [a for b in last for a in b['taught'] if a['operation']==op]
            immediate[op] = {
                'held_accuracy': mean([float(a['correct']) for a in rows]),
                'taught_accuracy': mean([float(a['correct']) for a in taught_rows]),
                'numeric_mae': mean([abs(a['model_predicted_value']-int(bytes.fromhex(a['target_hex']))) for a in rows if a['model_predicted_value'] is not None]),
            }
        last_records = [row for h in hs for row in h['ledger'][-64:]]
        first_events = [next(e for e in row['events'] if e['phase']=='reply') for row in last_records]
        eos_events = [row['events'][-1] for row in last_records]
        histories[history] = {'final_held': row_metrics(final), 'final_taught': row_metrics(taught),
            'checkpoint_curves': checkpoints, 'last_own_operation_block': immediate,
            'last64_records_first_reply_choice_correct': mean([float(e['prior_choice']==e['event']) for e in first_events]),
            'last64_records_eos_choice_correct': mean([float(e['prior_choice']==e['event']) for e in eos_events]),
            'last64_numeric_aux_mae': mean([abs(e['core_audit']['model_predicted_value']-e['core_audit']['numeric_observed_target']) for e in eos_events if e.get('core_audit') and e['core_audit']['numeric_observed_target'] is not None]),
            'aux_used_count': sum(bool(e.get('core_audit') and e['core_audit']['aux_used']) for h in hs for row in h['ledger'] for e in row['events']),
            'example_wrong_held': [{'prompt': bytes.fromhex(r['prompt_hex']).decode(), 'target': bytes.fromhex(r['target_hex']).decode(),
                'reply': bytes.fromhex(r['payload_hex']).decode('ascii', errors='backslashreplace'),
                'status':r['status'], 'value':r['model_predicted_value']} for r in final if not r['correct']][:12]}
    report['configurations'][config] = histories

path = ROOT/'ops/post_result_review_20261013/READ_ONLY_REANALYSIS.json'
path.write_text(json.dumps(report,indent=2)+'\n')
for config, hist in report['configurations'].items():
    w=hist['W']; fh=w['final_held']; ft=w['final_taught']
    print(json.dumps({'config':config,'held_macro':fh['accuracy_balanced_macro'],'taught_macro':ft['accuracy_balanced_macro'],
        'held_ended':fh['ended'],'held_zero':fh['zero_reply'],'taught_zero':ft['zero_reply'],
        'held_first_byte':fh['first_byte_correct'],'held_numeric_mae':fh['numeric_mae'],
        'held_numeric_nearest_integer':fh['numeric_nearest_integer_correct'],'taught_numeric_mae':ft['numeric_mae'],
        'held_top_replies':fh['top_replies'],'immediate':w['last_own_operation_block'],
        'held_by_op':fh['by_operation'],'taught_by_op':ft['by_operation'],
        'curve_zero':[c['held']['zero_reply'] for c in w['checkpoint_curves']],
        'curve_taught':[c['taught']['accuracy_balanced_macro'] for c in w['checkpoint_curves']],
        'curve_numeric_mae':[c['held']['numeric_mae'] for c in w['checkpoint_curves']]}))
print('saved',path)
