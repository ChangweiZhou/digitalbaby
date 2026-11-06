"""Recover omitted v630 relays and produce a structural FB plasticity candidate mask."""
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from anatomy import META, EDGE, hash_file

ROOT = Path(__file__).resolve().parent
SIGNS = {'acetylcholine': 1, 'gaba': -1}


def scan(graph):
    fields = ['root_id', 'cell_class', 'cell_sub_class', 'cell_type', 'hemibrain_type', 'top_nt', 'side']
    metadata = pq.read_table(META, columns=fields).to_pydict()
    labels = [t or h or '' for t, h in zip(metadata['cell_type'], metadata['hemibrain_type'])]
    roots, nt = metadata['root_id'], metadata['top_nt']
    roles = np.full(len(roots), '', dtype='<U8')
    roles[graph['parent']] = graph['roles']
    mbon = roles == 'MBON'
    cx = (roles != '') & ~np.isin(roles, ['PN', 'KC', 'APL', 'DAN', 'MBON'])
    incoming, outgoing = defaultdict(dict), defaultdict(dict)
    pfl3 = np.array([name == 'PFL3' for name in labels])
    dna02 = np.array([name == 'DNa02' for name in labels])
    glno = np.array([name == 'GLNO' for name in labels])
    pen_a = np.array([name.startswith('PEN_a') for name in labels])
    motor, yaw = [], []
    for batch in pq.ParquetFile(EDGE).iter_batches(batch_size=400000, columns=[
        'Presynaptic_Index', 'Postsynaptic_Index', 'Excitatory x Connectivity'
    ]):
        u, v, w = [np.asarray(column) for column in batch.columns]
        first = mbon[u] & (roles[v] == '') & (w != 0)
        second = (roles[u] == '') & cx[v] & (w != 0)
        for source, relay, strength in zip(u[first], v[first], w[first]):
            incoming[int(relay)][int(source)] = float(strength)
        for relay, target, strength in zip(u[second], v[second], w[second]):
            outgoing[int(relay)][int(target)] = float(strength)
        for i in np.flatnonzero(pfl3[u] & dna02[v] & (w != 0)):
            source, target, strength = int(u[i]), int(v[i]), float(w[i])
            motor.append(dict(source=roots[source], target=roots[target], target_side=metadata['side'][target],
                              raw_signed_weight=strength, consistent=bool(np.sign(strength) == SIGNS.get(nt[source], 0))))
        for i in np.flatnonzero(glno[u] & pen_a[v] & (w != 0)):
            source, target, strength = int(u[i]), int(v[i]), float(w[i])
            yaw.append(dict(source=roots[source], target=roots[target], source_side=metadata['side'][source],
                            target_side=metadata['side'][target], source_transmitter=nt[source], raw_signed_weight=strength,
                            consistent=bool(np.sign(strength) == SIGNS.get(nt[source], 0))))

    def cell(i):
        return dict(root_id=roots[i], type=labels[i], role=str(roles[i]), transmitter=nt[i])

    relays, recovered_dopamine = [], set()
    strength_counts = {level: Counter() for level in (1, 3, 5, 10)}
    for relay in sorted(incoming.keys() & outgoing.keys()):
        fb = metadata['cell_class'][relay] == 'CX' and metadata['cell_sub_class'][relay] == 'tangential'
        i, o = incoming[relay], outgoing[relay]
        signed_in = {s: w for s, w in i.items() if np.sign(w) == SIGNS.get(nt[s], 0)}
        signed_out = {t: w for t, w in o.items() if np.sign(w) == SIGNS.get(nt[relay], 0)}
        if fb and nt[relay] == 'dopamine' and signed_in:
            recovered_dopamine.update(roots[t] for t in o)
        for level, counter in strength_counts.items():
            first = sum(abs(w) >= level for w in signed_in.values())
            if first:
                counter['all_consistent_cholinergic_or_gaba_paths'] += first * sum(abs(w) >= level for w in signed_out.values())
                counter['consistent_cholinergic_or_gaba_paths_to_FC2_or_PFL3'] += first * sum(
                    abs(w) >= level and (roles[t] == 'FC2' or labels[t] == 'PFL3') for t, w in signed_out.items()
                )
                if fb:
                    counter['fb_tangential_anatomical_paths'] += first * sum(abs(w) >= level for w in o.values())
                    counter['fb_tangential_consistent_additive_paths'] += first * sum(abs(w) >= level for w in signed_out.values())
        relays.append(dict(**cell(relay), fb_tangential=fb,
                           source_edges=[dict(**cell(s), raw_signed_weight=w, consistent=s in signed_in) for s, w in sorted(i.items())],
                           target_edges=[dict(**cell(t), raw_signed_weight=w, consistent=t in signed_out) for t, w in sorted(o.items())]))

    # An anatomical contact with a PFL3 cell only addresses the cell, not the FC2 synapse on it.
    u, v = graph['u'], graph['v']
    candidates = graph['usable'] & (graph['roles'][u] == 'FC2') & (graph['labels'][v] == 'PFL3')
    candidate_ids = np.flatnonzero(candidates & np.isin(graph['ids'][v], sorted(recovered_dopamine)))
    return dict(materialization=630, metadata_sha256=hash_file(META), edge_table_sha256=hash_file(EDGE),
                relay_count=len(relays), fb_tangential_count=sum(r['fb_tangential'] for r in relays),
                fb_tangential_transmitters=dict(Counter(r['transmitter'] for r in relays if r['fb_tangential'])),
                counts_by_minimum_absolute_weight_on_both_hops=strength_counts,
                pfl3_to_dna02=motor,
                glno_to_pen_a=yaw,
                glutamatergic_delta7=[roots[i] for i in np.flatnonzero(roles == 'D7') if nt[i] == 'glutamate'],
                relays=relays,
                candidate_fb_mask=dict(route='FC2→PFL3, recipient contacted by an MBON-addressed FB dopaminergic relay',
                                       structural_contact_only=True, writes_enabled=False,
                                       available_consistent_fc2_pfl3_edges=int(candidates.sum()),
                                       candidate_edges=[dict(index=int(i), source=str(graph['ids'][u[i]]),
                                                             target=str(graph['ids'][v[i]])) for i in candidate_ids]))


def main():
    with np.load(ROOT / 'EXTRACT.npz', allow_pickle=False) as source:
        graph = dict(source)
    result = scan(graph)
    assert result['relay_count'] == json.loads((ROOT / 'BOUNDARY_AUDIT.json').read_text())['omitted_one_relay_candidates']
    (ROOT / 'RESOLVED_BOUNDARY.json').write_text(json.dumps(result, separators=(',', ':')) + '\n')
    print(result['fb_tangential_transmitters'], result['counts_by_minimum_absolute_weight_on_both_hops'])
    print('Candidate FB edges:', len(result['candidate_fb_mask']['candidate_edges']))


if __name__ == '__main__':
    main()
