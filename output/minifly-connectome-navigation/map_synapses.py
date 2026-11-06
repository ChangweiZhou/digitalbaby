"""Localize observed FlyWire v630 compass synapses using the EB and PB atlas meshes."""
import argparse
from collections import defaultdict
import gzip
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from map_compartments import ROOT, inside, mesh, mesh_axes, validate_reference

SOURCE = 'https://storage.googleapis.com/flywire-data/codex/data/fafb/630/synapse_coordinates.csv.gz'


def read_contacts(path, epg, pen, fb_pairs):
    epg, pen = ({k.encode(): k for k in ids} for ids in (epg, pen))
    fb_pairs = {(a.encode(), b.encode()): (a, b) for a, b in fb_pairs}
    points = defaultdict(list)
    pair = None
    pre = post = b''
    with gzip.open(path, 'rb') as file:
        if next(file).strip() != b'pre_root_id,post_root_id,x,y,z':
            raise ValueError('Unexpected FlyWire coordinates header')
        for line in file:
            if line.startswith(b',,'):
                xyz = line[2:]
            else:
                pre_value, post_value, xyz = line.split(b',', 2)
                pre, post = pre_value or pre, post_value or post
                pair = None
                if post in epg:
                    pair = (pen.get(pre), epg[post])
                elif pre in epg and post in pen:
                    pair = (epg[pre], pen[post])
                else:
                    pair = fb_pairs.get((pre, post))
            if pair is not None:
                points[pair].append(tuple(int(n) for n in xyz.split(b',')))
    return points


def roi_mask(points, vertices, triangles):
    result = np.all((points > vertices.min(axis=0)) & (points < vertices.max(axis=0)), axis=1)
    result[result] = inside(points[result], triangles)
    return result


def checksum(path):
    digest = sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def locate(synapses, atlas, rows, boundary):
    epg = {r['root_id']: r for r in rows if r['type'] == 'EPG'}
    pen = {r['root_id'] for r in rows if r['type'] == 'PEN_a(PEN1)'}
    fb_mask = {(r['source'], r['target']) for r in boundary['candidate_fb_mask']['candidate_edges']}
    modulator = {(r['root_id'], edge['root_id']) for r in boundary['relays']
                 if r['fb_tangential'] and r['transmitter'] == 'dopamine' and any(e['consistent'] for e in r['source_edges'])
                 for edge in r['target_edges'] if edge['type'] == 'PFL3'}
    with ZipFile(atlas) as archive:
        eb, pb, fb = (mesh(archive.read(f'{name}.ply')) for name in ('EB', 'PB', 'FB'))
    eb_axes, _ = mesh_axes(eb[0])
    eb_xy = eb[0] @ eb_axes.T
    center = (eb_xy.max(axis=0) + eb_xy.min(axis=0)) / 2
    inputs, da_sites = defaultdict(list), defaultdict(list)
    candidate_sites, modulator_pairs_with_coordinates = {}, 0
    connections = []
    for (a, b), xyz in read_contacts(synapses, epg, pen, fb_mask | modulator).items():
        points = np.asarray(xyz)
        if (a, b) in modulator or (a, b) in fb_mask:
            within = points[roi_mask(points, *fb)]
            if (a, b) in modulator:
                modulator_pairs_with_coordinates += 1
                da_sites[b].extend(within)
            else:
                candidate_sites[a, b] = within
            continue
        eb_contacts = roi_mask(points, *eb)
        if b in epg and eb_contacts.any():
            inputs[b].extend(points[eb_contacts])
        if a is not None:
            pb_contacts = roi_mask(points, *pb)
            connections.append(dict(source=a, target=b, total=len(points), eb=int(eb_contacts.sum()), pb=int(pb_contacts.sum())))
    if set(inputs) != set(epg):
        raise ValueError('Missing observed EB inputs on some v630 EPG neurons')
    results = []
    for k in sorted(inputs):
        xyz = np.asarray(inputs[k])
        xy = xyz @ eb_axes.T - center
        z = xy[:, 0] + 1j * xy[:, 1]
        mean = np.mean(z / abs(z).clip(1e-12))
        results.append(dict(root_id=k, type='EPG', side=epg[k]['side'],
                            EB=dict(phase_deg=float(np.rad2deg(np.angle(mean))),
                                    resultant=float(abs(mean)), input_contacts=len(xyz))))
    distances = []
    for (a, b), xyz in sorted(candidate_sites.items()):
        if len(xyz) and da_sites[b]:
            delta = xyz[:, None] - np.asarray(da_sites[b])
            distances.append(dict(source=a, target=b, nearest_fb_contact_um=float(np.min(np.linalg.norm(delta, axis=2)) / 1000)))
    fb_check = dict(structural_candidates=len(fb_mask), pairs_with_coordinates=len(candidate_sites),
                    pairs_with_contacts_inside_fb=sum(len(xyz) > 0 for xyz in candidate_sites.values()),
                    modulator_pairs_in_aggregate=len(modulator), modulator_pairs_with_coordinates=modulator_pairs_with_coordinates,
                    dopaminergic_contacts_inside_fb=sum(map(len, da_sites.values())),
                    recipients_with_dopaminergic_contacts_inside_fb=sum(bool(xyz) for xyz in da_sites.values()),
                    candidate_pairs_with_nearby_contact={um: sum(r['nearest_fb_contact_um'] <= um for r in distances)
                                                        for um in (1, 2, 5)},
                    distances=distances, writes_enabled=False)
    return results, sorted(connections, key=lambda r: (r['source'], r['target'])), fb_check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--reference-html', type=Path)
    args = parser.parse_args()
    source = args.directory / 'synapse_coordinates.csv.gz'
    atlas = args.directory / 'roi.zip'
    morphology = json.loads((ROOT / 'COMPARTMENT_MAP.json').read_text())
    boundary = json.loads((ROOT / 'RESOLVED_BOUNDARY.json').read_text())
    assert checksum(atlas) == morphology['atlas_sha256']
    neurons, connections, fb_check = locate(source, atlas, morphology['cells'], boundary)
    result = dict(materialization=630, url=SOURCE, synapse_csv_gzip_sha256=checksum(source),
                  atlas_sha256=morphology['atlas_sha256'], epg=neurons, epg_pen_a_connections=connections,
                  fb_plasticity_contact_check=fb_check)
    if args.reference_html:
        result['independent_geometry_check'] = validate_reference(neurons, args.reference_html)
    (ROOT / 'SYNAPSE_COMPARTMENTS.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'{len(neurons)} EPG neurons and {len(connections)} EPG–PEN1 directed pairs')


if __name__ == '__main__':
    main()
