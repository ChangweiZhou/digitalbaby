"""Derive v630 compartment coordinates from skeleton cable inside atlas meshes."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parent
SKELETONS = 'https://flyem.mrc-lmb.cam.ac.uk/flyconnectome/flywire_skeletons_630/'
ATLAS = 'https://raw.githubusercontent.com/navis-org/fafbseg-py/master/fafbseg/data/JFRC2NP.surf.fw.zip'
REFERENCE = 'https://www.immortalfly.app/docs/circuit/'


def wanted(graph):
    return np.isin(graph['roles'], ['EPG', 'PEN', 'PEG', 'D7', 'PFR', 'FC2']) | np.isin(
        graph['labels'], ['PFL3', 'PFNd', 'PFNv', 'hDeltaB', 'ER4d']
    )


def mesh(data):
    header, body = data.split(b'end_header\n', 1)
    assert b'format binary_little_endian 1.0' in header
    n = int(re.search(rb'element vertex (\d+)', header)[1])
    m = int(re.search(rb'element face (\d+)', header)[1])
    xyz = np.frombuffer(body, '<f4', count=3 * n).reshape(n, 3).astype(float)
    faces = np.frombuffer(body, '<i4', count=4 * m, offset=12 * n).reshape(m, 4)
    assert np.all(faces[:, 0] == 3)
    return xyz, xyz[faces[:, 1:]]


def inside(points, triangles):
    """Odd-ray intersection rule. The ray is oblique to avoid atlas grid ties."""
    origin = triangles[:, 0]
    a = triangles[:, 1] - origin
    b = triangles[:, 2] - origin
    ray = np.array([0.037, 0.109, 1.])
    p = np.cross(ray, b)
    det = np.einsum('ij,ij->i', a, p)
    usable = np.abs(det) > 1e-9
    origin, a, b, p, det = (x[usable] for x in (origin, a, b, p, det))
    masks = []
    for chunk in np.array_split(points, max(1, (len(points) + 127) // 128)):
        offset = chunk[:, None, :] - origin
        u = np.einsum('ijk,jk->ij', offset, p) / det
        q = np.cross(offset, a)
        v = (q @ ray) / det
        t = np.einsum('ijk,jk->ij', q, b) / det
        masks.append(np.count_nonzero((u >= 0) & (v >= 0) & (u + v < 1) & (t > 0), axis=1) % 2 == 1)
    return np.concatenate(masks)


def segments(data):
    n, m = np.frombuffer(data, '<u4', count=2)
    xyz = np.frombuffer(data, '<f4', count=3 * n, offset=8).reshape(n, 3).astype(float)
    edges = np.frombuffer(data, '<u4', count=2 * m, offset=8 + 12 * n).reshape(m, 2)
    a, b = xyz[edges].transpose(1, 0, 2)
    return (a + b) / 2, np.linalg.norm(a - b, axis=1)


def mesh_axes(vertices):
    _, _, axes = np.linalg.svd(vertices - vertices.mean(axis=0), full_matrices=False)
    # Anchor the EB plane to the projected atlas x and y axes.
    normal = axes[-1]
    x = np.array([1., 0., 0.]) - normal[0] * normal
    x /= np.linalg.norm(x)
    y = np.cross(normal, x)
    y *= np.sign(y[1])
    axes[0] *= np.sign(axes[0, 0])
    return np.array([x, y]), axes[0]


def map_neurons(graph, directory):
    with ZipFile(directory / 'roi.zip') as archive:
        roi = {name: mesh(archive.read(f'{name}.ply')) for name in ('EB', 'PB', 'FB')}
    eb_axes, _ = mesh_axes(roi['EB'][0])
    _, pb_axis = mesh_axes(roi['PB'][0])
    eb_xy = roi['EB'][0] @ eb_axes.T
    eb_center = (eb_xy.max(axis=0) + eb_xy.min(axis=0)) / 2
    rows = []
    for i in np.flatnonzero(wanted(graph)):
        root_id = str(graph['ids'][i])
        path = directory / f'{root_id}.skel'
        if not path.is_file():
            raise FileNotFoundError(f'Missing v630 skeleton {root_id}')
        data = path.read_bytes()
        mid, length = segments(data)
        row = dict(root_id=root_id, type=str(graph['labels'][i]), side=str(graph['sides'][i]),
                   sha256=sha256(data).hexdigest())
        for name, (vertices, triangles) in roi.items():
            bbox = np.all((mid > vertices.min(axis=0)) & (mid < vertices.max(axis=0)), axis=1)
            points, weights = mid[bbox], length[bbox]
            mask = inside(points, triangles) & (weights > 0)
            if not mask.any():
                continue
            points, weights = points[mask], weights[mask]
            centroid = np.average(points, weights=weights, axis=0)
            item = dict(cable_um=float(weights.sum() / 1000), centroid_um=(centroid / 1000).tolist())
            if name == 'EB':
                xy = points @ eb_axes.T - eb_center
                z = xy[:, 0] + 1j * xy[:, 1]
                unit = z / np.maximum(abs(z), 1e-9)
                mean = np.average(unit, weights=weights)
                item.update(phase_deg=float(np.rad2deg(np.angle(mean))), resultant=float(abs(mean)))
            if name == 'PB':
                projection = points @ pb_axis / 1000
                center = float(np.average(projection, weights=weights))
                item.update(position_um=center, spread_um=float(np.sqrt(np.average((projection - center)**2, weights=weights))))
            row[name] = item
        rows.append(row)
    return rows


def validate_reference(rows, html):
    # This benchmark uses only exact unchanged root IDs with matching type and side.
    data = html.read_bytes()
    matches = re.findall(r'<td>EPG</td><td>(L|R)</td><td>\d+</td><td>(\d+)°</td><td>\d+</td><td><a [^>]+>(\d+)</a></td>', data.decode())
    reference = {root: ('left' if side == 'L' else 'right', float(deg)) for side, deg, root in matches}
    paired = [r for r in rows if r['type'] == 'EPG' and 'EB' in r and r['root_id'] in reference]
    assert all(reference[r['root_id']][0] == r['side'] for r in paired)
    assert len(paired) >= 30
    paired.sort(key=lambda r: r['root_id'])
    a = np.deg2rad([r['EB']['phase_deg'] for r in paired])
    b = np.deg2rad([reference[r['root_id']][1] for r in paired])
    fit = np.arange(len(a)) % 3 == 0
    best = max((-1, 1), key=lambda handedness: abs(np.mean(np.exp(1j * (b[fit] - handedness * a[fit])))))
    rotation = np.angle(np.mean(np.exp(1j * (b[fit] - best * a[fit]))))
    errors = abs(np.rad2deg(np.angle(np.exp(1j * (b[~fit] - best * a[~fit] - rotation)))))
    return dict(url=REFERENCE, sha256=sha256(data).hexdigest(), alignment_cells=int(fit.sum()),
                held_cells=int((~fit).sum()), held_median_deg=float(np.median(errors)),
                held_p90_deg=float(np.quantile(errors, .9)), held_max_deg=float(max(errors)),
                alignment_used_by_runner=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, nargs='?')
    parser.add_argument('--reference-html', type=Path)
    parser.add_argument('--ids', action='store_true')
    args = parser.parse_args()
    with np.load(ROOT / 'EXTRACT.npz', allow_pickle=False) as extract:
        graph = {name: extract[name] for name in ('ids', 'labels', 'roles', 'sides')}
    if args.ids:
        print('\n'.join(graph['ids'][wanted(graph)]))
        return
    if not args.directory:
        parser.error('supply a skeleton directory or use --ids')
    rows = map_neurons(graph, args.directory)
    result = dict(materialization=630, skeletons=SKELETONS, atlas=ATLAS,
                  atlas_sha256=sha256((args.directory / 'roi.zip').read_bytes()).hexdigest(),
                  measure='Cable-weighted skeleton segment midpoints inside transformed atlas ROIs',
                  cells=rows)
    if args.reference_html:
        result['independent_geometry_check'] = validate_reference(rows, args.reference_html)
    (ROOT / 'COMPARTMENT_MAP.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
