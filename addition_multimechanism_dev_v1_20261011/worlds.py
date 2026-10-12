"""Environment-only latent meanings; no truth metadata crosses learner API."""
import importlib.util
import random
import config

spec = importlib.util.spec_from_file_location('sample_gen_geometry', config.ROOT.parent / 'addition_sample_generalization_v2_design_20261011' / 'benchmark_fixtures.py')
geometry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry)
FAMILIES = geometry.families()
# Frozen confirmation graph, for the environment/auditor, never learner input.
FAMILIES['TRIANGLE_TAIL']=[tuple(sorted(tuple(sorted(e)) for e in
    ((0,1),(1,2),(2,0),(2,3),(3,4))))]


def raw(pair, target=None, query=False):
    result = bytes((pair[0], 43, pair[1], 61))
    if target is not None:
        result += bytes((48+target,))
    return result + b'\n'


def generate(world, family, noise, ordinary):
    config.admit(world, family, noise, ordinary)
    if noise not in ('STANDARD', 'BLURRY') or type(ordinary) is not bool:
        raise ValueError('technical task roster')
    permutations = random.Random(world*100+1)
    old_map = list(range(5))
    if not ordinary:
        permutations.shuffle(old_map)
    new_map = list(range(5))
    permutations.shuffle(new_map)
    graphs = random.Random(world*100+2)
    old_mask = graphs.choice(FAMILIES[family])
    new_mask = graphs.choice(FAMILIES[family])
    old_bytes = b'01234' if ordinary else b'ABCDE'
    new_bytes = b'FGHIJ'
    order = random.Random(world*100+3)
    corrupt = random.Random(world*100+4)
    wrong = random.Random(world*100+5)
    shuffle = random.Random(world*100+6)
    correct, erroneous = (.8, .1) if noise == 'STANDARD' else (.6, .2)
    master = {'world': world, 'family': family, 'noise': noise, 'ordinary': ordinary,
              'technical': True, 'old_map_private': old_map, 'new_map_private': new_map,
              'old_alphabet': list(old_bytes), 'new_alphabet': list(new_bytes),
              'old_mask': list(map(list, old_mask)), 'new_mask': list(map(list, new_mask)),
              'old': [], 'new': []}
    for phase, count, alphabet, meaning, mask in (
        ('old', 32, old_bytes, old_map, old_mask), ('new', 8, new_bytes, new_map, new_mask)
    ):
        for epoch in range(count):
            records = list(geometry.slots(mask))
            order.shuffle(records)
            block = []
            for a, b in records:
                truth = meaning[a]+meaning[b]
                draw = corrupt.random()
                if draw < correct:
                    target, kind = truth, 'correct'
                elif draw < correct+erroneous:
                    target = wrong.choice([y for y in range(9) if y != truth])
                    kind = 'wrong'
                else:
                    target, kind = None, 'missing'
                block.append({'pair': [alphabet[a], alphabet[b]], 'truth_private': truth,
                              'target': target, 'kind_private': kind, 'epoch': epoch})
            labels = [r['target'] for r in block if r['target'] is not None]
            shuffle.shuffle(labels)
            k = 0
            for record in block:
                record['shuffled_target'] = None
                if record['target'] is not None:
                    record['shuffled_target'] = labels[k]
                    k += 1
            master[phase].extend(block)
    return master


def queries(master, phase, taught):
    alphabet = master[phase+'_alphabet']
    meaning = master[phase+'_map_private']
    mask = {tuple(row) for row in master[phase+'_mask']}
    chosen = mask if taught else geometry.ALL_ORBITS-mask
    result = []
    for a, b in sorted(chosen):
        orientations = ((a, b),) if a == b else ((a, b), (b, a))
        result.append({'orbit': [a, b], 'truth_private': meaning[a]+meaning[b],
                       'pairs': [[alphabet[i], alphabet[j]] for i, j in orientations]})
    return result
