"""Deterministic paraphrase generator for bAbI QA1-style files.
Rewrites only statement lines '<Name> <verb phrase> the <loc>.' ; questions/answers untouched.
Usage: python mkpara.py <in_file> <family> <out_file> [seed]
Families: verbs, entered, noarticle, isnow (round-12 dev families)
          located, hasgone, madeit, reversed (round-12 held-out families; evaluated once)
          newnames, newlocs (entity substitution; combinable with '+', e.g. newnames+isnow)
          qfound (stress probe: statement 'X can be found in the Y.' + question 'Where can X be found?')
"""
import re, sys, random
def rewrite(line, fam, rng):
    if fam == 'qfound' and '\t' in line:   # stress probe: question template shares words with the statement template
        return re.sub(r'Where is (\w+)\? ', r'Where can \1 be found? ', line)
    m = re.match(r'^(\d+) (\w+) (.*) the (\w+)\.$', line)
    if not m or '\t' in line: return line
    n, name, vp, loc = m.groups()
    if fam == 'verbs':     s = f'{name} {rng.choice(["ran","strolled","headed","wandered"])} to the {loc}.'
    elif fam == 'entered': s = f'{name} entered the {loc}.'
    elif fam == 'noarticle': s = f'{name} ran to {loc}.'
    elif fam == 'isnow':   s = f'{name} is now in the {loc}.'
    elif fam == 'located': s = f'{name} is located in the {loc}.'
    elif fam == 'hasgone': s = f'{name} has gone to the {loc}.'
    elif fam == 'madeit':  s = f'{name} made it to the {loc}.'
    elif fam == 'reversed': s = f'The {loc} is where {name} went.'
    elif fam == 'qfound':  s = f'{name} can be found in the {loc}.'
    else: raise SystemExit('unknown family ' + fam)
    return f'{n} {s}'
NAMES = {'John': 'Peter', 'Mary': 'Alice', 'Sandra': 'Clara', 'Daniel': 'Oscar'}
LOCS = {'hallway': 'garage', 'bathroom': 'attic', 'bedroom': 'cellar', 'kitchen': 'library', 'garden': 'porch', 'office': 'studio'}
def remap(line, table):   # whole-line word substitution (statements, questions and answers)
    return re.sub(r'\b(' + '|'.join(table) + r')\b', lambda m: table[m.group(1)], line)
if __name__ == '__main__':
    fin, fam, fout = sys.argv[1:4]; seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    rng = random.Random(seed)
    with open(fin) as f, open(fout, 'w') as g:
        for line in f:
            line = line.rstrip('\n')
            for part in fam.split('+'):   # e.g. 'newnames+isnow'
                if part == 'newnames': line = remap(line, NAMES)
                elif part == 'newlocs': line = remap(line, LOCS)
                else: line = rewrite(line, part, rng)
            g.write(line + '\n')
