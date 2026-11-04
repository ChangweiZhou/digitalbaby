"""QA2 entity-substitution generator (deterministic). Usage: python mkpara2.py in fam out ; fam in newnames,newobjs,newnames+newobjs"""
import re, sys
NAMES = {'John': 'Peter', 'Mary': 'Alice', 'Sandra': 'Clara', 'Daniel': 'Oscar'}
OBJS = {'football': 'ball', 'milk': 'book', 'apple': 'pen'}
def remap(line, table): return re.sub(r'\b(' + '|'.join(table) + r')\b', lambda m: table[m.group(1)], line)
fin, fam, fout = sys.argv[1:4]
with open(fin) as f, open(fout, 'w') as g:
    for line in f:
        for part in fam.split('+'):
            line = remap(line, NAMES if part == 'newnames' else OBJS)
        g.write(line)
