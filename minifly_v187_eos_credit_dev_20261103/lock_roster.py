"""Select fresh documents by fixed source order and train-only eligibility."""
import collections
import hashlib
import itertools
import json
from pathlib import Path
import re
import unicodedata

from qualify import DecaySensor, PARENT, np

def sha(data):
    return hashlib.sha256(data).hexdigest()

def normalized(record):
    return ' '.join(unicodedata.normalize('NFKC', record.decode('utf-8', 'replace')).casefold().split())

# Reuse the exact V181 correction construction and qualification census.
source = (PARENT / 'data.py').read_text()
start = source.index('  train=splits')
end = source.index('  out[w]', start)
body = ''.join(line[1:] + '\n' for line in source[start:end].splitlines())
exec(source[source.index('def corrected'):source.index('def prepare')])
exec('def correction(splits, sensor, w):\n a={}\n' + body + ' return a["correction"]\n')

ORDER = (1400, 730, 98, 768, 174, 46, 120)

def load_document(identifier):
    raw = Path(f'local/text/{identifier}.txt').read_bytes()
    lines = raw.splitlines()
    start = next(i for i,l in enumerate(lines) if l.startswith(b'*** START OF'))
    end = next(i for i,l in enumerate(lines) if l.startswith(b'*** END OF'))
    records = [line[i:i+127] for line in lines[start+1:end] if line.strip()
               for i in range(0,len(line),127)]
    return sha(raw), records

def select():
    documents = {i:load_document(i) for i in ORDER}
    sensor = DecaySensor()
    used, worlds, rejected = set(), {}, []
    for w in ('V187-F1','V187-F2'):
        cached_corrections = {}
        for a,b in itertools.permutations([i for i in ORDER if i not in used],2):
            splits = {}
            seen = set()
            for role,i,n in [('A',a,160),('B',b,48)]:
                records = []
                for r in documents[i][1]:
                    key = normalized(r)
                    if key and key not in seen:
                        seen.add(key)
                        records.append(r)
                splits[role+'_train']=records[:n]
                splits[role+'_calib']=records[n:n+32]
                splits[role+'_held']=records[-64:]
            try:
                if a not in cached_corrections:
                    meta = correction(splits,sensor,w)
                    cached_corrections[a]=(meta,splits['EDIT_train'],splits['NOEDIT_train'])
                else:
                    meta,splits['EDIT_train'],splits['NOEDIT_train']=cached_corrections[a]
            except AssertionError as error:
                rejected.append([w,a,b,str(error)])
                continue
            if not meta['qualified']:
                rejected.append([w,a,b,meta])
                continue
            worlds[w] = (a,b,splits,meta)
            used.update((a,b))
            break
    return documents,worlds,rejected

if __name__=='__main__':
    documents,worlds,rejected=select()
    manifest = {'candidate_order':ORDER,'rejected':rejected,'worlds':{}}
    for w,(a,b,splits,meta) in worlds.items():
        manifest['worlds'][w]={'sources':{str(i):{'url':f'https://www.gutenberg.org/cache/epub/{i}/pg{i}.txt','sha256':documents[i][0]} for i in (a,b)},
          'correction':meta,'splits':{k:[sha(r) for r in rs] for k,rs in splits.items()}}
    manifest['status']='LOCKED' if len(worlds)==2 else 'ROSTER_INELIGIBLE'
    Path('FRESH_ROSTER.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(manifest['status'],[(w,a,b) for w,(a,b,_,_) in worlds.items()],len(rejected))
