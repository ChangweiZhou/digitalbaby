"""Exactly four predeclared DEV worlds; no screen, confirmation or automatic extension."""
import json
import os
from pathlib import Path

from runner import run_world
from audit import audit_receipt


def main():
    directory = Path(__file__).parent / 'cycles/cycle3'
    for world in (810201, 810202, 810203, 810204):
        path = directory / f'WORLD_{world}.json'
        if path.exists():
            raise ValueError('Refusing to repeat a committed DEV world')
        result = run_world(world)
        audit = audit_receipt(result)
        temp = path.with_suffix('.json.tmp')
        with temp.open('w') as f:
            json.dump(result, f, sort_keys=True, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
        (directory / f'AUDIT_{world}.json').write_text(json.dumps(audit, indent=2))
        print(json.dumps(dict(world=world, audit=audit, resources=result['resources'],
            old=result['probes']['old']['W']['old']['metrics']['focal_accuracy'],
            retained=result['probes']['day2']['W']['old']['metrics']['focal_accuracy'],
            revised=result['probes']['revised']['W']['revised']['metrics']['focal_accuracy'],
            unchanged=result['probes']['revised']['W']['unchanged']['metrics']['focal_accuracy'])), flush=True)


if __name__ == '__main__':
    main()
