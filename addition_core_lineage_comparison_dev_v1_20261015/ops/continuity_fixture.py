"""Cycle-2-to-3 operations qualification; five generic-byte fixtures, no science."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import core
import checkpoint
from io_utils import write,now


def run():
    rows=[]
    for version in core.VERSIONS:
        original=core.create(version,7981)
        original.train(b'\x00\xff?',b'\x80')
        path=core.bootstrap.HERE/'scratch'/f'continuity_{version}.pickle.gz'
        checkpoint.save(path,original)
        restored=checkpoint.load_internal(path)
        assert restored.digest()==original.digest()
        for model in (original,restored):
            model.train(b'\x00\xfe?',b'\x81')
            model.rest(123.)
        assert restored.digest()==original.digest()
        assert restored.generate_batch([b'\x00\xff?'],cap=1)==original.generate_batch([b'\x00\xff?'],cap=1)
        rows.append(dict(version=version,accepted=True,checkpoint_bytes=path.stat().st_size,
                         training_record_applications=3))
        path.unlink()
    result=dict(utc=now(),science=False,accepted=True,fixtures=rows,training_record_applications=15,
                automatic_resume_authorized=False)
    write(core.bootstrap.HERE/'cycles/cycle2/CONTINUITY_QUALIFICATION.json',result)
    return result


if __name__=='__main__':
    print(run(),flush=True)
