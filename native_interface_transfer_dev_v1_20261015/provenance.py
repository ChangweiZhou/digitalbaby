"""Source identity for this development attempt; no birth or trajectory execution."""
import sys
import platform
from pathlib import Path
import bootstrap
from artifacts import write,read,digest,now


def build():
    import models
    import scipy,numba
    from centered_core.integrity import verify_sources
    verify_sources()
    files=set(bootstrap.HERE.glob('*.py'))|set(bootstrap.HERE.glob('*.md'))|set(bootstrap.HERE.glob('SPEC.json'))|set(bootstrap.HERE.glob('MODEL_MANIFEST.json'))|set(bootstrap.HERE.glob('OUTPUT_CODEBOOK.json'))|set((bootstrap.HERE/'tests').glob('*.py'))
    files.add(bootstrap.HERE/'ops/AUTHORIZATION.json')
    for module in list(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name:
            path=Path(name).resolve()
            if path.is_relative_to(bootstrap.ROOT) and path.is_file():files.add(path)
    # The released native lock anchors native source AND canonical asset identity.
    files.add(bootstrap.ROOT/'r_center_core_v1/SOURCE_LOCK.json')
    return dict(schema='NATIVE_INTERFACE_TRANSFER_SOURCE_LOCK_V1',utc=now(),
                files={str(p.relative_to(bootstrap.ROOT)):digest(p) for p in sorted(files)},
                environment=dict(python=sys.version,numpy=models.np.__version__,scipy=scipy.__version__,numba=numba.__version__,torch=models.torch.__version__,machine=platform.machine()),
                parent_codebook_identical=digest(bootstrap.HERE/'OUTPUT_CODEBOOK.json')==digest(bootstrap.ROOT/'addition_core_lineage_comparison_dev_v1_20261015/OUTPUT_CODEBOOK.json'),
                native_asset_gate='Released centered_core.integrity.verify_sources; canonical birth B checked independently per store')


def verify():
    lock=read(bootstrap.HERE/'SOURCE_LOCK.json')
    for name,expected in lock['files'].items():
        if digest(bootstrap.ROOT/name)!=expected:raise AssertionError('source changed: '+name)
    from centered_core.integrity import verify_sources
    verify_sources()
    return lock


if __name__=='__main__':
    path=bootstrap.HERE/'SOURCE_LOCK.json'
    if path.exists():raise FileExistsError('source already locked; no overwrite')
    write(path,build())
