from pathlib import Path
import sys,importlib.util
I=Path(__file__).resolve().parents[2]
sys.path.append(str(I))
_spec=importlib.util.spec_from_file_location('_frozen_native_bind_bootstrap',I/'bootstrap.py')
_base=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_base)
HERE=_base.HERE;ROOT=_base.ROOT;source_module=_base.source_module
