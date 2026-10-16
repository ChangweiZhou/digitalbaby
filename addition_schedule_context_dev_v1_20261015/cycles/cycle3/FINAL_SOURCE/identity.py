"""Small operative-source identity; retained asset gate remains the native release's."""
import hashlib
from pathlib import Path
import bootstrap
FILES=('bootstrap.py','artifacts.py','config.py','models.py','data.py','engine.py','checkpoint.py','identity.py','job.py','probes.py','audits.py','diagnostics.py','analysis.py','OUTPUT_CODEBOOK.json')
PARENTS=('native_interface_transfer_dev_v1_20261015/models.py','addition_core_lineage_comparison_dev_v1_20261015/core.py','arithmetic_gated_feedback_dev_v2_20261014/core.py','r_center_core_v1/SOURCE_LOCK.json')

def source_id():
    h=hashlib.sha256()
    for base,names in ((bootstrap.HERE,FILES),(bootstrap.ROOT,PARENTS)):
        for name in names:
            h.update(name.encode());h.update((base/name).read_bytes())
    return h.hexdigest()
