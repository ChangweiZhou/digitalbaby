"""User-selected bounded addition development core."""
from .core import ByteLearner
DEFAULT_RECIPE="COV-SUPPORT"
CORE_VERSION="v2-support-1"
def create_core(recipe=DEFAULT_RECIPE):
    return ByteLearner(recipe)
