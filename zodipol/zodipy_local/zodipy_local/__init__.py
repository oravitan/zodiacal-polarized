import pkg_resources

from zodipy_local import comps, source_params
from zodipy_local._contour import tabulate_density
from zodipy_local.model_registry import model_registry
from zodipy_local.zodipy_local import Zodipy

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except pkg_resources.DistributionNotFound:  # pragma: no cover
    ...

__all__ = (
    "Zodipy",
    "model_registry",
    "comps",
    "source_params",
    "tabulate_density",
)
