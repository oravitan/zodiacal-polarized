import pkg_resources

from zodipol.zodipy_local.zodipy import comps, source_params
from zodipol.zodipy_local.zodipy._contour import tabulate_density
from zodipol.zodipy_local.zodipy.model_registry import model_registry
from zodipol.zodipy_local.zodipy.zodipy import Zodipy

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
