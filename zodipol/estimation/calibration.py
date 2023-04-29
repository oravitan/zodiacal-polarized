"""
This module contains the Calibration class, which is used to calibrate
zodipol created images.
"""
import astropy.units as u

from zodipol.estimation.base_calibration import BaseCalibration
from zodipol.zodipol import Zodipol
from zodipol.utils.argparser import ArgParser


class Calibration(BaseCalibration):
    """
    This class is used to calibrate zodipol created images.
    """
    def __init__(self, obs: list, zodipol: Zodipol, parser: ArgParser):
        super().__init__(zodipol, parser)
        self.obs = obs

    def _calibrate_itr(self, images_orig: u.Quantity, mode="all", **kwargs) -> None:
        self.estimate_polarizance(images_orig, **kwargs)
        self.estimate_birefringence(images_orig, **kwargs)
