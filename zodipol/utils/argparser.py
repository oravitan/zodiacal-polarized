import argparse
import numpy as np
import astropy.units as u
from astropy.time import Time


class ArgParser:
    def __init__(self, *args, **kwargs):
        self.args = self.get_arguments(*args, **kwargs)
        self._parse_arguments()

    def __getitem__(self, item):
        return self.args.__getattribute__(item)

    def __dict__(self):
        return self.args.__dict__

    def _parse_arguments(self):
        self.args.direction = np.array(self.args.direction, dtype=float) * u.deg
        self.args.obs_time = Time(self.args.obs_time)
        self.args.fov = self.args.fov * u.deg
        self.args.imager_params = {x: eval(y) for x, y in zip(self.args.imager_params[::2], self.args.imager_params[1::2])}
        self.args.polarization_angle = np.linspace(0, np.pi, self.args.n_polarization_ang, endpoint=False)

    @staticmethod
    def get_arguments(*args, **kwargs):
        parser = argparse.ArgumentParser()
        parser.add_argument('--nside', type=int, default=64, help='nside of healpy maps')
        parser.add_argument('--polarizance', '-pol', type=float, default=1., help='Polarizance of imager')
        parser.add_argument('--n_freq', '-nf', type=int, default=20, help='Number of frequencies for imager response')
        parser.add_argument('--n_realizations', '-nr', type=int, default=20, help='Number of realizations')
        parser.add_argument('--n_polarization_ang', '-npa', type=int, default=4, help='Number of polarization angles')
        parser.add_argument('--fov', type=float, default=180., help='Field of view of imager in deg')
        parser.add_argument('--resolution', '-res', type=int, nargs=2, default=(612, 512), help='Resolution of imager')
        parser.add_argument('--direction', '-dir', type=float, nargs=2, default=(90, 0), help='Direction of imager (theta, phi) in deg')
        parser.add_argument('--obs_time', '-t', type=str, default="2022-06-14",  help='Observation time')
        parser.add_argument('--planetry', '-p', action='store_true', help='Include planetry emission')
        parser.add_argument('--isl', '-i', action='store_true', help='Include interstellar emission')
        parser.add_argument('--new_isl', '-ni', action='store_true', help='Create new interstellar emission')
        parser.add_argument('--parallel', '-par', action='store_true', help='Use parallel processing')
        parser.add_argument('--imager_params', '-ip', nargs='*', default=(), help='Imager parameters (alpha, beta)')
        return parser.parse_args(*args, **kwargs)
