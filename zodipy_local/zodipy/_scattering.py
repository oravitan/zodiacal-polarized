import PyMieScatt as ps
import numpy as np
import matplotlib.pyplot as plt


class MieScatteringFunction:
    """
    Calculate the scattering function of a sphere
    """
    def __init__(self, m, w, d, prc=0.1, n=100):
        """
        :param m: refractive index
        :param w: wavelength in nm
        :param d: particle diameter in nm
        :param prc: percentage of the diameter to vary
        :param n: number of points to calculate
        """
        self._m = m  # refractive index
        self._w = w  # wavelength in nm
        self._d = d  # particle diameter in nm
        self._prc = prc  # percentage of the diameter to vary
        self._n = n  # number of points to calculate
        self._theta, self._SL, self._SR, self._SU = None, None, None, None
        self.set_scattering_func()

    def __call__(self, theta):
        return self.get_scattering_func(theta)

    def set_scattering_func(self):
        theta, SL, SR, SU = self.calculate_average_scattering_function(m, w, d, prc=self._prc, n=self._n)
        self._theta = theta
        self._SL = SL
        self._SR = SR
        self._SU = SU

    def get_scattering_func(self, theta):
        """
        :param theta: scattering angle in degrees
        :return: scattering function (SL, SR, SU)
        """
        SL = np.interp(theta, self._theta, self._SL)
        SR = np.interp(theta, self._theta, self._SR)
        SU = np.interp(theta, self._theta, self._SU)
        return SL, SR, SU

    @property
    def m(self):
        return self._m

    @m.setter
    def m(self, value):
        self._m = value
        self.set_scattering_func()

    @property
    def w(self):
        return self._w

    @w.setter
    def w(self, value):
        self._w = value
        self.set_scattering_func()

    @property
    def d(self):
        return self._d

    @d.setter
    def d(self, value):
        self._d = value
        self.set_scattering_func()

    @staticmethod
    def calculate_average_scattering_function(m, w, d, prc=0.1, n=100):
        """
        calculate the average between the  scattering function of different spheres
        :param m: refractive index
        :param w: wavelength in nm
        :param d: particle diameter in nm
        :param prc: percentage of the diameter to vary
        :param n: number of points to calculate
        :return: scattering function (theta, SL, SR, SU)
        """
        d_range = np.linspace(d * (1 - prc), d * (1 + prc), n)
        theta, SL, SR, SU = tuple(zip(*[ps.ScatteringFunction(m, w, d) for d in d_range]))
        theta, SL, SR, SU = (np.mean(np.array(x), axis=0) for x in (theta, SL, SR, SU))
        return theta, SL, SR, SU


if __name__ == '__main__':
    m = 1.714+0.031j  # refractive index of silicate
    w = 532  # wavelength in nm
    d = 300  # particle diameter in nm
    msf = MieScatteringFunction(m, w, d)

