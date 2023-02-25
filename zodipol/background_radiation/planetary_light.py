import numpy as np
import healpy as hp
import astropy.units as u

from astropy.coordinates import CartesianRepresentation, SphericalRepresentation
from astropy.time import Time
from astropy.modeling.models import BlackBody
from astropy.coordinates import get_body


sun_temperature = 5778 * u.K
sun_radius = 696340 * u.km


class PlanetaryLight:
    def __init__(self):
        self.planet_radius = {
            'mercury': 2439.7 * u.km,
            'venus': 6051.8 * u.km,
            'mars': 3396.2 * u.km,
            'jupiter': 71492 * u.km,
            'saturn': 60268 * u.km,
            'uranus': 25559 * u.km,
            'neptune': 24764 * u.km
        }
        self.planet_bond_albedo = {
            'mercury': 0.088,
            'venus': 0.76,
            'mars': 0.25,
            'jupiter': 0.503,
            'saturn': 0.342,
            'uranus': 0.30,
            'neptune': 0.29
        }

    @staticmethod
    def _validate_time(time: str | Time):
        if not isinstance(time, Time):
            time = Time(time)
        return time

    def get_ang(self, time, theta, phi, wavelength, nside=64):
        """
        Get the integrated starlight flux at a given angle
        :param theta: theta angle
        :param phi: phi angle
        :return: integrated starlight flux
        """
        planets_map = self.make_planets_map(nside, time, wavelength)
        return planets_map[hp.ang2pix(nside, theta, phi), ...]

    def make_planets_map(self, nside: int, time: str | Time, wavelength):
        planet_flux = self.get_all_planet_flux(time, wavelength)
        sky_map = np.zeros((hp.nside2npix(nside), len(wavelength),)) * u.Unit('W / m^2 Hz sr')
        for planet in self._get_planet_names():
            planet_location = self._get_planet_location(planet, time)
            planet_location = planet_location.represent_as(SphericalRepresentation)
            lon = planet_location.lon.value
            lat = planet_location.lat.value + np.pi/2
            pixel = hp.ang2pix(nside, lat, lon)
            sky_map[pixel, :] += planet_flux[planet]
        return sky_map

    def get_all_planet_flux(self, time: str | Time, wavelength):
        """
        Get the flux of all planets at a given time and wavelength
        :param time: time of observation
        :param wavelength: wavelength of observation
        :return:
        """
        time = self._validate_time(time)
        planet_names = self._get_planet_names()
        planet_flux = {planet_name: self.get_planet_flux(planet_name, time, wavelength) for planet_name in planet_names}
        return planet_flux

    def get_planet_flux(self, planet_name: str, time: str | Time, wavelength):
        """
           Get the flux of a planet at a given time and wavelength
        :param planet_name: name of the planet
        :param time: time of observation
        :param wavelength: wavelength of observation
        :return:
        """
        planet_radius = self.planet_radius[planet_name]
        planet_bond_albedo = self.planet_bond_albedo[planet_name]

        planet_distance_from_sun = self._get_planet_distance_from_sun(planet_name, time)
        solar_flux = self._get_solar_flux_density_at_distance(planet_distance_from_sun, wavelength)
        planet_scattering_angle = self._get_planet_scattering_angle(planet_name, time)

        angle_function = self._lambert_angular_function(planet_scattering_angle)
        gamma = 2 * planet_bond_albedo / (3 * np.pi) * planet_radius ** 2 * solar_flux
        lambertian_scattering = gamma * angle_function
        return lambertian_scattering.to(u.Unit('W / m^2 Hz sr'))

    @staticmethod
    def _get_planet_names():
        return ['mercury', 'venus', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune']

    def _get_planet_location(self, planet_name: str, time: str | Time):
        time = self._validate_time(time)
        if planet_name not in self._get_planet_names():
            raise ValueError(f'planet name {planet_name} is not a valid planet name')
        return get_body(body=planet_name, time=time)

    def _get_planet_distance_from_sun(self, planet_name: str, time: str | Time):
        time = self._validate_time(time)
        planet = self._get_planet_location(planet_name, time)
        sun = get_body('sun', time)
        return sun.separation_3d(planet)

    def _get_planet_scattering_angle(self, planet_name: str, time: str | Time):
        time = self._validate_time(time)
        planet = self._get_planet_location(planet_name, time)
        sun = get_body('sun', time)
        planet_cart, sun_cart = planet.represent_as(CartesianRepresentation), sun.represent_as(CartesianRepresentation)
        planet_to_sun, planet_to_earth = (sun_cart-planet_cart).xyz, (-planet_cart).xyz
        vec_multi = (planet_to_sun @ planet_to_earth) / np.linalg.norm(planet_to_sun) / np.linalg.norm(planet_to_earth)
        return np.arccos(vec_multi)

    @staticmethod
    def _get_solar_flux_density_at_distance(distance, wavelength):
        bb = BlackBody(temperature=sun_temperature)
        return bb(wavelength) / (4 * np.pi * distance ** 2)

    @staticmethod
    def _lambert_angular_function(angle):
        angle_func = np.sin(angle) + (np.pi - angle.value) * np.cos(angle)
        return angle_func


if __name__ == '__main__':
    pl = PlanetaryLight()
    sky_map = pl.make_planets_map(32, '2022-01-01', [0.3, 0.4, 0.5, 0.6, 0.7] * u.um)

    import matplotlib.pyplot as plt
    hp.mollview(sky_map[:, 0])
    plt.show()
