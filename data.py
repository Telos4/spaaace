import numpy as np
class PlanetData:
    def __init__(self, orbit='LEO'):
        self.AE = 149597870700.0
        self.mars_trajectory_radius = 227940000000.0

        # mass of sun and planets
        self.m_S = 1.99e30 # sun
        self.m_E = 5.97e24 # earth
        self.m_M = 6.39e23 # mars

        # gravitational constant
        self.G = 6.67408e-11

        self.radius_earth = 6.371e6  # mean radius of earth
        self.earth_orbital_speed = 29.78e3  # mean orbital speed of earth around sun

        self.mars_orbital_speed = 24.14e3

        if orbit == 'GEO':
            self.initial_height = 35786e3  # geostationary orbit (GEO)
        elif orbit == 'LEO':
            self.initial_height = 200e3  # LEO

        self.orbital_speed= np.sqrt(self.G*self.m_E/self.initial_height)

    def sun_dist(self, x):
        return np.sqrt(x[0]**2 + x[1]**2)
