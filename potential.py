#!/usr/bin/env python

import numpy as np
from numpy import pi
from astropy import units as u
from astropy.constants import h, c, G, k_B

_u_kms = u.km/u.s

class Potential(object):
    def g(self, r):
        r = u.Quantity(r, u.kpc)
        _g = G*self.M(r)/(r*r)
        return _g.cgs

    def vc(self, r):
        r = u.Quantity(r, u.kpc)
        vc = np.sqrt(G*self.M(r)/r)
        return vc.to(u.km/u.s)

    def freefall(self, r):
        r = u.Quantity(r, u.kpc)
        tff = np.sqrt(2*r/self.g(r))
        return tff.to(u.yr)


class NFW(Potential):
    def __init__(self, Mh, Rh, ch, delta=500):
        self.Mh = u.Quantity(Mh, u.Msun)
        self.Rh = u.Quantity(Rh, u.kpc)
        self.c = ch
        self.delta = delta

    def M(self, r):
        def _nfw_term(rratio):
            return np.log(1+rratio) - rratio/(1+rratio)

        r = u.Quantity(r, u.kpc)
        rrs = self.c * r/self.Rh
        return self.Mh * _nfw_term(rrs)/_nfw_term(self.c)

class Hernquist(Potential):
    def __init__(self, Mh, a):
        self.Mh = u.Quantity(Mh, u.Msun)
        self.a = u.Quantity(a, u.kpc)

    def M(self, r):
        r = u.Quantity(r, u.kpc)
        return (self.Mh * r*r/(r+self.a)**2).to(u.Msun)


class Isothermal(Potential):
    def __init__(self, Mh, Rh):
        self.Mh = u.Quantity(Mh, u.Msun)
        self.Rh = u.Quantity(Rh, u.kpc)

    @property
    def sigma(self):
        self._sigma = np.sqrt(G*self.Mh/(2*self.Rh)).to(_u_kms)
        return self._sigma

    @sigma.setter
    def sigma(self, sig):
        self._sigma = u.Quantity(sig, _u_kms)
        # self.Rh = 10*u.kpc
        self.Mh = (2*self._sigma**2*self.Rh/G).to(u.Msun)

    def M(self, r):
        # Should work out so that g = 2 sigma^2/R
        # Thus GM/R^2 = 2 sigma^2/R
        # M = 2 sigma^2 R/G
        # Equivalently, sigma = sqrt(GM/2R)
        return (2*self.sigma**2*r/G).to(u.Msun)

class Hogan(Potential):
    def __init__(self, sigma, rs, nfwpot):
        self.sigma = u.Quantity(sigma, u.km/u.s)
        self.rs = u.Quantity(rs, u.kpc)
        self.nfwpot = u.Quantity(nfwpot, u.keV)
        self.rho = (self.nfwpot / (4*pi*G*0.62*u.M_p*self.rs**2)).to(u.Msun/u.kpc**3)

    def __call__(self, r):
        r = u.Quantity(r, u.kpc)
        Piso = 2 * self.sigma**2 * np.log(r/(1*u.kpc))
        Pnfw = -4*pi*G*self.rho*self.rs**2 * np.log(1+r/self.rs)/(r/self.rs)
        return (Piso+Pnfw).cgs

    @classmethod
    def from_file(cls, mass_file):
        with open(mass_file, 'r') as mf:
            lines = mf.readlines()
        sigma = float(lines[0].split()[1]) * u.km/u.s
        rs = float(lines[1].split()[1]) * u.kpc
        nfwpot = float(lines[2].split()[1]) * u.keV
        return cls(sigma, rs, nfwpot)

    def M(self, r):
        r = u.Quantity(r, u.kpc)
        Miso = 2*self.sigma**2/G * r
        Mnfw = 4*pi*self.rho*self.rs**3 * (np.log(1.+r/self.rs) \
                   -r/(r+self.rs))
        return (Miso+Mnfw).to(u.Msun)
