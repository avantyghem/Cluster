#!/usr/bin/env python

from __future__ import print_function

import os
import sys
import pdb

import numpy as np
from scipy.interpolate import interp1d
from scipy.constants import pi
from astropy.constants import h, k_B, c, G
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM as LCDM
from astropy.coordinates import SkyCoord
from astropy.table import Table
from uncertainties import ufloat

from Cluster import Cluster
from potential import Hernquist, NFW


class Cavity(object):
    '''
    # Region file format: CIAO version 1.0
    ellipse(8:21:02.055,+7:51:48.14,0.0387342',0.0215365',6.10166)
    '''
    def __init__(self, str):
        trim = str.split('(')[1].split(')')[0]
        toks = trim.split(',')
        ra, dec = toks[:2]
        self.coords = SkyCoord(ra, dec, frame='icrs', unit=(u.hourangle, u.deg))

        aunit = toks[2][-1]
        if aunit == "\"":
            unit = u.arcsec
        elif aunit == "'":
            unit = u.arcmin

        aa = float(toks[2][:-1]) * unit
        bb = float(toks[3][:-1]) * unit
        self.pa = float(toks[4])
            
        if aa >= bb:
            self.a_asec = aa
            self.b_asec = bb
        else:
            self.a_asec = bb
            self.b_asec = aa
            self.pa -= 90.

    @classmethod
    def from_file(cls, regfile):
        with open(regfile, 'r') as f:
            cavs = [cls(line.rstrip()) for line in f if line.startswith('ellipse')
                                                     or line.startswith('circle')]
        if len(cavs) == 1:
            return cavs[0]
        else:
            return cavs

    # @property
    # def volume(self):
    #     return ((4*pi/3) * (self.a*self.b)**1.5).to(u.cm**3)
        

    def do_calculations(self, cluster):
        cosmo = LCDM(70, 0.3)
        self.asec_per_kpc = cosmo.arcsec_per_kpc_proper(cluster.z)
        self.a = (self.a_asec / self.asec_per_kpc).to(u.kpc)
        self.b = (self.b_asec / self.asec_per_kpc).to(u.kpc)

        self.volume = ((4*pi/3) * (self.a*self.b)**1.5).to(u.cm**3)
        self.area = (pi*self.a*self.b).to(u.cm**2)

        R = cluster.centroid.separation(self.coords)
        self.R = (R/self.asec_per_kpc).to(u.kpc)

        p = cluster.interpolate('pressure', self.R) * u.erg/u.cm**3
        self.cavity_pressure = p
        ne = cluster.interpolate('density', self.R) / u.cm**3
        kT = cluster.interpolate('kT', self.R) * u.keV
        T = kT.to(u.K, equivalencies=u.temperature_energy())

        self.pV = p * self.volume
        self.enthalpy = 4*self.pV
        self.sound_speed = np.sqrt(5*k_B*T/(3*0.62*u.M_p)).to(u.km/u.s)
        self.sound_crossing_time = (self.R / self.sound_speed).to(1e8*u.yr)

        try:
            g = cluster.potential.g(self.R)
            self.buoyancy_time = (self.R*np.sqrt(0.75*self.area/(2*g*self.volume))).to(1e8*u.yr)
            self.age = self.buoyancy_time
        except AttributeError:
            self.age = self.sound_crossing_time

        self.Pcav = (self.enthalpy/self.age).to(u.erg/u.s)
        self.Mdisp = (1.+1/1.2) * (ne*self.volume*0.62*u.M_p).to(u.Msun)

    def calculation_summary(self):
        print('Major: {:.2g}'.format(self.a))
        print('Minor: {:.2g}'.format(self.b))
        print('Distance: {:.2g}'.format(self.R))
        print('Pressure: {:.3g}'.format(self.cavity_pressure))
        print('Volume: {:.2g}'.format(self.volume))
        print('Enthalpy: {:.2g}'.format(self.enthalpy))
        print('Sound Crossing Time: {:.3g}'.format(self.sound_crossing_time.to(u.Myr)))
        if hasattr(self, 'buoyancy_time'):
            print('Buoyancy Time: {:.3g}'.format(self.buoyancy_time.to(u.Myr)))
        print('Pcav: {:.2g}'.format(self.Pcav))
        print('Displaced Mass: {:.2g}'.format(self.Mdisp))


def extract_cluster_info(cluster, path="/home/adrian/Clusters/", filename="ClusterData.txt"):
    header = ["Cluster", "z", "NH", "Something", "Other", "xpos", "ypos", "r_in"]
    with open(path+filename) as cfile:
        for line in cfile:
            if not line.startswith(cluster):
                continue
            toks = line.split()
            for i in [1, 2, 5, 6, 7]:
                toks[i] = float(toks[i])
            info = dict(zip(header, toks))
    return info

if __name__ == '__main__':

    if len(sys.argv) != 4:
        print('USAGE: {} cavities.reg profiles.csv centroid.reg'.format(sys.argv[0]))
        sys.exit(-1)

    path = os.getcwd()
    cluster = path.lstrip('/home/adrian/Clusters').split('/')[0]
    z = extract_cluster_info(cluster)["z"]

    pot = Hernquist(5e11, 5.2)
    # mf = '/home/adrian/Clusters/'+cluster+'/Chandra/mass_profile.txt'

    cavity = Cavity.from_file(sys.argv[1])
    cluster = Cluster.from_files(z, sys.argv[2], centroid=sys.argv[3], potential=pot)

    cavity.do_calculations(cluster)
    cavity.calculation_summary()

