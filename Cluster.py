#!/usr/bin/env python

import os
import sys
import pdb

import numpy as np
from scipy.interpolate import interp1d
from scipy.constants import pi
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from astropy.constants import h, k_B, c, G
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM as LCDM
from astropy.coordinates import SkyCoord
from astropy.table import Table
from uncertainties import ufloat

# from BCES import run_MC_BCES
from potential import Hogan


class Cluster(object):

    def __init__(self, z, infile):
        self.z = z
        self.profiles = Table.read(infile, format='csv')

    @classmethod
    def from_files(cls, z, infile, centroid=None, mass_file=None, potential=None):
        c = cls(z, infile)
        if centroid is not None:
            c.centroid_from_file(centroid)
        if mass_file is not None:
            c.potential = Hogan.from_file(mass_file)
            # c.set_mass_profile(mass_file)
        if potential is not None:
            if mass_file is not None:
                raise Warning("Potential is overwriting the mass profile")
            c.potential = potential
        return c

    @property
    def centroid(self):
        return self._centroid

    @centroid.setter
    def centroid(self, coord):
        self._centroid = coord

    # def centroid(self, ra, dec, unit=(u.hourangle, u.deg)):
    #     self._centroid = SkyCoord(ra, dec, unit=unit)

    def centroid_from_file(self, centroid_file):
        with open(centroid_file) as f:
            line = f.readlines()[-1]
        trim = line.split(")")[0].split("(")[1]
        toks = trim.split(",")
        ra, dec = toks[:2]
        self.centroid = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))

    def interpolate(self, key, value, xkey='R', return_error=False):
        xx = self.profiles[xkey]
        yy = self.profiles[key]
        if isinstance(xx, u.Quantity):
            xx = xx.value
            xu = xx.unit
        if isinstance(yy, u.Quantity):
            yy = yy.value
            yu = yy.unit
        if isinstance(value, u.Quantity):
            try:
                value = value.to(xu).value
            except NameError:
                value = value.value
        if value < xx[0]:
            print("Warning: Attempting to extrapolate.")
            print("Returning innermost data point.")
            return yy[0]

        x = np.log10(xx)
        y = np.log10(yy)
        interp_fxn = interp1d(x, y, kind='linear')
        log_yp = interp_fxn(np.log10(value))
        try:
            yp = 10**log_yp * yu
        except NameError:
            yp = 10**log_yp

        if return_error:
            # Find one or two closest points along x-axis
            # Average the fractional uncertainties
            left_ind = np.argwhere(xx <= value).max()
            right_ind = np.argwhere(xx > value).min()
            left_unc_p = self.profiles[f"{key}_p"][left_ind] / self.profiles[key][left_ind]
            left_unc_m = self.profiles[f"{key}_m"][left_ind] / self.profiles[key][left_ind]
            right_unc_p = self.profiles[f"{key}_p"][right_ind] / self.profiles[key][right_ind]
            right_unc_m = self.profiles[f"{key}_m"][right_ind] / self.profiles[key][right_ind]
            unc_p = yp * np.mean([left_unc_p, right_unc_p])
            unc_m = yp * np.mean([left_unc_m, right_unc_m])
            return (yp, unc_p, unc_m)
        return yp

    def fit_powerlaw(self, key):
        raise NotImplementedError

    def plot_profile(self, key, Rkey="R", xlims=None, ylims=None, 
                     outfile=None, ax=None, **mpl_kwargs):
        ylabels = dict(density = r'Density (cm$^{-3}$)',
                       kT = r'Temperature (keV)',
                       Z = r'Abundance (Z$_{\odot}$)',
                       pressure = r'Pressure (erg cm$^{-3}$)',
                       entropy = r'Entropy (kev cm$^{2}$)',
                       Lx = r'L$_X$ (erg s$^{-1}$)',
                       tcool = r'$t_{\rm cool}$ (yr)',
                       M = r'Gas Mass ($M_{\odot}$)')

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7,6), constrained_layout=True)
            ax = plt.gca()

        xerr = self.profiles[f'{Rkey}_pm']
        yerr = (abs(self.profiles[key+'_m']), self.profiles[key+'_p'])
        ax.errorbar(self.profiles[Rkey], self.profiles[key], yerr, xerr, 
                    **mpl_kwargs)

        ax.set_xlabel(r'R (kpc)', fontsize=16)
        ax.set_ylabel(ylabels[key], fontsize=16)

        if xlims is not None:
            ax.set_xlim(xlims)
        else:
            ax.set_xlim(xmin=1)
        if ylims is not None:
            ax.set_ylim(ylims)

        ax.set_xscale('log')
        if key in ['kT', 'Z']:
            ax.set_yscale('linear')
        else:
            ax.set_yscale('log')

        ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
        ax.tick_params(axis='both', which='major', labelsize=12)

        if outfile is not None:
            # plt.tight_layout()
            # plt.axes().set_aspect('equal')
            plt.savefig(outfile, facecolor='white', transparent=True)
            plt.clf()
