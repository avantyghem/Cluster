#!/usr/bin/env python

from __future__ import print_function

import os
import sys
import pdb

from math import sqrt as msqrt
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

cosmo = LCDM(70, 0.3)


class Cavity(object):
    """
    # Region file format: CIAO version 1.0
    ellipse(8:21:02.055,+7:51:48.14,0.0387342',0.0215365',6.10166)
    """

    def __init__(self, str):
        trim = str.split("(")[1].split(")")[0]
        toks = trim.split(",")
        ra, dec = toks[:2]
        self.coords = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))

        aunit = toks[2][-1]
        if aunit == '"':
            unit = u.arcsec
        elif aunit == "'":
            unit = u.arcmin

        aa = float(toks[2][:-1])
        bb = float(toks[3][:-1])

        self.a_asec = np.max([aa, bb]) * unit
        self.b_asec = np.min([aa, bb]) * unit
        self.pa = float(toks[4])
        if aa < bb:
            self.pa -= 90

    @classmethod
    def from_file(cls, regfile):
        with open(regfile, "r") as f:
            cavs = [
                cls(line.rstrip())
                for line in f
                if line.startswith("ellipse") or line.startswith("circle")
            ]
        if len(cavs) == 1:
            return cavs[0]
        else:
            return cavs

    def do_calculations(self, cluster, rel_size_unc=0.0):
        self.asec_per_kpc = cosmo.arcsec_per_kpc_proper(cluster.z)
        self.a = (self.a_asec / self.asec_per_kpc).to(u.kpc)
        self.b = (self.b_asec / self.asec_per_kpc).to(u.kpc)
        self.ellip = self.a / self.b

        self.Vmax = ((4 * pi / 3) * (self.a * self.a * self.b)).to(u.cm ** 3)
        self.Vmin = ((4 * pi / 3) * (self.a * self.b * self.b)).to(u.cm ** 3)
        self.volume = np.sqrt(self.Vmax * self.Vmin)  # geometric mean -- (ab)^(3/2)
        self.area = (pi * self.a * self.b).to(u.cm ** 2)

        # Volume uncertainties, take the max of:
        # 1) Geometric effects, or
        # 2) Propagated from axis ratios assuming a certain value
        self.volume_p_frac = max(msqrt(self.ellip) - 1, 3 * rel_size_unc / msqrt(2))
        self.volume_m_frac = max(1 - 1 / msqrt(self.ellip), 3 * rel_size_unc / msqrt(2))
        self.volume_p = self.volume * self.volume_p_frac
        self.volume_m = self.volume * self.volume_m_frac

        R = cluster.centroid.separation(self.coords)
        self.R = (R / self.asec_per_kpc).to(u.kpc)

        p = cluster.interpolate("pressure", self.R, xkey="R_kpc", return_error=True)
        p = np.array(p) * u.erg / u.cm ** 3
        p, p_p, p_m = p
        self.cavity_pressure = p
        self.cavity_pressure_p = p_p
        self.cavity_pressure_m = p_m

        ne = cluster.interpolate("density", self.R, xkey="R_kpc") / u.cm ** 3

        kT = cluster.interpolate("kT", self.R, xkey="R_kpc", return_error=True)
        kT = np.array(kT) * u.keV
        kT, kT_p, kT_m = kT
        T = kT.to(u.K, equivalencies=u.temperature_energy())

        self.pV = p * self.volume
        self.pV_p = self.pV * msqrt((p_p / p) ** 2 + (self.volume_p_frac) ** 2)
        self.pV_m = self.pV * msqrt((p_m / p) ** 2 + (self.volume_m_frac) ** 2)

        self.enthalpy = 4 * self.pV
        self.enthalpy_p = 4 * self.pV_p
        self.enthalpy_m = 4 * self.pV_m

        self.sound_speed = np.sqrt(5 * k_B * T / (3 * 0.62 * u.M_p)).to(u.km / u.s)
        self.sound_crossing_time = (self.R / self.sound_speed).to(1e8 * u.yr)
        self.sound_crossing_time_p = self.sound_crossing_time * 0.5 * (kT_p / kT)
        self.sound_crossing_time_m = self.sound_crossing_time * 0.5 * (kT_m / kT)

        try:
            g = cluster.potential.g(self.R)
            g_interp = cluster.interpolate("g", self.R, xkey="R_kpc", return_error=True)
            _, g_p, g_m = np.array(g_interp) * g.unit
            self.buoyancy_time = (
                self.R * np.sqrt(0.75 * self.area / (2 * g * self.volume))
            ).to(1e8 * u.yr)
            self.buoyancy_time_p = self.buoyancy_time * msqrt(
                (0.5 * g_p / g) ** 2 + 1 / 8 * rel_size_unc ** 2
            )
            self.buoyancy_time_m = self.buoyancy_time * msqrt(
                (0.5 * g_m / g) ** 2 + 1 / 8 * rel_size_unc ** 2
            )
            age_type = "buoyancy_time"
        except AttributeError:
            age_type = "sound_crossing_time"
        self.age = getattr(self, age_type)
        self.age_p = getattr(self, f"{age_type}_p")
        self.age_m = getattr(self, f"{age_type}_m")

        self.Pcav = (self.enthalpy / self.age).to(u.erg / u.s)
        self.Pcav_p = self.Pcav * msqrt(
            (self.pV_p / self.pV) ** 2 + (self.age_p / self.age) ** 2
        )
        self.Pcav_m = self.Pcav * msqrt(
            (self.pV_m / self.pV) ** 2 + (self.age_m / self.age) ** 2
        )

        self.Mdisp = (1.0 + 1 / 1.2) * (ne * self.volume * 0.62 * u.M_p).to(u.Msun)
        self.Mdisp_p = self.Mdisp * self.volume_p_frac
        self.Mdisp_m = self.Mdisp * self.volume_m_frac

        self.Macc = (self.enthalpy / (0.1 * c * c)).to(u.Msun)
        self.Macc_p = self.Macc * self.enthalpy_p / self.enthalpy
        self.Macc_m = self.Macc * self.enthalpy_m / self.enthalpy

        self.Mdot_acc = (self.Pcav / (0.1 * c * c)).to(u.Msun / u.yr)
        self.Mdot_acc_p = self.Mdot_acc * self.Pcav_p / self.Pcav
        self.Mdot_acc_m = self.Mdot_acc * self.Pcav_m / self.Pcav

    def summary(self):
        print("Major: a={:.2g}".format(self.a))
        print("Major: a={:.2g}".format(self.a_asec))
        print("Minor: b={:.2g}".format(self.b))
        print("Minor: b={:.2g}".format(self.b_asec))
        print("Distance: R={:.2g}".format(self.R))
        print("Pressure: p={:.3g}".format(self.cavity_pressure))
        print("Volume: V={:.2g}".format(self.volume))
        print("Enthalpy: 4pV={:.2g}".format(self.enthalpy))
        print("             +{:.2g}".format(self.enthalpy_p))
        print("             -{:.2g}".format(self.enthalpy_m))
        print(
            "Sound Crossing Time: t_cs={:.3g}".format(
                self.sound_crossing_time.to(u.Myr)
            )
        )
        print(
            "                         +{:.3g}".format(
                self.sound_crossing_time_p.to(u.Myr)
            )
        )
        print(
            "                         -{:.3g}".format(
                self.sound_crossing_time_m.to(u.Myr)
            )
        )
        if hasattr(self, "buoyancy_time"):
            print("Buoyancy Time: t_buoy={:.3g}".format(self.buoyancy_time.to(u.Myr)))
            print("                     +{:.3g}".format(self.buoyancy_time_p.to(u.Myr)))
            print("                     -{:.3g}".format(self.buoyancy_time_m.to(u.Myr)))
        print("Pcav: P_cav={:.2g}".format(self.Pcav))
        print("           +{:.2g}".format(self.Pcav_p))
        print("           -{:.2g}".format(self.Pcav_m))
        print("Displaced Mass: {:.2g}".format(self.Mdisp))

    def table_format(
        self,
        columns=[
            "a",
            "b",
            "R",
            "enthalpy",
            "sound_crossing_time",
            "buoyancy_time",
            "Pcav",
            "Macc",
            "Mdot_acc",
            "Mdisp",
        ],
        units=[
            u.kpc,
            u.kpc,
            u.kpc,
            u.Unit(1e58 * u.erg),
            u.Myr,
            u.Myr,
            u.Unit(1e44 * u.erg / u.s),
            u.Unit(1e6 * u.Msun),
            u.Unit(u.Msun / u.yr),
            u.Unit(1e10 * u.Msun),
        ],
    ):
        def format_val(col, unit):
            val = getattr(self, col).to(unit).value
            if hasattr(self, f"{col}_p"):
                perr = getattr(self, f"{col}_p").to(unit).value
                nerr = getattr(self, f"{col}_m").to(unit).value
                if f"{perr:.2g}" == f"{nerr:.2g}":
                    return f"${val:.3g}\pm{perr:.2g}$"
                return f"${val:.3g}^{{+{perr:.2g}}}_{{-{nerr:.2g}}}$"
            return f"${val:.3g}$"

        return "  &  ".join(format_val(col, unit) for col, unit in zip(columns, units))


def extract_cluster_info(
    cluster, path="/home/adrian/Clusters/", filename="ClusterData.txt"
):
    header = ["Cluster", "z", "NH", "Something", "Other", "xpos", "ypos", "r_in"]
    with open(path + filename) as cfile:
        for line in cfile:
            if not line.startswith(cluster):
                continue
            toks = line.split()
            for i in [1, 2, 5, 6, 7]:
                toks[i] = float(toks[i])
            info = dict(zip(header, toks))
    return info


if __name__ == "__main__":

    if len(sys.argv) != 4:
        print("USAGE: {} cavities.reg profiles.csv centroid.reg".format(sys.argv[0]))
        sys.exit(-1)

    path = os.getcwd()
    cluster = path.lstrip("/home/adrian/Clusters").split("/")[0]
    z = extract_cluster_info(cluster)["z"]

    pot = Hernquist(5e11, 5.2)
    # mf = '/home/adrian/Clusters/'+cluster+'/Chandra/mass_profile.txt'

    cavity = Cavity.from_file(sys.argv[1])
    cluster = Cluster.from_files(z, sys.argv[2], centroid=sys.argv[3], potential=pot)

    cavity.do_calculations(cluster)
    cavity.summary()

