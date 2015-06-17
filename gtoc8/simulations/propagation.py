# coding: utf-8
import logging
import importlib
from multiprocessing import Pool

import numpy as np
from astropy import units as u
from poliastro.bodies import Earth
Earth.k = 398600.4329 * u.km**3 / u.s**2  # GTOC 8

import sys
sys.path.insert(0, "../..")
from gtoc8 import lib, io
from gtoc8.integrate import integrate_rk4_numba


def state_to_vector(ss):
    x0, y0, z0 = ss.r.to(u.km).value
    vx0, vy0, vz0 = ss.v.to(u.km / u.s).value
    return np.array([x0, y0, z0, vx0, vy0, vz0])


SIMULATION_NUMBER = 2
conf = importlib.import_module("gtoc8.simulations.conf%d.conf" % SIMULATION_NUMBER)

ss_A, ss_B, ss_C, t = conf.config()

t = t.to(u.s).value

u1 = state_to_vector(ss_A)
u2 = state_to_vector(ss_B)
u3 = state_to_vector(ss_C)

sol1 = integrate_rk4_numba(t, u1)
sol2 = integrate_rk4_numba(t, u2)
sol3 = integrate_rk4_numba(t, u3)

catalogue = io.load_catalogue("../../data/gtoc8_radiosources.txt")

# "the function that you give to map() must be accessible through
# an import of your module"
# http://stackoverflow.com/a/3336182/554319
def task(source_idx):
    logging.info("Comparing radio source %d" % source_idx)
    dist_one = np.zeros_like(t)
    for ii in range(len(t)):
        s1 = sol1[ii, :3]
        s2 = sol2[ii, :3]
        s3 = sol3[ii, :3]
        vec, _ = lib.observation_triangle(s1, s2, s3)
        # for jj in range(len(catalogue)):
        ra, dec = np.deg2rad(catalogue.iloc[source_idx].values)
        dist_one[ii] = lib.distance_radio_source(vec, ra, dec)

    np.savetxt("conf{:d}/data_{:03d}.txt".format(SIMULATION_NUMBER, source_idx), dist_one)

logging.basicConfig(level=logging.INFO)

pool = Pool()
pool.map(task, range(len(catalogue)))
