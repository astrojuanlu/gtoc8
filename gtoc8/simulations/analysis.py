# coding: utf-8
import importlib
from glob import glob
import logging

import sys
sys.path.insert(0, "../..")

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u


def main():
    fig = plt.figure()
    ax = fig.add_subplot(111)

    SIMULATION_NUMBER = 2
    conf = importlib.import_module("gtoc8.simulations.conf%d.conf" % SIMULATION_NUMBER)

    _, _, _, t = conf.config()

    lim = np.cos(0.1 * u.deg).value

    files = sorted(glob("conf%d/*.txt" % SIMULATION_NUMBER))
    with plt.style.context("pybonacci"):
        for fname in files:
            logging.info(fname)
            data = np.loadtxt(fname)
            ax.plot(t[::2], data[::2], color='black')
            ax.fill_between(t[::2][data > lim], data[::2][data > lim], color='red', alpha=0.1)

    fig.savefig("conf%d/results.png" % SIMULATION_NUMBER)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
