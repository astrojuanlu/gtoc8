# coding: utf-8
import numpy as np
import pandas as pd


def load_catalogue(fname):
    """Reads radio sources catalogue from filename.

    """
    catalogue_data = np.loadtxt(fname, comments="%")
    catalogue = pd.DataFrame(catalogue_data[:, 1:],
                             index=catalogue_data[:, 0],
                             columns=["RA (deg)", "DEC (deg)"])
    return catalogue