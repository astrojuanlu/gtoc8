# coding: utf-8
"""Objective function related code.

Author: Juan Luis Cano Rodríguez <juanlu001@gmail.com>

Examples
--------
>>> import numpy as np
>>> data = np.array([[0.0, 0.0], [180, 0.0]])
>>> problem = GTOC8(data)
>>> problem.register_observation(180, 0.0, [10])
Observation(P=1, h=10, dec=0.0)
>>> problem.performance_index()
12.0

"""
from __future__ import division

from collections import namedtuple

import numpy as np


class GTOC8(object):
    """Class to store all information related to GTOC 8 problem.

    """
    def __init__(self, radio_sources_data):
        """Constructor.

        """
        self._observations = {tuple(row): [] for row in radio_sources_data}

    def register_observation(self, ra, dec, altitudes):
        """Registers a new observation.

        Raises
        ------
        KeyError
            If the observed source was not in the original data catalogue.

        """
        past_observations = self._observations[(ra, dec)]
        new_h = min(altitudes)

        P = compute_P(new_h, past_observations)

        new_observation = Observation(P, new_h, dec)
        past_observations.append(new_observation)
        return new_observation

    def performance_index(self):
        """Computes performance index so far.

        """
        performance_index = 0.0
        for source, observations in self._observations.items():
            for obs in observations:
                performance_index += compute_delta_J(obs.P, obs.h, obs.dec)

        return performance_index


def compute_P(new_h, past_observations):
    """Computes weighting factor.

    Parameters
    ----------
    new_h : float
        Altitude of current observation.
    past_observations : list
        List of past ``Observation`` objects of the same source.

    """
    if len(past_observations) == 0:
        P = 1

    elif len(past_observations) == 1:
        first_observation = past_observations[0]
        hmin, hmax = sorted([first_observation.h, new_h])
        P = 3 if hmax / hmin >= 3 else 1

    elif len(past_observations) == 2:
        first_observation, second_observation = past_observations
        hmin, hmid, hmax = sorted([first_observation.h,
                                   second_observation.h,
                                   new_h])
        if hmax / hmid >= 3 and hmid / hmin >= 3:
            P = 6
        elif hmax / hmin >= 3 and second_observation.P == 1:
            P = 3
        else:
            P = 1

    else:
        P = 0

    return P


def compute_delta_J(P, h, dec):
    """Increment of performance index.

    Parameters
    ----------
    P : int
        Weighting factor.
    h : float
        Altitude (kilometers).
    dec : float
        Declination (degrees)

    Returns
    -------
    float
        Increment of performance index (kilometers).

    """
    return P * h * (0.2 + np.cos(np.deg2rad(dec))**2)

Observation = namedtuple("Observation", ["P", "h", "dec"])
