# coding: utf-8
import numpy as np
from numpy.testing import assert_array_equal

from gtoc8.lib import (observation_triangle, rank_radio_sources)


def test_observation_triangle():
    x1 = np.array([0, 0, 0])
    x2 = np.array([1, 0, 0])
    x3 = np.array([0, 1, 0])
    expected_vector = np.array([0, 0, 1])
    expected_altitudes = [1 / np.sqrt(2), 1.0, 1.0]

    vector, altitudes = observation_triangle(x1, x2, x3)

    assert_array_equal(vector, expected_vector)
    assert_array_equal(altitudes, expected_altitudes)


def test_rank_radio_sources():
    test_data = np.deg2rad(np.array([
        [0, 90],
        [45, 85],
        [90, 80],
        [0, 10],
        [180, 0],
        [210, -45],
        [300, -90]
    ]))
    expected_result = np.deg2rad(np.array([  # Sorts are stable
        [0, 90],
        [300, -90],
        [45, 85],
        [90, 80],
        [210, -45],
        [0, 10],
        [180, 0]
    ]))
    observation_vector = np.array([0, 0, 1])

    result = rank_radio_sources(observation_vector, test_data)

    assert_array_equal(result, expected_result)
