# coding: utf-8
from __future__ import division

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal

from gtoc8.problem import GTOC8, compute_delta_J, compute_P, Observation


test_data = {
    # Source,   h, P,        delta_J
    "S1": [
        ( 10000.0, 1,  10000.0 * 1.2),
        ( 30000.0, 3,  90000.0 * 1.2),
        ( 95000.0, 6, 570000.0 * 1.2)
    ],
    "S2": [
        (100000.0, 1, 100000.0 * 1.2),
        ( 30000.0, 3,  90000.0 * 1.2),
        ( 10000.0, 6,  60000.0 * 1.2)
    ],
    "S3": [
        (100000.0, 1, 100000.0 * 1.2),
        ( 10000.0, 3,  30000.0 * 1.2),
        ( 40000.0, 1,  40000.0 * 1.2)   # FAIL
    ],
    "S4": [
        ( 10000.0, 1,  10000.0 * 1.2),
        ( 20000.0, 1,  20000.0 * 1.2),  # FAIL
        ( 30000.0, 3,  90000.0 * 1.2)   # ?
    ],
    "S5": [
        (100000.0, 1, 100000.0 * 1.2),
        ( 30000.0, 1,  90000.0 * 1.2),
        ( 10000.0, 1,  60000.0 * 1.2)
    ]
}


def test_simple_performance_index_increment():
    P = 1
    h = 10
    dec = 0
    expected_delta_J = 12.0

    delta_J = compute_delta_J(P, h, dec)

    assert_equal(delta_J, expected_delta_J)


def test_weighting_factor_computation():
    past_observations = [
        Observation(1, 100000.0, 0.0),
        Observation(3,  10000.0, 0.0)
    ]
    expected_P = 1

    P = compute_P(40000, past_observations)

    assert_equal(P, expected_P)


def test_problem_sample_data():
    # Sorted keys
    keys = sorted(test_data)

    # Initialize different right ascension for each source
    data = np.zeros((len(test_data), 2), dtype=float)
    for ii in range(len(keys)):
        ra = ii / len(test_data) * 360
        data[ii, 0] = ra

    problem = GTOC8(data)

    # For each source in order
    for ii, source in enumerate(keys):
        observations = test_data[source]
        for h, P, expected_delta_J in observations:
            old_J = problem.performance_index()
            ra, dec = data[ii, :]

            problem.register_observation(ra, dec, [h])

            new_J = problem.performance_index()
            assert_almost_equal(new_J - old_J, expected_delta_J)
