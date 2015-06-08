# coding: utf-8
"""GTOC 8 library code

Author: Juan Luis Cano Rodríguez <juanlu001@gmail.com>

"""
import numpy as np


def dist_point_segment(xp, x1, x2):
    """Compute distance from point to segment.

    """
    dist = (np.linalg.norm(np.cross(x2 - x1, x1 - xp)) /
            np.linalg.norm(x2 - x1))
    return dist


def triangle_altitudes(xa, xb, xc):
    """Compute all three altitudes of a triangle in space.

    """
    h1 = dist_point_segment(xa, xb, xc)
    h2 = dist_point_segment(xc, xa, xb)
    h3 = dist_point_segment(xb, xc, xa)

    return h1, h2, h3


def cartesian_to_celestial(xp):
    """Convert (x, y, z) vector to (ra, dec).

    Returns
    -------
    ra : float
        Right ascension.
    dec : float
        Declination.

    """
    r = np.linalg.norm(xp)
    x, y, z = xp
    ra = np.arctan2(y, x) % (2 * np.pi)
    dec = np.arcsin(z / r)
    return ra, dec


def celestial_to_cartesian(ra, dec):
    """Convert (ra, dec) to (x, y, z) vector.

    This function uses unit radius.

    """
    r = 1.0
    x = r * np.cos(ra) * np.cos(dec)
    y = r * np.sin(ra) * np.cos(dec)
    z = r * np.sin(dec)
    return np.array([x, y, z])


def observation_triangle(x1, x2, x3):
    """Compute sorted altitudes and observation vector.

    Returns
    -------
    ndarray
        Observation vector in cartesian coordinates.
    list
        Sorted ascending list of triangle altitudes.

    """
    vector = np.cross(x2 - x1, x3 - x2)
    altitudes = triangle_altitudes(x1, x2, x3)
    return vector, sorted(altitudes)


def distance_radio_source(vec, ra, dec):
    """Measurement of distance between radio source and observation vector.

    Returns
    -------
    float
        A number between 0.0 and 1.0, higher if the radio source is closer
        to the observation vector.

    Notes
    -----
    The result is the absolute value of the cosine of the angle between both
    vectors.

    """
    vec_u = vec / np.linalg.norm(vec)
    rs_xyz = celestial_to_cartesian(ra, dec)
    return np.abs(np.dot(vec_u, rs_xyz))


def rank_radio_sources(vec, sources_data):
    """Sort radio sources in ``sources_data`` according to proximity
    to ``vec``.

    Parameters
    ----------
    vec : ndarray
        Vector of cartesian (x, y, z) observation coordinates.
    sources_data : array-like
        Array of Nx2 rows of (ra, dec) positions.

    """
    res = sorted(sources_data,
                 key=lambda row: distance_radio_source(vec, row[0], row[1]),
                 reverse=True)  # Descending
    return res
