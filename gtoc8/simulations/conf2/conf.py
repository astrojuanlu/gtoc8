import numpy as np
from astropy import units as u
from poliastro.twobody import State
from poliastro.bodies import Earth
Earth.k = 398600.4329 * u.km**3 / u.s**2  # GTOC 8

from poliastro.maneuver import Maneuver
from poliastro.util import norm


def config():
    t = np.linspace(0 * u.h, 180 * u.day, 50000)

    ss_A0 = State.circular(Earth, 400 * u.km)
    ss_B0 = State.circular(Earth, 400 * u.km)
    ss_C0 = State.circular(Earth, 400 * u.km)

    initial_period = ss_A0.period
    max_dv = 3 * u.km / u.s

    # Maniobra en el plano
    vec_A = ss_A0.v
    man_A = Maneuver.impulse(vec_A / norm(vec_A) * max_dv)

    # Maniobra con componente vertical hacia arriba
    vec_B = [0, 0, 1] * u.km / u.s
    man_B = Maneuver.impulse(vec_B / norm(vec_B) * max_dv)

    # Maniobra con componente vertical hacia abajo
    vec_C = [0, 0, -1] * u.km / u.s
    man_C = Maneuver((initial_period / 10, vec_C / norm(vec_C) * max_dv))

    ss_A = ss_A0.apply_maneuver(man_A).propagate(initial_period / 4)
    ss_B = ss_B0.apply_maneuver(man_B).propagate(initial_period / 4)
    ss_C = ss_C0.apply_maneuver(man_C).propagate(initial_period / 4 - initial_period / 10)

    return ss_A, ss_B, ss_C, t
