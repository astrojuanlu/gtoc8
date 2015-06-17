# coding: utf-8
import numpy as np


def func_kepler(v, t0, mu=398600.4329):
    """Differential equation for pure Keplerian motion.

    """
    x, y, z, vx, vy, vz = v
    den = (x**2 + y**2 + z**2)**1.5
    return np.array([vx, vy, vz, -mu * x / den, -mu * y / den, -mu * z / den])


def rk4(f, ti, yi, dt):
    # Source: http://doswa.com/2009/04/21/improved-rk4-implementation.html
    k1 = dt * f(yi          , ti          )
    k2 = dt * f(yi + .5 * k1, ti + .5 * dt)
    k3 = dt * f(yi + .5 * k2, ti + .5 * dt)
    k4 = dt * f(yi + k3     , ti + dt     )

    yf = yi + (1. / 6.) * (k1 + 2 * k2 + 2 * k3 + k4)

    return yf


def integrate_rk4(func, t, v0):
    dt = t[1] - t[0]
    sol_rk4 = np.zeros((len(t), len(v0)))
    sol_rk4[0, :] = v0
    for ii in range(1, len(t)):
        sol_rk4[ii, :] = rk4(func, t[ii], sol_rk4[ii - 1, :], dt)

    return sol_rk4


# numba versions
try:
    import numba
    if numba.__version__.split('.')[1] < 19:
        raise ImportError

    jit = numba.njit
except ImportError:
    print("numba >= 0.19 is required for numba versions")
    import poliastro.jit
    jit = poliastro.jit.ijit


@jit('float64[:](float64[:], float64)')
def func_numba(v, t0):
    mu = 398600
    x, y, z, vx, vy, vz = v[:]
    den = (x**2 + y**2 + z**2)**1.5
    sol = np.zeros(6)
    #sol[:] = (vx, vy, vz, -mu * x / den, -mu * y / den, -mu * z / den)
    sol[0] = vx
    sol[1] = vy
    sol[2] = vz
    sol[3] = -mu * x / den
    sol[4] = -mu * y / den
    sol[5] = -mu * z / den
    return sol


@jit('float64[:](float64, float64[:], float64)')
def rk4_numba(ti, yi, dt):
    # Source: http://doswa.com/2009/04/21/improved-rk4-implementation.html
    # Limitations: cannot pass function as an argument
    nn = len(yi)

    k1 = func_numba(yi, ti)

    yi2 = np.zeros(nn)
    for ii in range(nn):
        yi2[ii] = yi[ii] + .5 * k1[ii] * dt
    k2 = func_numba(yi2, ti + .5 * dt)

    yi3 = np.zeros(nn)
    for ii in range(nn):
        yi3[ii] = yi[ii] + .5 * k2[ii] * dt
    k3 = func_numba(yi3, ti + .5 * dt)

    yi4 = np.zeros(nn)
    for ii in range(nn):
        yi4[ii] = yi[ii] + k3[ii] * dt
    k4 = func_numba(yi4, ti + dt)

    yf = np.zeros(len(yi))
    for ii in range(len(k1)):
        yf[ii] = yi[ii] + (1. / 6.) * (k1[ii] * dt + 2 * k2[ii] * dt + 2 * k3[ii] * dt + k4[ii] * dt)

    return yf


@jit('float64[:, :](float64[:], float64[:])')
def integrate_rk4_numba(t, v0):
    dt = t[1] - t[0]
    sol_rk4 = np.zeros((len(t), len(v0)))
    for ii in range(len(v0)):
        sol_rk4[0, ii] = v0[ii]

    for ii in range(1, len(t)):
        sol = rk4_numba(t[ii], sol_rk4[ii - 1, :], dt)
        for jj in range(len(sol)):
            sol_rk4[ii, jj] = sol[jj]

    return sol_rk4
