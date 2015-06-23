# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 14:18:10 2015

@author: Siro, Juan Luis Cano

Este subprograma es llamado por el main.

La función orbiter es alimentada por el genoma y las condiciones temporales,
y devuelve los arrays de posiciones y velocidades de los satélites.

Estos datos son guardados en la carpeta "orbitas"
"""



from __future__ import division

import os
import numpy as np
from scipy import integrate

from astropy import units as u
from poliastro.twobody import State
from poliastro.bodies import Earth
Earth.k = 398600.4329 * u.km**3 / u.s**2  # GTOC 8

from poliastro.maneuver import Maneuver



def state_to_vector(ss):
    x0, y0, z0 = ss.r.to(u.km).value
    vx0, vy0, vz0 = ss.v.to(u.km / u.s).value
    return np.array([x0, y0, z0, vx0, vy0, vz0])
    
    
def func_kepler(v, t, mass_param, thrust,  mu=398600.4329):
    """Ecuación a integrar cuando hay impulso continuo

    """
    x, y, z, vx, vy, vz = v
    m1, t0, mdot = mass_param
    tx, ty, tz = thrust / 1000 #Pasar los newtons (m) a km
    mass = m1 - (t -t0) * mdot
    den = (x**2 + y**2 + z**2)**1.5
    ax = tx/mass -mu * x / den
    ay = ty/mass -mu * y / den
    az = tz/mass -mu * z / den
    return np.array([vx, vy, vz, ax ,ay ,az ])
    
    
def calculate_orbit(generation,orbit_number, genome, tiempos):
    '''Calcula las órbitas de los 3 satélites de cada caso, y guarda los datos
    '''
    print('')
    print('Calculando órbitas del caso ', orbit_number, ' de la generación ', generation)
    length = tiempos[0]
    num_steps = tiempos[2]
    dt = length / num_steps
    
    t = np.linspace(0 * u.h, length * u.day, num_steps)
    m0 = 4000
    g = 9.80665
    isp_chem = 450
    isp_const = 5000
    
    
    case_name = 'gen' + str(generation) + 'caso' + str(orbit_number)
    data_root = os.path.join("orbitas","data" + case_name + '.txt')    
    
    genome = genome.reshape((3,9))
    case_state_vector = np.zeros(( num_steps, 18))
    for sat in range(3):
        print('    Integrando órbita del satélite ', sat + 1)

        
        state_vector = np.zeros(( num_steps,6))        
        
        tquim = genome[sat, 0]
        dv = genome[sat, 1:4]
        tstart = genome[sat, 4]
        tend = genome[sat, 5]
        tconst = genome[sat, 6:9]
        m1 = m0 * np.exp(- np.linalg.norm(dv) /(g * isp_chem))
        mdot = -np.linalg.norm(tconst) / (g * isp_const)
        mass_param = (m1, tstart, mdot)
        
        orbit = State.circular(Earth, 400 * u.km)
        
        n_dv = int(tquim//dt) #Desde 0 hasta n_dv, la posición se propaga
        n_start = int(tstart//dt) #Desde entonces a n_start, se propaga
        n_end = int(tend//dt) #Desde entonces a n_end, se integra con empuje
        
        #primera propagación: hasta el momento del impulso químico
        
        state_vector[0, :] = state_to_vector(orbit)
        for ii in range(1,n_dv):
            orb_state = orbit.propagate(t[ii])
            state_vector[ii,:] = state_to_vector(orb_state)
        print('        Primera propagación, ok!')
        
        maniobra = Maneuver((tquim * u.day,  dv * u.km / u.s)) #aplicamos el impulso
        orbit = orbit.apply_maneuver(maniobra)
        
        #Segunda propagación: hasta el encendido del motor constante
        
        for ii in range(n_dv, n_start):
            orb_state = orbit.propagate(t[ii])
            state_vector[ii,:] = state_to_vector(orb_state)
        print('        Segunda propagación, ok!') 
        
        #Integración del motor de empuje constante
        
        state_vector[n_start:n_end, :] = integrate.odeint(func_kepler, state_to_vector(orb_state),
                                             t[n_start:n_end].to(u.s).value,
                                               args=(mass_param, tconst))
        print('        Integración del empuje, ok!')  
                                     
        #Tercera y última propagación, coast orbit tras el empuje
        
        #--------- COMPROBAR INDICES, ME LÍO MUCHO CON ELLOS!!!!!!!!!--------------
        r = state_vector[n_end-1, 0:3] * u.km #¡¡¡¡¡ COMPROBAR LOS INDICES!!!!!
        v = state_vector[n_end-1, 3:6] * u.km / u.s
        orbit = State.from_vectors(Earth, r, v, t[n_end-1]) 
        for ii in range(n_end, num_steps):
            orb_state = orbit.propagate(t[ii])
            state_vector[ii,:] = state_to_vector(orb_state)
        print('        Terera propagación, ok!')                                       
        case_state_vector[:, sat *6 : (sat + 1) * 6] = state_vector
    try:
        os.remove(data_root)
    except :
        pass
   
    np.savetxt(data_root, case_state_vector)

def calculate_orbit_population(generation, tiempos):
    '''Calcula las posiciones y velocidades de los satélites de toda la 
    población de casos para una generación
    '''
    
    genome_root = os.path.join('genome','generation'+ str(generation) + '.txt')    
    genome_matrix = np.loadtxt(genome_root, skiprows=1)    
    num_pop = genome_matrix.shape[0]
    
    
    
    for orbit_number in np.arange(1,num_pop+1,1):
        calculate_orbit(generation,
                        orbit_number,
                        genome_matrix[orbit_number-1,:],
                        tiempos)

if __name__ == '__main__':
    
    if not os.path.exists('orbitas'):
        os.makedirs('orbitas')
    calculate_orbit_population(0, (15, 5, 5000))   
    
    