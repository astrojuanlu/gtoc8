# coding: utf-8
'''

Created on Fri Feb 20 20:57:16 2015

@author: Siro Moreno

This is a submodule for the genetic algorithm that is explained in
https://docs.google.com/presentation/d/1_78ilFL-nbuN5KB5FmNeo-EIZly1PjqxqIB-ant-GfM/edit?usp=sharing

This script creates the initial population for the genetic algorithm.
It does so by adding a random deviation to a default profile genome.

'''


from __future__ import division

import os
import numpy as np
#import testing as test


def start_pop(pop_num, tiempos):
    '''Creates a randomly generated population of the size (pop_num)
    '''
    
    genome = np.zeros([pop_num,27])    
    
    #Lo siguiente son los valores mínimos para cada variable del satélite
    genes = np.array([0, #Tquim, tiempo del impulso químico
                  -3,           
                  -3,    # dvx, dvy, dvz del impulso químico       
                  -3,          
                  0,     # tcons y 
                  0,     # tcone son los tiempos del motor constante, se calculan después
                  -0.1,          
                  -0.1,          #Componentes del empuje constante
                  -0.1,])        


    
    #Lo siguiente son los valores de las amplitudes del intervalo de valores
    gen_deviation = np.array([tiempos[1], 
                  6,           
                  6,           
                  6,           
                  0,  
                  0,           
                  0.2,           
                  0.2,           
                  0.2,])          
                  
                  
    for profile in np.arange(0, pop_num, 1):
        #Satelite 1
        deviation = np.random.rand(9) * gen_deviation
        genome[profile,0:9] = genes + deviation
        #Satelite 2
        deviation = np.random.rand(9) * gen_deviation
        genome[profile,9:18] = genes + deviation
        #Satelite 3
        deviation = np.random.rand(9) * gen_deviation
        genome[profile,18:27] = genes + deviation
        
        #Ahora vamos a calcular los tiempos de inicio y parada
        #del motor constante
        
        #Satelite 1
        tquim = genome[profile, 0]
        intervalo = tiempos[0] - tquim
        tcons = tquim + intervalo * np.random.rand(1)[0]
        intervalo2 = tiempos[0] - tcons
        tcone = tquim + intervalo2 * np.random.rand(1)[0]
        genome[profile, 4] = tcons
        genome[profile, 5] = tcone
        #Satelite 2
        tquim = genome[profile, 9]
        intervalo = tiempos[0] - tquim
        tcons = tquim + intervalo * np.random.rand(1)[0]
        intervalo2 = tiempos[0] - tcons
        tcone = tquim + intervalo2 * np.random.rand(1)[0]
        genome[profile, 13] = tcons
        genome[profile, 14] = tcone
        #Satelite 3
        tquim = genome[profile, 18]
        intervalo = tiempos[0] - tquim
        tcons = tquim + intervalo * np.random.rand(1)[0]
        intervalo2 = tiempos[0] - tcons
        tcone = tquim + intervalo2 * np.random.rand(1)[0]
        genome[profile, 22] = tcons
        genome[profile, 23] = tcone
        
        #Ahora vamos a recortar los vectores de empuje cuyo módulo
        #exceda lo permitido
        
        #Satelite 1
        dv = (genome[profile, 1]**2 +
              genome[profile, 2]**2 +
              genome[profile, 3]**2) ** 0.5
        if dv > 3 :
            genome[profile, 1:4] = genome[profile, 1:4] * 3 / dv
            
        t = (genome[profile, 6]**2 +
             genome[profile, 7]**2 +
             genome[profile, 8]**2) ** 0.5
        if t > 0.1 :
            genome[profile, 6:9] = genome[profile, 6:9] * 0.1 / t
         #Satelite 2
        dv = (genome[profile, 10]**2 +
              genome[profile, 11]**2 +
              genome[profile, 12]**2) ** 0.5
        if dv > 3 :
            genome[profile, 10:13] = genome[profile, 1:4] * 3 / dv
            
        t = (genome[profile, 15]**2 +
             genome[profile, 16]**2 +
             genome[profile, 17]**2) ** 0.5
        if t > 0.1 :
            genome[profile, 15:18] = genome[profile, 6:9] * 0.1 / t 
         #Satelite 3
        dv = (genome[profile, 19]**2 +
              genome[profile, 20]**2 +
              genome[profile, 21]**2) ** 0.5
        if dv > 3 :
            genome[profile, 19:22] = genome[profile, 1:4] * 3 / dv
            
        t = (genome[profile, 24]**2 +
             genome[profile, 25]**2 +
             genome[profile, 26]**2) ** 0.5
        if t > 0.1 :
            genome[profile, 24:27] = genome[profile, 6:9] * 0.1 / t

    
        
    genome_root = os.path.join('genome','generation0.txt')
    title = 'generation 0 genome'
    
    try:
        os.remove(genome_root)
    except :
        pass
    archivo = open(genome_root, mode = 'w')
    archivo.write(title + '\n')
    
    for profile in np.arange(0, pop_num, 1):
        line = ''
        for gen in np.arange(0, 27,1):
            line = line + str(genome[profile, gen]) +'    '
        line = line + '\n'
        archivo.write(line)
    
    return genome

if __name__ == '__main__':
    if not os.path.exists('genome'):
        os.makedirs('genome')
    start_pop(2, (15, 10,500))
