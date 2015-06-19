'''

Created on Fri Feb 20 20:57:16 2015

@author: Siro Moreno

This is a submodule for the genetic algorithm that is explained in
https://docs.google.com/presentation/d/1_78ilFL-nbuN5KB5FmNeo-EIZly1PjqxqIB-ant-GfM/edit?usp=sharing

This script is the Mutation subprogramme. Its objective is to add diversity
to the population, in order to avoid stagnation in a not good solution.

Intensity of the mutation is propotional to the square root of the generation 
number, in order to refine the search for an optimal solution.

The parents of the generation are left unmutated in order to avoid the chance
of decreasing the quality obtained in a previous step.

'''




import numpy as np

def mutation(children, generation, num_parent, tiempos):
    '''Given a genome, mutates it in order to have a diverse population
    '''
    coeff = 0.5 / (1 + generation**0.5)
    gen_deviation_0 = np.array([1, #Tquim, tiempo del impulso químico
                  0.5,           
                  0.5,    # dvx, dvy, dvz del impulso químico       
                  0.5,          
                  4,     # tstart y 
                  4,     # tend son los tiempos del motor constante
                  0.05,          
                  0.05,          #Componentes del empuje constante
                  0.05,])          #dist s2
    gen_deviation = np.zeros(27)
    gen_deviation[0:9] = gen_deviation_0
    gen_deviation[9:18] = gen_deviation_0
    gen_deviation[18:27] = gen_deviation_0
    
    pop_num = children.shape[0]
    
    children_n = np.zeros_like(children[1, :])
    #Ahora comprobaremos para cada satélite de cada genoma que sus genes son viables
    for i in np.arange(num_parent, pop_num, 1):
        deviation = coeff * np.random.randn(27) * gen_deviation
        children_n[:] = children[i,:] + deviation
        for sat in range(3):
            dv = children_n[9*sat + 1: 9*sat + 4]
            thr = children_n[9*sat + 6: 9*sat + 9]
            
            dv_n = np.linalg.norm(dv)
            thr_n = np.linalg.norm(thr)
            
            if dv_n > 3 :
                dv[:] = dv * 3 / dv_n
            if thr_n > 0.1 :
                thr[:] = thr * 0.1 / thr_n
            
            tquim = children_n[9*sat + 0: 9*sat + 1]
            tconst = children_n[9*sat + 4: 9*sat + 6]
            if tquim[0] > tiempos [2] :
                tquim[0] = tiempos [2]
            if tconst[0] < tquim[0] :
                tconst[0] = tquim[0]
            if tconst[1] < tconst[0] :
                tconst[1] = tconst[0]
            
        children[i,:] = children_n[:]
    
    return children
