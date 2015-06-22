# coding: utf-8
'''

Created on Fri Feb 20 20:57:16 2015

@author: Siro Moreno

Este subprograma es el iniciador y el centro de control del algoritmo genético.
Aquí se definen las principales variables que lo controlan.

Primero se define la funcion principal, que ejecuta todo el algoritmo.

Despues, si este es archivo que se está ejecutando, establece unos valores para 
los parámetros y llama a esta función.

'''




import os
import orbiter as orbiter
import numpy as np
import initial as initial
import genetics as genetics
import ender as ender

#First, the main function is defined. This allows us to call it from a future
#different starting file (like a PyQT graphic interface).

#If this is the starting file, it will call the main function with the 
#parameters described below.



def main_program(all_parameters):
    '''The main function of the program, calls in order all the rest'''
    
    solutions_per_generation = all_parameters[0]
    total_generations = all_parameters[1]
    num_parent = all_parameters[2]
    num_winners = all_parameters[3]
    tiempos = all_parameters[4]

    
###--- Creating work directories
    
#    if not os.path.exists('aerodata'):
#        os.makedirs('aerodata')
        
    if not os.path.exists('genome'):
        os.makedirs('genome')
        
    if not os.path.exists('orbitas'):
        os.makedirs('orbitas')
        
    if not os.path.exists('results'):
        os.makedirs('results')

       
    if not os.path.exists(os.path.join('results', 'data')):
        os.makedirs(os.path.join('results', 'data'))




####--- Starting the population, analysis of the starting population


    generation = 0
    print('')
    print('')
    print('Iniciando cálculos de la generación 0, generada aleatoriamente')
    print('')

    initial.start_pop(solutions_per_generation, tiempos)

    orbiter.calculate_orbit_population(generation, tiempos)


####--- Genetic Algorithm


    for generation in np.arange(0,total_generations,1):
    

        genetics.genetic_step(generation,num_parent, tiempos)
        print('')
        print('Genoma de la generación ', generation + 1, ' calculado')
        print('')
    
        orbiter.calculate_orbit_population(generation + 1,  tiempos)
    
   

    ender.finish(total_generations, tiempos)    
    


#If this is the file from which we are starting, we define here the parameters:

if __name__ == '__main__':

####---------Primary Variables-----


    solutions_per_generation = 4
    total_generations = 1
    num_parent = 1

# Other parameters of the algorithm

    length = 15 #Length of the experiment in days
    maxtquim = 5 #Maximun time for using the chemical impulse
    number_steps = 5000 #Number of time steps
    tiempos = (length, maxtquim, number_steps)


    num_winners = 1

             
             
    all_parameters = (solutions_per_generation, total_generations, num_parent,
                      num_winners, tiempos )   


    main_program(all_parameters)
