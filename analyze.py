# coding: utf-8
'''

Created on Fri Feb 20 20:57:16 2015

@author: Siro Moreno

This is a submodule for the genetic algorithm that is explained in
https://docs.google.com/presentation/d/1_78ilFL-nbuN5KB5FmNeo-EIZly1PjqxqIB-ant-GfM/edit?usp=sharing

Este es el subprograma de análisis. El objetivo es asignar a cada caso de una generación 
dada un valor, según la mejor oportunidad de observación que se obtiene.

'''




import numpy as np
import os
from gtoc8.lib import *
from gtoc8.io import *



def pop_analice (generation, num_pop):
    '''Para cada generación, analiza los casos uno a uno y devuelve
    un vector con sus puntuaciones
    '''
    catalogue = load_catalogue("data/gtoc8_radiosources.txt")
    
    pop_results = np.zeros(num_pop)
    pop_vectors = np.zeros((num_pop, 3))
    pop_minh = np.zeros(num_pop)
    pop_precission = np.zeros(num_pop)
    pop_step = np.zeros(num_pop)
    for case_number in np.arange(1,num_pop+1,1):
       case_result = case_analice (generation, case_number, catalogue)
       pop_results[case_number - 1] = case_result[0]
       pop_vectors[case_number - 1, :] = case_result[1]
       pop_step[case_number - 1] = case_result[2][0]
       pop_minh[case_number - 1] = case_result[3]
       pop_precission[case_number - 1] = case_result[4]
      
#    print('puntuaciones: ', pop_results)
    
    result_name = 'generation' + str(generation) + 'results'
    result_root = os.path.join('results', 'data', result_name + '.txt')
    
    try:
        os.remove(result_root)
    except :
        pass
    result_file = open(result_root, mode = 'x')
    
    result_file.write('case  result  vector  minh  precission  step' + '\n')
    for ii in range(num_pop):
        result_file.write(str(ii) + '   ' +
                           str(pop_results[ii]) + '   ' +
                           str(pop_vectors[ii, 0]) + '   ' +
                           str(pop_vectors[ii, 1]) + '   ' +
                           str(pop_vectors[ii, 2]) + '   ' +
                           str(pop_minh[ii]) + '   ' +
                           str(pop_precission[ii]) + '   ' +
                           str(pop_step[ii]) + '   '  + '\n')
    result_file.close()
    return pop_results
    
    
def case_analice (generation, case, catalogue):
    '''Para cada caso, busca la mejor oportunidad de medición y devuelve
    su valor
    '''  
    print('')
    print('Analizando caso ' , case, ' de la generación ', generation)
    case_name = 'datagen' + str(generation) + 'caso' + str(case)+ '.txt'
    data_root = os.path.join("orbitas", case_name)
    datos = np.loadtxt(data_root, usecols=[0,1,2, 6,7,8, 12,13,14])
    
#    for ii in range(5):
#        print('datos de satelites: ', datos[ii])
    
    read_dim = np.array(datos.shape)
    num_steps = read_dim[0]
    
    apunt = np.zeros((num_steps, 3))
    min_altura = np.zeros(num_steps)
    print('    Calculando vectores de apuntamiento y alturas de triángulos')
    
    
    for ii in range(num_steps):
        apunt[ii], altura = observation_triangle(datos[ii, 0:3],
                                                 datos[ii, 3:6],
                                                 datos[ii, 6:9])
                            
        min_altura[ii] = min(altura)
#        print(ii, 'min_altura = ', min_altura[ii])
#    for ii in range(5):
#        print('vector de apuntamiento: ', apunt[ii])
    print('    Calculando el valor de las observaciones')
    observation_values = np.zeros((num_steps, 2))
    p_radio = 1 #En algoritmos posteriores, este valor irá cambiando
    print('        sentido 1')
    for ii in range(num_steps):
#        print(ii)
        vector = apunt[ii, :]
        modulo = np.linalg.norm(vector)
        if modulo > 0.01 :
            dec = np.arcsin(vector[2] / modulo)
            altura_mod = min_altura[ii] * (0.5 + 0.5 * np.tanh( min_altura[ii]/500 - 20))
#            print(ii, 'altura_mod = ', altura_mod)
            valor = altura_mod * p_radio * (0.2 + np.cos(dec)**2)
#            print('valor = ', valor)
            dist_radio = closest_radio_source(vector, catalogue)[1]
            dist_radio = np.arccos(dist_radio)*180/np.pi
            dist_radio_mod = 0.5 - 0.5 * np.tanh(8 * dist_radio - 3)
            valor_def = valor * dist_radio_mod
#            print('valor def = ', valor_def)
            observation_values[ii, 0] = valor_def
        else:
            observation_values[ii, 0] = 0
    #Valores para el sentido contrario del vector
    print('        sentido 2')
    for ii in range(num_steps):
#        print(ii)
        vector = -apunt[ii, :]
        modulo = np.linalg.norm(vector)
        if modulo > 0.1 :
            dec = np.arcsin(vector[2] / modulo)
            altura_mod = min_altura[ii] * (0.5 + 0.5 * np.tanh( min_altura[ii]/500 - 20))
            valor = altura_mod * p_radio * (0.2 + np.cos(dec)**2)
            dist_radio = closest_radio_source(vector, catalogue)[1]
            dist_radio = np.arccos(dist_radio)*180/np.pi
            dist_radio_mod = 0.5 - 0.5 * np.tanh(8 * dist_radio - 3)
            valor_def = valor * dist_radio_mod
            observation_values[ii, 1] = valor_def
        else:
            observation_values[ii, 1] = 0
     
    #Búsqueda de la mejor oportunidad de medición
    best_valor = np.max(observation_values)
    index = np.argmax(observation_values, axis = 0)
    
    if observation_values[index[0], 0] > observation_values[index[1], 1] :
        pos_best = [index[0], 0]
        best_vector = apunt[index[0], :]
    else:
        pos_best = [index[1], 1]
        best_vector = -apunt[index[0], :]
    best_hmin = min_altura[pos_best[0]]
    best_dist_radio = closest_radio_source(best_vector, catalogue)[1]
    best_dist_radio = np.arccos(best_dist_radio)*180/np.pi
    
    best_obs = (best_valor,best_vector, pos_best, best_hmin, best_dist_radio)
    
    print('La mejor observación tuvo una hmin de ', best_hmin,
          ', una precisión de apuntamiento de ', best_dist_radio,
          ' grados y una puntuaciónde ', best_valor)
    print()
    return best_obs
    
if __name__ == '__main__':
    if not os.path.exists(os.path.join('results', 'data')):
        os.makedirs(os.path.join('results', 'data'))
    pop_analice (0, 2)
    

