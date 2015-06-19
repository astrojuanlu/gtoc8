'''

Created on Fri Feb 20 20:57:16 2015

@author: Siro Moreno

This is a submodule for the genetic algorithm that is explained in
https://docs.google.com/presentation/d/1_78ilFL-nbuN5KB5FmNeo-EIZly1PjqxqIB-ant-GfM/edit?usp=sharing

This script is the main program. It will call the different submodules
and manage the data transfer between them in order to achieve the
genetic optimization of the profile.

'''




import os
import numpy as np
import analyze as analyze




def finish(generation, tiempos):
    '''Returns the genome of the (n+1)generation
    '''
    file_parent_name = 'generation'+ str(generation) + '.txt'
    genome_parent_root = os.path.join('genome', file_parent_name)    
    genome = np.loadtxt(genome_parent_root, skiprows=1)
    num_pop = genome.shape[0]
    analyze.pop_analice(generation, num_pop)
    
    print('')
    print('Algoritmo finalizado,  guardando resultados')   
    print('') 
    
    result_name = 'generation' + str(generation) + 'results'
    result_root = os.path.join('results', 'data', result_name + '.txt')
    
    results = np.loadtxt(result_root, skiprows = 1)
    
    winner_n = np.argmax(results[:, 1])
    
    winner = results[winner_n, 1:]
    gen_winner = genome[winner_n, :]
    
    
    file_name = 'generation'+ str(generation + 1) + '_final_report.txt'
    report_root = os.path.join('results', file_name)
    title = 'Informe final de resultados del algoritmo ' 
    
    
    try:
        os.remove(report_root)
    except :
        pass
    
    
    report_file = open(report_root, mode = 'x')
    report_file.write(title + 3 * '\n')
    
    
    txt = 'El algoritmo analizó '+ str(generation + 1) + ' generaciones, de '
    txt = txt + str(num_pop) + ' casos cada una.'+ 2 * '\n' 
    txt = txt +'La mejor oportunidad de observación encontrada por el algoritmo' + '\n' 
    txt = txt +'tuvo las siguientes características:'+ 2 * '\n' 
    txt = txt +'- Distancia del vector de medición a la radiofuente: ' 
    txt = txt + str(winner[5]) +' grados'+ '\n'
    
    txt = txt + '- Mínima altura del triángulo de observación: '
    txt = txt + str(winner[4]) +' Km'+ '\n'
    
    txt = txt + '- step de observación: '
    txt = txt + str(winner[6]) + 3 * '\n'
    
    txt = txt +'Descripción de los parámetros de los satélites:'+ 3 * '\n'
    
    for ii in range(3):
        dv = gen_winner[9*ii + 1 : 9*ii + 4]
        thr = gen_winner[9*ii + 6 : 9*ii + 9]
        
        
        txt = txt + 'Satélite ' + str(ii + 1) + 2 * '\n'
        txt = txt + '-Tiempo hasta activación del motor químico: '
        txt = txt + str(gen_winner[9 * ii]) + ' días'+ 2 * '\n'
        
        txt = txt + '-Vector de delta-v del motor químico: '
        txt = txt + str(dv) + ' km/s'+ '\n'
        txt = txt + '-Delta-v en módulo del motor químico: '
        txt = txt + str(np.linalg.norm(dv)) + ' km/s'+ 2 *'\n'
        
        txt = txt + '-Tiempo hasta activación del motor de empuje contstante: '
        txt = txt + str(gen_winner[9 * ii + 4]) + ' días'+ 2 * '\n'
        
        txt = txt + '-Tiempo hasta apagado del motor de empuje contstante: '
        txt = txt + str(gen_winner[9 * ii + 5]) + ' días'+ 2 * '\n'
        
        
        txt = txt + '-Vector de empuje del motor de empuje contstante: '
        txt = txt + str(thr) + ' N'+ '\n'
        txt = txt + '-Empuje en módulo del motor de empuje contstante: '
        txt = txt + str(np.linalg.norm(thr)) + ' N'+ 2 *'\n'
    
    report_file.write(txt)
    
    report_file.close()  
if __name__ == '__main__':
    if not os.path.exists(os.path.join('results', 'data')):
        os.makedirs(os.path.join('results', 'data'))
    finish(2, (15, 5, 5000))