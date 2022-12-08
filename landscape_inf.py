# -*- coding: utf-8 -*-
"""
This script will do inference on a simple test-landscape, in order to see
if the computer is able to handle it. 
"""

import landscapes
import simulation
import inference
import numpy as np
import time
import torch
import pandas as pd
from contextlib import redirect_stdout
from scipy.stats import betaprime, norm, gamma, lognorm, uniform

def log_min_max(c, delta = 0.000006):
    """
    Transformation of a value between 0 and 1 with a logarithmic scale. Used
    to rescale the concentration of nutrients to sognificant levels.
    
    param:
        c : array containing the concentration to rescale
        delta : value to enusre non zero concentration
    """
    return np.log((c+delta)/delta)/np.log((1+delta)/delta)

def log_min_max_inv(c, delta = 0.000006):
    return delta*(np.exp(c*np.log((1+delta)/delta))-1)



def initializer():
    """
    This function create a dictionary with all the necessary variables to 
    simulate the landscape.
    
    The parameters are then saved in 'dict_data.npy'
    """
    
    AGN_matrix("./timed_counts.csv")
    AGN = np.load('AGN.npy')/100
    delta = 0.000006
    AGN = np.log((AGN+delta)/delta)/np.log((1+delta)/delta)
    AGN = np.hstack((np.ones((AGN.shape[0], 1)), AGN))
    
    AGN = torch.tensor(AGN)
    Nsamples = 500
        
    Ncells = 300
    dt = 0.01
    dim_param = 2 # without account of sigma
    Nstate = 2 # regular, MII
    time_point = np.array([0.01, 11, 14, 17, 20, 23])#, 37, 111])
    step_sample = time_point/dt
    Nsteps = int(step_sample[-1])
    Nmeasure = len(time_point)
   # radius_tolerance = 0.2 # for the gmm cluster
    nbatch = 1000
    
    

    #-----------------------------------------------------------
    #------------------ MAP AND CONDITIONS ---------------------
    #-----------------------------------------------------------
    
    mapp = landscapes.cusp_full    
    dim_particles = dim_param*(AGN.shape[1]) + 1 # for each landscape param, there 3 Nutrient and 1 
    Nconditions = AGN.shape[0]
    N_prediction = Nmeasure*Nconditions
    
    # In the future, will perform inference on part of the data to see if we can catch the underlying truth
    
    # HERE DO STUFF TO SPARSE THE DATA
    
    #-----------------------------------------------------------
    #---------------------- CLUSTERING -------------------------
    #-----------------------------------------------------------
    
    centers = torch.Tensor([[-0.2, 0.0], # for the fate_seperate,
                            [0.2, 0]])
    
    #-----------------------------------------------------------
    #-------------- EULER AND INITAL CONDITIONS ----------------
    #-----------------------------------------------------------
    
    X0 = np.hstack([np.random.normal(-0.2, 0.01, size=(Ncells,1)), np.random.normal(0, 0.01, size=(Ncells,1))])
    X0 = torch.tensor(X0[np.newaxis,:,:])
    
    sim_param = { "Nsamples":Nsamples, "Ncells":Ncells, "dt":dt, "Nsteps":Nsteps,
                  "dim_param":dim_param, "Nstate":Nstate,
                  "Nmeasure":Nmeasure, "centers":centers,
                  "mapp":mapp, "AGN":AGN, "Nconditions":Nconditions,"X0":X0,
                  "dim_particles":dim_particles, 'steps':step_sample, 'nbatch':nbatch}
            
    
    np.save('dict_data.npy', sim_param)


def AGN_matrix(file):
    """Reads the data file tp produce the AGN matrix of nutrient conditions"""
    data = pd.read_csv(file)
    np.save('./AGN.npy', data[data['t']==0].sort_values('condition')[['A', 'G', 'N']].values)
    
    
def R_from_data(data):
    """
    Process the experimental data into a faith matrixd
    """
    
    N_conditions = len(np.unique(data['condition']))
    N_states = 2 #regular and MII for now
    N_measure = len(np.unique(data['t']))
    
    R = np.zeros((N_conditions, N_states, N_measure))
    conditions = np.sort(data['condition'].unique())
    time = np.sort(data['t'].unique())
    
    print("|##############|")
    print('| Ground Truth |')
    print('|##############|')
    
    print('\n -Conditions are : \n{}'.format(conditions))
    print(' -Time points are : \n\t{}'.format(time))
    print(' -Number of states : {}\n'.format(N_states), flush=True)
    
    for i, t in enumerate(time):
        for j, cond in enumerate(conditions):
            row = data[(data['condition']==cond)*(data['t']==t)]
            
            MII = row[['r1', 'r2', 'r3', 'r4']].values.sum()
            NS  = row[['rR']].values
                        
            R[j,:,i] = np.hstack([NS.reshape(-1), MII.reshape(-1)])
    
    return R[np.newaxis,:,:,:]   





def fitting(data_file):
    """
    Main function that call all necessary parts to perform inference.
    """
    
    starting_epoch = 0
    
    with open('consol_output.txt', 'w') as f:
        with redirect_stdout(f):
            print('STARTING SIMULATION \n')
            # Simulation parameters
            

            max_iter = 1e4 # Maximum number of iteration on one epoch without saving any particle
            epsilons = [49]
            n_steps = 15
            R = R_from_data(pd.read_csv(data_file))
            
            initializer()
            simulation_param = np.load("./dict_data.npy", allow_pickle=True).item()
            prior_lscp = prior()
            
            Nsamples = simulation_param['Nsamples']
            Ncells = simulation_param['Ncells']
            Nconditions = simulation_param['Nconditions']
            nbatch = simulation_param['nbatch']
            dim_param = simulation_param['dim_particles']
            
            if starting_epoch == 0:
                p_in = np.zeros((Nsamples, dim_param))
                w_in = np.zeros((Nsamples))
                d = np.zeros((Nsamples))
            else:
                p_in = np.load('./outputs/2022_11_29-particles-e8.npy')
                w_in = np.load('./outputs/2022_11_29-weights-e8.npy')
                d = np.load('./outputs/2022_11_29-distance-e8.npy')
            
            for i in range(starting_epoch, starting_epoch + n_steps):
                

                
                print("Total samples    : {}".format(Nsamples))
                print("Simulated cells  : {}".format(Ncells))
                print("Total conditions : {}".format(Nconditions), flush=True)
                
                start = time.perf_counter()
                
                (p_out, w_out, d_out) = inference.ABC_SMC_step_gpu(epsilons[i], R, p_in, w_in, d, Nsamples, prior_lscp, nbatch, i, max_iter)
                
                end = time.perf_counter()
                print('-- time elapsed for epoch {} : {} s \n'.format(i, end-start), flush=True)
                
                print('|####################|')
                print('| Processing Results |')
                print('|####################| \n', flush=True)
                
                current_date = time.localtime()
                prefix = '{}_{}_{}-'.format(current_date.tm_year, current_date.tm_mon, current_date.tm_mday)
        
                np.save('./outputs/{}particles-e{}.npy'.format(prefix, i), p_out)
                np.save('./outputs/{}weights-e{}.npy'.format(prefix, i), w_out)
                np.save('./outputs/{}distance-e{}.npy'.format(prefix, i), d_out)
                
                print("Proposition of threshold based on distances distribution")
                print("\t With 0.1 quantile : {:.5f} \n".format(np.quantile(d_out, 0.15)))
                a = np.quantile(d_out, 0.1)
                epsilons.append(a)
                print('The threshold list is now :')
                print('\t {}'.format(epsilons), flush=True)
                
                p_in = p_out
                w_in = w_out
                d = d_out
                np.save('./outputs/{}epsilons-e{}.npy'.format(prefix, i), epsilons)
        

# -----------------------------------------------------------------------------
# --------------------------- PRIOR -------------------------------------------
# -----------------------------------------------------------------------------

class prior():
    # Prior for the test landscape 
    def __init__(self):
        pass
        
    
    def sample(self, Nparam=1):
        b = uniform.rvs(-1.5, 3, size=4)
        #a = uniform.rvs(-1.5, 3, size=4)
        somme = -1
        while somme < 0: # the velocity cannot be negative
            v = np.hstack((uniform.rvs(0, 2, size=1), uniform.rvs(-1.5, 3, size=3)))
            somme = v[0] + v[v<0].sum()
        sigma = uniform.rvs(0, 0.2, size=1)
        return np.hstack((b, v, sigma))
    
    def evaluate(self, param):
        v = param[8:12]
        if v[0] + v[v<0].sum() < 0:
            return 0
        pab = uniform.pdf(param[0:4], -1.5, 3)
        pv0 = uniform.pdf(param[4], 0, 2)
        pv = uniform.pdf(param[5:8], -1.5, 3)
        ps = uniform.pdf(param[-1], 0, 0.2)
        return np.prod(np.hstack((pab, pv0, pv, ps)))        


if __name__ == '__main__':
    fitting('timed_counts_reduced.csv')