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
from scipy.stats import betaprime, norm, gamma, lognorm

def log_min_max(c, delta = 1e-6):
    """
    Transformation of a value between 0 and 1 with a logarithmic scale. Used
    to rescale the concentration of nutrients to sognificant levels.
    
    param:
        c : array containing the concentration to rescale
        delta : value to enusre non zero concentration
    """
    return np.log((c+delta)/delta)/np.log((1+delta)/delta)



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
    Nsamples = 300
        
    Ncells = 300
    dt = 0.01
    dim_param = 3 # without account of sigma
    Nstate = 2 # regular, MII
    #time_point = np.array([11, 14, 17, 20, 23, 37, 111])
    time_point = np.array([11, 14, 17, 20, 23, 37])
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
    
    print('\n -Conditions are : \n\t{}'.format(conditions))
    print(' -Time points are : \n\t{}'.format(time))
    print(' -Number of states : {}'.format(N_states))
    
    for i, t in enumerate(time):
        for j, cond in enumerate(conditions):
            row = data[(data['condition']==cond)*(data['t']==t)]
            
            MII = row[['r1', 'r2', 'r3', 'r4']].values.sum()
            NS  = row[['rR']].values
                        
            R[j,:,i] = np.hstack([NS.reshape(-1), MII.reshape(-1)])
    
    return R            


def fitting(data_file):
    """
    Main function that call all necessary parts to perform inference of the landscape
    """
    
    with open('consol_output.txt', 'w') as f:
        with redirect_stdout(f):
            print('STARTING SIMULATION \n')
            # Simulation parameters
            
            # Purely randomely, the error goes from 110 to 160
            # If only states to the first well, the error goes down 127
            max_iter = 1e4 # Maximum number of iteration on one epoch without saving any particle
            epsilons = [29]
            n_steps = 1
            
            
            initializer()
            simulation_param = np.load("./dict_data.npy", allow_pickle=True).item()
            prior_lscp = prior()
            
            Nsamples = simulation_param['Nsamples']
            Ncells = simulation_param['Ncells']
            Nconditions = simulation_param['Nconditions']
            nbatch = simulation_param['nbatch']
            dim_param = simulation_param['dim_particles']
            
            p_in = np.zeros((Nsamples, dim_param))
            w_in = np.zeros((Nsamples))
            d = np.zeros((Nsamples))
            
            for i in range(n_steps):
                
                R = np.load('R.npy')
                
                print("Total samples    : {}".format(Nsamples))
                print("Simulated cells  : {}".format(Ncells))
                print("Total conditions : {}".format(Nconditions), flush=True)
                                                
                start = time.perf_counter()
                
                (p_out, w_out, d_out) = inference.ABC_SMC_step_gpu(epsilons[i], R, p_in, w_in, d, Nsamples, prior_lscp, nbatch, i, max_iter)
                
                end = time.perf_counter()
                print('Time elapsed for epoch {} : {} s \n'.format(i, end-start), flush=True)
                
                print('|####################|')
                print('| Processing Results |')
                print('|####################| \n', flush=True)
                
                current_date = time.localtime()
                prefix = '{}_{}_{}-'.format(current_date.tm_year, current_date.tm_mon, current_date.tm_mday)
        
                np.save('./outputs/{}particles-e{}.npy'.format(prefix, i), p_out)
                np.save('./outputs/{}weights-e{}.npy'.format(prefix, i), w_out)
                np.save('./outputs/{}distance-e{}.npy'.format(prefix, i), d_out)
                
                print("Proposition of threshold based on distances distribution")
                print("\t With 0.15 quantile : {:.5f} \n".format(np.quantile(d_out, 0.15)))
                a = np.quantile(d_out, 0.15)
                epsilons.append(a)
                print('The threshold list is now :')
                print('\t {}'.format(epsilons), flush=True)
                
                p_in = p_out
                w_in = w_out
                d = d_out
        

# -----------------------------------------------------------------------------
# --------------------------- PRIOR -------------------------------------------
# -----------------------------------------------------------------------------

class prior():
    # Prior for the test landscape 
    def __init__(self):
        pass
        
    
    def sample(self, Nparam=1):
        b = norm.rvs(0, 0.8, size=4)
        a = norm.rvs(0, 0.8, size=4)
        v = np.hstack((betaprime.rvs(1,10,size=1), np.random.normal(0, 0.5, 3)))
        sigma = lognorm.rvs(1.5,loc=0.0001, scale=0.01, size=1)
        return np.hstack((b, a, v, sigma))
    
    def evaluate(self, param):
        pab = norm.pdf(param[0:8], 0, 0.8)
        pv0 = betaprime.pdf(param[8], 1, 10)
        pv = norm.pdf(param[9:12], 0, 0.5)
        ps = lognorm.pdf(param[-1], 1.5, loc=0.0001, scale=0.01)
        return np.prod(np.hstack((pab, pv0, pv, ps)))

if __name__ == '__main__':
    fitting()