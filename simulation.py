# -*- coding: utf-8 -*-
"""
The script regroup all functions that serve to simulate data
"""

import numpy as np
import torch
from clustering import KMeans_Clustering
#import gmm
import time

# def apply_along_axis(function, x, axis: int=0):
#     return torch.stack([function(x_i) for x_i in torch.unbind(x, dim=axis)], dim=axis)

def fitpam_to_landscape(fit_params, AGN, dim_param):
    """
    Adapat the parameters to a form compatible to the landscape
    param:
        fit_params: particle outputs of ABC SMC. Last position is sigma
        AGN : (Nmedia, Nsubstances) matrix for all media used
        dim_param : Dimension of the landscape parameters
    return:
        lands_parm: (Nmedia, dim_param) containing the landscape for each condition
        sigma : The noise term (float)
    """
    
    parameters = fit_params.reshape(-1)
    
    sigma = parameters[-1]
    lands_parm = parameters[:-1].reshape(dim_param, -1)
    lands_parm = AGN @ lands_parm.T
    return lands_parm, sigma

def generate_fate_gpu(parameters):
    """
    Generate the evolution and the fate probability with the given parameters.
    The function runs the euler and the clustering algorithm, returnong the
    final probabilities.
    
    param:
        parameters: Array(nbatch, dim_param * (Nvariable+1) + 1)
    return:
        R: Array(nbatch, Nconditions, Nstates, Nmeasure)
    """
    
    print("\t Loading and preparing simulation parameters", flush=True)
    
    # Load essential parameters from the initializer file
    simulation_param = np.load("./dict_data.npy", allow_pickle=True).item()
    
    AGN = simulation_param['AGN']       # Concentration matrix for each nutrient
    X0 = simulation_param['X0']         # Initial condition for the cells (Euler)
    F_lscpe = simulation_param['mapp']  # Landscape
    Nstate = simulation_param['Nstate'] # Number of state in the landscape
    Nmeasure = simulation_param['Nmeasure'] # Number of time point sampled
    dt = simulation_param['dt']         # Time step for the simulation
    Nsteps = simulation_param['Nsteps'] # Total iteration in Euler
    dim_param = simulation_param['dim_param']   # Dimension of the landscape parameter (!= particle dimension)
    centers = simulation_param['centers']   # Position of the center of the cluster
    Ncells = simulation_param['Ncells'] # Number of cell simulated for each condition
    Nconditions = AGN.shape[0]          # Number of nutrient conditions
    steps = simulation_param['steps']   # The steps at which to save Euler results 
    
    nbatch = parameters.shape[0]
    
    # Transform each batch particle into a compatible form with the landscape
    lands_param = torch.zeros((nbatch, Nconditions, dim_param))
    sigmas = np.zeros(nbatch)
    for i in range(nbatch):
        lands_param[i,:,:], sigmas[i] = fitpam_to_landscape(parameters[i], AGN, dim_param)
    
    # running stochastic euler
    print("\t Starting stochastic euler evolution", flush=True)
    start = time.perf_counter()
    
    X = euler_gpu(X0, sigmas, F_lscpe, lands_param, dt, Nsteps, steps, Nconditions, nbatch)
    
    torch.cuda.empty_cache()
    end = time.perf_counter()
    print('\t -- time elapsed for euler step : {} seconds'.format(end-start))

    # reshape the matrix to be compatible with the clustering algorithm
    X = X.reshape(nbatch*Nconditions*Ncells*Nmeasure, 2)
    
    if torch.cuda.is_available():
        centers = centers.cuda()
        X = X.cuda()
        
        
    #Cluster the data according to the specified algo
    print("\t Starting Clustering", flush=True)
    start = time.perf_counter()
    
    # /!\ This part is actually crucial ! Clustering all the conditions together leads to false results since it biases
    # The condition. Needs to be adressed at some point later !!
    assignement = KMeans_Clustering(X, centers)
    assignement = assignement.reshape(nbatch, Nconditions, Ncells, Nmeasure)
    assignement = assignement.cpu()
    assignement = assignement.numpy()
    
    end = time.perf_counter()
    print('\t -- time elapsed for clustering : {} seconds'.format(end-start), flush=True)
    
    R = np.zeros((nbatch, Nconditions, Nstate, Nmeasure))
    
    # Compute for all batch the faith probability
    for b in range(nbatch):
        for i in range(Nstate):
            R[b,:,i,:] = np.sum(assignement[b]==i, axis=-2)/Ncells
        
    # removing unncessary variables from gpu
    del X, centers, assignement
    torch.cuda.empty_cache()
    
    return R
                        

def euler_gpu(X0, sigma, F, parameters, dt, Nsteps, steps, Nconditions, nbatch):
    """
    This version of euler stochastic simulation make use of GPU
    param:
        X0 : (Nconditions, Ncells, 2) array of inital values
        sigma : (nbatch) array of noise parameter
        F : Field function of the form F(x, y, param)
        parameters : (nbatch, Ncondition, dim_param)
        dt : time step size float
        Nsteps : Number of steps to do in total
        steps : list of iteration at which to sample the data
        Nconditions : number of different media uesed
        nbatch : number of parameters that are sampled at the same time
        
    return:
        Xem: (nbatch, Nconditions, Ncells, 2, len(steps)) array of time evolution
    """
    
    Ncells = X0.shape[1]
    dim_param = parameters.shape[-1]
    
    dt = torch.tensor(dt)
    mean = torch.tensor([0.0])
    std = torch.tensor([1.0])   
    
    Xem = torch.zeros((nbatch, Nconditions, Ncells, len(steps), 2))
    
    Xtemp = torch.zeros((nbatch, Nconditions, Ncells, 2, 2)) 
    Xtemp[:,:,:,0, 0] += X0[:,:,0]
    Xtemp[:,:,:,0, 1] += X0[:,:,1]
    
    p = (torch.ones((nbatch, Nconditions, Ncells, dim_param))*parameters[:,:,np.newaxis,:])
    sigma = torch.tensor(sigma.reshape((nbatch, 1, 1)))
    
    if torch.cuda.is_available():
        dt = dt.cuda()
        mean = mean.cuda()
        std = std.cuda()
        Xtemp = Xtemp.cuda()
        p = p.cuda()
        sigma = sigma.cuda()
    else:
        print("The computation are runing on the CPU hence much slower than initially deseigned")
    
    # noise term
    dW = torch.distributions.normal.Normal(mean, std)    
    
    # remove unnecessary variable from GPU
    del mean, std
    torch.cuda.empty_cache()
    
    # reshaping parameters
    Xtemp = Xtemp.reshape(nbatch, Ncells*Nconditions, 2, 2) # put all condition queue to queue
    
    # allign parameters for direct computation of all conditions in one go
    p = p.reshape((nbatch, Ncells*Nconditions, dim_param)) 
        
    k = 1
    for i in range(Nsteps):
        Field = F(Xtemp[:,:,0,0], Xtemp[:,:,0,1], p).transpose(1,2)
        Xtemp[:,:,1,:] = Xtemp[:,:,0,:] + dt * Field + torch.sqrt(dt)*sigma*(dW.rsample((nbatch, Ncells*Nconditions, 2))).squeeze(-1)
        Xtemp[:,:,0,:] = Xtemp[:,:,1,:]
       
        #list of steps at which to save the results 
        if i+1 in steps:
            Xem[:,:,:,k-1,:] = Xtemp[:,:,1,:].reshape(nbatch, Nconditions, Ncells, 2).cpu()
            k += 1
           
    # remove unvessary variables from GPU
    del Xtemp, p, dW
    torch.cuda.empty_cache()
            
    return Xem
