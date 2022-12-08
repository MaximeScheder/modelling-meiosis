# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 14:14:00 2022

@author: scheder

A simple file to launch the generate landscape function
"""

from argparse import ArgumentParser
from simulation import generate_fate_gpu, fitpam_to_landscape
from benchmark_steps import euler_GPU_2
import clustering
import torch
import numpy as np
import time


def cusp_gpu(x, y, parameter):
    p = parameter
    if p.ndim == 1:
        p = p.reshape(1, -1)
    return -torch.stack([4*x**3 - 2*x/2 + p[:,:,0], y], dim=1)


def run_gpu(parameters):
    simulation_param = np.load("./dict_data.npy", allow_pickle=True).item()
    # nvmlInit()
    #Test Eueler
    start = time.perf_counter()
    
    AGN = simulation_param['AGN']
    
    Ncells = 50
    dt = 0.01
    dim_param = 1 # without account of sigma
    Nstate = 2 # regular, MII
    time_point = np.array([11, 14, 17, 20, 23])#, 37, 111])
    steps = time_point/dt
    Nsteps = int(steps[-1])
    Nmeasure = len(time_point)
    radius_tolerance = 0.2 # for the gmm cluster
    F_lscpe = cusp_gpu   
    dim_particles = dim_param*(AGN.shape[1]+1) + 1 # for each landscape param, there 3 Nutrient and 1 
    Nconditions = AGN.shape[0]
    X0 = np.hstack((np.random.normal(0, 0.08, (Ncells, 1)), np.random.normal(0.6, 0.05, (Ncells, 1))))
    X0 = X0[np.newaxis,:,:]
    
    centers = torch.Tensor([[0, 0.3], # for the fate_seperate,
                            [0.4, 0]])
    
    torch.cuda.empty_cache()
    
    ncpu = parameters.shape[0]
    
    lands_param = torch.zeros((ncpu, Nconditions, dim_param))
    sigmas = np.zeros((ncpu, 1, 1, 1))
    
    
    for i in range(ncpu):
        lands_param[i], sigmas[i] = fitpam_to_landscape(parameters[i], AGN.double(), dim_param)
    
    X0 = torch.tensor(X0)
    
    print('Starting Euler', flush=True)
    X = euler_GPU_2(X0, sigmas, F_lscpe, lands_param, dt, Nsteps, steps, Nconditions)
    
    X = X.reshape(ncpu*Nconditions*Ncells*Nmeasure, 2)
      
    assignement = clustering.KMeans_Clustering(X.cuda(), centers.cuda())
    assignement = assignement.reshape(ncpu, Nconditions, Ncells, Nmeasure)
    assignement = assignement.cpu()
    assignement = assignement.numpy()
    
    end = time.perf_counter()
    
    print('time elapsed GPU: {:.2f}'.format(end-start))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("batch_theta", type=str)
    args = parser.parse_args()
    param = torch.load(args.batch_theta)
    run_gpu(param)
    