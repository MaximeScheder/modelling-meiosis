# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 13:14:50 2022

@author: scheder

Take home message :
    The process are run much faster using the GPU. The multi-process on CPU is
    only able to simualte 12 particle at a time without beeing to slow. In 
    comparison, the GPU is able to simulate 1000 of particles in 100 seconds.
    
    What could be possible is maybe to paralllize and use GPU. But I do not
    know if it would help at all. It is allready much more efficient than what 
    it used to be...
"""

import numpy as np
import torch
import torch.multiprocessing as mp
import simulation
import time
import clustering
from scipy.stats import multivariate_normal
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from concurrent.futures import ProcessPoolExecutor
import subprocess
import os
from gmm import GaussianMixture

def cusp_gpu(x, y, parameter):
    p = parameter
    if p.ndim == 1:
        p = p.reshape(1, -1)
    return -torch.stack([4*x**3 - 2*x/2 + p[:,:,0], y], dim=1)

def cusp(x, y, parameter):
    p = parameter
    if p.ndim == 1:
        p = p.reshape(1, -1)
    return -np.stack([4*x**3 - 2*x/2 + p[:,:,0], y])

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
    
    return (lands_parm, sigma)



def fitpam_to_landscape_mp(queue, fit_params, AGN, dim_param):
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
    
    queue.put((lands_parm.reshape(-1), sigma))
    #return (lands_parm, sigma)

def euler(X0, sigma, F, parameters, dt, Nsteps, steps, Nconditions):
    """
    Do euler Step for stochastic simulation
    param:
        X0 : (Nconditions, Ncells, 2) array of inital values
        sigma : noise parameter (float)
        F : Field function of the form F(x, y, param)
        parameters
        dt : time step size
        N : Number of steps to do
        Nconditions : number of different media uesed
        steps : list of the step position at which to save the result
        
    return:
        Xem: (Nconditions, Ncells,2,N) array of time evolution
    """
    Ncells = X0.shape[1]
    dim_param = parameters.shape[1]
    
    # noise term
    dW = np.sqrt(dt)*np.random.normal(0, 1, size=(Nconditions*Ncells, 2 ,Nsteps))
    
    # results
    Xem = np.zeros((Nconditions, Ncells, len(steps), 2))
    
    Xtemp = np.zeros((Nconditions, Ncells, 2, 2))
    Xtemp[:,:,0, 0] = X0[:,:,0]
    Xtemp[:,:,0, 1] = X0[:,:,1]
    Xtemp = Xtemp.reshape(Ncells*Nconditions, 2, 2) # put all condition queue to queue
    
    # allign parameters for direct computation of all conditions in one go
    p = np.ones((Ncells, Nconditions, dim_param))*parameters
    p = p.transpose((1,0,2)).reshape((Ncells*Nconditions, dim_param)) # creat a matrix with same number of lines as Xtemp with corresponding conditions
    
    
    k = 1
    for i in range(Nsteps):
        Field = F(Xtemp[:,0,0], Xtemp[:,0,1], p).squeeze().T
        Xtemp[:,1,:] = Xtemp[:,0,:].squeeze() + dt * Field + sigma * dW[:,:,i].squeeze()
        Xtemp[:,0,:] = Xtemp[:,1,:]
        
        #list of steps at which to save the results 
        if i+1 in steps:
            Xem[:,:,k-1,:] = Xtemp[:,1,:].reshape(Nconditions, Ncells, 2) # Need to correct this
            k += 1
            
    return Xem

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
    
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    print(f'total    : {info.total}')
    print(f'\nfree     : {info.free}')
    print(f'used     : {info.used}') 
    
    
    Ncells = X0.shape[1]
    dim_param = parameters.shape[-1]
    
    dt = torch.tensor(dt)
    mean = torch.tensor([0.0])
    std = torch.tensor([1.0])   
    
    Xem = torch.zeros((nbatch, Nconditions, Ncells, len(steps), 2)).cuda()
    Xtemp = torch.zeros((nbatch, Nconditions, Ncells, 2, 2)).cuda()
    p = (torch.ones((nbatch, Nconditions, Ncells, dim_param))*parameters[:,:,np.newaxis,:])
    sigma = torch.tensor(sigma.reshape((nbatch, 1, 1, 1)))
    
    if torch.cuda.is_available():
        dt = dt.cuda()
        mean = mean.cuda()
        std = std.cuda()
        X0 = X0.cuda()
        Xem = Xem.cuda()
        Xtemp = Xtemp.cuda()
        p = p.cuda()
        sigma = sigma.cuda()
    else:
        print("The computation are runing on the CPU hence much slower than initially deseigned")
        
        info = nvmlDeviceGetMemoryInfo(h)
        print(f'\nfree     : {info.free}')
        print(f'used     : {info.used}') 
    
    # noise term
    dW = torch.sqrt(dt)*torch.distributions.normal.Normal(mean, std).rsample((nbatch, Ncells*Nconditions, 2, Nsteps))
    
    info = nvmlDeviceGetMemoryInfo(h)
    print(f'\nfree     : {info.free}')
    print(f'used     : {info.used}') 
    # results
    Xtemp[:,:,:,0, 0] += X0[:,:,0]
    Xtemp[:,:,:,0, 1] += X0[:,:,1]
    Xtemp = Xtemp.reshape(nbatch, Ncells*Nconditions, 2, 2) # put all condition queue to queue
    
    # allign parameters for direct computation of all conditions in one go
    p = p.reshape((nbatch, Ncells*Nconditions, dim_param)) 
        
    k = 1
    for i in range(Nsteps):
        Field = F(Xtemp[:,:,0,0], Xtemp[:,:,0,1], p).transpose(1,2)
        Xtemp[:,:,1,:] = Xtemp[:,:,0,:] + dt * Field + (sigma * dW[:,:,:,i]).squeeze(-1)
        Xtemp[:,:,0,:] = Xtemp[:,:,1,:]
        
        #list of steps at which to save the results 
        if i+1 in steps:
            Xem[:,:,:,k-1,:] = Xtemp[:,:,1,:].reshape(nbatch, Nconditions, Ncells, 2) # Need to correct this
            k += 1
            
            info = nvmlDeviceGetMemoryInfo(h)
            print(f'\nfree     : {info.free}')
            print(f'used     : {info.used}') 
            
    return Xem

def euler_GPU_2(X0, sigma, F, parameters, dt, Nsteps, steps, Nconditions):
    
    Ncells = X0.shape[1]
    dim_param = parameters.shape[-1]
    nbatch = parameters.shape[0]
    
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
            print('Arrived at step {}'.format(k), flush=True)
            T = time.localtime()
            print('\t time : {}:{}:{}'.format(T.tm_hour, T.tm_min, T.tm_sec))
            Xem[:,:,:,k-1,:] = Xtemp[:,:,1,:].reshape(nbatch, Nconditions, Ncells, 2).cpu()
            k += 1
           
    # remove unvessary variables from GPU
    del Xtemp, p, dW
    torch.cuda.empty_cache()
            
    return Xem

def run_cpu():
    simulation_param = np.load("./dict_data.npy", allow_pickle=True).item()

    #Test Eueler
    start = time.perf_counter()
    
    AGN = simulation_param['AGN'].numpy()
    
    Ncells = 300
    dt = 0.01
    dim_param = 1 # without account of sigma
    Nstate = 2 # regular, MII
    time_point = np.array([11, 14, 17, 20, 23, 37, 111])
    steps = time_point/dt
    Nsteps = int(steps[-1])
    Nmeasure = len(time_point)
    radius_tolerance = 0.2 # for the gmm cluster
    F_lscpe = cusp    
    dim_particles = dim_param*(AGN.shape[1]+1) + 1 # for each landscape param, there 3 Nutrient and 1 
    Nconditions = AGN.shape[0]
    X0 = np.hstack((np.random.normal(0, 0.08, (Ncells, 1)), np.random.normal(0.6, 0.05, (Ncells, 1))))
    X0 = X0[np.newaxis,:,:]
    
    centers = torch.Tensor([[0, 0.3], # for the fate_seperate,
                            [0.4, 0]])
    
    ncpu = 8 #mp.cpu_count()
    nbatch = 20
    parameters = np.ones((ncpu, 5))
    parameters[:,-1] *= 0.01
    
    lands_param = []
    
    for i in range(ncpu):
        lands_param.append(fitpam_to_landscape(parameters[i], AGN, dim_param))
            
    

    with mp.Manager() as manager:
        queue = manager.Queue()
        processes = []
        
        for i in range(ncpu):
            process = mp.Process(target=euler, args=(X0, lands_param[i][1], F_lscpe, lands_param[i][0].reshape(1,Nconditions, dim_param), dt, Nsteps, steps, Nconditions),) 
            processes.append(process)
            process.start()
           
            
        for p in processes:
            p.join()
        
        
        i=0
        while not queue.empty(): 
            lands_param.append(queue.get())
    
    #X = euler(X0, sigma, F_lscpe, lands_param, dt, Nsteps, steps, Nconditions)
    
    end = time.perf_counter()
    
    print('time elapsed CPU: {:.2f}'.format(end-start))

    
def run_gpu():
    simulation_param = np.load("./dict_data.npy", allow_pickle=True).item()
    # nvmlInit()

    #Test Eueler
    start = time.perf_counter()
    
    AGN = simulation_param['AGN']
    
    Ncells = 300
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
    
    ncpu = 500 #mp.cpu_count()
    parameters = torch.ones((ncpu, 5)).double()
    
    lands_param = torch.zeros((ncpu, Nconditions, dim_param))
    sigmas = np.zeros((ncpu, 1, 1, 1))
    
    
    for i in range(ncpu):
        lands_param[i], sigmas[i] = fitpam_to_landscape(parameters[i], AGN.double(), dim_param)
    
    X0 = torch.tensor(X0)
    
    X = euler_GPU_2(X0, sigmas, F_lscpe, lands_param, dt, Nsteps, steps, Nconditions)
        
    X = X.reshape(ncpu*Nconditions*Ncells*Nmeasure, 2)
    
    gm = GaussianMixture(2, 2).cuda()
    gm.fit(X.cuda())
    
    # for i in range(ncpu):
    #     assignement = clustering.KMeans_Clustering(X[i,:,:,:].view(Nconditions*Ncells*Nmeasure,2).cuda(), centers.cuda())
        #assignement = assignement.reshape(ncpu, Nconditions, Ncells, Nmeasure)
        #assignement = assignement.cpu()
        #assignement = assignement.numpy()
    
    end = time.perf_counter()
    
    print('time elapsed GPU: {:.2f}'.format(end-start))
    
    
def run_gpu_parallel():
    
    n_process = 10
    nbatch = 100
    start = time.perf_counter() 
    parameters = torch.ones((nbatch, 5)).double()
    
    batch_param = torch.split(parameters, nbatch//n_process)
    
    processes = []
    for i in range(n_process):
        name = './tensors/param-b{}'.format(i)
        torch.save(batch_param[i], name)
        f = open('file_{}.txt'.format(i), 'w')
        p = subprocess.Popen(['python', 'run_simulation_gpu.py', name], stdout=f)
        processes.append((p,f))
        
    for p, f in processes:
        p.wait()
        f.close()
        
    
    end = time.perf_counter()
    
    print('time elapsed GPU: {:.2f}'.format(end-start))
    
from scipy.stats import multinomial
def sample_candidate(epoch, weights, particles, Cov):
    """
    Sample a candidate particle for the ABC_SMC algorithm using the multivariate_OLMC
    
    epoch   : Int - epoch's number, critical if 0
    weights : array(N) the weights with which to draw the old particles
    particles:array(N, dim_particle) the particle to perturbe 
    
    return:
        theta : array(1, dim_particle) A new perturbed particle
    """
    
        
    # draw a particle from the old set with its weights
    index = multinomial.rvs(1, weights)
    index = np.where(index==1)
    theta = particles[index,:].squeeze()
    p = prior(0, 1)
    
    # Kernelize the selected particle
    not_in_prior = True
    while not_in_prior:
        theta_new = np.random.multivariate_normal(theta, Cov[:,:,index].squeeze())
        prior_value = p.evaluate(theta_new)
        if np.all(prior_value > 0): # /!\ wierd form
            not_in_prior = False
            theta = theta_new
    
    return theta, prior_value

# ----------------------------------------------------------------------------
# ---------------------------Definition of Prioirs--------------------------
# ----------------------------------------------------------------------------
   
from scipy.stats import norm
# define the system that allows to generate new particles
class prior():
    
    def __init__(self, a, b):
        self.a, self.b = a, b
    
    def sample(self, Nparam):
        return norm.rvs(self.a, self.b, Nparam)
    
    def evaluate(self, param):
        x = param.reshape(-1)
        return norm.pdf(x, self.a, self.b)
    
    

def Cov_matrix_computation():
    # Cost is marginal with respect to the simulation
    
    N_param = 500
    nbatch = 100
    d = 50
    theta = np.random.normal(size=(nbatch, d))
    p_in = np.random.normal(size=(N_param, d))
    w_in = np.abs(np.random.normal(size=(N_param)))
    w_in = w_in/w_in.sum()
    
    Cov = np.zeros((d, d, N_param))
    for i in range(N_param):
        A = np.random.normal(size=(d,d))
        Cov[:,:,i] = A @ A.T *0.01
        
    start = time.perf_counter()
    
    for i in range(nbatch):
        s = sample_candidate(0, w_in, p_in, Cov)
        
    norm = np.sum(np.stack(
        [w*multivariate_normal.pdf(theta, p, c.squeeze(), allow_singular=True) for w, p, c in zip(w_in, p_in, np.moveaxis(Cov, 2, 0))]
        ), axis=0)
    
    end = time.perf_counter()
    
    
    print('Execution in {:.3f} s'.format(end-start))
    

if __name__ == '__main__':

    # run_cpu()
    run_gpu()
    #run_gpu_parallel()
    
    #Cov_matrix_computation()

