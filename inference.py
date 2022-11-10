# -*- coding: utf-8 -*-
"""
This section regroups all functions that are necessary to perform inference 
with the ABC SMC methode
"""

#------------------
# LYBRARIES
#------------------

import numpy as np
import os
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.uniform import Uniform
from torch.distributions.multinomial import Multinomial

from datetime import datetime

#-------------------
# Functions
#-------------------

def multivariate_OLMC(particles, weights, index, batch=None):
    """
    This function compute and gives the covariance matrix and the mean for
    the optimal kernel based on last results.
    param:
        particles: all particles of the last iteration of dim (N, d) with 
            N number of sample and d the dimension
        weights: weight associated to the corresponding particles of size
            (N)
        index: the set of indexes of parameters that respect a distance lower than
            the next threshold
    
    return:
        Cov     : Tensor(d, d, N) of covariance matrix for each particle
    """
    print("\t Preparing the OLMC Kernel...")
    
    if os.path.isfile("./outputs/Cov-mix-b{}".format(batch)) and batch is not None:
        Cov = torch.load("./outputs/Cov-mix-b{}".format(batch))
    else:
        dim = particles.shape[1]
        N = weights.shape[0]
        diagonale = torch.diag(torch.ones(dim))*1e-4
        if torch.cuda.is_available():
            diagonale = diagonale.cuda()
        
        # Select the components that verify the distance conditions to current epoch
        w = weights[index]
        w = w / torch.sum(w)
        p_select = particles[index]
        Cov = torch.zeros((dim, dim, N))
        if torch.cuda.is_available():
            Cov.cuda()
    
            
        assert w.shape[0] > 0, "There should be particles that are below the current threshold"
        
        for i, p in enumerate(torch.unbind(particles)):
            for k in range(w.shape[0]):
                vec = (p_select[k]-particles[i]).reshape(dim, -1)
                Cov[:,:,i] += w[k]*(vec @ vec.T)
                
            Cov[:,:,i] += diagonale # Add diagonal for stability
        print("\t ...done !")    
    
    return Cov
    

    

def ABC_SMC_step(idx, eps, X, N, prior, 
                 distance, generate, epoch=0, maxiter = 1e6, sim_param=None):
    """
    This function proceed with one step of the ABC_SMC algorithm. This function
    is friendly with CPU multiprocessing in comparison to its GPU version.
    
    Note that if epoch is not 0, the algorithm will load files from "./outputs".
    The correct naming system has to be followed. The algorithm save the results
    in new files in "./outputs"
    
    param:
        idx     : int   - worker id
        eps     : float - current acceptation threshold
        X       : Tensor(Nconditions, Nstate, Nsteps) of ground Truth
        N       : Int   - Number of particles to be sampled
        prior   : fct   - either sample a new candidate particle or evaluate a particle
        generate: fct   - generate a prediction from particles
        epoch   : int   - current epoch
        maxiter : int   - maximum allowed number of iterations for one sample
        sim_param: dic  - Contain all parameters relevant to simulation (passed in generate)
        
        
        return:
            
    """
    dim_particles = sim_param["dim_particles"]

    # Initializing parameters
    p_out = torch.zeros((N, dim_particles))
    w_out = torch.ones(N)/N
    theta = torch.zeros((1, dim_particles))
    
    # Load old results if not first stage
    if epoch != 0:
        p_in = torch.load("./outputs/particles-mix-b{}".format(idx))
        w_in = torch.load("./outputs/weights-mix-b{}".format(idx))
        d = torch.load("./outputs/distances-mix-b{}".format(idx))
    else:
        p_in = torch.zeros(p_out.shape)
        w_in = torch.ones(w_out.shape)/w_out.shape[0]
        d = torch.zeros(w_out.shape)
        
    # Compute the OLMC covariance if not first epoch
    if epoch != 0:
        idd = torch.where(d < eps)
        Cov = multivariate_OLMC(p_in, w_in, idd, idx)
    
    niter = 1
    i = 0
    while i < N:
                
        if niter == maxiter:
            return
        
        # sample a proposal
        if epoch == 0:
            theta = prior(prior_parameters = sim_param["prior"])
            
        else:
            # draw a particle from the old set with its weights
            index = Multinomial(probs=w_in).sample()
            index = torch.where(index==1)
            theta = p_in[index,:].squeeze()
            
            # Kernelize the selected particle
            not_in_prior = True
            while not_in_prior:
                theta_new = MultivariateNormal(theta, Cov[:,:,index].squeeze()).sample()
                prior_value = prior(theta_new, sim_param["prior"])
                if prior_value > 0:
                    not_in_prior = False
                    theta = theta_new
                    
            
        # generate the corresponding predictions
        if sim_param is None:
            Y = generate(N, theta)
        else:
            Y = generate(N, theta, sim_param)
        # compute distance to ground truth
        dist = distance(X, Y)
            
        if dist <= eps:
            p_out[i,:] = theta.reshape(-1)
            d[i] = dist
            
            if epoch != 0:
            # computing the corresponding weights
                norm = torch.sum(torch.stack([w*torch.exp(MultivariateNormal(
                        p, c.squeeze()).log_prob(theta))
                    for w, p, c in zip(torch.unbind(w_in), torch.unbind(p_in), torch.unbind(Cov, dim=-1))]))
                    
                w_out[i] = prior_value/norm
            
            
            with open("./consol/follow-b{}-e{}.txt".format(idx, epoch), "a") as file:
                if os.path.exists("./consol/follow-b{}-e{}.txt".format(idx, epoch)):
                    file.write("Sample {}/{} : \n".format(i+1, N))
                    file.write("\t niter = {} \n".format(niter))
                    now = datetime.now()
                    current_time = now.strftime("%H:%M:%S")
                    file.write("\t Time : {} \n".format(current_time))
            
            i += 1
            niter = 0
        niter += 1
            
    # Normalizing weights
    w_out = w_out/torch.sum(w_out)
    torch.save(p_out, "./outputs/particles-b{}-e{}".format(idx, epoch))
    torch.save(w_out, "./outputs/weights-b{}-e{}".format(idx, epoch))
    torch.save(d, "./outputs/distances-b{}-e{}".format(idx, epoch)) 
        
    
# Definition which does not rely on saving results

def ABC_SMC_step_GPU(p_in, p_out, theta, w_in, w_out, d, eps, X, Y, N, prior, 
                 distance, generate, epoch=0, maxiter = 1e6, sim_param=None):
    """
    This function proceed with one step of the ABC_SMC algorithm
    param:
        p_in : (N, dim) array of parmeters from the last epoch
        p_out : (N, dim) array of the outputs parameters
        theta : (1, dim) array of one unique parmeter
        w_in : (N) array of the weights of last epoch
        w_out : (N) array of the future weights of current epoch
        d : (N) array containing the distance of the samples to data
        Cov : (dim, dim, N) array containing the covariance of each particle
        eps: current error threshold for distribution
        X : ground truth
        Y : Prediction with current parameters
        N : number of particles per batch
        prior : function that either return proba or sample new elements
        distance : function that compute distance between ground truth X and an estimate
        idx : id of the current batch
        epoch : epoch considered
        maxiter : maximum number of iterations
        
        return:
            The algorithm modifiy the tensor it is given directly.
        
    """
    
    if epoch != 0:
        idd = torch.where(d < eps)
        Cov = multivariate_OLMC(p_in, w_in, idd)
    
    niter = 1
    i = 0
    while i < N:
                
        if niter == maxiter:
            return
        
        if epoch == 0:
            theta = prior(prior_parameters = sim_param["prior"])
            
        else:
            # draw a particle from the old set with its weights
            index = Multinomial(probs=w_in).sample()
            index = torch.where(index==1)
            theta = p_in[index,:].squeeze()
            
            # Kernelize the selected particle
            not_in_prior = True
            while not_in_prior:
                # dumy variable checking that we are in the prior
                theta_new = MultivariateNormal(theta, Cov[:,:,index].squeeze()).sample()
                prior_value = prior(theta_new, sim_param["prior"])
                if prior_value > 0:
                    not_in_prior = False
                    theta = theta_new
                    
            
        # generate the corresponding predictions
        if sim_param is None:
            Y = generate(N, theta)
        else:
            Y = generate(N, theta, sim_param)
        # compute distance to ground truth
        dist = distance(X, Y)
            
        if dist <= eps:
            p_out[i,:] = theta.reshape(-1)
            d[i] = dist
            
            if epoch != 0:
            # computing the corresponding weights
                norm = torch.sum(torch.stack([w*torch.exp(MultivariateNormal(
                        p, c.squeeze()).log_prob(theta))
                    for w, p, c in zip(torch.unbind(w_in), torch.unbind(p_in), torch.unbind(Cov, dim=-1))]))
                    
                w_out[i] = prior_value/norm
            
            with open("./consol/follow-e{}.txt".format( epoch), "a") as file:
                
                if os.path.exists("./consol/follow-e{}.txt".format( epoch)):
                    file.write("Sample {}/{} : \n".format(i+1, N))
                    file.write("\t niter = {} \n".format(niter))
                    now = datetime.now()
                    current_time = now.strftime("%H:%M:%S")
                    file.write("\t Time : {} \n".format(current_time))
            
            i += 1
            niter = 0
        niter += 1
            
    # Normalizing weights
    w_out = w_out/torch.sum(w_out)
    
# ----------------------------------------------------------------------------
# ---------------------------Definition of distances--------------------------
# ----------------------------------------------------------------------------


        
# ----------------------------------------------------------------------------
# ---------------------------Definition of Prioirs--------------------------
# ----------------------------------------------------------------------------
   
def prior_toy(theta=None):
    a = -50
    b = 50
        
    if theta is None:
        return Uniform(a, b).sample(torch.Size([2, 1]))
    
    else:
        return torch.prod(Uniform(a,b).log_prob(theta))     
    

def prior_cusp(theta=None, prior_parameters = None):
    """
    Create samples for the inference or verify that they are into it. 
    
    param :
        theta   : tensor(d, 1)
        prior_param : dic - compile the information necessary to sample 
    """
    
    if prior_parameters is None:
        TypeError("prior_parameters have to be defined - {} - was given".format(type(prior_parameters())))
    
    if theta is None:
        maps_param = torch.cat([dist.sample(torch.Size([1, 1])) for i, dist in enumerate(prior_parameters["distribution"])])
        return maps_param
    
    else:
        # Check the validity of tensor
        
        out = torch.any((prior_parameters["left"].T > theta) + (theta > prior_parameters["right"].T))
        if not out: 
            prob = torch.sum(torch.cat([dist.log_prob(theta[i]).view(1, 1) for i, dist in enumerate(prior_parameters["distribution"])]))
            
            if prob < -100:
                return prior_parameters["minimal_proab"]
            else:
                return torch.exp(prob)
        else:
            return 0
        
        
    
    
    