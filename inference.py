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
from scipy.stats import multivariate_normal
from datetime import datetime
import simulation
import time

#-------------------
# Functions
#-------------------

def multivariate_OLMC(particles, weights, index):
    """
    This function compute and gives the covariance matrix and the mean for
    the optimal kernel based on last iteration.
    param:
        particles: all particles of the last iteration of dim (N, d) with 
            N number of sample and d the dimension
        weights: weight associated to the corresponding particles of size
            (N)
        index: indicies of parameters that have a distance lower than
            the next threshold
    
    return:
        Cov     : Tensor(d, d, N) of covariance matrix for each particle
    """
    
    print("- Preparing the OLMC Kernel\n")

    dim = particles.shape[1]
    N = weights.shape[0]
    diagonale = np.diag(np.ones(dim))*1e-4
    
    # Select the components that verify the distance conditions to current epoch
    w = weights[index]
    w = w / np.sum(w)
    p_select = particles[index]
    Cov = np.zeros((dim, dim, N))
        
    assert w.shape[0] > 0, "There should be particles that are below the current threshold"
    
    for i in range(particles.shape[0]):
        for k in range(w.shape[0]):
            vec = (p_select[k]-particles[i]).reshape(dim, -1)
            Cov[:,:,i] += w[k]*(vec @ vec.T)
            
        Cov[:,:,i] += diagonale # Add diagonal for stability
            
    return Cov


def sample_candidate(epoch, weights, particles, Cov, prior):
    """
    Sample a candidate particle for the ABC_SMC algorithm using the multivariate_OLMC
    
    epoch   : Int - epoch's number, critical if 0
    weights : array(N) the weights with which to draw the old particles
    particles:array(N, dim_particle) the particle to perturbe 
    Cov : Array(d, d, N) of the covariance matrix for each particle
    prior : class of prior to sample and evaluate the candidate
    
    return:
        theta : array(1, dim_particle) A new perturbed particle
    """
    
    if epoch == 0:
        theta = prior.sample()
        prior_value = prior.evaluate(theta)
        
    else:
        # draw a particle from the old set with its weights
        index = np.random.multinomial(1, weights)
        index = np.where(index==1)
        theta = particles[index,:].squeeze()
        
        # Kernelize the selected particle
        not_in_prior = True
        while not_in_prior:
            theta_new = np.random.multivariate_normal(theta, Cov[:,:,index].squeeze())
            prior_value = prior.evaluate(theta_new)
            if prior_value > 0: 
                not_in_prior = False
                theta = theta_new
    
    return theta, prior_value
    
    
def ABC_SMC_step_gpu(eps, X, p_in, w_in, d, N, prior, nbatch, epoch=0, maxiter = 1e6):
    """
    This function proceed with one step of the ABC_SMC algorithm. This function
    uses gpu to accelerate the computation and allow simulation of a great number
    of particles at the same time.
    
    param:
        eps     : float - current acceptation threshold
        X       : array(Nconditions, Nstate, Nsteps) of ground Truth
        p_in    : array(M, parameter_dim) of the last iteration 
        w_in    : array(M) weights of the particles from last iteration
        d       : array(M) distance of the particles from last iteration
        N       : Int   - Number of particles to be sampled
        prior   : Obj - class from which to draw and evaluate the candidate particles
        nbatch  : Int - number of particle to simualte at the same time
        epoch   : int - current epoch
        maxiter : int - maximum allowed number of iterations for one sample
        
        
    return:
        p_out : array(N, parameter_dim) - particles accepted for the current thershold
        w_out : array(N) - the weights corresponding to the particles p_out
        d : array(N) - the distance for each particle
            
    """
    
    print("\n|##################|")
    print("| Starting ABC-SMC |")
    print("|##################|")
    print("\nEpoch {}".format(epoch))
    print("Eps {} \n".format(eps), flush=True)
    
    
    # Initializing parameters
    p_out = np.zeros((N, p_in.shape[1]))
    w_out = np.ones(N)/N
    Cov = None
            
    # Compute the OLMC covariance if not first epoch
    if epoch != 0:
        idd = np.where(d < eps)
        Cov = multivariate_OLMC(p_in, w_in, idd)
    
    niter = 1
    i = 0
    total_iter = 1
    while i < N:
                
        if niter == maxiter: # stop simulation if stuck for too long
            return
        
        batch_theta = np.zeros((nbatch, p_in.shape[1]))
        batch_prior_value = np.zeros((nbatch))
        
        if niter == 1:
            print('Number of particles {} - Total iteration {}'.format(i, total_iter))
            print('________________________________________________________________ \n')
        else :
            print('Number of iteration {} - Total iteration {}'.format(niter, total_iter))
            print('__________________ \n')
        
        # draw new particles
        print(" - Sampling {} new particles".format(nbatch))
        
        percent = int(nbatch/20)
        print('\t____________________\n\t', end='', flush=True)
        for b in range(nbatch):
            batch_theta[b], batch_prior_value[b] = sample_candidate(epoch, w_in, p_in, Cov, prior)
            if (b % percent) == 0 and b >= percent:
                print("#", end='',flush=True)
        
        print('#\n')
            
        # Generate the statistics Y to be compared to the ground_truth
        print(" - Simulating the system :", flush=True)
        Y = simulation.generate_fate_gpu(batch_theta)
        
        dist = distance_gpu(Y, X)
        
        print(' - Average distance : {} +/- {}'.format(dist.mean(), dist.std()))
        
        # select the parameters that are lower than the distance, if any
        batch_theta = batch_theta[dist <= eps]
        n_accepted_sample = batch_theta.shape[0]
        
        print('\n - {} particles accepted'.format(n_accepted_sample), flush=True)

        if n_accepted_sample != 0:

            # select the correspoding prior values and distances
            batch_prior_value = batch_prior_value[dist <= eps]
            dist = dist[dist <= eps]
            
            print('\t - Distances : {}'.format(dist))
            
            
            if epoch != 0:
                print(' - Computing new weights', flush=True)
                start = time.perf_counter()
                
                # computing the corresponding weights for the accepted samples
                norm = np.sum(np.stack(
                    [w*multivariate_normal.pdf(batch_theta, p, c.squeeze()) for w, p, c in zip(w_in, p_in, np.moveaxis(Cov, 2, 0))]
                    ), axis=0)
        
                
                end = time.perf_counter()
                print('\t -- time elapsed for computation of weights : {:.2f} seconds'.format(end-start), flush=True)
                    
            for k in range(n_accepted_sample):
                if i < N:
                    if epoch != 0 and n_accepted_sample > 1:
                        w_out[i] = batch_prior_value[k]/norm[k]
                    elif epoch != 0:
                        w_out[i] = batch_prior_value / norm
                        
                    p_out[i,:] = batch_theta[k].reshape(-1)
                    d[i] = dist[k]
                    i += 1
                
            niter = 0
            
            print(' - {} / {} particles found \n'.format(i, N), flush=True)
        
        else:
            print('')
            
        niter += 1
        total_iter += 1
    
    print('\nEpoch {} finished'.format(epoch), flush=True)
    
    w_out = w_out/np.sum(w_out)
    
    return (p_out, w_out, d)
    
    
# ----------------------------------------------------------------------------
# ---------------------------Definition of distances--------------------------
# ----------------------------------------------------------------------------

def distance(R1, R2):
    return np.sum(np.abs(R1-R2))

def distance_gpu(Rgpu, R):
    return np.sum(np.abs(Rgpu-R), axis=(1,2,3))



        
        
    
    
    