# -*- coding: utf-8 -*-
"""
This script will do inference on a simple test-landscape, in order to see
if the computer is able to handle it. 
"""

import landscapes
import simulation
import inference
import numpy as np
import multiprocessing as mp
import time
import torch
import gmm
import torch.distributions.uniform.Uniform as unif


def initializer():
    """
    This function create a dictionary with all the necessary variables to 
    simulate a landscape.
    
    Note that initialization is more important when working on the GPU without
    multiprocessing. Initializing tensors before reduce a bit the computation.
    """
        
    # Constants and simulation parameters
    
    Nsamples = 240
    if not torch.cuda.is_available():
        # Seperate the number of samples into the available batches
        Nsamples = Nsamples // mp.cpu_count()
    
    Ncells = 100
    dt = torch.Tensor([0.01])
    Nsteps = 800 # So that it simulates 8h at each iterations
    maxiter = 1e6
    dim_param = 2 # without account of sigma
    Nstate = 3 # left / transitionning / right
    Nmeasure = 3
    radius_tolerance = 0.2 # for the gmm cluster

    #-----------------------------------------------------------
    #------------------ MAP AND CONDITIONS ---------------------
    #-----------------------------------------------------------
    
    mapp = landscapes.cusp
    AGN = torch.Tensor([[1, 1, 1, 1],
                        [0, 0, 1, 0],
                        [0, 0, 0, 0]])
    # AGN = torch.Tensor([[1, 1, 1, 1],
    #                     [1, 0, 0, 0],
    #                     [0, 1, 0, 0],
    #                     [0, 0, 1, 0],
    #                     [0, 0, 0, 1],
    #                     [0, 0, 0, 0]])  
    
    dim_particles = dim_param*(AGN.shape[1]) + 1
    Nconditions = AGN.shape[0]
    
    #-----------------------------------------------------------
    #----------------- GMM AND CENTERS -------------------------
    #-----------------------------------------------------------
    
    # Need a [1, k, d] shape for the gmm algorithm
    centers = torch.stack([torch.Tensor([[-0.8, 0],
                                  [0, 0],
                                  [0.8, 0]])]).double()

    model = gmm.GaussianMixture(Nstate, 2, mu_init=centers, r=radius_tolerance)
    
    #-----------------------------------------------------------
    #-------------- EULER AND INITAL CONDITIONS ----------------
    #-----------------------------------------------------------
    
    X0 = torch.ones((Nconditions, Ncells, 2)) * torch.Tensor([[-0.8, 0]]) 
    X = torch.ones((Nconditions, Ncells, Nmeasure, 2)) * torch.Tensor([[-0.8, 0]])
    Xtemp = torch.zeros((Nconditions, Ncells, 2, 2))
    p_mp = torch.zeros((Ncells, Nconditions, dim_param))
    
    #------------------------------------------------------------
    #---------------- INFERENCE PARAMETERS ----------------------
    #------------------------------------------------------------
    
    p_out = torch.zeros((Nsamples, dim_particles))
    w_out = torch.ones(Nsamples)/Nsamples
    theta = torch.zeros((1, p_out.shape[1]))
    Cov = torch.zeros((dim_particles, dim_particles, Nsamples))
    Y = torch.zeros((Nconditions, Nstate, Nmeasure))
    
    
    #-----------------------------------------------
    #------------------ PRIOR ----------------------
    #-----------------------------------------------
    
    # component for the prior
    a0, aa, ag, an = torch.Tensor([-2, -5, -8, -8])
    b0, ba, bg, bn = torch.Tensor([0.5, 0, 8, 8])
    a0_2, b0_2 = torch.Tensor([-2, 0])
    av0, ava, avg, avn =  torch.Tensor([0,-2, -4, -4])
    bv0, bva, bvg, bvn = torch.Tensor([1, 2, 4, 4])
    
    noise_a, noise_b = torch.Tensor([0, 0.1])
    minimal_proba = torch.exp(torch.Tensor([-100]))
    
    if torch.cuda.is_available():
        a0, aa, ag, an = a0.cuda(), aa.cuda(), ag.cuda(), an.cuda()
        b0, ba, bg, bn = b0.cuda(), ba.cuda(), bg.cuda(), bn.cuda()
        av0, ava, avg, avn = av0.cuda(), ava.cuda(), avg.cuda(), avn.cuda()
        bv0, bva, bvg, bvn = bv0.cuda(), bva.cuda(), bvg.cuda(), bvn.cuda()
        a0_2, b0_2 = a0_2.cuda(), b0_2.cuda()
        noise_a = noise_a.cuda()
        noise_b = noise_b.cuda()
        minimal_proba = minimal_proba.cuda()
    
    prior_map = [unif(a0, b0), unif(aa, ba), unif(ag, bg), unif(an, bn),
                 unif(a0_2, b0_2), unif(aa, ba), unif(ag, bg), unif(an, bn),
                 unif(av0, bv0), unif(ava, bva), unif(avg, bvg), unif(avn, bvn),
                 unif(av0, bv0), unif(ava, bva), unif(avg, bvg), unif(avn, bvn)]
    prior_noise = torch.distributions.uniform.Uniform(noise_a, noise_b)
    prior = {"distribution":[prior_map, prior_noise], "minimal_proba":minimal_proba}
    
    # Components for the Euleur algorithm
    if torch.cuda.is_available():
        dW = torch.distributions.normal.Normal(torch.Tensor([0]).cuda(), 
                                               torch.Tensor([1]).cuda())
    else:
        dW = torch.distributions.normal.Normal(torch.Tensor([0]), 
                                               torch.Tensor([1]))
        
    
    sim_param = { "Nsamples":Nsamples, "Ncells":Ncells, "dt":dt, "Nsteps":Nsteps,
                  "maxiter":maxiter, "dim_param":dim_param, "Nstate":Nstate,
                  "Nmeasure":Nmeasure, "centers":centers,
                  "mapp":mapp, "AGN":AGN, "Nconditions":Nconditions,"X0":X0, "X":X,
                  "Xtemp":Xtemp, "p_out":p_out, "theta":theta, "w_out":w_out,
                  "Cov":Cov,"Y":Y,"prior":prior, "dW":dW, "p_mp":p_mp, "model":model,
                  "dim_particles":dim_particles}

    
    # send to GPU if available
    if torch.cuda.is_available():
        #sim_param["gmm"] = sim_param["gmm"].cuda()
        for key in sim_param.keys():
            if type(sim_param[key]) is torch.Tensor:
                sim_param[key] = sim_param[key].cuda()
            
    
    return sim_param
        

def distance(R1, R2):
    return torch.sum(torch.abs(R1-R2))

def generate_GPU(N, parameters, sim_param):
    return simulation.generate_yeasts_fate_landscape_GPU(sim_param["AGN"], parameters, sim_param["X0"], sim_param["Y"],
                                              sim_param["mapp"], sim_param["Nstate"], sim_param["Nmeasure"], sim_param["model"],
                                              sim_param["dt"], sim_param["Nsteps"], sim_param["dim_param"],
                                              sim_param["X"], sim_param["Xtemp"],
                                              sim_param["dW"], sim_param["p_mp"])

def generate_CPU(N, parameters, sim_param):
        return simulation.generate_yeasts_fate_landscape_CPU(sim_param["AGN"], parameters, sim_param["X0"],
                                              sim_param["mapp"], sim_param["Nstate"], sim_param["Nmeasure"],
                                              sim_param["dt"], sim_param["Nsteps"], sim_param["dim_param"],
                                              sim_param["dW"], sim_param["centers"])

def mix_samples(nbatches, eps, epoch):
    """
    Mix the particles obtained at the last iterations so that
        1. One avoid a batch getting stuck in a non-optimal region
        2. Ensure that there are particles to compute the OLMC
    
    parameters:
        nbatches    : float - number of data set to create (according to cpu_count)
        eps         : float - current threshold for selection
        epoch       : Int   - current epoch for save file naming
        
    return:
        It does not return anything, but it saves file in "./outputs"
    """
    particles = torch.load("./outputs/particles-e{}".format(epoch))
    weights = torch.load("./outputs/weights-e{}".format(epoch))
    distances = torch.load("./outputs/distances-e{}".format(epoch))
    bad = torch.where(distances > eps)
    good = torch.where(distances <= eps)
    
    nper_batch = weights.shape[0]//nbatches
    assert torch.sum(distances <= eps) >= nbatches, "Not enough particules with distance smaller than next step !"

    
    good_p = particles[good]
    good_w = weights[good]
    good_d = distances[good]
    
    bad_p = particles[bad]
    bad_w = weights[bad]
    bad_d = distances[bad]
    
    id_good = torch.randperm(good_w.shape[0])
    id_bad = torch.randperm(bad_w.shape[0])

    
    batch_p = torch.split(particles, nper_batch, dim=0)
    batch_w = torch.split(weights, nper_batch)
    batch_d = torch.split(distances, nper_batch)
    
    N1 = good_w.shape[0]
    N2 = bad_w.shape[0]
    b = 0
    j = 0
    for i in range(N1 + N2):
        
        if i < N1:
            batch_p[b, j] = good_p[id_good[i]]
            batch_w[b, j] = good_w[id_good[i]]
            batch_d[b, j] = good_d[id_good[i]]
        else:
            batch_p[b, j] = bad_p[id_bad[i-N1]]
            batch_w[b, j] = bad_w[id_bad[i-N1]]
            batch_d[b, j] = bad_d[id_bad[i-N1]]
            
        b += 1
        if b == nbatches:
            b = 0
            j += 1
    
    for i in range(nbatches):
        torch.save(batch_p[i], "./outputs/particles-mix-b{}".format(i))
        torch.save(batch_w[i]/torch.sum(batch_w[i]), "./outputs/weights-mix-b{}".format(i))
        torch.save(batch_d[i], "./outputs/distances-mix-b{}".format(i))
    
    

def Create_R(true_param, sim_param):
    R = torch.zeros((sim_param["Nconditions"], sim_param["Nstate"], sim_param["Nmeasure"]))
    R = simulation.generate_yeasts_fate_landscape(sim_param["AGN"], true_param, sim_param["X0"], R,
                                              sim_param["mapp"], sim_param["Nstate"], sim_param["Nmeasure"], sim_param["model"],
                                              sim_param["dt"], sim_param["Nsteps"], sim_param["dim_param"],
                                              sim_param["X"], sim_param["Xtemp"],
                                              sim_param["dW"], sim_param["p_mp"])
    return R


def fitting():
    """
    Main function that call all necessary parts to perform inference.
    """
    # Simulation parameters
    Multiple_exp = False
    # epsilons = [2.1, 1.2, 0.44, 0.31, 0.26, 0.187, 0.144]
    epsilons = [7, 5.82,  4.5 , 4.05, 3.66, 3.37, 3.18]
    
    if Multiple_exp:
        begin = 3
        finish = begin + 1
    else:
        begin = len(epsilons)-1
        finish = len(epsilons)
    
    prior = inference.prior_cusp
    
    print("Initializing simulation parameters...")
    sim_param = initializer()

    # Generating a dataset
    # p = torch.Tensor([[-1, 0, 0, 0, -0.3, 0, 0, 0, 0.3]])
    # Create_R(p, sim_param)
    
    # Load ground truth
    R = torch.from_numpy(np.load("./R_test.npy"))
    
    # Remove old consol outputs
    import os, os.path
    mypath = "./consol"
    for root, dirs, files in os.walk(mypath):
        for file in files:
            os.remove(os.path.join(root, file))
            
    
    # Initialize the matrices necessary for outputs of inference
    if begin == 0:
        p_in = torch.zeros(sim_param["p_out"].shape)
        w_in = torch.zeros(sim_param["w_out"].shape)
        d = torch.zeros(sim_param["w_out"].shape)
    else:
        p_in = torch.load("./outputs/particles-e{}".format(begin-1))
        w_in = torch.load("./outputs/weights-e{}".format(begin-1))
        d = torch.load("./outputs/distances-e{}".format(begin-1))
        
    # Check if a GPU is available and send vector to GPU id yes
    device = "CPU"  
    if torch.cuda.is_available():
        d = d.cuda()
        w_in = w_in.cuda()
        p_in = p_in.cuda()
        R = R.cuda()
        device = "GPU"
        
    print("\t ... done ! \n")    
    
    
    if torch.cuda.is_available():
        print("\t Total samples    : {}".format(sim_param["Nsamples"]))
    else:
        print("\t Total samples    : {}".format(sim_param["Nsamples"]*mp.cpu_count()))
    
    print("\t Simulated cells  : {}".format(sim_param["Ncells"]))
    print("\t Total conditions : {}".format(sim_param["Nconditions"]))
    print("\t Working on {} \n".format(device))
    
    for t in range(begin, finish):
        print("Starting epoch {} with threshold {}".format(t, epsilons[t]))
        start = time.perf_counter()
        
        if device == "GPU":
            inference.ABC_SMC_step_GPU(
                                         p_in, sim_param["p_out"], sim_param["theta"],
                                         w_in, sim_param["w_out"], d,
                                         epsilons[t], R, sim_param["Y"], sim_param["Nsamples"], prior,
                                         distance, generate_GPU, t, sim_param["maxiter"], sim_param)
        
        elif device == "CPU":
            worker = mp.cpu_count()
            processes = []
            
            if t > 0:
                mix_samples(worker, epsilons[t], t-1)
            
            print("\t starting inference with {} workers...".format(worker)) 
            for i in range(worker):
               
                
                    
                processes.append(mp.Process(target=inference.ABC_SMC_step, args=(i,
                                          epsilons[t], R, sim_param["Nsamples"], prior,
                                          distance, generate_CPU, t, sim_param["maxiter"], sim_param, )))
    
            for p in processes:
                p.start()
                
            for p in processes:
                p.join()
                
             
            sim_param["p_out"] = torch.concat([torch.load("./outputs/particles-b{}-e{}".format(i, t)) for i in range(worker)], dim=0)
            sim_param["w_out"] = torch.concat([torch.load("./outputs/weights-b{}-e{}".format(i, t)) for i in range(worker)])  
            d = torch.concat([torch.load("./outputs/distances-b{}-e{}".format(i, t)) for i in range(worker)])
            
        end = time.perf_counter()
        
        
        print("\t Finished after {}".format(end-start))
    
    print("Proposition of threshold based on distances distribution")
    print("\t With 0.75 quantile : {:.5f}".format(torch.quantile(d, 0.75)))
    print("\t With 0.50 quantile : {:.5f}".format(torch.quantile(d, 0.50)))
    print("\t With 0.25 quantile : {:.5f}".format(torch.quantile(d, 0.25)))
    print("\t With 0.15 quantile : {:.5f}".format(torch.quantile(d, 0.15)))
    
        
    print("Saving final results")
    torch.save(sim_param["p_out"], "./outputs/particles-e{}".format(t))
    torch.save(sim_param["w_out"], "./outputs/weights-e{}".format(t))
    torch.save(d, "./outputs/distances-e{}".format(t))


if __name__ == '__main__':
    fitting()