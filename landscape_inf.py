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
import pandas as pd
from torch.distributions.uniform import Uniform as unif


def initializer():
    """
    This function create a dictionary with all the necessary variables to 
    simulate a landscape.
    
    Note that initialization is more important when working on the GPU without
    multiprocessing. Initializing tensors before reduce a bit the computation.
    """
        
    # Constants and simulation parameters
    data = pd.read_excel("./conditions.xlsx")
    #data = data.drop([9, 11, 20, 16, 15])
    #data = data.drop([0, 1, 2, 3, 8, 9, 10, 11, 12, 20, 16, 15, 17, 18, 21, 23, 24, 25])
    #data = data.drop([0, 1, 2, 3, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
    AGN = torch.from_numpy(np.hstack([np.ones((data["Acetate"].shape[0], 1)), data[["Acetate", "Glucose" ,"Nitrogen"]].values])).float()
    AGN = AGN * torch.Tensor([1, 1, 2, 5])
    AGN = AGN[:,1:]
    AGN[:,1:] = torch.log(AGN[:,1:])
    
    Nsamples = 400
    if not torch.cuda.is_available():
        # Seperate the number of samples into the available batches
        Nsamples = Nsamples // mp.cpu_count()
    
    Ncells = 200
    dt = torch.Tensor([0.05])
    Nsteps = 960 # So that it simulates 8h at each iterations
    maxiter = 1e4
    dim_param = 4 # without account of sigma
    Nstate = 3 # G / MI / MII
    Nmeasure = 1
    radius_tolerance = 0.2 # for the gmm cluster

    #-----------------------------------------------------------
    #------------------ MAP AND CONDITIONS ---------------------
    #-----------------------------------------------------------
    
    mapp = landscapes.fate_seperate    

    
    dim_particles = dim_param*(AGN.shape[1]) + 1
    Nconditions = AGN.shape[0]
    
    #-----------------------------------------------------------
    #----------------- GMM AND CENTERS -------------------------
    #-----------------------------------------------------------
    
    # Need a [1, k, d] shape for the gmm algorithm
    # centers = torch.Tensor([[-1.4, 0], double cusp
    #                         [-0.2, 0],
    #                         [1, 0]])
    
    centers = torch.Tensor([[0, 0.3], # for the fate_seperate
                            [-0.4, 0],
                            [0.4, 0]])

    # model = gmm.GaussianMixture(Nstate, 2, mu_init=centers, r=radius_tolerance)
    model = None
    
    #-----------------------------------------------------------
    #-------------- EULER AND INITAL CONDITIONS ----------------
    #-----------------------------------------------------------
    
    #X0 = torch.ones((Nconditions, Ncells, 2)) * torch.Tensor([[-1.4, 0]]) 
    X0 = torch.load("./X0")
    X = torch.ones((Nconditions, Ncells, Nmeasure, 2)) * torch.Tensor([[-1.4, 0]])
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
    
    # component for the prior in usual case
    left_born = torch.Tensor([-2, -1, -1, -2, -2, -2, 0, -1, -1, 0, -1, -1, 0]).reshape(-1, 1)
    right_born = torch.Tensor([1,  1,  1,  2,  2,  2, 1,  1,  1, 1,  1,  1, 0.1]).reshape(-1, 1)
    
    
    # a0, aa, ag, an = torch.Tensor([-2, -5, -8, -8])
    # b0, ba, bg, bn = torch.Tensor([0.5, 0, 8, 8])
    # a0_2, b0_2 = torch.Tensor([-2, 0])
    # av0, ava, avg, avn =  torch.Tensor([0,-2, -4, -4])
    # bv0, bva, bvg, bvn = torch.Tensor([1, 2, 4, 4])
    
    #Version with scaled data
    # left_born = torch.Tensor([-2, -2, -2, -2, -2, -2, -2, -2, 0, -2, -2, -2, 0,-2, -2, -2, 0]).reshape(-1, 1)
    # right_born = torch.Tensor([0.5, 0.5, 2, 2, 0, 0, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 0.1]).reshape(-1, 1)
    # a0, aa, ag, an = torch.Tensor([-2, -2, -2, -2])
    # b0, ba, bg, bn = torch.Tensor([0.5, 0.5, 2, 2])
    # a0_2, b0_2 = torch.Tensor([-2, 0])
    # av0, ava, avg, avn =  torch.Tensor([0,-2, -2, -2])
    # bv0, bva, bvg, bvn = torch.Tensor([1, 2, 2, 2])
    
    # left_born = torch.Tensor([  -1, -1, -1, -1, -1, -1, -1,-1, 0, -1, -1, -1, 0]).reshape(-1, 1)
    # right_born = torch.Tensor([0.5,  0,  1,  1,  1,  1,  1, 1, 1,  1,  1,  1, 0.1]).reshape(-1, 1)
    
    
    
    # noise_a, noise_b = torch.Tensor([0, 0.1])
    minimal_proba = torch.exp(torch.Tensor([-100]))
    
    # if torch.cuda.is_available():
    #     a0, aa, ag, an = a0.cuda(), aa.cuda(), ag.cuda(), an.cuda()
    #     b0, ba, bg, bn = b0.cuda(), ba.cuda(), bg.cuda(), bn.cuda()
    #     av0, ava, avg, avn = av0.cuda(), ava.cuda(), avg.cuda(), avn.cuda()
    #     bv0, bva, bvg, bvn = bv0.cuda(), bva.cuda(), bvg.cuda(), bvn.cuda()
    #     a0_2, b0_2 = a0_2.cuda(), b0_2.cuda()
    #     left_born = left_born.cuda()
    #     right_born = right_born.cuda()
    #     noise_a = noise_a.cuda()
    #     noise_b = noise_b.cuda()
    #     minimal_proba = minimal_proba.cuda()
        
    #prior_noise = torch.distributions.uniform.Uniform(noise_a, noise_b)
    # prior_map = [unif(a0, b0), unif(aa, ba), unif(ag, bg), unif(an, bn),
    #              unif(a0_2, b0_2), unif(aa, ba), unif(ag, bg), unif(an, bn),
    #              unif(av0, bv0), unif(ava, bva), unif(avg, bvg), unif(avn, bvn),
    #              unif(av0, bv0), unif(ava, bva), unif(avg, bvg), unif(avn, bvn),
    #              prior_noise]
    
    prior_map = [unif(left_born[i], right_born[i]) for i in range(dim_param*AGN.shape[1] + 1)]
    prior = {"distribution":prior_map, "minimal_proba":minimal_proba, "left":left_born,
             "right":right_born}
    
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
        sim_param["gmm"] = sim_param["gmm"].cuda()
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
                                              sim_param["dW"], sim_param["p_mp"], sim_param["centers"])

def generate_CPU(N, parameters, sim_param):
        return simulation.generate_yeasts_fate_landscape_CPU(sim_param["AGN"], parameters, sim_param["X0"],
                                              sim_param["mapp"], sim_param["Nstate"], sim_param["Nmeasure"],
                                              sim_param["dt"], sim_param["Nsteps"], sim_param["dim_param"],
                                              sim_param["dW"], sim_param["centers"])

def mix_samples(nbatches, eps, epoch, nomix=False):
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
    
    if nomix:
        Cov = make_OLMC(nbatches, eps, epoch)
        for i in range(nbatches):
            particles = torch.load("./outputs/particles-b{}-e{}".format(i, epoch))
            weights = torch.load("./outputs/weights-b{}-e{}".format(i, epoch))
            distances = torch.load("./outputs/distances-b{}-e{}".format(i, epoch))
            N = distances.shape[0]
            cov_mix = Cov[:,:,i*N:(i+1)*N]
            
            torch.save(particles, "./outputs/particles-mix-b{}".format(i))
            torch.save(weights/torch.sum(weights), "./outputs/weights-mix-b{}".format(i))
            torch.save(distances, "./outputs/distances-mix-b{}".format(i))
            torch.save(cov_mix, "./outputs/Cov-mix-b{}".format(i))
            
            assert torch.sum(distances <= eps) > 0, "Not enough particules with distance smaller than next step !"
    else:
        particles = torch.load("./outputs/particles-e{}".format(epoch))
        weights = torch.load("./outputs/weights-e{}".format(epoch))
        distances = torch.load("./outputs/distances-e{}".format(epoch))
        bad = torch.where(distances > eps)
        good = torch.where(distances <= eps)
        
        nper_batch = weights.shape[0]//nbatches
        assert torch.sum(distances <= eps) >= nbatches, "Not enough particules with distance smaller than next step !"
        
        Cov = make_OLMC(nbatches, eps, epoch)
        
        good_p = particles[good]
        good_w = weights[good]
        good_d = distances[good]
        good_Cov = Cov[:,:,good[0]]
        
        bad_p = particles[bad]
        bad_w = weights[bad]
        bad_d = distances[bad]
        bad_Cov = Cov[:,:,bad[0]]
        
        id_good = torch.randperm(good_w.shape[0])
        id_bad = torch.randperm(bad_w.shape[0])
    
        
        batch_p = torch.split(particles, nper_batch, dim=0)
        batch_w = torch.split(weights, nper_batch)
        batch_d = torch.split(distances, nper_batch)
        batch_Cov = torch.split(Cov, nper_batch, dim=-1)
        
        N1 = good_w.shape[0]
        N2 = bad_w.shape[0]
        b = 0
        j = 0
        for i in range(N1 + N2):
            
            if i < N1:
                batch_p[b][j] = good_p[id_good[i]]
                batch_w[b][j] = good_w[id_good[i]]
                batch_d[b][j] = good_d[id_good[i]]
                batch_Cov[b][:,:,j] = good_Cov[:,:,id_good[i]]
            else:
                batch_p[b][j] = bad_p[id_bad[i-N1]]
                batch_w[b][j] = bad_w[id_bad[i-N1]]
                batch_d[b][j] = bad_d[id_bad[i-N1]]
                batch_Cov[b][:,:,j] = bad_Cov[:,:,id_bad[i-N1]]
                
            b += 1
            if b == nbatches:
                b = 0
                j += 1
        
        for i in range(nbatches):
            torch.save(batch_p[i], "./outputs/particles-mix-b{}".format(i))
            torch.save(batch_w[i]/torch.sum(batch_w[i]), "./outputs/weights-mix-b{}".format(i))
            torch.save(batch_d[i], "./outputs/distances-mix-b{}".format(i))
            torch.save(batch_Cov[i], "./outputs/Cov-mix-b{}".format(i))
    
def make_OLMC(nbatches, eps, epoch):
    
    particles = torch.load("./outputs/particles-e{}".format(epoch))
    d = torch.load("./outputs/distances-e{}".format(epoch))
    weights = torch.load("./outputs/weights-e{}".format(epoch))
    idx = torch.where(d < eps)
    
    Cov = inference.multivariate_OLMC(particles, weights, idx)
    return Cov
    

def Create_R(true_param, sim_param, AGN, X0):
    R = simulation.generate_yeasts_fate_landscape_CPU(AGN, true_param, X0,
                                              sim_param["mapp"], sim_param["Nstate"], sim_param["Nmeasure"],
                                              sim_param["dt"], sim_param["Nsteps"], sim_param["dim_param"],
                                              sim_param["dW"], sim_param["centers"])
    return R


def fitting():
    """
    Main function that call all necessary parts to perform inference.
    """
    # Simulation parameters
    Multiple_exp = False
    nomix = False
    #epsilons = [7.6, 6.15, 5.5, 4.85, 4.64, 4.485, 4.435, 4.3, 4.2]
    # epsilons = [8.9, 8.41, 7.8, 7.14, 6.41, 5.8, 5.53, 5.325, 5.4]
    # epsilons = [5.5, 4, 2.8, 2.25, 2, 1.85, 0, 2.5, 2.36]
    epsilons = [0.5, 0.271]
    
    for i in range(1):
    
        #for i in range(5):
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
        # R = torch.from_numpy(np.load("./R_test.npy"))
        data = pd.read_excel("./conditions.xlsx")
        
    
        G1 = data["C1"]/(data["C1"]+data["C2"]+data["C3"])
        MI = data["C3"]/(data["C1"]+data["C2"]+data["C3"])
        MII = data["C2"]/(data["C1"]+data["C2"]+data["C3"])
        
        R = torch.zeros((sim_param["Nconditions"], sim_param["Nstate"],
                         sim_param["Nmeasure"]))
        
        R[:,0,:] = torch.from_numpy(G1.values.reshape(-1, 1))
        R[:,1,:] = torch.from_numpy(MI.values.reshape(-1, 1))
        R[:,2,:] = torch.from_numpy(MII.values.reshape(-1, 1))
        
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
                    mix_samples(worker, epsilons[t], t-1, nomix)
                    
                inference.ABC_SMC_step(0,epsilons[t], R, sim_param["Nsamples"], prior, distance, generate_CPU, t, sim_param["maxiter"], sim_param, )
                
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
        a = torch.quantile(d, 0.15)
        epsilons.append(a.item())
        
            
        print("Saving final results")
        torch.save(sim_param["p_out"], "./outputs/particles-e{}".format(t))
        torch.save(sim_param["w_out"], "./outputs/weights-e{}".format(t))
        torch.save(d, "./outputs/distances-e{}".format(t))


if __name__ == '__main__':
    fitting()