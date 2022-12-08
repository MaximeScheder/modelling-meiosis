# -*- coding: utf-8 -*-
"""
Testing inference on a created landscape with a few time points to see if
the programm can catch the dynamic.
"""
import torch
import matplotlib.pyplot as plt
import landscapes
import numpy as np
from simulation import generate_fate_gpu
from landscape_inf import initializer
import inference
import time
from contextlib import redirect_stdout
from scipy.stats import betaprime, norm, gamma, lognorm


def making_fake_ground_truth():
    
    "Test on a simple cusp landscape with the different media"
    true_parameter = torch.tensor(np.array([[-0.1, -0.4, 0, 0.8, -0.3, 0.3, 0.9, -0.1, 0.05, 0.01, -0.01, 0, 0.03]]))
    initializer()
    R = generate_fate_gpu(true_parameter)
    np.save('R.npy',R)

def test_landscape_2():
    
    p = np.load('lands_parm.npy')
    #p = torch.Tensor([[0, 0, 1]]).reshape(1,1,-1)
    p = torch.tensor(p)
    cond = 15
    
    R = np.load('R.npy')
    
    #Ploting and computing vector field
    x1, x2 = -1.5, 1.5
    y1, y2 = -1.5, 1.5
    x, y = torch.meshgrid(torch.linspace(x1, x2, 100), torch.linspace(y1, y2, 100)) 
    
    fig_2d, ax_2d = landscapes.plotLandscape(x, y, landscapes.cusp_full, p[0,cond].reshape((1,1,-1)), normMax=0.05)
    ax_2d.set_xlabel("x", size=50)
    ax_2d.set_ylabel("y", size=50)
    ax_2d.annotate("Unsporulated", (-0.3, 0.2),size=50)
    ax_2d.annotate("Sporulated", (0.2, 0.2),size=50)
    ax_2d.tick_params(axis='both', which='major', labelsize=50)
    
    X0 = np.load('X0.npy')
    X = np.load('X_time.npy') 
    alpha = [0.3, 0.4, 0.5, 0.6, 0.7, 1] 
    ax_2d.scatter(X0[:,0], X0[:,1], color='black')
    for i in range(X.shape[3]):
   
        ax_2d.scatter(X[0,cond,:,i,0], X[0,cond,:,i,1], alpha=alpha[i])
    
    print(R[0,cond])
    
    
class prior():
    # Prior for the test landscape 
    def __init__(self):
        pass
        
    
    def sample(self, Nparam=1):
        b = norm.rvs(0, 0.8, size=4)
        a = norm.rvs(0, 0.8, size=4)
        somme = -1
        while somme < 0: # the velocity cannot be negative
            v = np.hstack((betaprime.rvs(1,10,size=1), np.random.normal(0, 0.5, 3)))
            somme = v[0] + v[v<0].sum()
        sigma = lognorm.rvs(1.5,loc=0.0001, scale=0.01, size=1)
        return np.hstack((b, a, v, sigma))
    
    def evaluate(self, param):
        v = param[8:12]
        if v[0] + v[v<0] < 0:
            return 0
        pab = norm.pdf(param[0:8], 0, 0.8)
        pv0 = betaprime.pdf(param[8], 1, 10)
        pv = norm.pdf(param[9:12], 0, 0.5)
        ps = lognorm.pdf(param[-1], 1.5, loc=0.0001, scale=0.01)
        return np.prod(np.hstack((pab, pv0, pv, ps)))
    
def fitting():
    """
    Main function that call all necessary parts to perform inference.
    """
    
    starting_epoch = 0
    
    with open('consol_output.txt', 'w') as f:
        with redirect_stdout(f):
            print('STARTING SIMULATION \n')
            # Simulation parameters
            
            # Purely randomely, the error goes from 110 to 160
            # If only states to the first well, the error goes down 127
            max_iter = 1e4 # Maximum number of iteration on one epoch without saving any particle
            epsilons = [55]
            #epsilons = [29, 24.1, 19.4, 15.18, 11, 8, 6.2, 4.7, 3.8, 3.1]
            #epsilons = [29, 24.1, 19.4, 15.18, 11, 8, 6.2, 4.7, 3.8, 15]
            n_steps = 15
            
            
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
                
                R = np.load('R.npy')
                
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
                print("\t With 0.15 quantile : {:.5f} \n".format(np.quantile(d_out, 0.15)))
                a = np.quantile(d_out, 0.15)
                epsilons.append(a)
                print('The threshold list is now :')
                print('\t {}'.format(epsilons), flush=True)
                
                p_in = p_out
                w_in = w_out
                d = d_out
                np.save('./outputs/{}epsilons-e{}.npy'.format(prefix, i), epsilons)
                
def distance_gpu(Rgpu, R):
    return np.sum(np.abs(Rgpu-R), axis=(1,2,3))
                
def range_R():
    R = np.load('R.npy')
    N = 1000
    distances = np.zeros(N)
    for i in range(N):
        Y = np.random.uniform(0,1, (1, 24, 2, 6))
        norm = Y.sum(axis=2)[:,:,np.newaxis,:]
        d = distance_gpu(Y/norm, R)
        distances[i] = d
        
    plt.hist(distances)
    print(np.min(distances))

if __name__ == '__main__':
    # making_fake_ground_truth()
    # test_landscape_2()
    fitting()
    # range_R()
    