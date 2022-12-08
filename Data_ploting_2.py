# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 07:56:03 2022

@author: scheder
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from simulation import fitpam_to_landscape, generate_fate_gpu
from sklearn.mixture import GaussianMixture
from landscape_inf import initializer
from landscape_inf import log_min_max_inv
from landscape_inf import R_from_data
from landscapes import cusp_full, plotLandscape
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import torch
from simulation import euler_gpu


def plot_hist(particles, var_name):
    pass

def find_optimal_param(parameters, var_name, d=None):
    
    gm = GaussianMixture().fit(parameters)
    print(gm.means_)
    print(gm.weights_)
            
    return gm.means_[np.newaxis,:]

def compare_param(particle1, particle2):
    
    simulation_param = np.load("./dict_data.npy", allow_pickle=True).item()
    AGN = simulation_param['AGN']
    dim_param = simulation_param['dim_param']
    
    p_1, s_1 = fitpam_to_landscape(particle1, AGN, dim_param)
    p_2, s_2 = fitpam_to_landscape(particle2, AGN, dim_param)
    
    for c in range(AGN.shape[0]):
        print('Condition {}'.format(c))
        print('\tp1 : {}'.format(p_1[c].numpy()))
        print('\tp2 : {}'.format(p_2[c].numpy()))
    
    print('Difference : {}'.format(np.linalg.norm(p_1-p_2)))
    print('sigma 1 : {}'.format(s_1))
    print('sigma 2 : {}'.format(s_2))
    
def param_to_landscape_form(param):
    simulation_param = np.load("./dict_data.npy", allow_pickle=True).item()
    AGN = simulation_param['AGN']
    dim_param = simulation_param['dim_param']
    
    p, s = fitpam_to_landscape(param, AGN, dim_param)
    
    for c in range(AGN.shape[0]):
        print('Condition {}'.format(c))
        print('\t {}'.format(p[c].numpy()))
        
    print('sigma : {}'.format(s))
    
def simulate_landscape(param, R_exp = None):
    
    R = generate_fate_gpu(param)
    
    if type(R_exp) is np.ndarray:
        print(distance_gpu(R, R_exp))
    return R
    
def distance_gpu(Rgpu, R):
    return np.sum(np.abs(Rgpu-R), axis=(1,2,3))

from math import log10 , floor
def round_it(x, sig):
    if x == 0:
        return 0
    else:
        return round(x, sig-int(floor(log10(abs(x))))-1)

def plot_predicitons(R_sim, R_gt):
    
    R_sim = R_sim.squeeze()
    R_gt = R_gt.squeeze()
    simulation_param = np.load('dict_data.npy', allow_pickle=True).item()
    AGN = simulation_param['AGN'].numpy()
    Nstate = simulation_param['Nstate']
    time_steps = simulation_param['steps']*simulation_param['dt']
    
    plt.style.use("seaborn")
    col = sns.color_palette()
    state = ["UNSPO", "SPO"]
    
    for t, time in enumerate(time_steps):
        fig = plt.figure(figsize=(30,30))
        fig.suptitle('{} hours in media'.format(time), fontsize=40)
        for i in range(AGN.shape[0]):
            ax = plt.subplot(5, 5, i+1)
            
            # if i in unseen:
            #    ax.set_facecolor(col[-2])
            for k in range(Nstate):
                ground = [0, 0]
                d = 0.05
                
                
                p_e = R_gt[i, k, t]
                p_s = R_sim[i, k, t]
                if k > 0:
                    ground[0] = np.sum(R_gt[i,0:k,t])
                    ground[1] = np.sum(R_sim[i,0:k,t])
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)
                plt.title("({},{},{})".format(round_it(log_min_max_inv(AGN[i,1])*100, 1), 
                                                          round_it(log_min_max_inv(AGN[i,2])*100, 1),
                                                          round_it(log_min_max_inv(AGN[i,3])*100, 1)),
                          fontsize=20)
                    
                bar = plt.bar("Exp", p_e, color=col[k], bottom=ground[0], label=state[k])
                if p_e > d:
                   plt.text(bar[0].get_x() + bar[0].get_width()/2., ground[0]+p_e/2-0.02, str("{:.2f}".format(p_e)) , fontdict=dict(fontsize=20), color="white", fontweight="bold", ha="center")
                
                bar = plt.bar("Sim", p_s,  color=col[k], bottom=ground[1])
                if p_s > d:
                    plt.text(bar[0].get_x() + bar[0].get_width()/2., ground[1]+p_s/2-0.02, str("{:.2f}".format(p_s)) , fontdict=dict(fontsize=20), color="white", fontweight="bold", ha="center")
                    
    
        plt.subplot(5, 5,i+1)
        plt.legend(bbox_to_anchor = [2, 1], fontsize=30)
        plt.text(2, 0.45, "(Ac,Gl,Ni) in [%]", fontsize=30)
        
    
        plt.savefig("./figures/comparison_t{}.png".format(t), dpi=200)
        plt.clf()
        

    
def movie_landscape(condition):

    simulation_param = np.load("./dict_data.npy", allow_pickle=True).item()
    AGN = simulation_param['AGN'].numpy()     # Concentration matrix for each nutrient
    C = log_min_max_inv(AGN)*100
    C = np.around(C[condition], 3)
    HML = []
    for i in range(1,4):
        if C[i] in [1, 0.5]:
            HML.append('H')
        elif C[i] in [0.1, 0.005]:
            HML.append('L')
        elif C[i] in [0.05]:    
            HML.append('M')
        else:
            HML.append('_')
    
    fig, ax = plt.subplots(1, 1, figsize=(20,20))
    
    x1, x2 = -0.75, 0.75
    y1, y2 = -0.75, 0.75
    x, y = torch.meshgrid(torch.linspace(x1, x2, 100), torch.linspace(y1, y2, 100)) 
    
    p = torch.tensor(np.load('p_movie.npy'))
    X = np.load('X_movie.npy')
    
    n_frame = X.shape[-2]


    def animationFunction(frame):
        ax.cla()
        ax.set_ylim(-0.75, 0.75)
        ax.set_xlim(-0.75, 0.75)
        plotLandscape(x, y, cusp_full, p[0,condition].reshape((1,1,-1)), normMax=0.4, ax_2d=ax)
        ax.set_xlabel("x", size=30)
        ax.set_ylabel("y", size=30)
        ax.annotate("Unspo", (-0.3, 0.2),size=50)
        ax.annotate("Spo", (0.2, 0.2),size=50)
        ax.tick_params(axis='both', which='major', labelsize=30)
        
        
        ax.scatter(X[0,condition,:,frame,0], X[0,condition,:,frame,1])
        
        ax.set_title("Time {}h \n Ac: {} - Gl: {} - Ni: {}".format(frame*2, HML[0], HML[1], HML[2]), fontsize=30)
        
        
    anim = FuncAnimation(fig, animationFunction, n_frame)
    writervideo = animation.writers["ffmpeg"]
    writervideo = writervideo(fps=10)
    anim.save('./figures/movies/landscape_AGN-{}{}{}.mp4'.format(HML[0],HML[1],HML[2]), writer=writervideo)
        

def X_movie(nbr_frame, parameter):
    
    simulation_param = np.load("./dict_data.npy", allow_pickle=True).item()
    AGN = simulation_param['AGN']       # Concentration matrix for each nutrient
    X0 = simulation_param['X0']         # Initial condition for the cells (Euler)
    F_lscpe = simulation_param['mapp']  # Landscape
    dt = simulation_param['dt']         # Time step for the simulation
    Nsteps = simulation_param['Nsteps'] # Total iteration in Euler
    dim_param = simulation_param['dim_param']   # Dimension of the landscape parameter (!= particle dimension)
    Nconditions = AGN.shape[0]          # Number of nutrient conditions
    
    steps = np.linspace(0, Nsteps, nbr_frame, dtype=int)
    
    
    
    nbatch = parameter.shape[0]
    
    # Transform each batch particle into a compatible form with the landscape
    lands_param = torch.zeros((nbatch, Nconditions, dim_param))
    sigmas = np.zeros(nbatch)
    
    for i in range(nbatch):
        lands_param[i,:,:], sigmas[i] = fitpam_to_landscape(parameter[i], AGN, dim_param)
    
    
    X = euler_gpu(X0, sigmas, F_lscpe, lands_param, dt, Nsteps, steps, Nconditions, nbatch)
    np.save('X_movie.npy', X)
    np.save('p_movie.npy', lands_param)
    
    
if __name__ == '__main__':
    
    d_last = np.load('./outputs/2022_12_5-distance-e12.npy')
    p_last = np.load('./outputs/2022_12_5-particles-e12.npy')
    d_init = np.load('./outputs/2022_12_2-distance-e0.npy')
    p_init = np.load('./outputs/2022_12_2-particles-e0.npy')
    p_mid = np.load('./outputs/2022_12_3-particles-e6.npy')

    var_name = ["b_0", 'b_a', "b_g", "b_n", 
                "a_0", 'a_a', "a_g", "a_n",
                "v_0", 'v_a', "v_g", "v_n",
                "sigma"]

    epoch = np.ones(800, dtype=int)

    data =  pd.DataFrame(p_last, columns=var_name)
    data['epoch'] = epoch*12

    data_0 =  pd.DataFrame(p_init, columns=var_name)
    data_0['epoch'] = epoch*0

    data_mid =  pd.DataFrame(p_mid, columns=var_name)
    data_mid['epoch'] = epoch*6

    data_08 = pd.concat([data_0, data_mid, data], ignore_index=True)
    
    # sns.pairplot(data_08, vars=var_name, hue="epoch")
    # sns.pairplot(data, vars=var_name)
    p_opt = find_optimal_param(p_last, var_name)
    p_opt=p_opt.reshape((1,13))
    
    # Ploting R matrix
    
    # R = R_from_data(pd.read_csv('timed_counts.csv'))
    
    # param_to_landscape_form(p_opt)
    # R_sim = simulate_landscape(p_opt, R)
    
    # plot_predicitons(R_sim, R)
    
    # MOVIE LANDSCAPE
    #X_movie(55, p_opt)
    for i in range(24):
        print('Starting movie {}'.format(i))
        movie_landscape(i)