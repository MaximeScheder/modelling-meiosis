# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 15:49:46 2022

@author: scheder
"""
import numpy as np
import torch
from simulation import fitpam_to_landscape
import time
from clustering import KMeans_Clustering
from simulation import euler_gpu
from inference import distance_gpu
import matplotlib.pyplot as plt

#%% SImulation of the whole system
parameters = np.load('./outputs/2022_11_29-particles-e10.npy')
R_gt = np.load('R.npy')


simulation_param = np.load("./dict_data.npy", allow_pickle=True).item()

AGN = simulation_param['AGN']       # Concentration matrix for each nutrient
X0 = simulation_param['X0']         # Initial condition for the cells (Euler)
F_lscpe = simulation_param['mapp']  # Landscape
Nstate = simulation_param['Nstate'] # Number of state in the landscape
Nmeasure = simulation_param['Nmeasure'] # Number of time point sampled
Nsteps = simulation_param['Nsteps'] # Total iteration in Euler
dim_param = simulation_param['dim_param']   # Dimension of the landscape parameter (!= particle dimension)
centers = simulation_param['centers']   # Position of the center of the cluster
Nconditions = AGN.shape[0]          # Number of nutrient conditions
steps = simulation_param['steps']   # The steps at which to save Euler results 

Ncells = 300
dt = 0.01
time_point = np.array([11, 14, 17, 20, 23, 37])
step_sample = time_point/dt
Nsteps = int(step_sample[-1])
Nmeasure = len(time_point)

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
X_fit = X.reshape(nbatch*Nconditions*Ncells*Nmeasure, 2)

if torch.cuda.is_available():
    centers = centers.cuda()
    X_fit = X_fit.cuda()
    
    
#Cluster the data according to the specified algo
print("\t Starting Clustering", flush=True)
start = time.perf_counter()

assignement = KMeans_Clustering(X_fit, centers)
assignement = assignement.reshape(nbatch, Nconditions, Ncells, Nmeasure)
assignement = assignement.cpu()
assignement_all = assignement.numpy()

end = time.perf_counter()
print('\t -- time elapsed for clustering : {} seconds'.format(end-start), flush=True)

R_all = np.zeros((nbatch, Nconditions, Nstate, Nmeasure))

# Compute for all batch the faith probability
for b in range(nbatch):
    for i in range(Nstate):
        R_all[b,:,i,:] = np.sum(assignement_all[b]==i, axis=-2)/Ncells
    
# removing unncessary variables from gpu
torch.cuda.empty_cache()
d_all_batch = distance_gpu(R_all, R_gt)

#%% Small
threshold = 22

d_small = d_all_batch[d_all_batch < threshold]
param_small = parameters[d_all_batch < threshold]


nbatch = param_small.shape[0]

# Transform each batch particle into a compatible form with the landscape
lands_param = torch.zeros((nbatch, Nconditions, dim_param))
sigmas = np.zeros(nbatch)
for i in range(nbatch):
    lands_param[i,:,:], sigmas[i] = fitpam_to_landscape(param_small[i], AGN, dim_param)

# running stochastic euler
print("\t Starting stochastic euler evolution", flush=True)
start = time.perf_counter()

X_small = euler_gpu(X0, sigmas, F_lscpe, lands_param, dt, Nsteps, steps, Nconditions, nbatch)

torch.cuda.empty_cache()
end = time.perf_counter()
print('\t -- time elapsed for euler step : {} seconds'.format(end-start))

# reshape the matrix to be compatible with the clustering algorithm
X_small_fit = X_small.reshape(nbatch*Nconditions*Ncells*Nmeasure, 2)

if torch.cuda.is_available():
    centers = centers.cuda()
    X_small_fit = X_small_fit.cuda()
    
    
#Cluster the data according to the specified algo
print("\t Starting Clustering", flush=True)
start = time.perf_counter()

assignement = KMeans_Clustering(X_small_fit, centers)
assignement = assignement.reshape(nbatch, Nconditions, Ncells, Nmeasure)
assignement = assignement.cpu()
assignement_small = assignement.numpy()

end = time.perf_counter()
print('\t -- time elapsed for clustering : {} seconds'.format(end-start), flush=True)

R_small = np.zeros((nbatch, Nconditions, Nstate, Nmeasure))

# Compute for all batch the faith probability
for b in range(nbatch):
    for i in range(Nstate):
        R_small[b,:,i,:] = np.sum(assignement_small[b]==i, axis=-2)/Ncells
    
# removing unncessary variables from gpu
torch.cuda.empty_cache()

i = np.argmin(d_small)

p_min = param_small[i][np.newaxis,:]
print(np.argmax(parameters==p_min)//13)
print(np.argmin(d_small))

#%% Minimal value


# p = np.ones(parameters.shape)*p_min
# p_min = p
p_min = np.array([[-0.1, -0.4, 0, 0.8, -0.3, 0.3, 0.9, -0.1, 0.05, 0.01, -0.01, 0, 0.03]])
nbatch = p_min.shape[0]

# Transform each batch particle into a compatible form with the landscape
lands_param = torch.zeros((nbatch, Nconditions, dim_param))
sigmas = np.zeros(nbatch)
for i in range(nbatch):
    lands_param[i,:,:], sigmas[i] = fitpam_to_landscape(p_min[i], AGN, dim_param)

# running stochastic euler
print("\t Starting stochastic euler evolution", flush=True)
start = time.perf_counter()

X_min = euler_gpu(X0, sigmas, F_lscpe, lands_param, dt, Nsteps, steps, Nconditions, nbatch)

torch.cuda.empty_cache()
end = time.perf_counter()
print('\t -- time elapsed for euler step : {} seconds'.format(end-start))

# reshape the matrix to be compatible with the clustering algorithm
X_min_fit = X_min.reshape(nbatch*Nconditions*Ncells*Nmeasure, 2)

if torch.cuda.is_available():
    centers = centers.cuda()
    X_min_fit = X_min_fit.cuda()
    
    
#Cluster the data according to the specified algo
print("\t Starting Clustering", flush=True)
start = time.perf_counter()

assignement = KMeans_Clustering(X_min_fit, centers)
assignement = assignement.reshape(nbatch, Nconditions, Ncells, Nmeasure)
assignement = assignement.cpu()
assignement_min = assignement.numpy()

end = time.perf_counter()
print('\t -- time elapsed for clustering : {} seconds'.format(end-start), flush=True)

R_min = np.zeros((nbatch, Nconditions, Nstate, Nmeasure))

# Compute for all batch the faith probability
for b in range(nbatch):
    for i in range(Nstate):
        R_min[b,:,i,:] = np.sum(assignement_min[b]==i, axis=-2)/Ncells
    
# removing unncessary variables from gpu
torch.cuda.empty_cache()

#%% PLOT stuff
from landscapes import plotLandscape

condition = 11

def cusp_full(x, y, p):
    " p : velocity, a, b"
    if p.ndim == 1:
        p = p.reshape(1, -1)
    return p[:,2].item()*cusp(x, y, p[:,0:2])

def cusp(x, y, parameter):
    p = parameter    
    return -torch.stack([4*x**3 - 2*x/2*p[:,1] + p[:,0], y], dim=0)

fig, ax = plt.subplots(2,1, sharex=True, figsize=(15, 30))

x1, x2 = -0.5, 0.5
y1, y2 = -0.5, 0.5
x, y = torch.meshgrid(torch.linspace(x1, x2, 50), torch.linspace(y1, y2, 50)) 

for i in range(2):
    ax[i] = plotLandscape(x,y, cusp_full, lands_param[0,condition,:], ax_2d=ax[i])
    ax[i].annotate("UNSPO", (-0.3, 0.2),size=25)
    ax[i].annotate("SPO", (0.3, 0.2),size=25)
    ax[i].tick_params(axis='both', which='major', labelsize=25)




alpha = [1, 1, 1, 1, 1, 1]

i = 0
#well = 1
#where_all = assignement_all[392, condition, :, i] == well
#where_small = assignement_small[60, condition, :, i] == well
#where_min = assignement_min[0, condition, :, i] == well

ax[0].scatter(X_min[0,condition,:,i,0], X_min[0,condition,:,i,1], alpha=alpha[i])
# ax[1].scatter(X_small[60,condition,:,i,0], X_small[60,condition,:,i,1], alpha=alpha[i])
#ax[1].scatter(X[302,condition,:,i,0][where_all], X[302,condition,:,i,1][where_all], alpha=alpha[i])

# ax[0].scatter(X_t_min_2[i,condition,:,0], X_t_min_2[i,condition,:,1])
# ax[1].scatter(X_t_all_2[i,condition,:,0], X_t_all_2[i,condition,:,1])

# print('MIN - prop of cell in cluster : {}'.format(np.sum(where_min)/Ncells))
# print('SML - prop of cell in cluster : {}'.format(np.sum(where_small)/Ncells))
# print('ALL - prop of cell in cluster : {}'.format(np.sum(where_all)/Ncells))

