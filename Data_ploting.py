# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 09:31:22 2022

@author: msche
"""
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from landscape_inf import Create_R, initializer, generate_CPU

from math import log10 , floor
def round_it(x, sig):
    if x == 0:
        return 0
    else:
        return round(x, sig-int(floor(log10(abs(x))))-1)
#%%

p_last = torch.load("./outputs/particles-e6")
p_before = torch.load("./outputs/particles-e5")
p_init = torch.load("./outputs/particles-e0")
p_last = p_last.numpy()
p_before = p_before.numpy()
p_init = p_init.numpy()
var_name = ["g_1_0", "g_1_g", "g_1_n",
            "g_2_0", "g_2_g", "g_2_n",
            "v_1_0", "v_1_g", "v_1_n",
            "v_2_0", "v_2_g", "v_2_n",
            "sigma"]

parts1 = pd.DataFrame(p_last, columns=var_name)

epoch = np.ones(240, dtype=int)
parts1["epoch"] = epoch*6

parts2 = pd.DataFrame(p_before, columns=var_name)

parts2["epoch"] = epoch*5

parts3 = pd.DataFrame(p_init, columns=var_name)
parts3["epoch"] = epoch*0


parts_67 = pd.concat([parts1, parts2], ignore_index=True)
parts_07 = pd.concat([parts1, parts3], ignore_index=True)

#%% only last epoch

sns.pairplot(parts1, vars=var_name)
#%%last and before
sns.pairplot(parts_67, vars=var_name, hue="epoch")

#%%
sns.pairplot(parts_07, vars=var_name, hue="epoch")

#%%
opt_param = np.zeros((13))
std_param = np.zeros((13))

for i, name in enumerate(var_name):
    hist = np.histogram(p_last[:,i])
    p = hist[1][np.argmax(hist[0])]
    std = np.std(p_last[:,i])
    opt_param[i] = p
    std_param[i] = std
    print(name + " : {:.3f} +/- {:.3f}".format(p, std))

    
torch.save(opt_param, "./outputs/opti_param")
torch.save(std_param, "./outputs/opti_param_std")

#%% Testing the fit
data = pd.read_excel("./results.xlsx")
# Transforming data for logscale case
# sub_data = data.drop([9, 11, 20, 16, 15])
# AGN_i = torch.from_numpy(np.hstack([np.ones((sub_data["Acetate"].shape[0], 1)), sub_data[["Acetate", "Glucose" ,"Nitrogen"]].values])).float()
# AGN_i = AGN_i * torch.Tensor([1, 1, 2, 5])
# w = torch.where(AGN_i[:,1:] > 0)
# min_val = torch.min(torch.log(AGN_i[:,1:][w]/100))



# #Loading all data and transform data to suit the simu
# AGN = torch.from_numpy(np.hstack([np.ones((data["Acetate"].shape[0], 1)), data[["Acetate", "Glucose" ,"Nitrogen"]].values])).float()
# AGN = AGN * torch.Tensor([1, 1, 2, 5])
# w = torch.where(AGN[:,1:] > 0)
# AGN[:,1:][w] = torch.log(AGN[:,1:][w]/100)/min_val

# data = pd.read_excel("./results.xlsx")
# data = data.drop([9, 11, 20, 16, 15])
# AGN = torch.from_numpy(np.hstack([np.ones((data["Acetate"].shape[0], 1)), data[["Acetate", "Glucose" ,"Nitrogen"]].values])).float()
# AGN = AGN * torch.Tensor([1, 1, 2, 5])
# w = torch.where(AGN[:,1:] > 0)
# AGN[:,1:][w] = torch.log(AGN[:,1:][w]/100)/torch.min(torch.log(AGN[:,1:][w]/100))

data = data.drop([0, 1, 2, 3, 8, 10, 11, 12, 20, 15, 17, 18, 21, 23, 24, 25])
AGN = torch.from_numpy(np.hstack([np.ones((data["Acetate"].shape[0], 1)), data[["Acetate", "Glucose" ,"Nitrogen"]].values])).float()
AGN = AGN * torch.Tensor([1, 1, 2, 5])
AGN = AGN[:,1:]

# True proportions
G1 = data["C1"]/(data["C1"]+data["C2"]+data["C3"])
MI = data["C3"]/(data["C1"]+data["C2"]+data["C3"])
MII = data["C2"]/(data["C1"]+data["C2"]+data["C3"])

R = torch.zeros((AGN.shape[0], 3,1))

R[:,0,:] = torch.from_numpy(G1.values.reshape(-1, 1))
R[:,1,:] = torch.from_numpy(MI.values.reshape(-1, 1))
R[:,2,:] = torch.from_numpy(MII.values.reshape(-1, 1))





param = torch.load("./outputs/opti_param")
sim_param = initializer()

#%%
#X0 = torch.ones((AGN.shape[0],sim_param["Ncells"], 2)) * torch.Tensor([[-1.4, 0]]) 
R_sim = generate_CPU(1, param, sim_param)


#%%
plt.style.use("seaborn")
col = sns.color_palette()
f = plt.figure(figsize=(30,30))
state = ["G", "MG", "MII"]
unseen = [4, 7]

R_sim = R_saved/100

for i in range(AGN.shape[0]):
    ax = plt.subplot(3, 4, i+1)
    
    if i in unseen:
       ax.set_facecolor(col[-2])
    for k in range(3):
        ground = [0, 0]
        d = 0.05
        
        
        p_e = R[i, k, 0].item()
        p_s = (R_sim[i, k, 0].item())
        if k > 0:
            ground[0] = torch.sum(R[i,0:k,0])
            ground[1] = torch.sum(R_sim[i,0:k,0])
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.title("({},{},{})".format(round_it(AGN[i,1-1].item(), 1), 
                                                  round_it(AGN[i,2-1].item(), 1),
                                                  round_it(AGN[i,3-1].item(), 1)),
                  fontsize=20)
            
        bar = plt.bar("Exp", p_e, color=col[k], bottom=ground[0], label=state[k])
        if p_e > d:
           plt.text(bar[0].get_x() + bar[0].get_width()/2., ground[0]+p_e/2-0.02, str("{:.2f}".format(p_e)) , fontdict=dict(fontsize=20), color="white", fontweight="bold", ha="center")
        
        bar = plt.bar("Sim", p_s,  color=col[k], bottom=ground[1])
        if p_s > d:
            plt.text(bar[0].get_x() + bar[0].get_width()/2., ground[1]+p_s/2-0.02, str("{:.2f}".format(p_s)) , fontdict=dict(fontsize=20), color="white", fontweight="bold", ha="center")
            

plt.subplot(3, 4,i+1)
plt.legend(bbox_to_anchor = [2, 1], fontsize=30)
plt.text(2, 0.45, "(Ac,Gl,Ni) in [%]", fontsize=30)

plt.savefig("./fig/comparison.png", dpi=200)

#%%
d = 6
R_saved = R_sim * 0
for i in range(100):
    
    R_sim = generate_CPU(240, opt_param, sim_param)
    dist = torch.sum(torch.abs(R-R_sim))
    if dist < d:
        R_saved += R_sim
    print(i, torch.sum(torch.abs(R-R_sim)))
    
    
#%%
x = np.array([0.005, 0.05, 0.5])
y = np.array([0.3449, 0.1936,0])
plt.style.use("seaborn")
l = np.linspace(0.005, 0.55, 100)
f = lambda x: 0.3678*np.exp(-12.83*x)

plt.plot(x, y, 'o', label="Data points")
plt.plot(l, f(l), '--', label="fit : 0.37 exp(-12.83x)")
plt.xlabel("nutrient concentration [%]", fontsize=20)
plt.ylabel("MII proportion", fontsize=20)
plt.legend(fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

#%%

n = np.linspace(0.005, 0.5, 100)
f = lambda x: (0.486-np.sqrt(8/27)+0.813*n)/1.245
err = lambda x: 0.02/1.245 + np.abs((0.486-np.sqrt(8/27)+0.813*x)/1.245**2*0.8) + 0.6/1.245*x
g = f(n)
plt.plot(n[g > 0], g[g>0], '-k', label="bifurcation of G state")
e = err(n)
plt.fill_between(n[g>0], g[g>0], g[g>0]+e[g>0], alpha = 0.5, color="c", label="uncertainty")
ids = g-e < 0
e[ids] = 0
plt.fill_between(n[g>0], g[g>0], -e[g>0], alpha = 0.5, color="c")
plt.xlabel("Nitrogen [%]", fontsize=24)
plt.ylabel("Glucose [%]",fontsize=24)
plt.legend(fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.title("Bifurcation of state G", fontsize=24)
plt.savefig("./fig/bifurcation.png", dpi=200)



