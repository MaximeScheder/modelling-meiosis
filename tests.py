# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 15:31:08 2021

@author: msche
"""
#asdkasldkaslkdaslkd
import landscapes as maps
import inference as inf
import clustering as clst
import simulation as sim
import numpy as np
import torch
from inference import prior_toy
import gmm
import time
import matplotlib.pyplot as plt
import seaborn as sns

############################################################################
################ LANDSCAPES ################################################
############################################################################

def test_landscape_1():
    p = torch.Tensor([0, 0, 0, 0, 2, 1.5, 1.5, 1, 1])
    
    #Ploting and computing vector field
    x1, x2 = -1.5, 5
    y1, y2 = -4, 1.5
    x, y = torch.meshgrid(torch.linspace(x1, x2, 50), torch.linspace(y1, y2, 50)) 
    
    fig_2d, ax_2d = maps.plotLandscape(x,y, maps.field_yeast_fate, p)
    ax_2d.annotate("G0", (-0.8, 0.6),size=50)
    ax_2d.annotate("G1", (0.9, 0),size=50)
    ax_2d.annotate("SM", (2.1, 0),size=50)
    ax_2d.annotate("G0", (3.7, 0),size=50)
    ax_2d.annotate("Sm", (-0.1, -1),size=50)
    ax_2d.annotate("MI/II", (-0.1, -2.5),size=50)
    ax_2d.tick_params(axis='both', which='major', labelsize=50)
    plt.savefig("./fig/simplified_landscape.png", dpi=200)
    
def test_landscape_2():
    p = torch.Tensor([[0, 0, 1, 1]])
    #p = torch.Tensor([[-1]]) # flip
    #p = torch.Tensor([[-1, 0]]) # flip
    
    
    #Ploting and computing vector field
    x1, x2 = -1.5, 1.5
    y1, y2 = -1.5, 1.5
    x, y = torch.meshgrid(torch.linspace(x1, x2, 100), torch.linspace(y1, y2, 100)) 
    
    fig_2d, ax_2d = maps.plotLandscape(x, y, maps.fate_seperate, p, normMax=1, pot=maps.fate_seperate_V)
    ax_2d.annotate("Regular", (-0.1, 0.5),size=50)
    ax_2d.annotate("Dyads", (-0.6, -0.2),size=50)
    ax_2d.annotate("Tetrads", (0.4, -0.2),size=50)
    ax_2d.set_xlabel("x", size=50)
    ax_2d.set_ylabel("y", size=50)
    ax_2d.tick_params(axis='both', which='major', labelsize=50)
    plt.savefig("./fig/simplified_landscape_2.png", dpi=200)
    
    


    
    
############################################################################
################ SIMULATIONS ################################################
############################################################################

    
def test_simulation_1():
    dt = torch.Tensor([0.05])
    N = 100
    Ncondition = 1
    Nmeasure = 1
    cells = 250
    sigma = 0.08
    p = torch.Tensor([[0,0,2,2]]) #bicusp
    p = torch.Tensor([[0.3, -0.2, 0.5,0.5]]) # flip-cusp
    #p = torch.Tensor([[0,0.1,0.5]])

    dim_param = 4
    wiener_dist = torch.distributions.normal.Normal(0, 1)
    
    x1, x2 = -1.5, 1.5
    y1, y2 = -1.5, 1.5
    n = 50
    x, y = torch.meshgrid(torch.linspace(x1, x2, n+50), torch.linspace(y1, y2, 50))
    
    
    X0 = torch.ones((Ncondition, cells, 2)) * torch.Tensor([[0, 0.5]])
    Xtemp = torch.zeros((Ncondition, cells, 2, 2))
    Xem = torch.zeros((Ncondition, cells, Nmeasure, 2))
    param = torch.zeros((cells, Ncondition, dim_param))
    
    F = maps.fate_seperate
       
    X = sim.euler(X0, sigma, F, p, dt, N, Nmeasure, wiener_dist, Xem, Xtemp, param, Ncondition)
    
    torch.save(X[:, :, 0, :], "./X0")
    
    colors = ["k", "g", "b"]
    print(X.shape)
    for j in range(Ncondition):
        fig, ax = maps.plotLandscape(x, y, F, p[j], 1)
        for i in range(Nmeasure):
            ax.plot(X[j,:,i,0], X[0,:,i,1], 'o', color=colors[i], markersize=12)
    

############################################################################
################ CLUSTERING ###############################################
############################################################################

def test_cluster_1():
    
    #p = torch.Tensor([[-0.5, 0.5, 0.5]])
    p = torch.Tensor([[0.55, -0.2, 0.5,0.5]]) # flip-cusp
    wiener_dist = torch.distributions.normal.Normal(0, 1)

    
    normmax = 2
    dt = torch.Tensor([0.05])
    steps = 960
    Nmeasure = 1
    Ncells = 250
    sigma = 0.08
    param = torch.zeros((Ncells, 1, 4))
    Nconditions=1
    
    x1, x2 = -1.5, 1.5
    y1, y2 = -0.5, 1.5
    n = 50
    x, y = torch.meshgrid(torch.linspace(x1, x2, n+50), torch.linspace(y1, y2, 50))
    
    
    
    #X0 = torch.ones((1, Ncells, 2)) * torch.Tensor([[-0.8, 0]])
    X0 = torch.load("./X0")
    Xtemp = torch.zeros((1, Ncells, 2, 2 ))
    Xem = torch.zeros((1, Ncells, Nmeasure, 2))
    
    #centers = [np.array([-0.5, 0]), np.array([0.7, 0]), np.array([0,0])]
    centers = torch.Tensor([[0, 0.3],
                            [-0.4, 0],
                            [0.4, 0]])
    
    
    n_centers = 3
    colors = ["k", "g", "b"]
    
    # start = time.perf_counter()
    # #model = gmm.GaussianMixture(n_centers, 2, mu_init = centers)#, r=0.2)   
    # end = time.perf_counter()
    # print("\t Time of modeling : {:2f}".format(end-start))

    
        
   
    start = time.perf_counter()
    X = sim.euler(X0, sigma, maps.fate_seperate, p, dt, steps,Nmeasure, wiener_dist, Xem, Xtemp, param, Nconditions=1)
    X_fit = X.reshape(Nconditions*Ncells*Nmeasure, 2)
    end = time.perf_counter()
    
    print("Time to simulate : ", end-start)
    # X_mean = X_fit.mean(dim=0)
    # X_std = X_fit.std(dim=0)
    # X_ = (X_fit-X_mean)/X_std
    # model.mu_init = ((centers-X_mean)/X_std)
    
    # model.fit(X_fit)
    # assignment = model.predict(X_fit)
    
    start = time.perf_counter()
    assignment = clst.KMeans(X_fit, centers)
    end = time.perf_counter()
    
    print("Simply for assigning : ", end-start)
    
    start = time.perf_counter()
    assignment = clst.KMeans_Clustering(X_fit, centers)
    end = time.perf_counter()
    
    print("Time for clustering : ", end-start)
    
    assignment = assignment.reshape(Nconditions, Ncells, Nmeasure)
    for i in range(Nmeasure): 
        fig, ax = maps.plotLandscape(x, y, maps.fate_seperate, p, normmax)
        for j in range(n_centers):
            x_a, y_a = X[:,:,i,:][torch.where(assignment[:,:,i]==j)].T
            ax.plot(x_a, y_a, marker = 'o', color=colors[j], linestyle="", markersize=12)
            ax.plot(centers[j, 0], centers[j, 1], marker='s', color=colors[j], markersize=20)
            
    ax.annotate("G1", (0, 0.4),size=50)
    ax.annotate("D", (-0.4, -0.1),size=50)
    ax.annotate("T", (0.4, -0.1),size=50)
    ax.set_xlabel("x", fontsize=50)
    ax.set_ylabel("y",fontsize=50)
    ax.tick_params(axis='both', which='major', labelsize=50)
    #plt.savefig("./fig/clustering.png", dpi=200)
        
    
def test_cluster_2():
    mu = torch.Tensor([[[1, 2], [-1, -0.5]]]).double()
    mu_init = torch.Tensor([[[1, 0], [-1, 0]]]).double()
    S = torch.Tensor([[[[0.5, 0.5], [0.5, 1]], [[0.3, 0.], [0, 0.5]]]]).double()
    S_init = torch.Tensor([[[[1, 0], [0, 1]], [[1, 0.], [0, 1]]]]).double()
    X1 = torch.distributions.multivariate_normal.MultivariateNormal(mu[0,0], S[0,0]).sample(torch.Size([100]))
    X2 = torch.distributions.multivariate_normal.MultivariateNormal(mu[0,1], S[0,1]).sample(torch.Size([100]))
    
    x = torch.vstack([X1, X2])
    
    model = gmm.GaussianMixture(2, 2, mu_init=mu_init, var_init=S_init)
    model.fit(x)
    
############################################################################
################ INFERENCE ###############################################
############################################################################

def test_inference_1():
    
    import time
    import multiprocessing as mp
    from scipy.stats import uniform
    
    # Counting number of worker available
    ncpu = mp.cpu_count()
    
    #defining the prior

    # define the system that allows to generate new particles

        
    prior = prior_toy()
    param1, param2 = 8, 4
    batchsize = 50 # number of sample in each batch
    maxiter = 1e6
    
    # Create the system
    system = toy_system(param1, param2)
    

    
    X = system.generate(batchsize*ncpu) # generated data
    batches = np.split(X, ncpu, axis=0) # Split the data
    
    
    # The espilon ladder definition
    epsilons = [0.95, 0.9]#, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15]#, 0.28, 0.26, 0.23, 0.2]#, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15] #160, 120, 80, 60, 40, 30, 20, 15, 10, 8, 7, 6, 4, 3]
        



    T = len(epsilons)   
    print("{} processes with batch size {} :".format(ncpu, batchsize))
    print("\t Total samples : {}".format(ncpu*batchsize))
    
    for t in range(T):
        print("Starting epoch {} with threshold {}".format(t, epsilons[t]))
        processes = []
        start = time.perf_counter()
        
        if T == T-1:
            inf.ABC_SMC_step(epsilons[t], epsilons[t+1], X, X.shape[0], prior,
                              distance, generate, 0, t, maxiter)
        else:
            inf.ABC_SMC_step(epsilons[t], epsilons[t], X, X.shape[0], prior,
                              distance, generate, 0, t, maxiter)
            
        
        for i, batch in enumerate(batches):
            if t != T-1:
                processes.append(mp.Process(target= inf.ABC_SMC_step, args=(epsilons[t], epsilons[t+1],
                                                                            batch, batchsize, prior,
                                                                            distance, generate, i, t,
                                                                            maxiter, )))
            else:
                processes.append(mp.Process(target= inf.ABC_SMC_step, args=(epsilons[t], epsilons[t],
                                                                            batch, batchsize, prior,
                                                                            distance, generate, i, t,
                                                                            maxiter, )))
    
        for p in processes:
          p.start()
    
        for p in processes:
          p.join()
    
        end = time.perf_counter()
        
        print("\t Finished after {}".format(end-start))
    
    particles = np.concatenate([np.load("./outputs/particles-b{}-e{}.npy".format(i, t)) for i in range(ncpu)], axis=0)
    weights = np.concatenate([np.load("./outputs/weights-b{}-e{}.npy".format(i, t)) for i in range(ncpu)])  
        
    np.save("./outputs/particles", particles)
    np.save("./outputs/weights", weights)
    
class toy_system():
    
    def __init__(self, a, b):
        self.a, self.b = a, b
    
    def generate(self, N):
        return np.random.normal((self.a-2*self.b)**2 + 
                                             (self.b-4), 1, (N,1))
    
    def update_param(self, param):
        x = param.reshape(-1)
        self.a, self.b = x[0], x[1]
        
              

def distance(X, Y):
    hx = np.histogram(X, bins=100, range = (-50, 50))[0]
    hy = np.histogram(Y, bins=100, range = (-50, 50))[0]
    return np.sum(np.abs(hx-hy)/np.sum(hx))
    
def generate(N,theta):
    return np.random.normal((theta[0]-2*theta[1])**2 + (theta[1]-4), 1, N)

def test_movie_1():
    from matplotlib.animation import FuncAnimation
    import matplotlib.animation as animation
    sns.set_style("whitegrid")
    
    fig = plt.figure(figsize =(14, 9))
    ax = plt.axes(projection ='3d')
    
    N = 50
    #p = torch.Tensor([[[-0.6]]])
    p = torch.linspace(-0.5, 0.5, N).reshape(-1, 1, 1)
    
    x1, x2 = -0.8, 0.8
    y1, y2 = -1, 1
    n = 50
    x, y = torch.meshgrid(torch.linspace(x1, x2, n), torch.linspace(y1, y2, n))
    
    V = maps.cuspX_V    
    #N = 30
    def animationFunction(frame):
        ax.cla()
        z = V(x, y, p[frame])
        #vec = mapp.F(x, y)
        #ax_2d.quiver(x, y, vec[0].squeeze(), vec[1].squeeze())
        #ax_2d.contour(x, y, potential, levels=100, cmap=cm.coolwarm)
        #ax_2d.set_title("f1 : {}".format(f))
        ax.plot_surface(x.numpy(), y.numpy(), z.numpy()+0.5, cmap='magma_r')
        ax.contour(x.numpy(), y.numpy(), z.numpy(), 10, cmap='magma_r', linestyles="solid", offset=0)
        ax.set_zlim(0, 1.5)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title("b = {:.2f}".format(p[frame, 0, 0]))
        
    anim = FuncAnimation(fig, animationFunction, N)
    writervideo = animation.writers["ffmpeg"]
    writervideo = writervideo(fps=10)
    anim.save('landscape-motion.mp4', writer=writervideo)
    
def test_movie_2():
    from matplotlib.animation import FuncAnimation
    import matplotlib.animation as animation
    
    t = np.linspace(-0.41, 0.41, 100)
    a = -6*t**2
    b = 8*t**3
    

    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    N = 50
    #p = torch.Tensor([[[-0.6]]])
    p = np.linspace(-0.5, 0.5, N)
    
    #N = 30
    def animationFunction(frame):
        ax.cla()  
        ax.plot(b, a, linewidth=2)
        ax.set_ylim(-0.8, 0.1)    
        ax.set_xlabel("b", fontsize=24)
        ax.set_ylabel("a", fontsize=24)
        ax.tick_params(axis='both', which='major', labelsize=22)
        ax.set_title("bifurcation")
        ax.plot(p[frame], -0.6, 'kx', markersize=24)
        
        
    anim = FuncAnimation(fig, animationFunction, N)
    writervideo = animation.writers["ffmpeg"]
    writervideo = writervideo(fps=10)
    anim.save('bifurcation.mp4', writer=writervideo)


if __name__ == '__main__':
    #test_landscape_1()
    test_landscape_2()
    #test_simulation_1()
    #test_cluster_1()
    #test_cluster_2()
    #test_inference_1()
    #test_movie_1()
    #test_movie_2()
        
  
    