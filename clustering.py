# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 17:16:09 2021

This code is used in order to cluster data into different states

@author: msche
"""

import torch
#from autograd.scipy.stats import multivariate_normal as mn
import projections


def GMM_EM(centers_init, centers, sigmas, weights, obs, q = None,
           diag =1e-2*torch.diag(torch.Tensor([1, 1])), tolerance=0.01, Niter = 100,
           radius = None):
    """
    This function allows to make a GMM fitting of a 2D data through the Expectation Maximization
    algorithm (EM). It allows to constrain the centers of the clusters into L2 balls
    of a wanted size.
        param:
            centers: (Nstate, 2) array of inital centers of GMM
            sigmas: (Nstate, 2, 2) array of covariance matrix
            weights: (Nstate) array of GMM weights
            obs: (N,2) array with N observation of 2D positions
            tolerance: wanted precision in the difference between maximum
                likelihood of two steps
            Niter: Maximum number of iterations
            radius: maximum allowed L2 ball radius for the centers to move. If
                None, this constrain is ignored.
        return:
            centers, sigmas, weights : The fitted new array
            assignment: (N,1) array with values between (0,1,...,n) which
                correspond to the assigned gaussian.
    """
    
    # Initialising parameters
    epsilon = 10
    L_new = 1e10
    iteration = 0
    
    N = obs.shape[0]
    k = weights.shape[0]
    
    while epsilon > tolerance and iteration < Niter:
        Q, q, norm = ComputeQ(obs, centers, sigmas, k, N, diag, q)
        
        # Marginal likelihood step
        L_old = L_new
        L_new = torch.sum(torch.log(torch.sum(q, axis=1)))
        epsilon = torch.abs((L_old-L_new)/L_old) # normalized difference
        
        # Maximization step 
        
        Mus = Q.T @ obs / norm.reshape(-1,1)
            
        for i in range(k):
            centers[i] = Mus[i,:]
            
            if radius is not None:
                # Verify if new center is out of L2 ball
                r = centers[i]-centers_init[i]
                norm_r = torch.linalg.norm(r)
                recompute_Q = False
                
                if norm_r > radius:
                    vec_dir = r/norm_r # vector pointing to best direction
                    centers[i] = centers_init[i]+radius*vec_dir # projection on L2 ball
                    recompute_Q = True
                    
                if recompute_Q: #recompute Q only if centers have changed
                    Q, q, norm = ComputeQ(obs, centers, sigmas, k, N, diag, q)
            
            #Compute the new covariance matrices
            sigma_temp = torch.sum(torch.stack([Q[j,i]*(obs[j,:]-centers[i]).reshape(-1,1)@(obs[j,:]-centers[i]).reshape(1,-1) for j in range(N)]), axis=0)
            # sigma_temp = torch.zeros((dim,dim), device=centers.device)
            # for j in range(N):
            #     sigma_temp += Q[j,i]*(obs[j,:]-centers[i]).reshape(-1,1)@(obs[j,:]-centers[i]).reshape(1,-1)
             
            sigmas[i] = sigma_temp / norm[i]
            
            # Compute the new weights
            weights[i] = norm[i]/N
            
        iteration += 1
    return torch.argmax(Q, axis=1)

        

def ComputeQ(x, centers, sigmas, k, N, diag, qq=None):
     """
     Function that compute the assignment matrix given the observation
     matrix x.
         return: 
             Q: the assignment matrix used for further calculations
             q: unormalized assignment matrix
             norm: sum of the Lines of Q
     """                
     if qq == None:
         qq = torch.zeros((N, k))
         
     for i in range(k):
         # We add a small diagonal matrix to ensure none singular matrices
         mult_norm = torch.distributions.multivariate_normal.MultivariateNormal(centers[i], sigmas[i] + diag)
         
         qq[:,i] = torch.stack([mult_norm.log_prob(x_j) for x_j in torch.unbind(x)])
         # for j in range(N):
         #     qq[j,i] = mult_norm.log_prob(x[j,:])
     index = torch.where(qq < -100)
     qq[index] = -100
     qq = torch.exp(qq)


     QQ = qq/torch.sum(qq, axis=1).reshape(N,1)
     normm = torch.sum(QQ, axis=0)
     return QQ, qq, normm
 
    
def KMeans(x, centers):
    """Implements Lloyd's algorithm for the Euclidean metric."""
    
    N, D = x.shape  # Number of samples, dimension of the ambient space
    K = centers.shape[0]
    c = centers # must be a (1, K, D)
    D_ij = torch.pow((x.reshape(N,1,D) - c.view(1, K, D)), 2).sum(-1)  # (N, K) symbolic squared distances
    cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

    return cl

def KMeans_Clustering(x, centers, Niter=10, radius = 0.4):
    """Implements Lloyd's algorithm for the Euclidean metric."""
    

    N, D = x.shape  # Number of samples, dimension of the ambient space
    K = centers.shape[0]

    c = centers.clone() # Simplistic initialization for the centroids

    x_i = x.view(N, 1, D)  # (N, 1, D) samples
    c_j = c.view(1, K, D)  # (1, K, D) centroids
    projector = projections.L2Ball()
    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in range(Niter):

        # E step: assign points to the closest cluster -------------------------
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)
        
        # Divide by the number of points per cluster:
        Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
        c /= Ncl  # in-place division to compute the average
        
        if radius is not None:
            norm = torch.linalg.norm(c-centers, dim=1)
            c[norm != norm] = centers[norm != norm]
            to_change = (norm > radius)

            if torch.any(to_change):
                c[to_change] = centers[to_change] + projector.project(c-centers, radius)[0][to_change]
            
        


    return cl