# -*- coding: utf-8 -*-
"""
Here are present the landscape for the euler simulations.
"""

#I am adding this stuff to understand github !!!

#--------- LYBRARIES

import torch
import matplotlib.pyplot as plt

#--------- LANDSCAPES

def cuspX_V(x, y, p):
    if p.ndim == 1:
        p = p.reshape(1, -1)
    return x**4 - 1*x**2/2 + p[:,0]*x + y**2/2
    
def cuspX_F(a, b, x, y):
    return -torch.stack([4*x**3 + a*2*x + b, 4*y])
        
def cuspY_V( b, x, y):
    if b.ndim == 1:
        b = b.reshape(1, -1)
    return y**4 - 1*y**2/2 + b*y + x**2/2
    
def cuspY_F(a, b, x, y):
    return -torch.stack([4*x, 4*y**3 + a*2*y + b])
        
def binaryFlip_V(x, y, p):
    #x, y = 0.71*x-0.71*y, 0.71*x+0.71*y
    return x**4 + y**4 + x**3 - 2*x*y**2 - x**2 + p[:,0]*x + p[:,1]*y
    
def binaryFlip_F(a, b, x, y):
    x, y = 0.71*x-0.71*y, 0.71*x+0.71*y
    f1 = 4*x**3 + 0.5*3*x**2 -0.5*2*y**2 - 2*x + a
    f2 = 4*y**3 -0.5*2*x*y + b
    return -0.71*torch.stack([f1+f2,-f1+f2])

# def cusp(x, y, parameter):
#     p = parameter
#     if p.ndim == 1:
#         p = p.reshape(1, -1)
#     return -torch.stack([4*x**3 -2*x + p[:,0], 4*y**3])

def cusp(x, y, parameter):
    p = parameter
    if p.ndim == 1:
        p = p.reshape(1, -1)
    return -torch.stack([4*x**3 - 2*x/2 + p[:,0], y])

def cuspy(x, y, p):
    if p.ndim == 1:
        p = p.reshape(1, -1)
    return -torch.stack([x, 4*y**3 - 2*y/2 + p[:,0]])
        
        
def flip_V(x, y, p):
    if p.ndim == 1:
        p = p.reshape(1, -1)
    return y**3/3 + x**2/2 + p[:,0]*y
    
def flip(x, y, p):
    if p.ndim == 1:
        p = p.reshape(1, -1)
    return -torch.stack([x, y**2 + p[:,0]])

def fate_seperate(x, y, p):
    if p.ndim == 1:
        p = p.reshape(1, -1)
    return p[:,2]*(1-glueCusp(y))*cuspy(x, y, p[:,0]) + p[:,3]*(glueCusp(y))*cusp(x, y, p[:,1])

def fate_seperate_V(x, y, p):
    if p.ndim == 1:
        p = p.reshape(1, -1)
    return p[:,2]*(1-glueCusp(y))*cuspY_V(p[:,0], x, y) + p[:,3]*(glueCusp(y))*cuspX_V(x, y, p[:,1])

def binary(x, y, p):
    if p.ndim == 1:
        p = p.reshape(1, -1)
    return -p[:,2]*torch.stack([4*x**3+  3*x**2 - 2*y**2-2*x+p[:,0], 4*y**3-4*y*x+p[:,1]])

def cycle_F(mu, w, b, x, y):
    r = torch.linalg.norm(torch.stack([x, y]), dim=0)
    cos = x/r
    sin = y/r
    rdot = mu*r  - torch.pow(r, 3)
    thetadot = (w + b*torch.pow(r, 2))*r
    return torch.stack([rdot*cos + sin*thetadot, rdot*sin - cos*thetadot])

def glueCusp(x):
    return (torch.tanh(-10*(x))+1)/2
    

def simplified_fate(x, y, parameters):
    if parameters.ndim == 1:
        parameters = parameters.reshape(1, -1)
    return parameters[:,2]*glueCusp(x)*cusp(x+0.7, y, parameters[:,0]) + parameters[:,3]*(1-glueCusp(x))*cusp(x-0.7,y,parameters[:,1])
    
def simplified_fate_V(x, y, parameters):
    return parameters[:,2]*glueCusp(x)*cuspX_V(x+0.7, y, parameters[:,0]) + parameters[:,3]*(1-glueCusp(x))*cuspX_V(x-0.7,y,parameters[:,1])
    

def glueCycle(x, y):
    """Gluing function for the cylce in the final landscape"""
    r = torch.linalg.norm(torch.stack([x-2.2, y-0.2]), dim = 0)
    return (torch.tanh(-10*(r-1)) + 1)/2

def glueMI(x, y):
    """Glue the MI phase to the rest of the final landscape"""
    return (torch.tanh(-10*(y+1.25))+1)/2

def glueG0(x, y):
    """Glue the inital 3 attractor binary landscape to the rest of the mapps"""
    return (1-glueMI(x,y))*(1-glueCycle(x,y))

def field_yeast_fate(x, y, p):
    """ Final landscape, note that p must be the parameters of the landscape 
    p = [bf1, bf2, csp1, cyc1, cyc2, vbf, vcsp]"""
    if p.ndim == 1:
        p = p.reshape(1, -1)
        
    return (glueG0(x,y)*p[:,5]*binaryFlip_F(p[:,0], p[:,1], x, y) +
            glueMI(x,y)*p[:,6]*cuspY_F(-1, p[:,2], x+0.05, y+1.95) +
            glueCycle(x,y)*cycle_F(p[:,3], p[:,4], 0, x-2.2, y-0.2))
    

#--------- UTULITARY
    
def plotLandscape(x, y, mapp, parameters, normMax = 8, pot = None):
    
    vec = mapp(x, y, parameters)

    vx = vec[0].squeeze()
    vy= vec[1].squeeze()
    color = torch.abs(vx)+torch.abs(vy)
    mask = torch.where(color < normMax)
    
    fig_2d = plt.figure(figsize =(28, 28))
    ax_2d = plt.axes()
    
    if pot is not None:
        xx = torch.full(x.shape, torch.nan)
        yy = torch.full(y.shape, torch.nan)
        xx[mask] = x[mask]
        yy[mask] = y[mask]
        potential = pot(xx, yy, parameters)
        ax_2d.contour(xx, yy, potential, levels=50)
        

    ax_2d.quiver(x[mask], y[mask], vx[mask], vy[mask], color[mask])
    ax_2d.grid()
    

    
    
    return fig_2d, ax_2d
    
