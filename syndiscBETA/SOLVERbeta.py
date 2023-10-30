# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 11:33:54 2023

@author: Mathis  Wetterwald
"""

import numpy as np

from scipy.spatial.distance import cdist 
from scipy.stats import entropy
import pypoman as pm
from syndisc.solver import lp_sol
from scipy.optimize import minimize, Bounds, LinearConstraint
from itertools import product

def synsolve1D(pA, pBgA, polyA):
    """
    from syndisc.solver
    Computes the synergistic self-disclosure capacity (alpha-synergy).

    Parameters
    ----------
    pA : np.ndarray
        column vector with the pmf of A
    PBgA : np.ndarray
        conditional probability of B given A.
    polyA : np.ndarray
        N-by-D array with N vertices of the polytope
    
    Returns 
    -------
    Is : float
        synergistic disclosure capacity
        
    """
    #find extremal entropies
    dist_for_ent = np.matmul(pBgA, polyA.T)
    c = [entropy(dist_for_ent[:,i],base=2) for i in range(len(polyA))]
    
    #given them, find the distribution u that minimizes joint entropy
    u, minH = lp_sol(np.array(c), polyA.T, pA)
    
    #compute I_s
    Is = entropy(np.matmul(pBgA,pA),base=2) - minH
    return Is


def synsolvebeta(pY, pX, pYgX, pXgU, pYgV, **kwargs):
    """
    Computes the synergistic self-disclosure capacity. (alphabeta-synergy)
    Parameters
    ----------
    pY : np.ndarray
        column vecto with the pmf of Y^m
    pX : np.ndarray
        column vector with the pmf of X^n
    pYgX : np.ndarray
        conditional probability of Y given X.
    pXgU : np.ndarray
        optimal conditional probability of X given U : given by the vertices of the polytope
    pYgV : np.ndarray
    optimal conditional probability of Y given V : given by the vertices of the polytope
    
    
    Returns 
    -------
    Is : float
        synergistic disclosure capacity
        
    """
    
    #find shapes of X,U,Y,V
    a,c = pXgU.shape
    b,d = pYgV.shape
    
    #computing p(x,y)/(p(x)p(y))
    C_yx = np.zeros_like(pYgX)
    y_length, x_length = C_yx.shape
    for y in range(y_length):
        for x in range(x_length):
            if pY[y] == 0 or pYgX[y,x] == 0:
                C_yx[y,x] = 0
            else :
                C_yx[y,x] = pYgX[y,x]/pY[y]
    
    #Computing C_vu
    C_vu = np.zeros((a,b))
    A = np.zeros((a,b))
    for u in range(a):
        for v in range(b):
            #print(np.array(pYgV)[v,:].shape, C_yx.T.shape)
            #print(C_yx.shape, c,d)
            C_vu[u,v] = np.matmul(pXgU[u,:].T, np.matmul(C_yx.T, pYgV[v,:]))
            if C_vu[u,v] !=0 :
                A[u,v] = C_vu[u,v] * np.log2(C_vu[u,v])
    
    #use scipy to optimize over pU and pV
    if 'method' in kwargs:
        if kwargs['method'] == 'scipy':
            B = np.zeros((a+b,a+b))
            B[0:a,a:a+b] = A
            C = np.zeros(c+d+2)
            C[0:c] = pX
            C[c:c+d] = pY
            C[c+d] = 1
            C[c+d+1] = 1
            D = np.zeros((c+d+2,a+b))
            D[0:c,0:a] = pXgU.T
            D[c:c+d,a:a+b] = pYgV.T
            D[c+d,0:a] = np.ones(a)
            D[c+d+1,a:a+b] = np.ones(b)
            return scipopt(B,C,D,a)
        
        elif kwargs['method'] != 'polytope':
            raise Exception('method should be "scipy" or "polytope". Currently, it is "%s".' % kwargs['method'])
    
    ##use some polytopes to optimize faster
    
    #Compute the polytope for pU
    A1 = np.vstack([pXgU.T, -pXgU.T, - np.eye(pXgU.shape[0])])
    C1 = np.hstack([pX, -pX, np.zeros(pXgU.shape[0])])
    V1 = pm.compute_polytope_vertices(A1, C1)
    V1 = np.vstack(V1)
    
    #To avoid numerical problems, manually cast all small negative values to 0
    if np.any(V1 < -1e-6):
        raise RuntimeError("Polytope vertices computation failed : (found negative vertices).")
    V1[V1<0] = 0
    
    #normalize the conditionnal pmfs
    V1 = V1/(V1.sum(axis=1)[:,np.newaxis])
    
    #Eliminate vertices too close of each other
    SS = cdist(V1,V1)
    BB = SS<1e-10
    indices = [BB[:i,i].sum() for i in range(len(V1))]
    idx = [bool(1-indices[i]) for i in range(len(indices))]
    V1 = V1[idx]

    #compute the polytope for pV
    A2 = np.vstack([pYgV.T, -pYgV.T, - np.eye(pYgV.shape[0])])
    C2 = np.hstack([pY, -pY, np.zeros(pYgV.shape[0])])
    V2 = pm.compute_polytope_vertices(A2, C2)
    V2 = np.vstack(V2)
    
    #To avoid numerical problems, manually cast all small negative values to 0
    if np.any(V2 < -1e-6):
        raise RuntimeError("Polytope vertices computation failed : (found negative vertices).")
    V2[V2<0] = 0
    
    #normalize the conditional pmfs
    V2 = V2/(V2.sum(axis=1)[:,np.newaxis])
    
    #Eliminate vertices too close of each other
    SS = cdist(V2,V2)
    BB = SS<1e-10
    indices = [BB[:i,i].sum() for i in range(len(V2))]
    idx = [bool(1-indices[i]) for i in range(len(indices))]
    V2 = V2[idx]
    

    #find best match
    I = -1
    for pU, pV in product(V1,V2):
        res = np.dot(pV.T, np.dot(A.T, pU))
        if I < res:
            I = res
    return I

    
    

def scipopt(B,C,D,a):
    def bar(x):
        return - x.dot(B).dot(x)
    b = len(B[0]) - a
    x0 = np.array([1/a]*a+[1/b]*b)
    boundaries = Bounds(lb=np.zeros(a+b), ub=np.ones(a+b))
    constraints = LinearConstraint(D, lb=C- 0.0000001*np.ones_like(C), ub=C + 0.0000001*np.ones_like(C))
    method = "trust-constr"
    u = - bar(minimize(bar, x0, method=method, bounds=boundaries, constraints=constraints).x)
    return  u
