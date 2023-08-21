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
    dist_for_ent = np.matmul(pBgA, polyA.T)
    c = [entropy(dist_for_ent[:,i],base=2) for i in range(len(polyA))]
    u, minH = lp_sol(np.array(c), polyA.T, pA)
    Is = entropy(np.matmul(pBgA,pA),base=2) - minH
    #Is = Is[0]
    return Is


def synsolvebeta(pY, pX, pYgX, pXgU, pYgV, **kwargs):
    a,c = pXgU.shape
    b,d = pYgV.shape
    
    C_yx = np.zeros_like(pYgX)
    y_length, x_length = C_yx.shape
    for y in range(y_length):
        for x in range(x_length):
            if pY[y] == 0 or pYgX[y,x] == 0:
                C_yx[y,x] = 0
            else :
                C_yx[y,x] = pYgX[y,x]/pY[y]
    C_vu = np.zeros((a,b))
    A = np.zeros((a,b))
    for u in range(a):
        for v in range(b):
            #print(np.array(pYgV)[v,:].shape, C_yx.T.shape)
            #print(C_yx.shape, c,d)
            C_vu[u,v] = np.matmul(pXgU[u,:].T, np.matmul(C_yx.T, pYgV[v,:]))
            if C_vu[u,v] !=0 :
                A[u,v] = C_vu[u,v] * np.log2(C_vu[u,v])
    
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
            print('scipy')
            return scipopt(B,C,D,a)
        
        elif kwargs['method'] != 'polytope':
            raise Exception('method should be "scipy" or "polytope". Currently, it is "%s".' % kwargs['method'])
    
    A1 = np.vstack([pXgU.T, -pXgU.T, - np.eye(pXgU.shape[0])])
    C1 = np.hstack([pX, -pX, np.zeros(pXgU.shape[0])])
    V1 = pm.compute_polytope_vertices(A1, C1)
    V1 = np.vstack(V1)
    SS = cdist(V1,V1)
    BB = SS<1e-10
    indices = [BB[:i,i].sum() for i in range(len(V1))]
    idx = [bool(1-indices[i]) for i in range(len(indices))]
    if np.any(V1<0):
        V1[V1<0] = 0
    V1 = V1/(V1.sum(axis=1)[:,np.newaxis])
    
    A2 = np.vstack([pYgV.T, -pYgV.T, - np.eye(pYgV.shape[0])])
    C2 = np.hstack([pY, -pY, np.zeros(pYgV.shape[0])])
    V2 = pm.compute_polytope_vertices(A2, C2)
    V2 = np.vstack(V2)
    SS = cdist(V2,V2)
    BB = SS<1e-10
    indices = [BB[:i,i].sum() for i in range(len(V2))]
    idx = [bool(1-indices[i]) for i in range(len(indices))]
    if np.any(V2<0):
        V2[V2<0] = 0
    V2 = V2/(V2.sum(axis=1)[:,np.newaxis])


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
    if u<-0.01:
        print(u,B)
    return  u
    #return minimize(bar, x0, method=method, bounds=boundaries, constraints=constraints)