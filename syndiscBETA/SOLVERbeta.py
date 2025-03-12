"""
Backend functions to solve algebraic and optimisation problems related to
synergistic channels.

References:

    F. Rosas*, P. Mediano*, B. Rassouli and A. Barrett (2019). An operational
    information decomposition via synergistic disclosure.

    B. Rassouli, Borzoo, F. Rosas, and D. Gündüz (2018). Latent Feature
    Disclosure under Perfect Sample Privacy. In 2018 IEEE WIFS, pp. 1-7.

Distributed under the modified BSD licence. See LICENCE for details.

Pedro Mediano and Fernando Rosas, 2019
"""
import numpy as np

from numpy.linalg import svd
from scipy.optimize import linprog, minimize, Bounds, LinearConstraint
from scipy.spatial.distance import cdist 
from cvxopt import matrix, solvers
from scipy.stats import entropy
from itertools import product
import pypoman as pm


def extreme_points(P, Px, SVDmethod='standard'):
    """
    Calculation of extreme points of the polytope S
    Parameters
    ----------
    P : np.ndarray
        binary matrix of transitions
    Px : np.ndarray
        distribution of the dataset
    SVDmethod : str
        'standard' does the full SVD 
        'fast' uses a trick described in Apendix XX of journal paper

    Returns
    -------
    sols : np.ndarray
        N-by-D array with N vertices of the polytope
    """
    if SVDmethod == 'standard':
        U, S, Vh = svd(P)

    elif SVDmethod == 'fast':
        PtimesPt = P*P.transpose()

        # multiplication by 1+epsilon for numerical stability
        U0, S0, Vh0 = svd((1+1e-20)*PtimesPt)
        S = np.sqrt(S0)

        # Faster than using U.transpose() as it avoids the transpose operation 
        Vh = Vh0 * P
        
    else:
        raise ValueError("SVDMethod not recognised. Must be either \
                'standard' or 'fast'.")


    # Extract reduced A matrix using a small threshold to avoid numerical
    # problems
    rankP = np.sum(S > 1e-6)
    A = Vh[:rankP,:]
    b = np.matmul(A,Px)
    
    # Turn np.matrix into np.array for polytope computations
    A = np.array(A)
    b = np.array(b).flatten()

    # Build polytope and find extremes
    A = np.vstack([A, -A, -np.eye(A.shape[1])])
    b = np.hstack([b, -b, np.zeros(A.shape[1])])
    V = pm.compute_polytope_vertices(A, b)  # <-- magic happens here
    V = np.vstack(V)
    
    # To avoid numerical problems, manually cast all small negative values to 0
    if np.any(V < -1e-6):
        raise RuntimeError("Polytope vertices computation failed \
                (found negative vertices).")
    V[V < 0] = 0
    
    # Normalise the rows of V to 1, since they are conditional PMFs
    V = V/(V.sum(axis=1)[:,np.newaxis])
    
    # Eliminate vertices that are too similar (distance less than 1e-10)
    SS = cdist(V,V)
    BB = SS<1e-10
    indices = [BB[:i,i].sum() for i in range(len(V))]
    idx = [bool(1-indices[i]) for i in range(len(indices))]
    
    return V[idx]


def lp_sol(c, A, b, mode='cvx'):
    """
    Solve the linear programme given by 
    
    min cT*x subject to Ax = b and x > 0.
    
    Parameters
    ----------
    c : np.ndarray
    A : np.ndarray
    b : np.ndarray
    mode : str
        'cvx' is the standard. 'scipy' uses the scipy solver (not recommended).

    Returns
    -------
    u : np.ndarray
        D-by-1 array that achieves the minimum
    minH : float
        minimum value of the objective (in this case an entropy)
    """
    
    if mode=='cvx':
        c = matrix(c)
        G = matrix(-np.eye(len(c)))
        h = matrix(np.zeros(len(c))) 
        A = matrix(A)
        b = matrix(b)
    
        options={'glpk':{'msg_lev':'GLP_MSG_OFF'}}
        res = solvers.lp(c, G, h, A, b, solver='glpk', options=options)
        u = np.array(res['x']).flatten()
        minH = res['dual objective']
               
    elif mode=='scipy':
        res = linprog(c, A_eq=A, b_eq=b)
        u = res['x']
        minH = res['fun']

    return u, minH



def synsolve1D(pA, pBgA, polyA, direction='XtoY'):
    """
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
    #handle the direction
    if direction!= 'XtoY' and direction !='YtoX':
        raise Exception('direction should be "XtoY" or "YtoX" ! ')
    
    #find extremal entropies
    dist_for_ent = np.matmul(pBgA, polyA.T)
    c = [entropy(dist_for_ent[:,i],base=2) for i in range(len(polyA))]
    
    #given them, find the distribution u that minimizes joint entropy
    u, minH = lp_sol(np.array(c), polyA.T, pA)
    
    #compute I_s
    Is = entropy(np.matmul(pBgA,pA),base=2) - minH
    
    # compute mappings
    u = quantize(u,1e-12)
    polyA_nonzero = polyA[np.nonzero(u)]
    pAgV = quantize(polyA_nonzero.transpose(),1e-4)
    
    u_nonzero = u[np.nonzero(u)]
    pA = pA + 1e-40     #avoid numerical issues
    pA = pA/pA.sum()    # (singular matrices) 
    pVgA = np.diag(u_nonzero)@polyA_nonzero@np.linalg.inv(np.diagflat(pA.T))
    
    #build the dictionary
    if direction =='XtoY':
        channel_dict = {'pU': u[np.nonzero(u)], 'pXgU': pAgV, 'pUgX': pVgA}
    
    if direction =='YtoX':
        channel_dict = {'pV': u[np.nonzero(u)], 'pYgV': pAgV, 'pVgY': pVgA}
    
    return Is, channel_dict


def synsolvebeta(pY, pX, pYgX, pXgU, pYgV, **kwargs):
    """
    Computes the synergistic self-disclosure capacity. (alphabeta-synergy)
    Parameters
    ----------
    pY : np.ndarray
        column vector with the pmf of Y^m
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
            return scipopt(B,C,D,a),0
        
        elif kwargs['method'] != 'polytope':
            raise Exception('method should be "scipy" or "polytope". Currently, it is "%s".' % kwargs['method'])
    
    ##use some polytopes to optimize faster
    
    #Compute the polytope for pU
    A1 = np.vstack([pXgU.T, -pXgU.T, - np.eye(pXgU.shape[0])])
    C1 = np.hstack([pX+1e-12, -pX-1e-12, np.zeros(pXgU.shape[0])])
    V1 = pm.compute_polytope_vertices(A1, C1)
    if len(V1)==0:
        raise Exception('should use scipy on this one !')
    V1 = np.vstack(V1)
    
    #To avoid numerical problems, manually cast all small negative values to 0
    if np.any(V1 < -1e-6):
        raise RuntimeError("Polytope vertices computation failed : (found negative vertices).")
    V1[V1<0] = 0
    
    #Eliminate vertices too close of each other
    SS = cdist(V1,V1)
    BB = SS<1e-10
    indices = [BB[:i,i].sum() for i in range(len(V1))]
    idx = [bool(1-indices[i]) for i in range(len(indices))]
    V1 = V1[idx]
    
    #normalize the conditionnal pmfs
    V1 = V1/(V1.sum(axis=1)[:,np.newaxis])
    
    #compute the polytope for pV
    A2 = np.vstack([pYgV.T, -pYgV.T, - np.eye(pYgV.shape[0])])
    C2 = np.hstack([pY+1e-8, -pY-1e-8, np.zeros(pYgV.shape[0])])
    V2 = pm.compute_polytope_vertices(A2, C2)
    V2 = np.vstack(V2)
    
    #To avoid numerical problems, manually cast all small negative values to 0
    if np.any(V2 < -1e-6):
        raise RuntimeError("Polytope vertices computation failed : (found negative vertices).")
    V2[V2<0] = 0
    
    #Eliminate vertices too close of each other
    SS = cdist(V2,V2)
    BB = SS<1e-10
    indices = [BB[:i,i].sum() for i in range(len(V2))]
    idx = [bool(1-indices[i]) for i in range(len(indices))]
    V2 = V2[idx]
    
    #normalize the conditional pmfs
    V2 = V2/(V2.sum(axis=1)[:,np.newaxis])
    
    #find best match between V1,V2
    I = -1
    for pU, pV in product(V1,V2):
        res = np.dot(pV.T, np.dot(A.T, pU))
        if I < res:
            I = res
            u = pU
            v = pV
    channel_dict = {'pU': u, 'pV': v, 'pXgU': pXgU}#, 'pYgV' : pYgV, 'pUgX': pVgA}
    return I, channel_dict

    
    

def scipopt(B,C,D,a):
    '''

    Parameters
    ----------
    B : presented in the paper
    C : presented in the paper
    D : presented in the paper
    a : presented in the paper

    Returns
    -------
    the alpha-beta synergy for the node considered

    '''
    def bar(x):
        return - x.dot(B).dot(x)
    b = len(B[0]) - a
    x0 = np.array([1/a]*a+[1/b]*b)
    boundaries = Bounds(lb=np.zeros(a+b), ub=np.ones(a+b))
    constraints = LinearConstraint(D, lb=C- 0.0000001*np.ones_like(C), ub=C + 0.0000001*np.ones_like(C))
    method = "trust-constr"
    u = - bar(minimize(bar, x0, method=method, bounds=boundaries, constraints=constraints).x)
    return  u

def quantize(array,pres=0.01):
    """
    Function to reduce the decimal precision of an input array.

    Parameters
    ----------
    array : np.array
        numerical array to be quantized
        
    pres : float
        desired precision

    Returns
    -------
    np.array with quantized values
    """
    return ((pres**(-1))*array).round()*pres
