# -*- coding: utf-8 -*-


"""
Created on Wed Jun 28 11:33:06 2023

@author: Mathis  Wetterwald
"""

import numpy as np


from dit.pid.pid import BasePID
from lattices.utils import powerset
from lattices.constraints import is_antichain
from itertools import product
from lattices.lattice import Lattice
from operator import le
from copy import deepcopy
import networkx as nx
from syndisc.syndisc import build_constraint_matrix
from syndisc.solver import extreme_points
from .SOLVERbeta import synsolvebeta, synsolve1D
from dit.multivariate import coinformation




def _transform(lattice):
    """
    Based on dit.pid.pid
    Transform a free distributive lattice from being tuples of frozensets of frozensets
    of tuples of integers to being tuples of tuples of tuples of integers.

    Parameters
    ----------
    lattice : Lattice
        The lattice to transform.

    Returns
    -------
    tupled_lattice : Lattice
        The lattice, but with tuples in place of frozensets.
    """
    def tuplefy(n):
        return (tuple(sorted((tuple(sorted(sum(_, ()))) for _ in n[0]), key=lambda tup: (len(tup), tup))),tuple(sorted((tuple(sorted(sum(_, ()))) for _ in n[1]), key=lambda tup: (len(tup), tup))))

    def freeze(n):
        return (frozenset(frozenset((__,) for __ in _) for _ in n[0]),frozenset(frozenset((__,) for __ in _) for _ in n[1]))

    tuple_lattice = deepcopy(lattice)

    tuple_edges = [(tuplefy(e[0]), tuplefy(e[1])) for e in lattice._lattice.edges]
    tuple_lattice._lattice = nx.DiGraph(tuple_edges)

    tuple_lattice._relationship = lambda a, b: lattice._relationship(freeze(a), freeze(b))
    tuple_lattice.top = tuplefy(lattice.top)
    tuple_lattice.bottom = tuplefy(lattice.bottom)
    tuple_lattice._ts = [tuplefy(n) for n in lattice._ts]

    return tuple_lattice


def le2D(alpha1,alpha2): 
    """
    Based on lattices.orderings
    Compares two nodes of the lattice, returns alpha1 <= alpha2 iff alpha1 and alpha2 follow the order relationship presented in the paper
    
    Parameters
    ----------
    alpha: tuple of tuples of integers
        the first node to be compared
    beta: tuple of tuples of integers
        the second node to be compared
    
    Returns
    -------
    a<=b : boolean
    wether alpha <= beta
    """
    
    def r_le(x,y):
        for a in x:
            if not any(le(a,b)for b in y):
                return False
        return True
    return r_le(alpha1[0],alpha2[0]) and r_le(alpha1[1],alpha2[1])


def full_constraint_lattice_2D(inputs, outputs):
    '''
    Returns a lattice of constrained marginals, in the same way than in syndisc, but over X AND Y.
    
    Parameters
    ----------
    inputs : iter of iter
        indices of the marginals X_i
    outputs : iter of iter
        indices of the marginals Y_j
    
    Returns
    -------
    lattice : nx.DiGraph
        The lattice of antichains.
    '''
    depinput = [dep for dep in powerset(powerset(inputs, 1)) if is_antichain(dep)]
    depoutput = [dep for dep in powerset(powerset(outputs, 1)) if is_antichain(dep)]
    
    dependencies = [k for k in product(depinput, depoutput)]
    lattice = Lattice(dependencies, le2D, '꞉⋮•')
    
    return _transform(lattice).inverse()



class PID_SD_beta(BasePID):
    '''
    The information decomposition
    '''
    _name = "I_dis"
    
    def __init__(self, dist, n = 1, **kwargs):
        '''
        Parameters
        ----------
        dist : Distribution
            The distribution to compute the decomposition on
        n : integer
            The number of marginals in X. X will be the n first elements of dist.rvs, Y will be the others
        '''
        def separate(dist, n):
            '''
            from dist, gets the probabilities associated with X and Y
            
            Parameters
            ----------
            dist : Distribution
                the distribution to compute the synergy on
            n : integer
                the first n elements of dist.rvs are the marginals of X.
            
            Returns
            -------
            pX : The probability distribution of X
            pY : The probability distribution of Y
            pYgX : The probability distribution of Y, given X
            pXgY : The probability distribution of X, given Y
            
            '''
            inputs = range(0,n)
            outputs = range(n,dist.outcome_length())
            
            pX, pYgX = dist.condition_on(inputs, rv_mode = 'indexes')
            pY, pXgY = dist.condition_on(outputs, rv_mode = 'indexes')
            
            return pX, pY, pYgX, pXgY
        
        self._dist = dist
        self.pX, self.pY, pYgX, pXgY = separate(dist, n)
        
        #make sure inputs and outputs are tuples (should be as follows : (integer,) )
        inputs = [var for var in self.pX.rvs]
        outputs = [var for var in self.pY.rvs]
        self._inputs = tuple(map(tuple, inputs))
        self._outputs = tuple(map(tuple, outputs))
        
        #build the lattice, init dictionaries
        self._lattice = full_constraint_lattice_2D(self._inputs, self._outputs)
        self._reds = {}
        self._pis = {}
        self._kwargs = kwargs
        
        self._poly_vert_X = {}
        self._poly_vert_Y = {}
        
        #ensure probability distributions are dense
        self.pX.make_dense()
        self.pY.make_dense()
        for p in pYgX:
            p.make_dense()
        for p in pXgY:
            p.make_dense()
        
        #build pXgY and pYgX as arrays
        input_alphabet  = np.prod([len(dist.alphabet[i[0]]) for i in self._inputs])
        output_alphabet  = np.prod([len(dist.alphabet[i[0]]) for i in self._outputs])
        
        self.pYgX = np.zeros((output_alphabet, input_alphabet))
        count=0
        for i,(_,p) in enumerate(self.pX.zipped()):
            if p>0 :
                self.pYgX[:,i] = pYgX[count].pmf
                count += 1
        
        self.pXgY = np.zeros((input_alphabet, output_alphabet))
        count=0
        for i,(_,p) in enumerate(self.pY.zipped()):
            if p>0 :
                self.pXgY[:,i] = pXgY[count].pmf
                count += 1
        
        #for each node, build the polytopes
        for node in self._lattice._lattice:
            pass
        
    
    def get_red(self, node):
        """
        from dit.pid.pid
        Get the redundancy value associated with `node`.

        Parameters
        ----------
        node : tuple of iterable of iterables
            The node to get the partial information of.

        Returns
        -------
        pi : float
            The partial information associated with `node`.
        """
        
        if node not in self._reds:
            if node not in self._lattice: 
                raise Exception('Cannot get redundancy associated with node "%s" because it is in the lattice'
                                % str(node) )
            self._reds[node] = float(self._measure(node))

        return self._reds[node]


    def _measure(self, node):
        """
        Compute synergistic disclosure.
        
        Parameters
        ----------
        self : PID_SD_beta(BasePID)
            The PID on which we are working.
        node : tuple of iterables of iterables
            The node we want to compute the partial information associated with.
        
        Returns
        -------
        synsolvebeta : float
            The value of I_dis
        """
        
        #obtain alpha and beta from the node
        alpha, beta = node
        
        #if necessary, compute the polytope associated with alpha
        if len(node[0]) !=0 and node[0] not in self._poly_vert_X: 
            Const_X = build_constraint_matrix(node[0], self._dist.coalesce(self._inputs))
            Px = self.pX.pmf + 10**-40
            Px = Px/Px.sum()
            Px = np.array([Px]).T
            self._poly_vert_X[node[0]] = extreme_points(Const_X,Px)
        
        #if necessary, compute the polytope associated with beta
        if len(node[1]) !=0 and node[1] not in self._poly_vert_Y: 
            Const_Y = build_constraint_matrix(node[1], self._dist.coalesce(self._outputs))
            Py = self.pY.pmf + 10**-40
            Py = Py/Py.sum()
            Py = np.array([Py]).T
            self._poly_vert_Y[node[1]] = extreme_points(Const_Y,Py)
        
        #if alpha = {} and beta = {}, return coinforation between X and Y
        if len(node[1]) == 0 and len(node[0])==0:
            def foo(x):
                return x[0]
            inputs = list(map(foo,self._inputs))
            outputs = list(map(foo,self._outputs))
            return coinformation(self._dist, [inputs, [k+len(self._inputs) for k in outputs]])
        
        #if beta = {}, compute alpha-synergy on X
        if len(node[1]) == 0 :
            return synsolve1D(self.pX.pmf, self.pYgX, self._poly_vert_X[node[0]])
        
        #if alpha = {}, compute beta-synergy on Y
        if len(node[0]) == 0 :
            return synsolve1D(self.pY.pmf, self.pXgY, self._poly_vert_Y[node[1]])
        
        #if alpha != {} and beta != {}, compute alphabetasynergy on X and Y
        else:
            return synsolvebeta(self.pY.pmf, self.pX.pmf, self.pYgX, self._poly_vert_X[node[0]], self._poly_vert_Y[node[1]], **self._kwargs)
