"""
Synergistic disclosure and self-disclosure in discrete random variables.

References:

    F. Rosas*, P. Mediano*, B. Rassouli and A. Barrett (2019). An operational
    information decomposition via synergistic disclosure.

    B. Rassouli, Borzoo, F. Rosas, and D. Gündüz (2018). Latent Feature
    Disclosure under Perfect Sample Privacy. In 2018 IEEE WIFS, pp. 1-7.

Distributed under the modified BSD licence. See LICENCE for details.

Pedro Mediano and Fernando Rosas, 2019
"""
import numpy as np
import dit
from itertools import combinations,chain
from dit.utils import build_table

def build_constraint_matrix(cons, d):
    """
    Build constraint matrix.

    The constraint matrix is a matrix P that is the vertical stack
    of all the preserved marginals.

    Parameters
    ----------
    cons : iter of iter
        List of variable indices to preserve.
    d : dit.Distribution
        Distribution for which to design the constraints

    Returns
    -------
    P : np.ndarray
        Constraint matrix

    """
    # Initialise a uniform distribution to make sure it has full support
    u = dit.distconst.uniform_like(d)
    n = len(u.rvs)
    l = u.rvs
    u = u.coalesce(l + l)

    # Generate one set of rows of P per constraint
    P_list = []
    for c in cons:
        pX123, pX1gX123 = u.condition_on(crvs=range(n, 2*n), rvs=c)

        pX123.make_dense()
        for p in pX1gX123:
          p.make_dense()

        P_list.append(np.hstack([p.pmf[:,np.newaxis] for p in pX1gX123]))

    # Stack rows and return
    P = np.vstack(P_list)

    return P

def build_list_atoms(system,**kwargs):
    if kwargs == {}:
        n = len(system._inputs)
        inputs = tuple(chain.from_iterable(system._inputs))
        outputs = tuple(chain.from_iterable(system._outputs))
        list_atoms = []
        for i in range(n+1):
            node = tuple(combinations(inputs, i)),tuple(combinations(outputs,i))
            list_atoms.append(node)
        return list_atoms
    else:
        if not 'u' in kwargs or not 'v' in kwargs:
            raise Exception('u and v must be defined')
        u = kwargs['u']
        v = kwargs['v']
        n,m = len(system._inputs), len(system._outputs)
        inputs = tuple(chain.from_iterable(system._inputs))
        outputs = tuple(chain.from_iterable(system._outputs))
        list_atoms = []
        for i in range(n+m+1):
            node = tuple(combinations(inputs, u(i))),tuple(combinations(outputs,v(i)))
            list_atoms.append(node)
        return list_atoms
def decompose_information(system,**kwargs):
    if len(system._inputs) != len(system._outputs) and kwargs == {} :
        raise Exception('if nothing is specified, X and Y should have the same length')
    list_atoms = build_list_atoms(system,**kwargs)
    if 'u' in kwargs:
        u = kwargs['u']
    else:
        u = lambda x : x
    if 'v' in kwargs:
        v = kwargs['v']
    else:
        v = lambda x : x
    table = build_table(['i,j',r'B^{i,j}',r'dB^{i,j}'],'synergy atoms')
    table.float_format = '{}.{}'.format(6, 4)
    list_values = []
    for i in range(len(list_atoms)):
        list_values.append(system.disclosure(list_atoms[i])[0])
    list_deltas = []
    for i in range(0,len(list_atoms)):
        list_deltas.append(list_values[len(list_atoms)-1-i]-sum(list_deltas))
    for i in range(len(list_atoms)):
        row = [(u(i),v(i)),list_values[i],list_deltas[len(list_atoms)-1-i]]
        table.add_row(row)
    print(table)
    return list_atoms,list_values,list_deltas
