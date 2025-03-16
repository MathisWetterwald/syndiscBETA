# syndiscBETA
Python for AlphaBetaSynergy

Computes $\alpha\beta$-synergy.

This package gives multiple ways of playing around with synergy.

## 0 : Create a PID object
```python
#imports
import numpy as np
import dit
from syndiscBETA.pid import PID_SD_beta

#create distribution
pmf = [1/4,1/4,1/4,1/4]
outcomes = ['0000','0101','1001','1111']
d = dit.Distribution(outcomes,pmf)

#create PID object
pid = PID_SD_beta(d,X=[0,1],Y=[2,3]) #X=[0,1] : X system is the first two atoms ; Y=[2,3] : Y system is the last two
```

## 1 : Assess the full synergistic decomposition of information
```python
pid1 = PID_SD_beta(Xor_And(), X=[0,1], Y=[2,3],method='polytope', table='2D')
pid2 = PID_SD_beta(Xor_And(), X=[0,1], Y=[2,3],method='scipy', table='2D')
pid3 = PID_SD_beta(Xor_And(), X=[0,1,2], Y=[3],method='polytope', table='1D')
print(pid1)
print(pid2)
print(pid3)
```
```method='polytope'``` is faster but fails in rare cases. In those cases, use ```method='scipy'```.
```table='1D'``` and ```table='2D'``` are used to choose the display of the table. In case $X$ and $Y$ each got more than one atom, ```table='2D'``` is required.

## 2 : Assess one node
```python
node = ((),((0,),))
node1, node2 = [] , [[0]]
print(pid.disclosure(node)[0])
print(pid.disclosure_practice(node1,node2)[0])
```
These two functions return the measure for the same node (no privacy constrait for $X$, privacy constraint for the first atom of $Y$). As it is clearer, it is recommended to use ```pid.disclosure_practice```. Here is an example of the node corresponding to privacy constraints on the first atom of $X$, the joint law of the two following atoms, and the first atom of $Y$ :
```python
node1,node2 = [[0],[1,2]], [0]
```
The ```[0]``` in ```pid.disclosure(node)[0]``` means we are only interested in the amount of information that respects the privacy constraints. To get an idea of which $U$ and $V$ achieve it, ```pid.disclosure(node)[1]``` is a ```dict``` with all information needed.

## 3 : Assess the backbone (work in progress)
```python
from syndiscBETA.syndisc import decompose_information

_ = decompose_information(pid)
```
will print the backbone decomposition, along with the corresponding Möbius inversion.
