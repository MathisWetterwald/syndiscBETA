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

