## 2. Inconsistent Systems ##

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,20,1000)
y1 = -2*x+1.25
y2 = -2*x+2.5
plt.plot(x,y1,color = 'blue')
plt.plot(x,y2,color = 'blue')
plt.show()

## 5. Homogenous Systems ##

def test_homog(x3):
    x1 = 4/3*x3
    x2 = 0*x3
    a = 3*x1 + 5*x2 - 4*x3
    if a==0 and x2 == 0:
        return True
    else:
        return False
b_one = test_homog(1)
b_two = test_homog(-10)
