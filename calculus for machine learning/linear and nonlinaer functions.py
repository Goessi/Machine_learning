## 1. Why Learn Calculus? ##

import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0,3,num=301)
y = -x**(1/2) + x*3 - 1
plt.plot(x,y)
plt.show()

# find the derivative = 0 (x,y)

## 4. Math Behind Slope ##

def slope(x1,x2,y1,y2):
    slope = (y2 - y1)/(x2 - x1)
    return slope
slope_one = slope(0,4,1,13)
slope_two = slope(5,-1,16,-2)

## 6. Secant Lines ##

import seaborn
seaborn.set(style='darkgrid')

def draw_secant(x_values):
    x = np.linspace(-20,30,100)
    y = -1*(x**2) + x*3 - 1
    plt.plot(x,y)
    y_values = x_values
    for i in range(len(x_values)):
        y_values[i] = -1*(x_values[i]**2) + x_values[i]*3 - 1
    slope = (x_values[0]-x_values[1])/(y_values[0]-y_values[1])
    b = y_values[0] - slope*x_values[0]
    plt.plot(x,slope*x+b,color='g')
    plt.show()
    
draw_secant([3,5])
draw_secant([3,10])
draw_secant([3,15])

