## 2. Geometric Intuition Of Vectors ##

import matplotlib.pyplot as plt
import numpy as np

# This code draws the x and y axis as lines.
plt.axhline(0, c='black', lw=0.5)
plt.axvline(0, c='black', lw=0.5)
plt.xlim(-3,3)
plt.ylim(-4,4)

plt.quiver(0,0,2,3,scale_units='xy',angles='xy',scale=1,color='b')
plt.quiver(0,0,-2,-3,scale_units='xy',angles='xy',scale=1,color='b')
plt.quiver(0,0,1,1,scale_units='xy',angles='xy',scale=1,color='gold')
plt.quiver(0,0,2,3,scale_units='xy',angles='xy',scale=1,color='gold')
plt.show()

## 3. Vector Operations ##

# This code draws the x and y axis as lines.
plt.axhline(0, c='black', lw=0.5)
plt.axvline(0, c='black', lw=0.5)
plt.xlim(-4,4)
plt.ylim(-1,4)

plt.quiver(0,0,3,0,angles='xy',scale_units='xy',scale=1,color = 'green')
plt.quiver(3,0,0,3,angles='xy',scale_units='xy',scale=1,color='green')
plt.quiver(0,0,3,3,angles='xy',scale_units='xy',scale=1,color='green')
plt.show()

## 4. Scaling Vectors ##

# This code draws the x and y axis as lines.
plt.axhline(0, c='black', lw=0.5)
plt.axvline(0, c='black', lw=0.5)
plt.xlim(0,10)
plt.ylim(0,5)

plt.quiver(0,0,3,1,color = 'b',scale = 1)
plt.quiver(0,0,6,2,color = 'g',scale = 1)
plt.quiver(0,0,9,3,color = 'orange',scale = 1)
plt.show()

## 5. Vectors In NumPy ##

import numpy as np

vector_one = np.asarray([
    [1],
    [2],
    [1]
], dtype=np.float32)
vector_two = np.asarray([
                    [3],
                    [0],
                    [1]
],dtype = np.float32)
vector_linear_combination = 2*vector_one + 5*vector_two

## 6. Dot Product ##

vector_one = np.asarray([
    [1],
    [2],
    [1]
], dtype=np.float32)

vector_two = np.asarray([
    [3],
    [0],
    [1]
], dtype=np.float32)

dot_product = np.dot(vector_one[:,0], vector_two)
print(dot_product)

## 7. Linear Combination ##

w = np.asarray([
    [1],
    [2]
])
v = np.asarray([
    [3],
    [1]
])
end_point = 2*v - 2*w
