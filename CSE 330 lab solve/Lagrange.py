# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

class Lagrange_polynomial:
  def lag(x,y,num):
    """
    'num' is a set of given points, use x and y to figure out lagrange
    polynomial and calculate all the corresponding values for num

    Implement as you wish but your 'total' numpy array
    has to return all the results 
    """
    
    assert(len(x)==len(y))
    total  = np.zeros(len(num))

    #place your code here!!!!!!!!!!!!!!!!!!!!!!!!!
    sum = 1 
    val=np.array([])
    for z in range (0 , len(num)) :
        for i in range(0, len(x)):
            for j in range(0,len(x)):
                if (j!=i) :
                    sum*= (num[z]-x[j])/(x[i]-x[j])
            sum*= y[i] 
            val = np.append(val ,sum)
            sum=1
        total[z]=val.sum()
        val=np.array([])
        
    return total
 

data_x = np.array([-3.,-2.,-1.,0.,1.,3.,4.])
data_y = np.array([-60.,-80.,6.,1.,45.,30.,16.])

#generating 50 points from -3 to 4 in order to create a smooth line
X = np.linspace(-3, 4, 50, endpoint=True)
F = Lagrange_polynomial.lag(data_x, data_y, X)
plt.plot(X,F)
plt.plot(data_x, data_y, 'ro')
plt.show()


