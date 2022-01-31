# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 02:10:32 2019

@author: Armaan
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations 
from numpy.polynomial import Polynomial

def l(k, x):
    n = len(x)
    assert (k < len(x))
    
    x_k = x[k]
    x_copy = np.delete(x, k)
    
    denominator = np.prod(x_copy - x_k)
    
    coeff = []
    
    for i in range(n):
        coeff.append(sum([np.prod(x) for x in combinations(x_copy, i)]) * (-1)**(i) / denominator)
    
    coeff.reverse()
    
    return Polynomial(coeff)

def h(k, x):
    l_k = l(k,x)
    l_k_sqr = l_k*l_k
    l_k_prime = l_k.deriv(1)
   

    coeff =[1+2*l_k_prime(x[k])*x[k]  ,  -2*l_k_prime(x[k])]
    p = Polynomial(coeff)
    
    return p*l_k_sqr

def h_hat(k, x):
    l_k = l(k,x)
    l_k_sqr = l_k*l_k

    coeff = np.array([-x[k],1])
    p =Polynomial(coeff)
    return p*l_k_sqr

def hermit(x, y, y_prime):
    assert( len(x) == len(y))
    assert( len(y) == len(y_prime))
    
    f = Polynomial([0.0])
    for i in range(len(x)):
        f+=y[i]*h(i,x)+y_prime[i]*h_hat(i,x)
        # f += ?
        
    return f




##############################################TEST##########################################
pi = np.pi

x       = np.array([0.0, pi/2.0,  pi, 3.0*pi/2.0])
y       = np.array([0.0,    1.0, 0.0,       -1.0])
y_prime = np.array([1.0,    0.0, 1.0,        0.0])


n = 1
f3     = hermit(x[:(n+1)], y[:(n+1)], y_prime[:(n+1)])
data   = f3.linspace(n=50, domain=[-3, 3])
test_x = np.linspace(-3, 3, 50, endpoint=True)
test_y = np.sin(test_x)

plt.plot(data[0], data[1])
plt.plot(test_x, test_y)
plt.show()

n = 2
f5     = hermit(x[:(n+1)], y[:(n+1)], y_prime[:(n+1)])
data   = f5.linspace(n=50, domain=[-0.7, 3])
test_x = np.linspace(-2*pi, 2*pi, 50, endpoint=True)
test_y = np.sin(test_x)

plt.plot(test_x, test_y)
plt.plot(data[0], data[1])
plt.show()

n = 3
f7     = hermit(x[:(n+1)], y[:(n+1)], y_prime[:(n+1)])
data   = f7.linspace(n=50, domain=[-0.3, 3])
test_x = np.linspace(-2*pi, 2*pi, 50, endpoint=True)
test_y = np.sin(test_x)

plt.plot(data[0], data[1])
plt.plot(test_x, test_y)
plt.show()


x       = np.array([0.0, 1.0,          2.0       ])
y       = np.array([1.0, 2.71828183,  54.59815003])
y_prime = np.array([0.0, 5.43656366, 218.39260013])
f7      = hermit( x, y, y_prime)
data    = f7.linspace(n=50, domain=[-0.5, 2.2])
test_x  = np.linspace(-0.5, 2.2, 50, endpoint=True)
test_y  = np.exp(test_x**2)

plt.plot(data[0], data[1])
plt.plot(test_x, test_y)
plt.show()

x       = np.array([1.0, 3.0, 5.0])
y       = np.array([5.0, 1.0, 5.0])
y_prime = np.array([-4.0, 0.0, 4.0])
f7      = hermit( x, y, y_prime)
data    = f7.linspace(n=50, domain=[-10, 10])
test_x  = np.linspace(-10, 10, 50, endpoint=True)
test_y  = (test_x-3)**2 + 1

plt.plot(data[0], data[1])
plt.plot(test_x, test_y)
plt.show()