# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 11:20:23 2019

@author: Armaan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial

#===============================================
def dh(f, h, x):
    '''
    Input:
        f: np.polynomial.Polynonimial type data. 
        h: floating point data.
        x: np.array type data.
    Output:
        return np.array type data of slope at each point x.
        
    '''
    m=f(x+h)
    n=f(x-h)
    central=(m-n)/(2*h)
    return central
    # return <write your code here>


def dh1(f, h, x):
    '''
    Input:
        f: np.polynomial.Polynonimial type data. 
        h: floating point data.
        x: np.array type data.
    Output:
        return np.array type data of slope at each point x.
    '''

    m=dh(f, h/2, x)
    n=dh(f, h, x)
    
    final=(4*m-n)/3
    
    return final


    
def error(f, hs, x_i):
    '''
    Input:
        f  : np.polynomial.Polynonimial type data. 
        hs : np.array type data. list of h.
        x_i: floating point data. single value of x.
    Output:
        return two np.array type data of errors by two methods..
    '''
    f_prime = f.deriv(1)
    Y_actual = f_prime(x_i)
    
    print(Y_actual-dh1(f,hs,x_i))

    diff_error = [0.0] * len(hs)
    diff2_error = [0.0] * len(hs)
    
    diff_error=Y_actual-dh(f,hs,x_i)
    diff2_error=Y_actual-dh1(f,hs,x_i)
    """ 
     for h in hs:
       
        m=Y_actual-dh(f,hs,x_i )
        diff_error.append(m)
        n=Y_actual-dh1(f,hs,x_i)
        diff2_error.append(n)
    """
    #for h in hs:
    #    diff_error.insert(h,Y_actual-dh(f,hs[h],x_i))
    #   diff2_error.insert(h,Y_actual-dh1(f,hs[h],x_i))
        
        # for each values of hs calculate the error using both methods
        # and append those values into diff_error and diff2_error list.

        # write your code here
    print(pd.DataFrame({"h": hs, "Diff": diff_error, "Diff2": diff2_error}))
    
      
    return diff_error,diff2_error

      
    
    print(pd.DataFrame({"h": hs, "Diff": diff_error, "Diff2": diff2_error}))

    return diff_error, diff2_error

#========================TESTER====================================
def draw_graph(f, ax, domain=[-10, 10], label=None):
    data = f.linspace(domain=domain)
    ax.plot(data[0], data[1], label='Function')

########
# Draw the graph and it's derivative
#######
fig, ax = plt.subplots()
ax.axhline(y=0, color='k')

p = Polynomial([2.0, 1.0, -6.0, -2.0, 2.5, 1.0])
p_prime = p.deriv(1)
draw_graph(p, ax, [-2.4, 1.5], 'Function')
draw_graph(p_prime, ax, [-2.4, 1.5], 'Derivative')

ax.legend()

########
# Draw the actual derivative and richardson derivative using
# h=1 and h=-0.1 as step size.
#######
fig, ax = plt.subplots()
ax.axhline(y=0, color='k')

draw_graph(p_prime, ax, [-2.4, 1.5], 'actual')

h = 1
x = np.linspace(-2.4, 1.5, 50, endpoint=True)
y = dh1(p, h, x)
ax.plot(x, y, label='Richardson; h=1')

h = 0.1
x = np.linspace(-2.4, 1.5, 50, endpoint=True)
y = dh1(p, h, x)
ax.plot(x, y, label='Richardson; h=0.1')

ax.legend()

########
# Draw h-vs-error cuve
#######
fig, ax = plt.subplots()
ax.axhline(y=0, color='k')
hs = np.array([1., 0.55, 0.3, .17, 0.1, 0.055, 0.03, 0.017, 0.01])
e1, e2 = error(p, hs, 2.0)
ax.plot(hs, e1, label='e1')
ax.plot(hs, e2, label='e2')

ax.legend()


