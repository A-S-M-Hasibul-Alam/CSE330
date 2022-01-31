import numpy as np
import matplotlib.pyplot as plt

class Newtons_Divided_Differences:
  def __init__(self, differences):
    self.differences = differences
    
  def __call__(self, x):
    """
    this function is for calculating y from given x using all the difference coefficients
    x can be a single value or a numpy
    the formula being used:
    f(x) = f [x0] + (x-x0) f[x0,x1] + (x-x0) (x-x1) f[x0,x1,x2] + . . . + (x-x0) (x-x1) . . . (x-xk-1) f[x0, x1, . . ., xk]
    
    work on this after implementing 'calc_div_diff'. Then you should have
    f[x0], f[x0,x1]. . . . . ., f[x0, x1, . . ., xk] stored in self.differences

    'res' variable must return all the results (corresponding y for x)
    """
    def minterms(i,val,p):  
      p1=1 
      for j in range(0,i):  
        p1=p1*(val - p[j])  
      return p1
    k=0
    def calculate(val): 
      sum1=differences[0] 
      for i in range(1,len(data_x)): 
        sum1= sum1+(minterms(i, val, data_x) * differences[i])    
      new=sum1
      return new
    res = np.zeros(len(x)) #Initialization to avoid runtime error. You can change this line if you wish
    # place your code here!!!!!!!!!!!!!!!!!!!!!!!
    for i in x:
      res[k]=calculate(i)
      k+=1   
    return res

#basic rule for calculating the difference, implanted in the lambda function. You may use it if you wish
difference = lambda y2, y1, x2, x1: (y2-y1)/(x2-x1)


def calc_div_diff(x,y):
  assert(len(x)==len(y))
  #write this function to calculate all the divided differences in the list 'b'
  b = []
  ln=len(x);
  matrix = np.zeros([ln, ln])
  matrix[::,0] =y 
  for j in range(1,ln):
    for i in range(ln-j):
      matrix[i][j] = (matrix[i+1][j-1] - matrix[i][j-1]) / (x[i+j] - x[i])
  for i in range(0, ln): 
      b.append(matrix[0][i]);   
  return b


data_x = np.array([-3.,-2.,-1.,0.,1.,3.,4.])
data_y = np.array([-60.,-80.,6.,1.,45.,30.,16.])
differences = calc_div_diff(list(data_x), list(data_y))
obj = Newtons_Divided_Differences(list(differences))

#generating 50 points from -3 to 4 in order to create a smooth line
X = np.linspace(-3, 4, 50, endpoint=True)
F = obj(X)
plt.plot(X,F)
plt.plot(data_x, data_y, 'ro')
plt.show()
